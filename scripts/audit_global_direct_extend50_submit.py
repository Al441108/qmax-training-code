#!/usr/bin/env python3
from __future__ import annotations

"""CPU-only, non-mutating gate for Global Direct epoch-15 to epoch-50 jobs."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import train_m2_prnf_global_direct_extend50 as trainer  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def require_file(path: Path, errors: list[str]) -> Path:
    path = path.resolve()
    if not path.is_file():
        errors.append(f"Missing required file: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot_dir",
        default="outputs/m2_prnf_fusion/pilot15_R8_bs4/global_direct_seed42",
    )
    parser.add_argument(
        "--output_dir",
        default=(
            "outputs/m2_prnf/formal_R8_bs4/"
            "global_direct_prnf_no_need_seed42_ep50"
        ),
    )
    parser.add_argument(
        "--metadata_csv",
        default=(
            "/lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/"
            "multicoil_lesion_split/metadata/"
            "reorganised_dataset_split_isambard.csv"
        ),
    )
    parser.add_argument(
        "--manifest_root",
        default="outputs/m2_prnf/preflight_R8_bs4",
    )
    parser.add_argument(
        "--condition_manifest",
        default=(
            "outputs/m2_prnf_fusion/preflight_R8_bs4/"
            "condition_manifest.json"
        ),
    )
    parser.add_argument(
        "--pilot_decision",
        default=(
            "outputs/evaluation/"
            "m2_prnf_fusion_pilot15_R8_bs4_seed42_evalfix1/"
            "fusion_pilot_decision.json"
        ),
    )
    parser.add_argument(
        "--pilot_audit",
        default=(
            "outputs/evaluation/"
            "m2_prnf_fusion_pilot15_R8_bs4_seed42_evalfix1/"
            "fusion_pilot_evaluation_audit.json"
        ),
    )
    parser.add_argument(
        "--slurm",
        default="slurm/continue_m2_prnf_global_direct_R8_ep15_to_ep50.slurm",
    )
    args = parser.parse_args()

    errors: list[str] = []
    pilot_dir = Path(args.pilot_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    trainer_path = Path(trainer.__file__).resolve()
    slurm_path = require_file(Path(args.slurm), errors)

    required_inputs = [
        Path(args.metadata_csv),
        Path(args.manifest_root) / "full_clean_manifest.json",
        Path(args.manifest_root) / "robustness_manifest.json",
        Path(args.condition_manifest),
        Path(args.pilot_decision),
        Path(args.pilot_audit),
        pilot_dir / "model_last.pt",
        pilot_dir / "model_best.pt",
        pilot_dir / "config.json",
        pilot_dir / "training_log.csv",
        pilot_dir / "train_patient_ids.txt",
        pilot_dir / "val_patient_ids.txt",
        pilot_dir / "run_corruption_audit.json",
    ]
    for path in required_inputs:
        require_file(path, errors)

    if errors:
        print(json.dumps({"status": "failed", "errors": errors}, indent=2))
        raise SystemExit(1)

    # Compile the trainer source without launching a training graph.
    compile(trainer_path.read_text(encoding="utf-8"), str(trainer_path), "exec")

    slurm_text = slurm_path.read_text(encoding="utf-8")
    if not slurm_text.startswith("#!/bin/bash"):
        errors.append("SLURM script does not start with #!/bin/bash")
    if "set -euo pipefail" not in slurm_text:
        errors.append("SLURM script is missing set -euo pipefail")
    required_slurm_tokens = [
        "--run_stage pilot_extension_15_to_50",
        "--fusion_design global_direct",
        "--variant prnf_no_need",
        "--epochs 50",
        "--batch_size 4",
        "--grad_accum_steps 1",
        '--resume "$OUTPUT_DIR/model_last.pt"',
    ]
    for token in required_slurm_tokens:
        if token not in slurm_text:
            errors.append(f"SLURM is missing locked token: {token}")

    source_checkpoint = torch.load(
        pilot_dir / "model_last.pt", map_location="cpu", weights_only=False
    )
    source_config = source_checkpoint.get("config", {})
    expected_source = {
        "variant": "prnf_no_need",
        "fusion_design": "global_direct",
        "need_scope": "residual",
        "run_stage": "pilot",
        "epochs": 15,
        "seed": 42,
        "acceleration": 8,
        "pd_aux_acceleration": 2,
        "batch_size": 4,
        "grad_accum_steps": 1,
    }
    observed_source = {
        key: source_config.get(key) for key in expected_source
    }
    if observed_source != expected_source:
        errors.append(
            "Pilot checkpoint identity mismatch: "
            + json.dumps(
                {"expected": expected_source, "observed": observed_source}
            )
        )

    checkpoint_requirements = {
        "epoch": 15,
        "sampler_next_epoch": 15,
    }
    for key, expected in checkpoint_requirements.items():
        if int(source_checkpoint.get(key, -1)) != expected:
            errors.append(
                f"Pilot checkpoint {key}={source_checkpoint.get(key)!r}, "
                f"expected {expected}"
            )
    if len(source_checkpoint.get("history", [])) != 15:
        errors.append("Pilot checkpoint does not contain 15 history rows")
    for key in (
        "model_state_dict",
        "optimizer_state_dict",
        "rng_state",
        "run_corruption_audit",
        "code_hashes",
    ):
        if key not in source_checkpoint:
            errors.append(f"Pilot checkpoint is missing {key}")

    with (pilot_dir / "training_log.csv").open(
        newline="", encoding="utf-8"
    ) as file:
        log_rows = list(csv.DictReader(file))
    if len(log_rows) != 15 or int(log_rows[-1]["epoch"]) != 15:
        errors.append("Pilot training log is not complete through epoch 15")

    # This exactly reproduces the transition semantics and the JSON round trip
    # which caused the historical tuple/list failure.
    current_config = dict(source_config)
    current_config["run_stage"] = "pilot_extension_15_to_50"
    current_config["epochs"] = 50
    try:
        trainer.validate_resume_config(source_config, current_config)
        trainer.validate_resume_config(
            json.loads(json.dumps(source_config)), current_config
        )
    except Exception as exc:
        errors.append(f"Semantic resume comparison failed: {exc}")

    if canonical(json.loads(json.dumps(current_config))) != canonical(
        current_config
    ):
        errors.append("Extension config is not stable under JSON round trip")

    installed_hashes = trainer.locked_code_hashes()
    if source_checkpoint.get("code_hashes") != installed_hashes:
        errors.append("Pilot checkpoint code hashes differ from installed code")

    # Strict architecture and optimizer-state compatibility, CPU only.
    try:
        model = trainer.M2PRNFFusionPilotVarNet(
            model_variant=source_config["variant"],
            fusion_design=source_config["fusion_design"],
            need_scope=source_config["need_scope"],
            residual_scale=source_config["residual_scale"],
            num_cascades=source_config["num_cascades"],
            sens_chans=source_config["sens_chans"],
            sens_pools=source_config["sens_pools"],
            chans=source_config["chans"],
            pools=source_config["pools"],
            controller_chans=source_config["controller_chans"],
            initial_aux_alpha=source_config["initial_aux_alpha"],
            initial_gate_probability=source_config[
                "initial_gate_probability"
            ],
            initial_need_probability=source_config[
                "initial_need_probability"
            ],
            need_floor=source_config["need_floor"],
        )
        model.load_state_dict(
            source_checkpoint["model_state_dict"], strict=True
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(source_config["learning_rate"])
        )
        optimizer.load_state_dict(
            source_checkpoint["optimizer_state_dict"]
        )
        del optimizer, model
    except Exception as exc:
        errors.append(f"Model/optimizer strict-load audit failed: {exc}")

    output_state: dict[str, Any]
    if output_dir.exists():
        provenance_path = output_dir / "extension_provenance.json"
        checkpoint_path = output_dir / "model_last.pt"
        if not provenance_path.is_file() or not checkpoint_path.is_file():
            errors.append(
                "Existing output directory is incomplete or unprovenanced"
            )
            output_state = {"exists": True, "valid": False}
        else:
            provenance = json.loads(
                provenance_path.read_text(encoding="utf-8")
            )
            recorded_hash = provenance.get(
                "continuation_trainer_sha256"
            )
            current_hash = sha256_file(trainer_path)
            if recorded_hash != current_hash:
                errors.append(
                    "Existing continuation provenance uses a different "
                    "trainer SHA; archive and reinitialise the output directory"
                )
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            output_state = {
                "exists": True,
                "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
                "next_epoch": int(checkpoint.get("epoch", -1)) + 1,
                "recorded_trainer_sha256": recorded_hash,
                "current_trainer_sha256": current_hash,
            }
    else:
        output_state = {
            "exists": False,
            "checkpoint_epoch": 15,
            "next_epoch": 16,
            "action": "preparer will initialise from locked pilot",
        }

    result = {
        "status": "passed" if not errors else "failed",
        "trainer": str(trainer_path),
        "trainer_sha256": sha256_file(trainer_path),
        "pilot_checkpoint_epoch": int(source_checkpoint.get("epoch", -1)),
        "expected_next_epoch": output_state.get("next_epoch", 16),
        "output_state": output_state,
        "json_roundtrip_semantics": "passed" if not any(
            "Semantic resume" in error or "JSON round trip" in error
            for error in errors
        ) else "failed",
        "strict_model_optimizer_load": "passed" if not any(
            "strict-load" in error for error in errors
        ) else "failed",
        "errors": errors,
    }
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
