#!/usr/bin/env python3
from __future__ import annotations

"""CPU-only, non-mutating gate for the RPCRF H/4 formal-50 job."""

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

from scripts import train_m2_prnf_rpcrf as trainer  # noqa: E402


PROTOCOL = "M2-PRNF-R8-v1.7.3-RPCRF-H4-formal50-Isambard-audited"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, errors: list[str]) -> Path:
    path = path.resolve()
    if not path.is_file():
        errors.append(f"Missing required file: {path}")
    return path


def construct_model() -> torch.nn.Module:
    return trainer.M2PRNFRPCRFVarNet(
        model_variant="prnf_no_need",
        fusion_design="hybrid_direct_residual",
        need_scope="residual",
        residual_scale=0.1,
        residual_reliability_power=1.5,
        residual_levels=(2,),
        num_cascades=12,
        sens_chans=8,
        sens_pools=4,
        chans=18,
        pools=4,
        controller_chans=16,
        initial_aux_alpha=0.1,
        initial_gate_probability=0.95,
        initial_need_probability=0.95,
        need_floor=0.25,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--output_dir",
        default=(
            "outputs/m2_prnf_rpcrf_h4/formal_R8_bs4/"
            "rpcrf_h4_final_seed42_ep50"
        ),
    )
    parser.add_argument(
        "--slurm",
        default="slurm/train_m2_prnf_rpcrf_h4_R8_formal50.slurm",
    )
    parser.add_argument(
        "--protocol",
        default="RPCRF_FINAL_PROTOCOL_R8.json",
    )
    args = parser.parse_args()

    errors: list[str] = []
    output_dir = Path(args.output_dir).resolve()
    trainer_path = Path(trainer.__file__).resolve()
    slurm_path = require_file(Path(args.slurm), errors)
    protocol_path = require_file(Path(args.protocol), errors)

    required_inputs = [
        Path(args.metadata_csv),
        Path(args.manifest_root) / "full_clean_manifest.json",
        Path(args.manifest_root) / "robustness_manifest.json",
        Path(args.condition_manifest),
        PROJECT_ROOT / "src/m2_prnf_rpcrf_varnet.py",
        PROJECT_ROOT / "scripts/train_m2_prnf_rpcrf.py",
    ]
    for path in required_inputs:
        require_file(path, errors)

    if errors:
        print(json.dumps({"status": "failed", "errors": errors}, indent=2))
        raise SystemExit(1)

    compile(trainer_path.read_text(encoding="utf-8"), str(trainer_path), "exec")

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != PROTOCOL:
        errors.append("Unexpected RPCRF protocol version")
    formal = protocol.get("formal_training", {})
    expected_formal = {
        "acceleration": 8,
        "pd_aux_acceleration": 2,
        "epochs": 50,
        "seed": 42,
        "batch_size": 4,
        "gradient_accumulation": 1,
        "learning_rate": 0.0003,
        "random_initialisation": True,
    }
    for key, expected in expected_formal.items():
        if formal.get(key) != expected:
            errors.append(
                f"Protocol {key}={formal.get(key)!r}; expected {expected!r}"
            )

    slurm_text = slurm_path.read_text(encoding="utf-8")
    if not slurm_text.startswith("#!/bin/bash"):
        errors.append("SLURM does not start with #!/bin/bash")
    if "set -euo pipefail" not in slurm_text:
        errors.append("SLURM is missing set -euo pipefail")
    required_tokens = [
        "--variant prnf_no_need",
        "--fusion_design hybrid_direct_residual",
        "--run_stage rpcrf_h4_formal50",
        "--need_scope residual",
        "--residual_scale 0.1",
        "--residual_reliability_power 1.5",
        "--lambda_correction 0.05",
        "--lambda_overshoot 0.01",
        "--lambda_residual_gain 0.10",
        "--epochs 50",
        "--batch_size 4",
        "--grad_accum_steps 1",
        "--seed 42",
        'if [[ -f "$OUTPUT_DIR/model_last.pt" ]]',
        'RESUME_ARGS=(--resume "$OUTPUT_DIR/model_last.pt")',
    ]
    for token in required_tokens:
        if token not in slurm_text:
            errors.append(f"SLURM is missing locked token: {token}")

    # Regression test for the historical tuple-versus-list resume failure.
    previous = {
        key: None for key in trainer.IMMUTABLE_RESUME_KEYS
    }
    previous.update(
        {
            "corruption_config": {
                "shift_magnitudes": (2, 4, 8),
                "scale_relief": (0.0, 0.15, 0.3, 0.45),
            },
            "run_stage": "rpcrf_h4_formal50",
            "epochs": 50,
            "batch_size": 4,
            "grad_accum_steps": 1,
        }
    )
    json_roundtrip = json.loads(json.dumps(previous))
    try:
        trainer.validate_resume_config(previous, json_roundtrip)
        trainer.validate_resume_config(json_roundtrip, previous)
    except Exception as exc:
        errors.append(f"JSON round-trip resume regression failed: {exc}")

    try:
        installed_hashes = trainer.locked_code_hashes()
    except Exception as exc:
        installed_hashes = {}
        errors.append(f"Locked-code hash audit failed: {exc}")

    parameter_count = -1
    try:
        model = construct_model()
        parameter_count = sum(
            parameter.numel() for parameter in model.parameters()
        )
    except Exception as exc:
        model = None
        errors.append(f"CPU model-construction audit failed: {exc}")

    output_state: dict[str, Any]
    checkpoint_path = output_dir / "model_last.pt"
    config_path = output_dir / "config.json"
    if checkpoint_path.is_file():
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            required_checkpoint_keys = {
                "model_state_dict",
                "optimizer_state_dict",
                "config",
                "rng_state",
                "code_hashes",
                "epoch",
                "sampler_next_epoch",
                "history",
                "run_corruption_audit",
            }
            missing = sorted(required_checkpoint_keys - set(checkpoint))
            if missing:
                errors.append(f"Resume checkpoint is missing keys: {missing}")
            epoch = int(checkpoint.get("epoch", -1))
            if int(checkpoint.get("sampler_next_epoch", -2)) != epoch:
                errors.append("Checkpoint sampler_next_epoch is inconsistent")
            if len(checkpoint.get("history", [])) != epoch:
                errors.append("Checkpoint history length is inconsistent")
            if checkpoint.get("code_hashes") != installed_hashes:
                errors.append(
                    "Checkpoint code hashes differ from installed RPCRF code"
                )
            if not config_path.is_file():
                errors.append("Resume output has no config.json")
            else:
                installed_config = json.loads(
                    config_path.read_text(encoding="utf-8")
                )
                trainer.validate_resume_config(
                    checkpoint.get("config", {}), installed_config
                )
            if model is not None:
                model.load_state_dict(
                    checkpoint["model_state_dict"], strict=True
                )
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=float(
                        checkpoint.get("config", {}).get(
                            "learning_rate", 3e-4
                        )
                    ),
                )
                optimizer.load_state_dict(
                    checkpoint["optimizer_state_dict"]
                )
                del optimizer
            log_path = output_dir / "training_log.csv"
            if not log_path.is_file():
                errors.append("Resume output has no training_log.csv")
            else:
                with log_path.open(
                    newline="", encoding="utf-8"
                ) as file:
                    rows = list(csv.DictReader(file))
                if len(rows) != epoch:
                    errors.append(
                        "training_log row count differs from checkpoint epoch"
                    )
            output_state = {
                "mode": "resume",
                "checkpoint_epoch": epoch,
                "next_epoch": epoch + 1,
            }
        except Exception as exc:
            errors.append(f"Full resume audit failed: {exc}")
            output_state = {"mode": "resume", "valid": False}
    else:
        existing_files = (
            sorted(
                str(path.relative_to(output_dir))
                for path in output_dir.rglob("*")
                if path.is_file()
            )
            if output_dir.is_dir()
            else []
        )
        if existing_files:
            errors.append(
                "Output directory has files but no model_last.pt: "
                + json.dumps(existing_files[:20])
            )
        output_state = {
            "mode": "fresh",
            "start_epoch": 1,
            "next_epoch": 1,
            "output_directory_exists": output_dir.exists(),
        }

    del model
    result = {
        "status": "passed" if not errors else "failed",
        "protocol_version": protocol.get("protocol_version"),
        "trainer": str(trainer_path),
        "trainer_sha256": sha256_file(trainer_path),
        "model_source_sha256": sha256_file(
            PROJECT_ROOT / "src/m2_prnf_rpcrf_varnet.py"
        ),
        "parameter_count": parameter_count,
        "json_roundtrip_resume": "passed" if not any(
            "JSON round-trip" in error for error in errors
        ) else "failed",
        "locked_code_file_count": len(installed_hashes),
        "output_state": output_state,
        "slurm_walltime": next(
            (
                line.split("=", 1)[1]
                for line in slurm_text.splitlines()
                if line.startswith("#SBATCH --time=")
            ),
            None,
        ),
        "errors": errors,
    }
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
