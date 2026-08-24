#!/usr/bin/env python3
from __future__ import annotations

"""Evaluate one locked epoch-30 QMax arm with a four-arm common schema."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_qmax_counterfactuals import (  # noqa: E402
    CONDITION_MANIFEST_PROTOCOL_VERSION,
    CONDITIONS,
    MANIFEST_PROTOCOL_VERSION,
    ROBUST_CONDITIONS,
    ManifestDataset,
    add_robustness_composite,
    evaluate_mode,
    patient_rows,
    summaries,
)
from scripts.qmax_common import (  # noqa: E402
    IndexedDataset,
    ShapeBucketBatchSampler,
    code_hashes,
    make_dataset,
    set_seed,
    sha256_file,
    write_csv,
)
from scripts.qmax_stage_b_versioning import (  # noqa: E402
    manifest_digest,
    stage_b_structure_hashes,
)
from src.m2_prnf_qmax_compactswin_varnet import (  # noqa: E402
    QMaxCompactSwinAuxPDVarNet,
)
from src.m2_prnf_qmax_varnet import QMaxAuxPDVarNet  # noqa: E402


PROTOCOL_VERSION = "QMax-epoch30-four-arm-evaluator-v1"
ARM_SPECS: Dict[str, Dict[str, str]] = {
    "stagea_core": {
        "stage": "stage_a",
        "qmax_variant": "qmax_core",
        "backbone_variant": "convolutional",
    },
    "stagea_full": {
        "stage": "stage_a",
        "qmax_variant": "qmax_full",
        "backbone_variant": "convolutional",
    },
    "stageb_core": {
        "stage": "stage_b",
        "qmax_variant": "qmax_core",
        "backbone_variant": "compactswin",
    },
    "stageb_full": {
        "stage": "stage_b",
        "qmax_variant": "qmax_full",
        "backbone_variant": "compactswin",
    },
}


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_checkpoint_location(path: Path) -> None:
    if path.name != "model_last.pt" or path.parent.name != "epoch30":
        raise RuntimeError(
            "Evaluation accepts only epoch30/model_last.pt, not a top-level "
            "model_last.pt or model_best checkpoint"
        )


def _validate_input_hashes(
    config: Mapping[str, Any], paths: Mapping[str, Path]
) -> None:
    expected = {
        "metadata_csv": config.get("metadata_sha256"),
        "full_clean_manifest": config.get(
            "full_clean_manifest_sha256"
        ),
        "robustness_manifest": config.get(
            "robustness_manifest_sha256"
        ),
        "condition_manifest": config.get(
            "condition_manifest_sha256"
        ),
    }
    missing = [key for key, value in expected.items() if value is None]
    if missing:
        raise RuntimeError(
            f"Checkpoint config lacks locked input hashes: {missing}"
        )
    mismatches = {
        key: {
            "checkpoint": expected[key],
            "installed": sha256_file(paths[key]),
        }
        for key in expected
        if expected[key] != sha256_file(paths[key])
    }
    if mismatches:
        raise RuntimeError(
            "Locked evaluator inputs differ from training:\n"
            + json.dumps(mismatches, indent=2)
        )


def _build_model(
    *,
    arm: str,
    checkpoint: Mapping[str, Any],
) -> torch.nn.Module:
    spec = ARM_SPECS[arm]
    config = checkpoint["config"]
    observed_variant = str(config.get("qmax_variant"))
    if observed_variant != spec["qmax_variant"]:
        raise RuntimeError(
            f"{arm} expected {spec['qmax_variant']}, got {observed_variant}"
        )
    kwargs = dict(config["model_kwargs"])
    declared_backbone = kwargs.pop(
        "backbone_variant", config.get("backbone_variant")
    )
    if spec["stage"] == "stage_a":
        if declared_backbone not in (None, "convolutional"):
            raise RuntimeError(
                f"{arm} is not a convolutional Stage-A checkpoint"
            )
        installed_hashes = code_hashes(PROJECT_ROOT)
        if config.get("code_hashes") != installed_hashes:
            raise RuntimeError(
                "Installed Stage-A scientific code differs from checkpoint"
            )
        model = QMaxAuxPDVarNet(
            qmax_variant=observed_variant, **kwargs
        )
    else:
        if declared_backbone != "compactswin":
            raise RuntimeError(
                f"{arm} is not a CompactSwin Stage-B checkpoint"
            )
        installed_hashes = stage_b_structure_hashes(PROJECT_ROOT)
        checkpoint_hashes = config.get(
            "structure_hashes", config.get("code_hashes")
        )
        if checkpoint_hashes != installed_hashes:
            raise RuntimeError(
                "Installed Stage-B scientific structure differs from "
                "checkpoint"
            )
        checkpoint_digest = config.get("structure_digest")
        if (
            checkpoint_digest is not None
            and checkpoint_digest != manifest_digest(installed_hashes)
        ):
            raise RuntimeError("Stage-B structure digest mismatch")
        model = QMaxCompactSwinAuxPDVarNet(
            qmax_variant=observed_variant, **kwargs
        )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model


def _enrich(
    rows: list[dict[str, Any]],
    *,
    arm: str,
    checkpoint_sha256: str,
) -> None:
    spec = ARM_SPECS[arm]
    for row in rows:
        row.update(
            {
                "arm": arm,
                "stage": spec["stage"],
                "qmax_variant": spec["qmax_variant"],
                "backbone_variant": spec["backbone_variant"],
                "checkpoint_epoch": 30,
                "checkpoint_sha256": checkpoint_sha256,
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=sorted(ARM_SPECS))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--full_clean_manifest", required=True)
    parser.add_argument("--robustness_manifest", required=True)
    parser.add_argument("--condition_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    if args.seed != 42:
        raise ValueError("Locked four-arm evaluation requires seed=42")
    if not args.amp:
        raise ValueError("Locked four-arm evaluation requires AMP")
    if not torch.cuda.is_available():
        raise RuntimeError("Epoch-30 evaluation requires CUDA")
    device = torch.device("cuda")
    set_seed(args.seed)

    paths: Dict[str, Path] = {}
    for name in (
        "checkpoint",
        "metadata_csv",
        "full_clean_manifest",
        "robustness_manifest",
        "condition_manifest",
    ):
        path = Path(getattr(args, name)).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        paths[name] = path
    _validate_checkpoint_location(paths["checkpoint"])

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "slice": output_dir / "epoch30_slice_metrics.csv",
        "patient": output_dir / "epoch30_patient_metrics.csv",
        "summary": output_dir / "epoch30_summary.csv",
        "scale": output_dir / "epoch30_scale_cascade_diagnostics.csv",
        "audit": output_dir / "epoch30_evaluation_audit.json",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise RuntimeError(
            "Refusing to overwrite an existing evaluation: "
            + ", ".join(existing)
        )

    checkpoint = torch.load(
        paths["checkpoint"], map_location="cpu", weights_only=False
    )
    required = {"epoch", "config", "model_state_dict"}
    missing = sorted(required - set(checkpoint))
    if missing:
        raise RuntimeError(f"Checkpoint missing keys: {missing}")
    if int(checkpoint["epoch"]) != 30:
        raise RuntimeError(
            f"Expected checkpoint epoch 30, got {checkpoint['epoch']}"
        )
    config = checkpoint["config"]
    _validate_input_hashes(config, paths)
    model = _build_model(arm=args.arm, checkpoint=checkpoint).to(device)
    model.eval()

    source = IndexedDataset(
        make_dataset(
            str(paths["metadata_csv"]),
            "val",
            acceleration=8,
            pd_aux_acceleration=2,
        )
    )
    full_clean_manifest = _load_json(paths["full_clean_manifest"])
    robustness_manifest = _load_json(paths["robustness_manifest"])
    for cohort, manifest in (
        ("full_clean", full_clean_manifest),
        ("robustness", robustness_manifest),
    ):
        if (
            manifest.get("protocol_version") != MANIFEST_PROTOCOL_VERSION
            or manifest.get("cohort") != cohort
        ):
            raise RuntimeError(
                f"{cohort} manifest protocol/cohort mismatch"
            )
    condition_manifest = _load_json(paths["condition_manifest"])
    if (
        condition_manifest.get("protocol_version")
        != CONDITION_MANIFEST_PROTOCOL_VERSION
        or int(condition_manifest.get("seed", -1)) != args.seed
    ):
        raise RuntimeError("Condition manifest protocol/seed mismatch")
    condition_lookup = {
        int(entry["source_index"]): entry
        for entry in condition_manifest["entries"]
    }
    if len(condition_lookup) != int(condition_manifest["num_entries"]):
        raise RuntimeError("Duplicate source index in condition manifest")

    clean_dataset = ManifestDataset(source, full_clean_manifest)
    robust_dataset = ManifestDataset(source, robustness_manifest)
    clean_loader = DataLoader(
        clean_dataset,
        batch_sampler=ShapeBucketBatchSampler(
            clean_dataset, args.batch_size, False, args.seed
        ),
        num_workers=args.num_workers,
        pin_memory=True,
    )
    robust_loader = DataLoader(
        robust_dataset,
        batch_sampler=ShapeBucketBatchSampler(
            robust_dataset, args.batch_size, False, args.seed
        ),
        num_workers=args.num_workers,
        pin_memory=True,
    )

    all_slice_rows: list[dict[str, Any]] = []
    all_scale_rows: list[dict[str, Any]] = []
    rows, scale_rows = evaluate_mode(
        model=model,
        loader=clean_loader,
        source_dataset=source,
        condition_lookup=condition_lookup,
        device=device,
        amp=args.amp,
        cohort="full_clean",
        condition="correct",
        mode="full",
        constant_q=None,
    )
    all_slice_rows.extend(rows)
    all_scale_rows.extend(scale_rows)
    for condition in CONDITIONS:
        rows, scale_rows = evaluate_mode(
            model=model,
            loader=robust_loader,
            source_dataset=source,
            condition_lookup=condition_lookup,
            device=device,
            amp=args.amp,
            cohort="robustness",
            condition=condition,
            mode="full",
            constant_q=None,
        )
        all_slice_rows.extend(rows)
        all_scale_rows.extend(scale_rows)

    patient_level = patient_rows(all_slice_rows)
    add_robustness_composite(patient_level)
    summary_level = summaries(patient_level)
    checkpoint_hash = sha256_file(paths["checkpoint"])
    _enrich(
        all_slice_rows,
        arm=args.arm,
        checkpoint_sha256=checkpoint_hash,
    )
    _enrich(
        patient_level,
        arm=args.arm,
        checkpoint_sha256=checkpoint_hash,
    )
    _enrich(
        summary_level,
        arm=args.arm,
        checkpoint_sha256=checkpoint_hash,
    )
    _enrich(
        all_scale_rows,
        arm=args.arm,
        checkpoint_sha256=checkpoint_hash,
    )

    write_csv(outputs["slice"], all_slice_rows)
    write_csv(outputs["patient"], patient_level)
    write_csv(outputs["summary"], summary_level)
    write_csv(outputs["scale"], all_scale_rows)

    missing_rows = [
        row
        for row in all_slice_rows
        if row["cohort"] == "robustness"
        and row["condition"] == "missing"
    ]
    missing_exact_zero = bool(missing_rows) and all(
        float(row["missing_direct_exact_zero"]) == 1.0
        and float(row["missing_correction_exact_zero"]) == 1.0
        for row in missing_rows
    )
    audit = {
        "protocol_version": PROTOCOL_VERSION,
        "arm": args.arm,
        **ARM_SPECS[args.arm],
        "checkpoint": str(paths["checkpoint"]),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_training_protocol": config.get("protocol_version"),
        "strict_state_dict_load": True,
        "installed_training_structure_validated": True,
        "evaluation_mode": "full",
        "conditions": list(CONDITIONS),
        "robustness_composite_conditions": list(ROBUST_CONDITIONS),
        "input_hashes": {
            key: sha256_file(value) for key, value in paths.items()
        },
        "num_slice_rows": len(all_slice_rows),
        "num_patient_rows": len(patient_level),
        "full_clean_num_patients": len(
            {
                row["patient_id"]
                for row in patient_level
                if row["cohort"] == "full_clean"
            }
        ),
        "robustness_num_patients": len(
            {
                row["patient_id"]
                for row in patient_level
                if row["cohort"] == "robustness"
            }
        ),
        "missing_exact_zero": missing_exact_zero,
        "outputs": {key: str(value) for key, value in outputs.items()},
        "status": "passed" if missing_exact_zero else "failed",
    }
    outputs["audit"].write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2), flush=True)
    if audit["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
