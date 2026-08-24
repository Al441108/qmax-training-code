#!/usr/bin/env python3
from __future__ import annotations

"""Locked validation evaluation for StageA-Full epoch60/model_last.pt.

This is an additive runtime tool.  It deliberately imports the original,
checkpoint-bound Stage-A evaluation implementation instead of modifying it.
It evaluates only the formal validation evidence required immediately after
training: actual q, q=1, the definition-fixed constant q, and the missing-PD
condition.  Component counterfactuals and held-out test evaluation remain
separate tasks.
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compare_qmax_stage_a import q_functionality_gate  # noqa: E402
from scripts.evaluate_qmax_counterfactuals import (  # noqa: E402
    CONDITION_MANIFEST_PROTOCOL_VERSION,
    CONDITIONS,
    MANIFEST_PROTOCOL_VERSION,
    METRICS,
    ROBUST_CONDITIONS,
    ManifestDataset,
    add_robustness_composite,
    evaluate_mode,
    paired_bootstrap,
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
from src.m2_prnf_qmax_varnet import QMaxAuxPDVarNet  # noqa: E402


PROTOCOL_VERSION = "StageA-Full-epoch60-locked-validation-v1"
MODES = ("full", "q1", "constant_q")
MODE_LABELS = {
    "full": "actual-q",
    "q1": "q=1",
    "constant_q": "constant-q",
}
EXPECTED_LR_SCHEDULE = {
    "epochs_1_40": 3e-4,
    "epochs_41_50": 1e-4,
    "epochs_51_60": 3e-5,
}
COMPARISON_ENDPOINTS = (
    ("full_clean", "correct"),
    ("robustness", "correct"),
    ("robustness", "shift8"),
    ("robustness", "wrong_slice"),
    ("robustness", "wrong_patient"),
    ("robustness", "robustness_composite"),
)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_checkpoint_location(path: Path) -> None:
    if path.name != "model_last.pt" or path.parent.name != "epoch60":
        raise RuntimeError(
            "Formal validation accepts only epoch60/model_last.pt; "
            "model_best and earlier model_last checkpoints are forbidden"
        )


def _validate_input_hashes(
    config: Mapping[str, Any], paths: Mapping[str, Path]
) -> None:
    expected = {
        "metadata_csv": config.get("metadata_sha256"),
        "full_clean_manifest": config.get("full_clean_manifest_sha256"),
        "robustness_manifest": config.get("robustness_manifest_sha256"),
        "condition_manifest": config.get("condition_manifest_sha256"),
    }
    missing = sorted(key for key, value in expected.items() if value is None)
    if missing:
        raise RuntimeError(
            f"Checkpoint config lacks locked input hashes: {missing}"
        )
    mismatches = {}
    for key, expected_hash in expected.items():
        observed = sha256_file(paths[key])
        if observed != expected_hash:
            mismatches[key] = {
                "checkpoint": expected_hash,
                "installed": observed,
            }
    if mismatches:
        raise RuntimeError(
            "Locked validation inputs differ from training:\n"
            + json.dumps(mismatches, indent=2)
        )


def _validate_training_log(run_dir: Path) -> Dict[str, Any]:
    log_path = run_dir / "training_log.csv"
    if not log_path.is_file():
        raise RuntimeError(f"Training log is missing: {log_path}")
    with log_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    epochs = [int(row["epoch"]) for row in rows]
    if epochs != list(range(1, 61)):
        raise RuntimeError(
            "training_log.csv must contain exactly epochs 1..60; "
            f"observed={epochs[:5]}...{epochs[-5:] if epochs else []}"
        )
    finite_fields = (
        "train_total_loss",
        "train_recon_loss",
        "train_clean_l1",
        "train_corrupt_l1",
        "val_patient_l1",
    )
    nonfinite = []
    for row in rows:
        for field in finite_fields:
            if field not in row or not math.isfinite(float(row[field])):
                nonfinite.append({"epoch": int(row["epoch"]), "field": field})
    if nonfinite:
        raise RuntimeError(
            "Non-finite required training-log values: "
            + json.dumps(nonfinite[:20], indent=2)
        )
    return {
        "path": str(log_path),
        "sha256": sha256_file(log_path),
        "num_rows": len(rows),
        "epochs_exactly_1_to_60": True,
        "required_loss_fields_finite": True,
    }


def _validate_checkpoint(
    checkpoint_path: Path,
    checkpoint: Mapping[str, Any],
    installed_code_hashes: Mapping[str, str],
) -> Dict[str, Any]:
    required = {
        "epoch",
        "config",
        "model_state_dict",
        "optimizer_state_dict",
        "grad_scaler_state_dict",
        "rng_state",
        "history",
        "code_hashes",
        "run_corruption_audit",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise RuntimeError(f"Epoch-60 checkpoint missing keys: {missing}")
    if int(checkpoint["epoch"]) != 60:
        raise RuntimeError(
            f"Expected checkpoint epoch 60, got {checkpoint['epoch']}"
        )
    config = checkpoint["config"]
    if str(config.get("qmax_variant")) != "qmax_full":
        raise RuntimeError(
            "Formal StageA-Full evaluation requires qmax_variant=qmax_full"
        )
    if config.get("formal_structure_selection_checkpoint") != (
        "epoch60/model_last.pt"
    ):
        raise RuntimeError("Checkpoint does not declare epoch60/model_last.pt")
    observed_schedule = config.get("learning_rate_schedule")
    if observed_schedule != EXPECTED_LR_SCHEDULE:
        raise RuntimeError(
            "Learning-rate schedule mismatch:\n"
            + json.dumps(
                {
                    "expected": EXPECTED_LR_SCHEDULE,
                    "observed": observed_schedule,
                },
                indent=2,
            )
        )
    history = list(checkpoint["history"])
    history_epochs = [int(row["epoch"]) for row in history]
    if history_epochs != list(range(1, 61)):
        raise RuntimeError("Checkpoint history is not exactly epochs 1..60")
    if checkpoint.get("code_hashes") != installed_code_hashes:
        raise RuntimeError("Checkpoint top-level scientific code hashes drifted")
    if config.get("code_hashes") != installed_code_hashes:
        raise RuntimeError("Checkpoint config scientific code hashes drifted")

    optimizer = checkpoint["optimizer_state_dict"]
    param_groups = list(optimizer.get("param_groups", []))
    if not param_groups:
        raise RuntimeError("Optimizer state contains no parameter groups")
    final_lrs = [float(group["lr"]) for group in param_groups]
    if any(not math.isclose(lr, 3e-5, rel_tol=0.0, abs_tol=1e-12) for lr in final_lrs):
        raise RuntimeError(f"Epoch-60 optimizer LR is not 3e-5: {final_lrs}")

    run_dir = checkpoint_path.parent.parent
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": 60,
        "qmax_variant": "qmax_full",
        "history_epochs_exactly_1_to_60": True,
        "learning_rate_schedule": observed_schedule,
        "optimizer_final_lrs": final_lrs,
        "optimizer_state_present": True,
        "grad_scaler_state_present": True,
        "rng_state_present": True,
        "corruption_state_present": True,
        "scientific_code_hashes_match": True,
        "training_log": _validate_training_log(run_dir),
    }


def _build_model(
    checkpoint: Mapping[str, Any], device: torch.device
) -> QMaxAuxPDVarNet:
    config = checkpoint["config"]
    model = QMaxAuxPDVarNet(
        qmax_variant="qmax_full", **dict(config["model_kwargs"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


def _enrich(
    rows: list[dict[str, Any]], checkpoint_sha256: str
) -> None:
    for row in rows:
        mode = str(row["mode"])
        row.update(
            {
                "arm": "stagea_full",
                "qmax_variant": "qmax_full",
                "backbone_variant": "convolutional",
                "checkpoint_epoch": 60,
                "checkpoint_sha256": checkpoint_sha256,
                "mode_label": MODE_LABELS[mode],
            }
        )


def _assert_finite_metrics(rows: list[dict[str, Any]]) -> None:
    failures = []
    for row in rows:
        for metric in METRICS:
            value = float(row[metric])
            if not math.isfinite(value):
                failures.append(
                    {
                        "mode": row.get("mode"),
                        "cohort": row.get("cohort"),
                        "condition": row.get("condition"),
                        "patient_id": row.get("patient_id"),
                        "metric": metric,
                    }
                )
    if failures:
        raise RuntimeError(
            "Non-finite validation metrics:\n"
            + json.dumps(failures[:20], indent=2)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--full_clean_manifest", required=True)
    parser.add_argument("--robustness_manifest", required=True)
    parser.add_argument("--condition_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--bootstrap_resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    if args.seed != 42:
        raise ValueError("Locked StageA-Full evaluation requires seed=42")
    if not args.amp:
        raise ValueError("Locked StageA-Full evaluation requires AMP")
    if args.bootstrap_resamples < 1000:
        raise ValueError("At least 1000 bootstrap resamples are required")
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("Invalid DataLoader settings")
    if not torch.cuda.is_available():
        raise RuntimeError("StageA-Full validation evaluation requires CUDA")
    set_seed(args.seed)
    device = torch.device("cuda")

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
        "slice": output_dir / "stagea_full_epoch60_slice_metrics.csv",
        "patient": output_dir / "stagea_full_epoch60_patient_metrics.csv",
        "summary": output_dir / "stagea_full_epoch60_summary.csv",
        "scale": output_dir
        / "stagea_full_epoch60_scale_cascade_diagnostics.csv",
        "paired": output_dir
        / "stagea_full_epoch60_actual_q_paired_metrics.csv",
        "audit": output_dir
        / "stagea_full_epoch60_validation_audit.json",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise RuntimeError(
            "Refusing to overwrite existing formal evaluation outputs: "
            + ", ".join(existing)
        )

    checkpoint = torch.load(
        paths["checkpoint"], map_location="cpu", weights_only=False
    )
    installed_hashes = code_hashes(PROJECT_ROOT)
    checkpoint_audit = _validate_checkpoint(
        paths["checkpoint"], checkpoint, installed_hashes
    )
    config = checkpoint["config"]
    _validate_input_hashes(config, paths)
    model = _build_model(checkpoint, device)

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
            raise RuntimeError(f"{cohort} manifest protocol/cohort mismatch")

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

    # The numerical value is recomputed for epoch60, while its definition is
    # fixed: patient-equal mean actual q on the locked full-clean cohort.
    actual_clean, actual_clean_scale = evaluate_mode(
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
    q_by_patient: dict[str, list[float]] = defaultdict(list)
    for row in actual_clean:
        q_by_patient[str(row["patient_id"])].append(float(row["q"]))
    if not q_by_patient:
        raise RuntimeError("Locked full-clean cohort produced no q values")
    constant_q = float(
        np.mean([np.mean(values) for values in q_by_patient.values()])
    )

    all_slice_rows = list(actual_clean)
    all_scale_rows = list(actual_clean_scale)
    for mode in MODES:
        if mode != "full":
            rows, scale_rows = evaluate_mode(
                model=model,
                loader=clean_loader,
                source_dataset=source,
                condition_lookup=condition_lookup,
                device=device,
                amp=args.amp,
                cohort="full_clean",
                condition="correct",
                mode=mode,
                constant_q=constant_q,
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
                mode=mode,
                constant_q=constant_q,
            )
            all_slice_rows.extend(rows)
            all_scale_rows.extend(scale_rows)

    patient_level = patient_rows(all_slice_rows)
    add_robustness_composite(patient_level)
    summary_level = summaries(patient_level)
    _assert_finite_metrics(patient_level)
    checkpoint_hash = checkpoint_audit["checkpoint_sha256"]
    for rows in (
        all_slice_rows,
        patient_level,
        summary_level,
        all_scale_rows,
    ):
        _enrich(rows, checkpoint_hash)

    paired = []
    for reference_mode in ("q1", "constant_q"):
        for cohort, condition in COMPARISON_ENDPOINTS:
            for metric in METRICS:
                result = paired_bootstrap(
                    patient_level,
                    candidate="full",
                    reference=reference_mode,
                    cohort=cohort,
                    condition=condition,
                    metric=metric,
                    resamples=args.bootstrap_resamples,
                    seed=args.seed,
                )
                result.update(
                    {
                        "candidate_label": "actual-q",
                        "reference_label": MODE_LABELS[reference_mode],
                        "interpretation": (
                            "negative delta favors actual-q for L1/NMSE; "
                            "positive delta favors actual-q for PSNR/SSIM"
                        ),
                    }
                )
                paired.append(result)

    q_gate = q_functionality_gate(
        patient_level, args.bootstrap_resamples, args.seed
    )
    missing_rows = [
        row
        for row in all_slice_rows
        if row["mode"] == "full"
        and row["cohort"] == "robustness"
        and row["condition"] == "missing"
    ]
    missing_exact_zero = bool(missing_rows) and all(
        float(row["missing_direct_exact_zero"]) == 1.0
        and float(row["missing_correction_exact_zero"]) == 1.0
        for row in missing_rows
    )
    if not missing_exact_zero:
        raise RuntimeError(
            "Formal missing condition did not produce exact-zero direct and "
            "correction paths"
        )

    write_csv(outputs["slice"], all_slice_rows)
    write_csv(outputs["patient"], patient_level)
    write_csv(outputs["summary"], summary_level)
    write_csv(outputs["scale"], all_scale_rows)
    write_csv(outputs["paired"], paired)

    audit = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "passed",
        "scope": "locked validation only; held-out test not accessed",
        "arm": "stagea_full",
        "checkpoint_audit": checkpoint_audit,
        "strict_state_dict_load": True,
        "evaluation_modes": list(MODES),
        "mode_labels": MODE_LABELS,
        "conditions": list(CONDITIONS),
        "robustness_composite_conditions": list(ROBUST_CONDITIONS),
        "shift2_shift4_status": (
            "not evaluated: not part of the locked condition manifest or "
            "original robustness composite"
        ),
        "constant_q_definition": (
            "patient-equal mean actual q on the locked full-clean cohort"
        ),
        "constant_q": constant_q,
        "constant_q_num_patients": len(q_by_patient),
        "q_functionality": q_gate,
        "q_functionality_is_scientific_result_not_runtime_pass_gate": True,
        "missing_direct_and_correction_exact_zero": missing_exact_zero,
        "bootstrap_unit": "patient",
        "bootstrap_resamples": args.bootstrap_resamples,
        "seed": args.seed,
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
        "input_hashes": {
            key: sha256_file(value) for key, value in paths.items()
        },
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    outputs["audit"].write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
