#!/usr/bin/env python3
from __future__ import annotations

"""Unified locked-validation evaluation for QMax-Full replication seeds.

The evaluator is read-only and accepts only the formal epoch60/model_last.pt
checkpoints for model seeds 123 and 2026.  All data ordering, corruption
conditions and bootstrap randomness use the already locked analysis seed 42,
so model seed is never allowed to change the evaluation cohort.
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

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
from scripts.qmax_multiseed_common import (  # noqa: E402
    IndexedDataset,
    ShapeBucketBatchSampler,
    code_hashes,
    install_amp_diagnostic_quantile_compatibility,
    make_dataset,
    set_seed,
    sha256_file,
    write_csv,
)
from src.m2_prnf_qmax_varnet import QMaxAuxPDVarNet  # noqa: E402


PROTOCOL_VERSION = "QMax-Full-three-seed-locked-evaluation-v1"
ALLOWED_MODEL_SEEDS = (123, 2026)
ANALYSIS_SEED = 42
MODES = (
    "full",
    "q1",
    "constant_q",
    "detail_neutral",
    "alignment_off",
    "correction_off",
    "dc_zero",
)
MODE_LABELS = {
    "full": "actual-q / all-components-on",
    "q1": "q=1",
    "constant_q": "constant-q",
    "detail_neutral": "detail-neutral (G=1)",
    "alignment_off": "alignment-off (Delta=0)",
    "correction_off": "correction-off (C=0)",
    "dc_zero": "DC-zero after RMS normalization",
}
MODE_GROUPS = {
    "full": "shared_reference",
    "q1": "q_counterfactual",
    "constant_q": "q_counterfactual",
    "detail_neutral": "component_counterfactual",
    "alignment_off": "component_counterfactual",
    "correction_off": "component_counterfactual",
    "dc_zero": "component_counterfactual",
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
            "Only the formal epoch60/model_last.pt checkpoint is accepted"
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
        raise RuntimeError(f"Checkpoint lacks locked input hashes: {missing}")
    mismatches = {}
    for key, expected_hash in expected.items():
        observed = sha256_file(paths[key])
        if observed != expected_hash:
            mismatches[key] = {"checkpoint": expected_hash, "installed": observed}
    if mismatches:
        raise RuntimeError(
            "Locked evaluation inputs differ from training:\n"
            + json.dumps(mismatches, indent=2)
        )


def _validate_training_log(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "training_log.csv"
    if not path.is_file():
        raise RuntimeError(f"Training log is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    epochs = [int(row["epoch"]) for row in rows]
    if epochs != list(range(1, 61)):
        raise RuntimeError(f"training_log.csv is not exactly epochs 1..60: {epochs}")
    required = (
        "train_total_loss",
        "train_recon_loss",
        "train_clean_l1",
        "train_corrupt_l1",
        "val_patient_l1",
    )
    failures = []
    for row in rows:
        for field in required:
            if field not in row or not math.isfinite(float(row[field])):
                failures.append({"epoch": row.get("epoch"), "field": field})
    if failures:
        raise RuntimeError(
            "Non-finite required training-log values:\n"
            + json.dumps(failures[:20], indent=2)
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "num_rows": len(rows),
        "epochs_exactly_1_to_60": True,
        "required_fields_finite": True,
    }


def _validate_checkpoint(
    path: Path,
    checkpoint: Mapping[str, Any],
    installed_hashes: Mapping[str, str],
    model_seed: int,
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
        raise RuntimeError(f"Expected epoch 60, got {checkpoint['epoch']}")
    config = checkpoint["config"]
    if int(config.get("seed", -1)) != model_seed:
        raise RuntimeError(
            f"Checkpoint seed {config.get('seed')} != requested {model_seed}"
        )
    if str(config.get("qmax_variant")) != "qmax_full":
        raise RuntimeError("Replication evaluator accepts qmax_full only")
    if config.get("formal_structure_selection_checkpoint") != (
        "epoch60/model_last.pt"
    ):
        raise RuntimeError("Checkpoint does not declare epoch60/model_last.pt")
    if config.get("learning_rate_schedule") != EXPECTED_LR_SCHEDULE:
        raise RuntimeError("Frozen learning-rate schedule mismatch")
    history_epochs = [int(row["epoch"]) for row in checkpoint["history"]]
    if history_epochs != list(range(1, 61)):
        raise RuntimeError("Checkpoint history is not exactly epochs 1..60")
    if checkpoint.get("code_hashes") != installed_hashes:
        raise RuntimeError("Checkpoint top-level scientific hashes drifted")
    if config.get("code_hashes") != installed_hashes:
        raise RuntimeError("Checkpoint config scientific hashes drifted")
    groups = list(checkpoint["optimizer_state_dict"].get("param_groups", []))
    if not groups:
        raise RuntimeError("Optimizer state contains no parameter groups")
    final_lrs = [float(group["lr"]) for group in groups]
    if any(
        not math.isclose(value, 3e-5, rel_tol=0.0, abs_tol=1e-12)
        for value in final_lrs
    ):
        raise RuntimeError(f"Epoch-60 optimizer LR is not 3e-5: {final_lrs}")
    return {
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
        "checkpoint_epoch": 60,
        "model_seed": model_seed,
        "history_epochs_exactly_1_to_60": True,
        "learning_rate_schedule": EXPECTED_LR_SCHEDULE,
        "optimizer_final_lrs": final_lrs,
        "optimizer_state_present": True,
        "grad_scaler_state_present": True,
        "rng_state_present": True,
        "corruption_state_present": True,
        "scientific_code_hashes_match": True,
        "training_log": _validate_training_log(path.parent.parent),
    }


def _build_model(
    checkpoint: Mapping[str, Any], device: torch.device
) -> QMaxAuxPDVarNet:
    model = QMaxAuxPDVarNet(
        qmax_variant="qmax_full",
        **dict(checkpoint["config"]["model_kwargs"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


def _enrich(
    rows: Iterable[Dict[str, Any]], model_seed: int, checkpoint_hash: str
) -> None:
    for row in rows:
        mode = str(row["mode"])
        row.update(
            {
                "arm": "stagea_full",
                "qmax_variant": "qmax_full",
                "backbone_variant": "convolutional",
                "model_seed": model_seed,
                "analysis_seed": ANALYSIS_SEED,
                "checkpoint_epoch": 60,
                "checkpoint_sha256": checkpoint_hash,
                "mode_label": MODE_LABELS[mode],
                "mode_group": MODE_GROUPS[mode],
            }
        )


def _assert_finite_metrics(rows: Iterable[Mapping[str, Any]]) -> None:
    failures = []
    for row in rows:
        for metric in METRICS:
            if not math.isfinite(float(row[metric])):
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
            "Non-finite metrics:\n" + json.dumps(failures[:20], indent=2)
        )


def _missing_invariance(rows: list[dict[str, Any]]) -> Dict[str, Any]:
    selected: Dict[tuple[str, str], Dict[str, float]] = {}
    for row in rows:
        if row["cohort"] != "robustness" or row["condition"] != "missing":
            continue
        key = (str(row["mode"]), str(row["patient_id"]))
        if key in selected:
            raise RuntimeError(f"Duplicate missing patient row: {key}")
        selected[key] = {metric: float(row[metric]) for metric in METRICS}
    full_patients = {patient for mode, patient in selected if mode == "full"}
    if not full_patients:
        raise RuntimeError("No full/missing patient rows were produced")
    maximum = 0.0
    for mode in MODES[1:]:
        patients = {patient for current, patient in selected if current == mode}
        if patients != full_patients:
            raise RuntimeError(f"Missing patient sets differ for full and {mode}")
        for patient in full_patients:
            for metric in METRICS:
                maximum = max(
                    maximum,
                    abs(
                        selected[(mode, patient)][metric]
                        - selected[("full", patient)][metric]
                    ),
                )
    return {
        "max_abs_patient_metric_delta": float(maximum),
        "tolerance": 1e-12,
        "passed": bool(maximum <= 1e-12),
        "num_patients": len(full_patients),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--full_clean_manifest", required=True)
    parser.add_argument("--robustness_manifest", required=True)
    parser.add_argument("--condition_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_seed", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--bootstrap_resamples", type=int, default=10000)
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    if args.model_seed not in ALLOWED_MODEL_SEEDS:
        raise ValueError(f"model_seed must be one of {ALLOWED_MODEL_SEEDS}")
    if not args.amp:
        raise ValueError("Locked evaluation requires AMP")
    if args.bootstrap_resamples < 1000:
        raise ValueError("At least 1000 bootstrap resamples are required")
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("Invalid DataLoader settings")
    if not torch.cuda.is_available():
        raise RuntimeError("Replication evaluation requires CUDA")

    install_amp_diagnostic_quantile_compatibility()
    set_seed(ANALYSIS_SEED)
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
    prefix = f"qmax_full_seed{args.model_seed}_epoch60"
    outputs = {
        "slice": output_dir / f"{prefix}_slice_metrics.csv",
        "patient": output_dir / f"{prefix}_patient_metrics.csv",
        "summary": output_dir / f"{prefix}_summary.csv",
        "scale": output_dir / f"{prefix}_scale_cascade_diagnostics.csv",
        "paired": output_dir / f"{prefix}_paired_metrics.csv",
        "audit": output_dir / f"{prefix}_evaluation_audit.json",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise RuntimeError("Refusing to overwrite outputs: " + ", ".join(existing))

    checkpoint = torch.load(
        paths["checkpoint"], map_location="cpu", weights_only=False
    )
    checkpoint_audit = _validate_checkpoint(
        paths["checkpoint"],
        checkpoint,
        code_hashes(PROJECT_ROOT),
        args.model_seed,
    )
    _validate_input_hashes(checkpoint["config"], paths)
    model = _build_model(checkpoint, device)

    source = IndexedDataset(
        make_dataset(
            str(paths["metadata_csv"]),
            "val",
            acceleration=8,
            pd_aux_acceleration=2,
        )
    )
    clean_manifest = _load_json(paths["full_clean_manifest"])
    robust_manifest = _load_json(paths["robustness_manifest"])
    for cohort, manifest in (
        ("full_clean", clean_manifest),
        ("robustness", robust_manifest),
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
        or int(condition_manifest.get("seed", -1)) != ANALYSIS_SEED
    ):
        raise RuntimeError("Condition manifest protocol/analysis-seed mismatch")
    condition_lookup = {
        int(entry["source_index"]): entry
        for entry in condition_manifest["entries"]
    }
    if len(condition_lookup) != int(condition_manifest["num_entries"]):
        raise RuntimeError("Duplicate source index in condition manifest")

    clean_dataset = ManifestDataset(source, clean_manifest)
    robust_dataset = ManifestDataset(source, robust_manifest)
    clean_loader = DataLoader(
        clean_dataset,
        batch_sampler=ShapeBucketBatchSampler(
            clean_dataset, args.batch_size, False, ANALYSIS_SEED
        ),
        num_workers=args.num_workers,
        pin_memory=True,
    )
    robust_loader = DataLoader(
        robust_dataset,
        batch_sampler=ShapeBucketBatchSampler(
            robust_dataset, args.batch_size, False, ANALYSIS_SEED
        ),
        num_workers=args.num_workers,
        pin_memory=True,
    )

    actual_clean, actual_scale = evaluate_mode(
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
        raise RuntimeError("Full-clean cohort produced no q values")
    constant_q = float(
        np.mean([np.mean(values) for values in q_by_patient.values()])
    )

    all_slice_rows = list(actual_clean)
    all_scale_rows = list(actual_scale)
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
    for rows in (all_slice_rows, patient_level, summary_level, all_scale_rows):
        _enrich(rows, args.model_seed, checkpoint_hash)

    paired = []
    for reference_mode in MODES[1:]:
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
                    seed=ANALYSIS_SEED,
                )
                result.update(
                    {
                        "model_seed": args.model_seed,
                        "analysis_seed": ANALYSIS_SEED,
                        "candidate_label": MODE_LABELS["full"],
                        "reference_label": MODE_LABELS[reference_mode],
                        "evidence_group": MODE_GROUPS[reference_mode],
                        "interpretation": (
                            "negative delta favors full for L1/NMSE; "
                            "positive delta favors full for PSNR/SSIM"
                        ),
                    }
                )
                paired.append(result)

    missing_rows = [
        row
        for row in all_slice_rows
        if row["cohort"] == "robustness" and row["condition"] == "missing"
    ]
    missing_zero = bool(missing_rows) and all(
        float(row["missing_direct_exact_zero"]) == 1.0
        and float(row["missing_correction_exact_zero"]) == 1.0
        for row in missing_rows
    )
    missing_invariance = _missing_invariance(patient_level)
    if not missing_zero or not missing_invariance["passed"]:
        raise RuntimeError(
            "Missing-PD safety failed: "
            + json.dumps(
                {"exact_zero": missing_zero, "invariance": missing_invariance},
                indent=2,
            )
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
        "model_seed": args.model_seed,
        "analysis_seed": ANALYSIS_SEED,
        "checkpoint_audit": checkpoint_audit,
        "strict_state_dict_load": True,
        "modes": list(MODES),
        "mode_labels": MODE_LABELS,
        "conditions": list(CONDITIONS),
        "robustness_composite_conditions": list(ROBUST_CONDITIONS),
        "shift2_shift4_status": (
            "not evaluated because they are absent from the frozen condition "
            "manifest and original composite"
        ),
        "constant_q_definition": (
            "patient-equal mean actual q on locked full-clean cohort, "
            "computed separately within checkpoint"
        ),
        "constant_q": constant_q,
        "constant_q_num_patients": len(q_by_patient),
        "q_functionality": q_functionality_gate(
            patient_level, args.bootstrap_resamples, ANALYSIS_SEED
        ),
        "q_functionality_is_scientific_result_not_runtime_gate": True,
        "missing_direct_and_correction_exact_zero_all_modes": missing_zero,
        "missing_mode_metric_invariance": missing_invariance,
        "bootstrap_unit": "patient",
        "bootstrap_resamples": args.bootstrap_resamples,
        "input_hashes": {key: sha256_file(value) for key, value in paths.items()},
        "num_slice_rows": len(all_slice_rows),
        "num_patient_rows": len(patient_level),
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    outputs["audit"].write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
