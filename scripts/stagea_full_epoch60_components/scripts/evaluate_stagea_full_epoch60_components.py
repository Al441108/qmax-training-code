#!/usr/bin/env python3
from __future__ import annotations

"""Locked component counterfactuals for StageA-Full epoch60/model_last.pt.

This is a read-only runtime evaluator. It does not change the model, checkpoint,
data, manifests, or scientific code. All component modes keep learned q active:
full, detail_neutral, alignment_off, correction_off, and dc_zero.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
from scripts.evaluate_stagea_full_epoch60_validation import (  # noqa: E402
    _assert_finite_metrics,
    _build_model,
    _load_json,
    _validate_checkpoint,
    _validate_checkpoint_location,
    _validate_input_hashes,
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


PROTOCOL_VERSION = "StageA-Full-epoch60-component-counterfactuals-v1"
MODES = (
    "full",
    "detail_neutral",
    "alignment_off",
    "correction_off",
    "dc_zero",
)
MODE_LABELS = {
    "full": "all-components-on",
    "detail_neutral": "detail-neutral (G=1)",
    "alignment_off": "alignment-off (Delta=0)",
    "correction_off": "correction-off (C=0)",
    "dc_zero": "DC-zero after RMS normalization",
}
COMPARISON_ENDPOINTS = (
    ("full_clean", "correct"),
    ("robustness", "correct"),
    ("robustness", "shift8"),
    ("robustness", "wrong_slice"),
    ("robustness", "wrong_patient"),
    ("robustness", "robustness_composite"),
)


def _enrich(rows: Iterable[Dict[str, Any]], checkpoint_hash: str) -> None:
    for row in rows:
        mode = str(row["mode"])
        row.update(
            {
                "arm": "stagea_full",
                "qmax_variant": "qmax_full",
                "backbone_variant": "convolutional",
                "checkpoint_epoch": 60,
                "checkpoint_sha256": checkpoint_hash,
                "mode_label": MODE_LABELS[mode],
            }
        )


def _missing_metric_invariance(
    rows: list[dict[str, Any]],
) -> Dict[str, Any]:
    selected: Dict[tuple[str, str], Dict[str, float]] = {}
    for row in rows:
        if row["cohort"] != "robustness" or row["condition"] != "missing":
            continue
        key = (str(row["mode"]), str(row["patient_id"]))
        if key in selected:
            raise RuntimeError(f"Duplicate missing patient row: {key}")
        selected[key] = {metric: float(row[metric]) for metric in METRICS}

    full_patients = {
        patient for mode, patient in selected if mode == "full"
    }
    if not full_patients:
        raise RuntimeError("No full/missing patient rows were produced")

    max_delta = 0.0
    for mode in MODES[1:]:
        mode_patients = {
            patient for current, patient in selected if current == mode
        }
        if mode_patients != full_patients:
            raise RuntimeError(
                f"Missing patient sets differ for full and {mode}"
            )
        for patient in sorted(full_patients):
            for metric in METRICS:
                delta = abs(
                    selected[(mode, patient)][metric]
                    - selected[("full", patient)][metric]
                )
                max_delta = max(max_delta, delta)
    return {
        "definition": (
            "maximum absolute patient-metric difference between full and "
            "each component counterfactual when availability m=0"
        ),
        "max_abs_metric_delta": float(max_delta),
        "tolerance": 1e-12,
        "passed": bool(max_delta <= 1e-12),
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
        raise RuntimeError("Component counterfactual evaluation requires CUDA")

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
        "slice": output_dir
        / "stagea_full_epoch60_component_slice_metrics.csv",
        "patient": output_dir
        / "stagea_full_epoch60_component_patient_metrics.csv",
        "summary": output_dir
        / "stagea_full_epoch60_component_summary.csv",
        "scale": output_dir
        / "stagea_full_epoch60_component_scale_cascade.csv",
        "paired": output_dir
        / "stagea_full_epoch60_component_paired_metrics.csv",
        "audit": output_dir
        / "stagea_full_epoch60_component_audit.json",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise RuntimeError(
            "Refusing to overwrite component outputs: " + ", ".join(existing)
        )

    checkpoint = torch.load(
        paths["checkpoint"], map_location="cpu", weights_only=False
    )
    installed_hashes = code_hashes(PROJECT_ROOT)
    checkpoint_audit = _validate_checkpoint(
        paths["checkpoint"], checkpoint, installed_hashes
    )
    if str(checkpoint["config"].get("qmax_variant")) != "qmax_full":
        raise RuntimeError("This evaluator accepts qmax_full only")
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
        or int(condition_manifest.get("seed", -1)) != args.seed
    ):
        raise RuntimeError("Condition manifest protocol/seed mismatch")
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
    for mode in MODES:
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
                mode=mode,
                constant_q=None,
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

    comparisons: list[dict[str, Any]] = []
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
                    seed=args.seed,
                )
                result.update(
                    {
                        "candidate_label": MODE_LABELS["full"],
                        "reference_label": MODE_LABELS[reference_mode],
                        "interpretation": (
                            "negative delta favors all-components-on for "
                            "L1/NMSE; positive delta favors it for PSNR/SSIM"
                        ),
                    }
                )
                comparisons.append(result)

    missing_zero = bool(all_slice_rows) and all(
        float(row["missing_direct_exact_zero"]) == 1.0
        and float(row["missing_correction_exact_zero"]) == 1.0
        for row in all_slice_rows
        if row["cohort"] == "robustness"
        and row["condition"] == "missing"
    )
    missing_invariance = _missing_metric_invariance(patient_level)
    if not missing_zero or not missing_invariance["passed"]:
        raise RuntimeError(
            "Missing-PD component safety failed: "
            + json.dumps(
                {
                    "direct_correction_exact_zero": missing_zero,
                    "metric_invariance": missing_invariance,
                },
                indent=2,
            )
        )

    write_csv(outputs["slice"], all_slice_rows)
    write_csv(outputs["patient"], patient_level)
    write_csv(outputs["summary"], summary_level)
    write_csv(outputs["scale"], all_scale_rows)
    write_csv(outputs["paired"], comparisons)

    audit = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "passed",
        "scope": "locked validation only; held-out test not accessed",
        "checkpoint_audit": checkpoint_audit,
        "strict_state_dict_load": True,
        "all_modes_use_actual_q": True,
        "modes": list(MODES),
        "mode_labels": MODE_LABELS,
        "conditions": list(CONDITIONS),
        "robustness_composite_conditions": list(ROBUST_CONDITIONS),
        "bootstrap_unit": "patient",
        "bootstrap_resamples": args.bootstrap_resamples,
        "seed": args.seed,
        "component_results_are_scientific_results_not_runtime_pass_gates": True,
        "missing_direct_and_correction_exact_zero_all_modes": missing_zero,
        "missing_component_metric_invariance": missing_invariance,
        "input_hashes": {
            key: sha256_file(value) for key, value in paths.items()
        },
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
