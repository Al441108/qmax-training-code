#!/usr/bin/env python3
from __future__ import annotations

"""Run only the Stage-A epoch-30 counterfactuals needed for selection.

The already completed actual-q/full-mode unified evaluation is reused.  This
script evaluates correction-off, q=1 and the frozen constant-q definition on
the same locked cohorts, then performs patient-paired within-checkpoint tests.
"""

import argparse
import csv
import json
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

from scripts.compare_qmax_stage_a import (  # noqa: E402
    q_functionality_gate,
    within_arm_bootstrap,
)
from scripts.evaluate_qmax_counterfactuals import (  # noqa: E402
    CONDITIONS,
    CONDITION_MANIFEST_PROTOCOL_VERSION,
    MANIFEST_PROTOCOL_VERSION,
    ManifestDataset,
    add_robustness_composite,
    evaluate_mode,
    patient_rows,
    summaries,
)
from scripts.evaluate_qmax_epoch30_unified import (  # noqa: E402
    ARM_SPECS,
    _build_model,
    _enrich,
    _load_json,
    _validate_checkpoint_location,
    _validate_input_hashes,
)
from scripts.qmax_common import (  # noqa: E402
    IndexedDataset,
    ShapeBucketBatchSampler,
    make_dataset,
    set_seed,
    sha256_file,
    write_csv,
)


PROTOCOL_VERSION = "QMax-epoch30-stagea-selection-counterfactual-v1"
ALLOWED_ARMS = ("stagea_core", "stagea_full")
COUNTERFACTUAL_MODES = ("correction_off", "q1", "constant_q")
ENDPOINTS = (
    ("primary_clean", "full_clean", "correct"),
    ("primary_robustness_composite", "robustness", "robustness_composite"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Empty CSV: {path}")
    return rows


def validate_actual_evaluation(
    *,
    arm: str,
    actual_dir: Path,
    checkpoint_path: Path,
    locked_paths: Mapping[str, Path],
) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    audit_path = actual_dir / "epoch30_evaluation_audit.json"
    slice_path = actual_dir / "epoch30_slice_metrics.csv"
    patient_path = actual_dir / "epoch30_patient_metrics.csv"
    scale_path = actual_dir / "epoch30_scale_cascade_diagnostics.csv"
    for path in (audit_path, slice_path, patient_path, scale_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("status") != "passed"
        or audit.get("arm") != arm
        or int(audit.get("checkpoint_epoch", -1)) != 30
    ):
        raise RuntimeError("Actual-mode epoch-30 evaluation audit is invalid")
    checkpoint_hash = sha256_file(checkpoint_path)
    if audit.get("checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("Actual evaluation used a different checkpoint")
    for key, path in locked_paths.items():
        observed = sha256_file(path)
        if audit.get("input_hashes", {}).get(key) != observed:
            raise RuntimeError(f"Actual evaluation locked input drift: {key}")
    slice_rows = read_csv(slice_path)
    patient_level = read_csv(patient_path)
    scale_rows = read_csv(scale_path)
    if any(row.get("mode") != "full" for row in slice_rows + patient_level):
        raise RuntimeError("Actual evaluation contains a non-full mode")
    return audit, slice_rows, patient_level, scale_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=ALLOWED_ARMS)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--actual_eval_dir", required=True)
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

    if args.seed != 42 or not args.amp:
        raise ValueError("Locked evaluation requires seed=42 and AMP")
    if args.bootstrap_resamples < 1000:
        raise ValueError("At least 1000 bootstrap resamples are required")
    if not torch.cuda.is_available():
        raise RuntimeError("Counterfactual evaluation requires CUDA")
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
    actual_dir = Path(args.actual_eval_dir).resolve()
    locked_paths = {
        key: paths[key]
        for key in (
            "metadata_csv",
            "full_clean_manifest",
            "robustness_manifest",
            "condition_manifest",
        )
    }
    actual_audit, actual_slice, actual_patient, actual_scale = (
        validate_actual_evaluation(
            arm=args.arm,
            actual_dir=actual_dir,
            checkpoint_path=paths["checkpoint"],
            locked_paths=locked_paths,
        )
    )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "slice": output_dir / "epoch30_stagea_counterfactual_slice_metrics.csv",
        "patient": output_dir / "epoch30_stagea_counterfactual_patient_metrics.csv",
        "summary": output_dir / "epoch30_stagea_counterfactual_summary.csv",
        "scale": output_dir / "epoch30_stagea_counterfactual_scale_cascade.csv",
        "paired": output_dir / "epoch30_stagea_within_checkpoint_paired_l1.csv",
        "audit": output_dir / "epoch30_stagea_counterfactual_audit.json",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise RuntimeError("Refusing to overwrite: " + ", ".join(existing))

    checkpoint = torch.load(
        paths["checkpoint"], map_location="cpu", weights_only=False
    )
    if int(checkpoint.get("epoch", -1)) != 30:
        raise RuntimeError("Counterfactual checkpoint is not epoch 30")
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

    q_by_patient: dict[str, list[float]] = defaultdict(list)
    for row in actual_slice:
        if row["cohort"] == "full_clean" and row["condition"] == "correct":
            q_by_patient[str(row["patient_id"])].append(float(row["q"]))
    if not q_by_patient:
        raise RuntimeError("Actual full-clean q values are absent")
    constant_q = float(
        np.mean([np.mean(values) for values in q_by_patient.values()])
    )

    cf_slice: list[dict[str, Any]] = []
    cf_scale: list[dict[str, Any]] = []
    for mode in COUNTERFACTUAL_MODES:
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
        cf_slice.extend(rows)
        cf_scale.extend(scale_rows)
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
            cf_slice.extend(rows)
            cf_scale.extend(scale_rows)

    cf_patient = patient_rows(cf_slice)
    add_robustness_composite(cf_patient)
    checkpoint_hash = sha256_file(paths["checkpoint"])
    for rows in (cf_slice, cf_patient, cf_scale):
        _enrich(rows, arm=args.arm, checkpoint_sha256=checkpoint_hash)
    combined_slice = [*actual_slice, *cf_slice]
    combined_patient = [*actual_patient, *cf_patient]
    combined_scale = [*actual_scale, *cf_scale]
    summary = summaries(combined_patient)
    _enrich(summary, arm=args.arm, checkpoint_sha256=checkpoint_hash)

    comparisons = []
    for reference_mode in COUNTERFACTUAL_MODES:
        for endpoint_role, cohort, condition in ENDPOINTS:
            result = within_arm_bootstrap(
                combined_patient,
                candidate_mode="full",
                reference_mode=reference_mode,
                cohort=cohort,
                condition=condition,
                metric="l1",
                resamples=args.bootstrap_resamples,
                seed=args.seed,
            )
            result.update(
                {
                    "arm": args.arm,
                    "endpoint_role": endpoint_role,
                    "interpretation": "negative delta favors actual/full mode",
                }
            )
            comparisons.append(result)
    q_gate = q_functionality_gate(
        combined_patient, args.bootstrap_resamples, args.seed
    )

    write_csv(outputs["slice"], combined_slice)
    write_csv(outputs["patient"], combined_patient)
    write_csv(outputs["summary"], summary)
    write_csv(outputs["scale"], combined_scale)
    write_csv(outputs["paired"], comparisons)
    missing_safe = all(
        float(row["missing_direct_exact_zero"]) == 1.0
        and float(row["missing_correction_exact_zero"]) == 1.0
        for row in cf_slice
        if row["cohort"] == "robustness" and row["condition"] == "missing"
    )
    audit = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "passed" if missing_safe else "failed",
        "arm": args.arm,
        **ARM_SPECS[args.arm],
        "checkpoint": str(paths["checkpoint"]),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "actual_evaluation_audit": str(
            actual_dir / "epoch30_evaluation_audit.json"
        ),
        "actual_evaluation_checkpoint_sha256": actual_audit[
            "checkpoint_sha256"
        ],
        "constant_q_definition": (
            "patient-equal mean actual q on the locked full-clean cohort"
        ),
        "constant_q": constant_q,
        "counterfactual_modes": list(COUNTERFACTUAL_MODES),
        "q_functionality_gate": q_gate,
        "missing_counterfactual_paths_exact_zero": missing_safe,
        "bootstrap_unit": "patient",
        "bootstrap_resamples": args.bootstrap_resamples,
        "input_hashes": {
            key: sha256_file(value) for key, value in paths.items()
        },
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    outputs["audit"].write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2), flush=True)
    if audit["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
