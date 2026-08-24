#!/usr/bin/env python3
from __future__ import annotations

"""Compare the three valid epoch-30 arms and retain StageB-Core failure."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np


PROTOCOL_VERSION = "QMax-epoch30-valid-three-arm-comparison-v1"
VALID_ARMS = ("stagea_core", "stagea_full", "stageb_full")
ALL_ARMS = ("stagea_core", "stagea_full", "stageb_core", "stageb_full")
PAIR_SPECS = (
    ("stagea_full_vs_stagea_core", "stagea_full", "stagea_core"),
    ("stageb_full_vs_stagea_core", "stageb_full", "stagea_core"),
    ("stageb_full_vs_stagea_full", "stageb_full", "stagea_full"),
)
ENDPOINTS = (
    ("primary_clean", "full_clean", "correct"),
    ("robustness_clean_input", "robustness", "correct"),
    ("primary_robustness_composite", "robustness", "robustness_composite"),
    ("robustness_shift8", "robustness", "shift8"),
    ("robustness_wrong_slice", "robustness", "wrong_slice"),
    ("robustness_wrong_patient", "robustness", "wrong_patient"),
    ("robustness_missing", "robustness", "missing"),
)


def read_csv(path: Path) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = tuple(reader.fieldnames or ())
    if not rows or not fields:
        raise RuntimeError(f"Empty or headerless CSV: {path}")
    return rows, fields


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Iterable[str] | None = None,
) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    fields = list(fieldnames or rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_actual_arm(arm: str, directory: Path) -> Dict[str, Any]:
    audit_path = directory / "epoch30_evaluation_audit.json"
    patient_path = directory / "epoch30_patient_metrics.csv"
    slice_path = directory / "epoch30_slice_metrics.csv"
    for path in (audit_path, patient_path, slice_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("status") != "passed"
        or audit.get("arm") != arm
        or int(audit.get("checkpoint_epoch", -1)) != 30
    ):
        raise RuntimeError(f"Invalid epoch-30 audit for {arm}")
    patient_rows, patient_fields = read_csv(patient_path)
    slice_rows, slice_fields = read_csv(slice_path)
    for label, rows in (("patient", patient_rows), ("slice", slice_rows)):
        if any(row.get("arm") != arm for row in rows):
            raise RuntimeError(f"{arm} mislabeled {label} rows")
        if any(row.get("mode") != "full" for row in rows):
            raise RuntimeError(f"{arm} has non-full rows in actual evaluation")
    return {
        "directory": str(directory),
        "audit": audit,
        "patient_rows": patient_rows,
        "patient_fields": patient_fields,
        "slice_rows": slice_rows,
        "slice_fields": slice_fields,
    }


def unique_keys(
    arm: str,
    rows: Sequence[Mapping[str, str]],
    keys: Sequence[str],
) -> set[tuple[str, ...]]:
    observed: set[tuple[str, ...]] = set()
    for row in rows:
        key = tuple(str(row[name]) for name in keys)
        if key in observed:
            raise RuntimeError(f"Duplicate key in {arm}: {key}")
        observed.add(key)
    return observed


def select_patient(
    rows: Sequence[Mapping[str, str]], cohort: str, condition: str
) -> Dict[str, Dict[str, float]]:
    selected: Dict[str, Dict[str, float]] = {}
    for row in rows:
        if row["cohort"] != cohort or row["condition"] != condition:
            continue
        patient = str(row["patient_id"])
        if patient in selected:
            raise RuntimeError(f"Duplicate patient: {cohort}/{condition}/{patient}")
        selected[patient] = {
            "l1": float(row["l1"]),
            "num_slices": float(row["num_slices"]),
        }
    if not selected:
        raise RuntimeError(f"No patient rows for {cohort}/{condition}")
    return selected


def paired_comparison(
    *,
    comparison_id: str,
    candidate_arm: str,
    reference_arm: str,
    endpoint_role: str,
    cohort: str,
    condition: str,
    loaded: Mapping[str, Mapping[str, Any]],
    resamples: int,
    seed: int,
) -> Dict[str, Any]:
    candidate = select_patient(
        loaded[candidate_arm]["patient_rows"], cohort, condition
    )
    reference = select_patient(
        loaded[reference_arm]["patient_rows"], cohort, condition
    )
    if set(candidate) != set(reference):
        raise RuntimeError(
            f"Patient sets differ: {comparison_id}/{cohort}/{condition}"
        )
    mismatches = {
        patient: (
            candidate[patient]["num_slices"],
            reference[patient]["num_slices"],
        )
        for patient in candidate
        if candidate[patient]["num_slices"]
        != reference[patient]["num_slices"]
    }
    if mismatches:
        raise RuntimeError(f"Slice counts differ: {dict(list(mismatches.items())[:8])}")
    patients = sorted(candidate)
    c = np.asarray([candidate[p]["l1"] for p in patients], dtype=np.float64)
    r = np.asarray([reference[p]["l1"] for p in patients], dtype=np.float64)
    diff = c - r
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(patients), size=(resamples, len(patients)))
    c_draws = c[indices].mean(axis=1)
    r_draws = r[indices].mean(axis=1)
    delta_draws = c_draws - r_draws
    relative_draws = delta_draws / r_draws
    delta = float(diff.mean())
    reference_mean = float(r.mean())
    return {
        "comparison_id": comparison_id,
        "candidate_arm": candidate_arm,
        "reference_arm": reference_arm,
        "endpoint_role": endpoint_role,
        "cohort": cohort,
        "condition": condition,
        "metric": "l1",
        "candidate_mean_l1": float(c.mean()),
        "reference_mean_l1": reference_mean,
        "delta_l1_candidate_minus_reference": delta,
        "relative_delta_percent": 100.0 * delta / reference_mean,
        "ci95_delta_l1_low": float(np.quantile(delta_draws, 0.025)),
        "ci95_delta_l1_high": float(np.quantile(delta_draws, 0.975)),
        "ci95_relative_delta_percent_low": float(
            100.0 * np.quantile(relative_draws, 0.025)
        ),
        "ci95_relative_delta_percent_high": float(
            100.0 * np.quantile(relative_draws, 0.975)
        ),
        "candidate_better_patients": int((diff < 0).sum()),
        "reference_better_patients": int((diff > 0).sum()),
        "tied_patients": int((diff == 0).sum()),
        "num_patients": len(patients),
        "negative_delta_favors_candidate": True,
    }


def load_counterfactual(arm: str, directory: Path, actual: Mapping[str, Any]):
    audit_path = directory / "epoch30_stagea_counterfactual_audit.json"
    patient_path = directory / "epoch30_stagea_counterfactual_patient_metrics.csv"
    paired_path = directory / "epoch30_stagea_within_checkpoint_paired_l1.csv"
    for path in (audit_path, patient_path, paired_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("status") != "passed"
        or audit.get("arm") != arm
        or int(audit.get("checkpoint_epoch", -1)) != 30
        or audit.get("checkpoint_sha256")
        != actual["audit"].get("checkpoint_sha256")
    ):
        raise RuntimeError(f"Invalid or mismatched counterfactual audit: {arm}")
    patient_rows, _ = read_csv(patient_path)
    paired_rows, _ = read_csv(paired_path)
    required_modes = {"full", "correction_off", "q1", "constant_q"}
    if {row["mode"] for row in patient_rows} != required_modes:
        raise RuntimeError(f"Counterfactual modes incomplete for {arm}")
    return {"audit": audit, "patient_rows": patient_rows, "paired_rows": paired_rows}


def summarize_training_log(arm: str, path: Path, expected_complete: bool):
    rows, _ = read_csv(path)
    epochs = [int(row["epoch"]) for row in rows]
    if epochs != list(range(1, max(epochs) + 1)):
        raise RuntimeError(f"Non-contiguous training log for {arm}")
    if expected_complete and max(epochs) != 30:
        raise RuntimeError(f"{arm} did not log exactly epochs 1-30")
    l1 = np.asarray([float(row["val_patient_l1"]) for row in rows])
    best_index = int(np.argmin(l1))
    seconds = np.asarray([float(row["epoch_seconds"]) for row in rows])
    memory = np.asarray([float(row["peak_gpu_memory_gb"]) for row in rows])
    return {
        "arm": arm,
        "training_status": (
            "completed_epoch30" if expected_complete else "failed_before_epoch30"
        ),
        "last_logged_epoch": max(epochs),
        "best_logged_epoch": epochs[best_index],
        "best_val_patient_l1": float(l1[best_index]),
        "epoch30_val_patient_l1": (
            float(l1[-1]) if max(epochs) == 30 else ""
        ),
        "mean_epoch_seconds": float(seconds.mean()),
        "peak_gpu_memory_gb": float(memory.max()),
        "training_log": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in VALID_ARMS:
        parser.add_argument(f"--{arm}_eval_dir", required=True)
        parser.add_argument(f"--{arm}_training_log", required=True)
    for arm in ("stagea_core", "stagea_full"):
        parser.add_argument(f"--{arm}_counterfactual_dir", required=True)
    parser.add_argument("--stageb_core_training_log", required=True)
    parser.add_argument("--arm_status_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bootstrap_resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.seed != 42 or args.bootstrap_resamples < 1000:
        raise ValueError("Locked comparison requires seed=42 and >=1000 resamples")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "status": output_dir / "epoch30_four_arm_status.csv",
        "patient": output_dir / "epoch30_valid_three_arm_patient_metrics.csv",
        "between": output_dir / "epoch30_valid_three_arm_paired_l1.csv",
        "within": output_dir / "epoch30_stagea_within_checkpoint_paired_l1.csv",
        "resources": output_dir / "epoch30_training_resource_summary.csv",
        "report": output_dir / "epoch30_valid_three_arm_report.json",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise RuntimeError("Refusing to overwrite: " + ", ".join(existing))

    loaded = {
        arm: load_actual_arm(
            arm, Path(getattr(args, f"{arm}_eval_dir")).resolve()
        )
        for arm in VALID_ARMS
    }
    patient_schemas = {loaded[arm]["patient_fields"] for arm in VALID_ARMS}
    slice_schemas = {loaded[arm]["slice_fields"] for arm in VALID_ARMS}
    if len(patient_schemas) != 1 or len(slice_schemas) != 1:
        raise RuntimeError("Valid-arm evaluator schemas differ")
    patient_keys = {
        arm: unique_keys(
            arm,
            loaded[arm]["patient_rows"],
            ("cohort", "condition", "patient_id"),
        )
        for arm in VALID_ARMS
    }
    slice_keys = {
        arm: unique_keys(
            arm,
            loaded[arm]["slice_rows"],
            ("cohort", "condition", "patient_id", "slice_idx"),
        )
        for arm in VALID_ARMS
    }
    if len({frozenset(value) for value in patient_keys.values()}) != 1:
        raise RuntimeError("Valid-arm patient keys differ")
    if len({frozenset(value) for value in slice_keys.values()}) != 1:
        raise RuntimeError("Valid-arm slice keys differ")
    hash_keys = (
        "metadata_csv",
        "full_clean_manifest",
        "robustness_manifest",
        "condition_manifest",
    )
    locked_hashes = {
        key: {loaded[arm]["audit"]["input_hashes"][key] for arm in VALID_ARMS}
        for key in hash_keys
    }
    if any(len(values) != 1 for values in locked_hashes.values()):
        raise RuntimeError("Valid arms used different locked inputs")

    comparisons = [
        paired_comparison(
            comparison_id=comparison_id,
            candidate_arm=candidate,
            reference_arm=reference,
            endpoint_role=endpoint_role,
            cohort=cohort,
            condition=condition,
            loaded=loaded,
            resamples=args.bootstrap_resamples,
            seed=args.seed,
        )
        for comparison_id, candidate, reference in PAIR_SPECS
        for endpoint_role, cohort, condition in ENDPOINTS
    ]
    counterfactual = {
        arm: load_counterfactual(
            arm,
            Path(getattr(args, f"{arm}_counterfactual_dir")).resolve(),
            loaded[arm],
        )
        for arm in ("stagea_core", "stagea_full")
    }
    within_rows = [
        row
        for arm in ("stagea_core", "stagea_full")
        for row in counterfactual[arm]["paired_rows"]
    ]

    status_document = json.loads(
        Path(args.arm_status_json).resolve().read_text(encoding="utf-8")
    )
    if set(status_document.get("arms", {})) != set(ALL_ARMS):
        raise RuntimeError("Arm status document does not contain exactly four arms")
    if status_document["arms"]["stageb_core"].get("status") != "training_failed_nonfinite_loss":
        raise RuntimeError("StageB-Core failure status is missing")
    status_rows = []
    for arm in ALL_ARMS:
        record = status_document["arms"][arm]
        status_rows.append(
            {
                "arm": arm,
                "status": record["status"],
                "eligible_for_epoch30_comparison": record[
                    "eligible_for_epoch30_comparison"
                ],
                "epoch30_checkpoint_available": arm in VALID_ARMS,
                "included_in_numeric_ranking": arm in VALID_ARMS,
                "failure_reason": record.get("failure_reason", ""),
            }
        )

    resource_rows = [
        summarize_training_log(
            arm,
            Path(getattr(args, f"{arm}_training_log")).resolve(),
            True,
        )
        for arm in VALID_ARMS
    ]
    resource_rows.append(
        summarize_training_log(
            "stageb_core",
            Path(args.stageb_core_training_log).resolve(),
            False,
        )
    )

    combined_patient = [
        row for arm in VALID_ARMS for row in loaded[arm]["patient_rows"]
    ]
    write_csv(outputs["status"], status_rows)
    write_csv(
        outputs["patient"],
        combined_patient,
        next(iter(patient_schemas)),
    )
    write_csv(outputs["between"], comparisons)
    write_csv(outputs["within"], within_rows)
    write_csv(outputs["resources"], resource_rows)

    stagea_clean = next(
        row
        for row in comparisons
        if row["comparison_id"] == "stagea_full_vs_stagea_core"
        and row["endpoint_role"] == "primary_clean"
    )
    stagea_robust = next(
        row
        for row in comparisons
        if row["comparison_id"] == "stagea_full_vs_stagea_core"
        and row["endpoint_role"] == "primary_robustness_composite"
    )
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "selection_epoch": 30,
        "numeric_ranking_arms": list(VALID_ARMS),
        "excluded_arm": {
            "arm": "stageb_core",
            **status_document["arms"]["stageb_core"],
        },
        "best_within_30_substitution_forbidden": True,
        "bootstrap_unit": "patient",
        "bootstrap_resamples": args.bootstrap_resamples,
        "between_arm_direction": "candidate minus reference; negative favors candidate",
        "stagea_primary_clean": stagea_clean,
        "stagea_primary_robustness_composite": stagea_robust,
        "stagea_q_functionality": {
            arm: counterfactual[arm]["audit"]["q_functionality_gate"]
            for arm in ("stagea_core", "stagea_full")
        },
        "selection_guidance": {
            "full_clean_superiority": (
                "StageA-Full clean CI upper bound < 0"
            ),
            "robustness_guard": (
                "interpret together with the pre-specified robustness margin; "
                "this script does not silently invent a margin"
            ),
            "if_stagea_ci_crosses_zero": (
                "continue both StageA arms to epoch 40; if still unclear, "
                "select the simpler Core"
            ),
        },
        "locked_input_hashes": {
            key: next(iter(values)) for key, values in locked_hashes.items()
        },
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    outputs["report"].write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
