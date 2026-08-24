#!/usr/bin/env python3
from __future__ import annotations

"""Strict four-arm patient-paired epoch-30 L1 comparison."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np


PROTOCOL_VERSION = "QMax-epoch30-four-arm-comparison-v1"
ARM_ORDER = (
    "stagea_core",
    "stagea_full",
    "stageb_core",
    "stageb_full",
)
PAIR_SPECS = (
    ("stagea_full_vs_stagea_core", "stagea_full", "stagea_core"),
    ("stageb_full_vs_stageb_core", "stageb_full", "stageb_core"),
    ("stageb_core_vs_stagea_core", "stageb_core", "stagea_core"),
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
REQUIRED_FILES = {
    "slice": "epoch30_slice_metrics.csv",
    "patient": "epoch30_patient_metrics.csv",
    "audit": "epoch30_evaluation_audit.json",
}


def _read_csv(path: Path) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = tuple(reader.fieldnames or ())
    if not rows or not fields:
        raise RuntimeError(f"Empty or headerless CSV: {path}")
    return rows, fields


def _write_csv(
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


def _load_arm(
    arm: str, directory: Path
) -> Dict[str, Any]:
    paths = {
        key: directory / filename
        for key, filename in REQUIRED_FILES.items()
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{arm} evaluation is incomplete: {missing}"
        )
    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    if (
        audit.get("status") != "passed"
        or audit.get("arm") != arm
        or int(audit.get("checkpoint_epoch", -1)) != 30
    ):
        raise RuntimeError(f"Invalid epoch-30 evaluation audit for {arm}")
    patient_rows, patient_fields = _read_csv(paths["patient"])
    slice_rows, slice_fields = _read_csv(paths["slice"])
    for label, rows in (("patient", patient_rows), ("slice", slice_rows)):
        if any(row.get("arm") != arm for row in rows):
            raise RuntimeError(f"{arm} mislabeled rows in {label} CSV")
        if any(row.get("mode") != "full" for row in rows):
            raise RuntimeError(f"{arm} contains non-full modes in {label}")
        if any(int(row.get("checkpoint_epoch", -1)) != 30 for row in rows):
            raise RuntimeError(f"{arm} contains non-epoch30 {label} rows")
    return {
        "directory": str(directory),
        "paths": {key: str(value) for key, value in paths.items()},
        "audit": audit,
        "patient_rows": patient_rows,
        "patient_fields": patient_fields,
        "slice_rows": slice_rows,
        "slice_fields": slice_fields,
    }


def _unique_key_check(
    arm: str,
    rows: Sequence[Mapping[str, str]],
    keys: Sequence[str],
    label: str,
) -> set[tuple[str, ...]]:
    observed: set[tuple[str, ...]] = set()
    for row in rows:
        key = tuple(str(row[name]) for name in keys)
        if key in observed:
            raise RuntimeError(f"Duplicate {arm} {label} key: {key}")
        observed.add(key)
    return observed


def _select_patient_l1(
    rows: Sequence[Mapping[str, str]],
    cohort: str,
    condition: str,
) -> Dict[str, Dict[str, float]]:
    selected: Dict[str, Dict[str, float]] = {}
    for row in rows:
        if row["cohort"] != cohort or row["condition"] != condition:
            continue
        patient = str(row["patient_id"])
        if patient in selected:
            raise RuntimeError(
                f"Duplicate patient for {cohort}/{condition}: {patient}"
            )
        selected[patient] = {
            "l1": float(row["l1"]),
            "num_slices": float(row["num_slices"]),
        }
    if not selected:
        raise RuntimeError(f"No patients for {cohort}/{condition}")
    return selected


def _paired_comparison(
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
    candidate = _select_patient_l1(
        loaded[candidate_arm]["patient_rows"], cohort, condition
    )
    reference = _select_patient_l1(
        loaded[reference_arm]["patient_rows"], cohort, condition
    )
    if set(candidate) != set(reference):
        raise RuntimeError(
            f"Patient sets differ for {comparison_id} {cohort}/{condition}; "
            f"candidate_only={sorted(set(candidate)-set(reference))[:8]}, "
            f"reference_only={sorted(set(reference)-set(candidate))[:8]}"
        )
    slice_mismatch = {
        patient: (
            candidate[patient]["num_slices"],
            reference[patient]["num_slices"],
        )
        for patient in candidate
        if candidate[patient]["num_slices"]
        != reference[patient]["num_slices"]
    }
    if slice_mismatch:
        raise RuntimeError(
            f"Per-patient slice counts differ for {comparison_id}: "
            f"{dict(list(slice_mismatch.items())[:8])}"
        )
    patients = sorted(candidate)
    candidate_values = np.asarray(
        [candidate[patient]["l1"] for patient in patients],
        dtype=np.float64,
    )
    reference_values = np.asarray(
        [reference[patient]["l1"] for patient in patients],
        dtype=np.float64,
    )
    differences = candidate_values - reference_values
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(patients), size=(int(resamples), len(patients))
    )
    candidate_draws = candidate_values[indices].mean(axis=1)
    reference_draws = reference_values[indices].mean(axis=1)
    delta_draws = candidate_draws - reference_draws
    relative_draws = delta_draws / reference_draws
    reference_mean = float(reference_values.mean())
    delta = float(differences.mean())
    return {
        "comparison_id": comparison_id,
        "candidate_arm": candidate_arm,
        "reference_arm": reference_arm,
        "endpoint_role": endpoint_role,
        "cohort": cohort,
        "condition": condition,
        "metric": "l1",
        "candidate_mean_l1": float(candidate_values.mean()),
        "reference_mean_l1": reference_mean,
        "delta_l1_candidate_minus_reference": delta,
        "relative_delta_candidate_minus_reference": (
            delta / reference_mean
        ),
        "relative_delta_percent": 100.0 * delta / reference_mean,
        "ci95_delta_l1_low": float(np.quantile(delta_draws, 0.025)),
        "ci95_delta_l1_high": float(np.quantile(delta_draws, 0.975)),
        "ci95_relative_delta_low": float(
            np.quantile(relative_draws, 0.025)
        ),
        "ci95_relative_delta_high": float(
            np.quantile(relative_draws, 0.975)
        ),
        "ci95_relative_delta_percent_low": float(
            100.0 * np.quantile(relative_draws, 0.025)
        ),
        "ci95_relative_delta_percent_high": float(
            100.0 * np.quantile(relative_draws, 0.975)
        ),
        "candidate_better_patients": int((differences < 0).sum()),
        "reference_better_patients": int((differences > 0).sum()),
        "tied_patients": int((differences == 0).sum()),
        "num_patients": len(patients),
        "favorable_direction": (
            "negative delta means lower L1 for candidate"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ARM_ORDER:
        parser.add_argument(f"--{arm}_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bootstrap_resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.seed != 42:
        raise ValueError("Locked comparison requires seed=42")
    if args.bootstrap_resamples < 1000:
        raise ValueError("At least 1000 bootstrap resamples are required")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "combined_slice": output_dir
        / "epoch30_four_arm_slice_metrics.csv",
        "combined_patient": output_dir
        / "epoch30_four_arm_patient_metrics.csv",
        "comparisons_csv": output_dir
        / "epoch30_four_arm_paired_l1_comparisons.csv",
        "comparisons_json": output_dir
        / "epoch30_four_arm_paired_l1_comparisons.json",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise RuntimeError(
            "Refusing to overwrite comparison outputs: "
            + ", ".join(existing)
        )

    loaded = {
        arm: _load_arm(
            arm, Path(getattr(args, f"{arm}_dir")).resolve()
        )
        for arm in ARM_ORDER
    }
    patient_schemas = {
        loaded[arm]["patient_fields"] for arm in ARM_ORDER
    }
    slice_schemas = {loaded[arm]["slice_fields"] for arm in ARM_ORDER}
    if len(patient_schemas) != 1 or len(slice_schemas) != 1:
        raise RuntimeError("Four-arm evaluator CSV schemas differ")

    locked_hash_keys = (
        "metadata_csv",
        "full_clean_manifest",
        "robustness_manifest",
        "condition_manifest",
    )
    common_input_hashes = {
        key: {
            loaded[arm]["audit"]["input_hashes"][key]
            for arm in ARM_ORDER
        }
        for key in locked_hash_keys
    }
    drift = {
        key: values
        for key, values in common_input_hashes.items()
        if len(set(values.values())) != 1
    }
    if drift:
        raise RuntimeError(
            "Four arms were not evaluated on identical locked inputs:\n"
            + json.dumps(drift, indent=2)
        )

    patient_keys = {
        arm: _unique_key_check(
            arm,
            loaded[arm]["patient_rows"],
            ("cohort", "condition", "patient_id"),
            "patient",
        )
        for arm in ARM_ORDER
    }
    slice_keys = {
        arm: _unique_key_check(
            arm,
            loaded[arm]["slice_rows"],
            ("cohort", "condition", "patient_id", "slice_idx"),
            "slice",
        )
        for arm in ARM_ORDER
    }
    if len({frozenset(value) for value in patient_keys.values()}) != 1:
        raise RuntimeError("Four-arm patient evaluation keys differ")
    if len({frozenset(value) for value in slice_keys.values()}) != 1:
        raise RuntimeError("Four-arm slice evaluation keys differ")

    comparisons = [
        _paired_comparison(
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
    combined_slice = [
        row for arm in ARM_ORDER for row in loaded[arm]["slice_rows"]
    ]
    combined_patient = [
        row for arm in ARM_ORDER for row in loaded[arm]["patient_rows"]
    ]
    _write_csv(
        outputs["combined_slice"],
        combined_slice,
        next(iter(slice_schemas)),
    )
    _write_csv(
        outputs["combined_patient"],
        combined_patient,
        next(iter(patient_schemas)),
    )
    _write_csv(outputs["comparisons_csv"], comparisons)

    result = {
        "protocol_version": PROTOCOL_VERSION,
        "bootstrap_unit": "patient",
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.seed,
        "metric": "patient-level mean L1",
        "comparison_direction": "candidate minus reference",
        "negative_delta_favors_candidate": True,
        "arms": {
            arm: {
                "evaluation_directory": loaded[arm]["directory"],
                "checkpoint": loaded[arm]["audit"]["checkpoint"],
                "checkpoint_sha256": loaded[arm]["audit"][
                    "checkpoint_sha256"
                ],
                "checkpoint_epoch": loaded[arm]["audit"][
                    "checkpoint_epoch"
                ],
            }
            for arm in ARM_ORDER
        },
        "locked_input_hashes": {
            key: next(iter(values.values()))
            for key, values in common_input_hashes.items()
        },
        "strict_patient_keys_identical": True,
        "strict_slice_keys_identical": True,
        "comparisons": comparisons,
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    outputs["comparisons_json"].write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
