#!/usr/bin/env python3
from __future__ import annotations

"""Paired patient-level Stage-A comparison of QMax-Full versus QMax-Core."""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def paired_values(
    core_rows: List[Mapping[str, str]],
    full_rows: List[Mapping[str, str]],
    cohort: str,
    condition: str,
    metric: str,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    def select(rows):
        return {
            str(row["patient_id"]): float(row[metric])
            for row in rows
            if row["mode"] == "full"
            and row["cohort"] == cohort
            and row["condition"] == condition
        }

    core = select(core_rows)
    full = select(full_rows)
    if set(core) != set(full):
        raise RuntimeError(
            f"Patient mismatch for {cohort}/{condition}/{metric}"
        )
    patients = sorted(core)
    return (
        patients,
        np.asarray([core[patient] for patient in patients]),
        np.asarray([full[patient] for patient in patients]),
    )


def paired_bootstrap(
    core_rows,
    full_rows,
    cohort,
    condition,
    metric,
    resamples,
    seed,
):
    patients, core, full = paired_values(
        core_rows, full_rows, cohort, condition, metric
    )
    difference = full - core
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(patients), size=(int(resamples), len(patients))
    )
    draws = difference[indices].mean(axis=1)
    return {
        "cohort": cohort,
        "condition": condition,
        "metric": metric,
        "delta_full_minus_core": float(difference.mean()),
        "core_mean": float(core.mean()),
        "full_mean": float(full.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "full_better_patients": int(
            (difference < 0).sum()
            if metric in {"l1", "nmse"}
            else (difference > 0).sum()
        ),
        "num_patients": len(patients),
    }


def within_arm_bootstrap(
    rows,
    *,
    candidate_mode,
    reference_mode,
    cohort,
    condition,
    metric,
    resamples,
    seed,
):
    def select(mode):
        selected = {}
        for row in rows:
            if (
                row["mode"] != mode
                or row["cohort"] != cohort
                or row["condition"] != condition
            ):
                continue
            patient = str(row["patient_id"])
            if patient in selected:
                raise RuntimeError(
                    f"Duplicate patient row for {mode} "
                    f"{cohort}/{condition}/{metric}: {patient}"
                )
            selected[patient] = float(row[metric])
        return selected

    candidate = select(candidate_mode)
    reference = select(reference_mode)
    if not candidate or not reference:
        raise RuntimeError(
            f"No within-arm pairs for {candidate_mode}/{reference_mode} "
            f"{cohort}/{condition}/{metric}"
        )
    if set(candidate) != set(reference):
        candidate_only = sorted(set(candidate) - set(reference))
        reference_only = sorted(set(reference) - set(candidate))
        raise RuntimeError(
            "Within-arm patient sets differ for "
            f"{candidate_mode}/{reference_mode} "
            f"{cohort}/{condition}/{metric}; "
            f"candidate_only={candidate_only[:8]}, "
            f"reference_only={reference_only[:8]}"
        )
    patients = sorted(candidate)
    difference = np.asarray(
        [
            candidate[patient] - reference[patient]
            for patient in patients
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(patients), size=(int(resamples), len(patients))
    )
    draws = difference[indices].mean(axis=1)
    return {
        "candidate_mode": candidate_mode,
        "reference_mode": reference_mode,
        "cohort": cohort,
        "condition": condition,
        "metric": metric,
        "delta_candidate_minus_reference": float(difference.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "candidate_better_patients": int((difference < 0).sum()),
        "num_patients": len(patients),
    }


def q_separation_bootstrap(rows, resamples, seed):
    def select(condition):
        selected = {}
        for row in rows:
            if (
                row["mode"] != "full"
                or row["cohort"] != "robustness"
                or row["condition"] != condition
            ):
                continue
            patient = str(row["patient_id"])
            if patient in selected:
                raise RuntimeError(
                    "Duplicate robustness-cohort q row for "
                    f"{condition}: {patient}"
                )
            selected[patient] = {
                "q": float(row["q"]),
                "num_slices": int(row["num_slices"]),
            }
        return selected

    clean = select("correct")
    corrupt = select("robustness_composite")
    if set(clean) != set(corrupt) or not clean:
        clean_only = sorted(set(clean) - set(corrupt))
        corrupt_only = sorted(set(corrupt) - set(clean))
        raise RuntimeError(
            "Robustness-cohort correct/corrupt q patient pairing is "
            f"incomplete; clean_only={clean_only[:8]}, "
            f"corrupt_only={corrupt_only[:8]}"
        )
    patients = sorted(clean)
    slice_count_mismatches = {
        patient: (
            clean[patient]["num_slices"],
            corrupt[patient]["num_slices"],
        )
        for patient in patients
        if clean[patient]["num_slices"]
        != corrupt[patient]["num_slices"]
    }
    if slice_count_mismatches:
        raise RuntimeError(
            "Robustness-cohort correct/composite q slice counts differ: "
            f"{dict(list(slice_count_mismatches.items())[:8])}"
        )
    difference = np.asarray(
        [
            clean[patient]["q"] - corrupt[patient]["q"]
            for patient in patients
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(patients), size=(int(resamples), len(patients))
    )
    draws = difference[indices].mean(axis=1)
    return {
        "definition": (
            "robustness-cohort correct q minus "
            "robustness-composite corrupt q"
        ),
        "delta_q_clean_minus_corrupt": float(difference.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "clean_q_higher_patients": int((difference > 0).sum()),
        "num_patients": len(patients),
    }


def q_functionality_gate(rows, resamples, seed):
    actual_vs_q1 = within_arm_bootstrap(
        rows,
        candidate_mode="full",
        reference_mode="q1",
        cohort="robustness",
        condition="robustness_composite",
        metric="l1",
        resamples=resamples,
        seed=seed,
    )
    actual_vs_constant = within_arm_bootstrap(
        rows,
        candidate_mode="full",
        reference_mode="constant_q",
        cohort="robustness",
        condition="robustness_composite",
        metric="l1",
        resamples=resamples,
        seed=seed,
    )
    separation = q_separation_bootstrap(rows, resamples, seed)
    missing_rows = [
        row
        for row in rows
        if row["mode"] == "full"
        and row["cohort"] == "robustness"
        and row["condition"] == "missing"
    ]
    if not missing_rows:
        raise RuntimeError("Missing-condition rows are absent")
    missing_exact_zero = all(
        float(row["missing_direct_exact_zero"]) == 1.0
        and float(row["missing_correction_exact_zero"]) == 1.0
        for row in missing_rows
    )
    actual_beats_q1 = (
        actual_vs_q1["ci95_high"] < 0.0
        and actual_vs_q1["candidate_better_patients"]
        > actual_vs_q1["num_patients"] / 2
    )
    actual_beats_constant = (
        actual_vs_constant["ci95_high"] < 0.0
        and actual_vs_constant["candidate_better_patients"]
        > actual_vs_constant["num_patients"] / 2
    )
    q_separates = (
        separation["ci95_low"] > 0.0
        and separation["clean_q_higher_patients"]
        > separation["num_patients"] / 2
    )
    return {
        "missing_exact_zero": missing_exact_zero,
        "actual_q_beats_q1": actual_beats_q1,
        "actual_q_beats_constant_q": actual_beats_constant,
        "q_clean_exceeds_q_corrupt": q_separates,
        "actual_q_vs_q1": actual_vs_q1,
        "actual_q_vs_constant_q": actual_vs_constant,
        "q_separation": separation,
        "passed": (
            missing_exact_zero
            and actual_beats_q1
            and actual_beats_constant
            and q_separates
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core_patient_csv", required=True)
    parser.add_argument("--full_patient_csv", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--bootstrap_resamples", type=int, default=10000)
    parser.add_argument(
        "--robust_noninferiority_margin_relative",
        type=float,
        default=0.005,
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    core_path = Path(args.core_patient_csv).resolve()
    full_path = Path(args.full_patient_csv).resolve()
    if not core_path.is_file() or not full_path.is_file():
        raise FileNotFoundError("Core/Full patient CSV is missing")
    core_rows = read_rows(core_path)
    full_rows = read_rows(full_path)
    q_gates = {
        "qmax_core": q_functionality_gate(
            core_rows, args.bootstrap_resamples, args.seed
        ),
        "qmax_full": q_functionality_gate(
            full_rows, args.bootstrap_resamples, args.seed
        ),
    }
    comparisons = []
    for cohort, condition in (
        ("full_clean", "correct"),
        ("robustness", "robustness_composite"),
        ("robustness", "shift8"),
        ("robustness", "wrong_slice"),
        ("robustness", "wrong_patient"),
        ("robustness", "missing"),
    ):
        for metric in ("l1", "nmse", "psnr", "ssim"):
            comparisons.append(
                paired_bootstrap(
                    core_rows,
                    full_rows,
                    cohort,
                    condition,
                    metric,
                    args.bootstrap_resamples,
                    args.seed,
                )
            )
    lookup = {
        (row["cohort"], row["condition"], row["metric"]): row
        for row in comparisons
    }
    clean = lookup[("full_clean", "correct", "l1")]
    robust = lookup[
        ("robustness", "robustness_composite", "l1")
    ]
    robust_margin_absolute = (
        args.robust_noninferiority_margin_relative * robust["core_mean"]
    )
    clean_superior = clean["ci95_high"] < 0.0
    robust_noninferior = (
        robust["ci95_high"] < robust_margin_absolute
    )
    core_eligible = bool(q_gates["qmax_core"]["passed"])
    full_eligible = bool(q_gates["qmax_full"]["passed"])
    if core_eligible and full_eligible:
        selected = (
            "qmax_full"
            if clean_superior and robust_noninferior
            else "qmax_core"
        )
        selection_reason = "both_pass_q_gate_then_apply_performance_rule"
    elif core_eligible:
        selected = "qmax_core"
        selection_reason = "qmax_full_failed_q_or_missing_gate"
    elif full_eligible and robust_noninferior and clean[
        "delta_full_minus_core"
    ] <= 0.0:
        selected = "qmax_full"
        selection_reason = (
            "only_qmax_full_passed_q_gate_and_it_was_not_clean_worse"
        )
    else:
        selected = "no_eligible_qmax"
        selection_reason = "no_arm_satisfied_all_frozen_safety_rules"

    decision = {
        "scope": (
            "Stage-A mechanism selection between QMax-Core and QMax-Full; "
            "historical Fifth/P0 is not an automatically paired 30-epoch arm"
        ),
        "rule": (
            "first require missing exact-zero, actual-q superiority over q=1 "
            "and constant-q, and robustness-cohort correct q greater than "
            "corrupt-composite q with paired 95% CI; then apply clean "
            "superiority/robustness non-inferiority"
        ),
        "q_functionality_and_missing_gate": q_gates,
        "clean_superiority": clean_superior,
        "robustness_noninferiority": robust_noninferior,
        "robustness_margin_relative": (
            args.robust_noninferiority_margin_relative
        ),
        "robustness_margin_absolute": robust_margin_absolute,
        "selected_mechanism": selected,
        "selection_reason": selection_reason,
    }
    result = {
        "protocol_version": "QMax-StageA-comparison-v3",
        "core_patient_csv": str(core_path),
        "full_patient_csv": str(full_path),
        "bootstrap_resamples": args.bootstrap_resamples,
        "seed": args.seed,
        "comparisons": comparisons,
        "q_functionality_gates": q_gates,
        "decision": decision,
    }
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
