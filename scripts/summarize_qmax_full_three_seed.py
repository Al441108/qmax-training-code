#!/usr/bin/env python3
from __future__ import annotations

"""Strict three-seed aggregation for QMax-Full locked validation results."""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np


PROTOCOL_VERSION = "QMax-Full-three-seed-summary-v1"
SEEDS = (42, 123, 2026)
METRICS = ("l1", "nmse", "psnr", "ssim")
MODES = (
    "full",
    "q1",
    "constant_q",
    "detail_neutral",
    "alignment_off",
    "correction_off",
    "dc_zero",
)
REFERENCES = MODES[1:]
ENDPOINTS = (
    ("full_clean", "correct"),
    ("robustness", "correct"),
    ("robustness", "shift8"),
    ("robustness", "wrong_slice"),
    ("robustness", "wrong_patient"),
    ("robustness", "missing"),
    ("robustness", "robustness_composite"),
)
ANALYSIS_SEED = 42


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_passed_audit(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "passed":
        raise RuntimeError(f"Evaluation audit did not pass: {path}")
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def patient_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row["mode"]),
        str(row["cohort"]),
        str(row["condition"]),
        str(row["patient_id"]),
    )


def merge_seed42(
    validation_rows: Iterable[dict[str, str]],
    component_rows: Iterable[dict[str, str]],
) -> list[dict[str, Any]]:
    merged: Dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for source, rows in (
        ("seed42_validation", validation_rows),
        ("seed42_components", component_rows),
    ):
        for original in rows:
            row: dict[str, Any] = dict(original)
            row["model_seed"] = 42
            key = patient_key(row)
            if key in merged:
                for metric in METRICS:
                    delta = abs(float(merged[key][metric]) - float(row[metric]))
                    if delta > 1e-10:
                        raise RuntimeError(
                            f"Seed42 duplicate full result differs at {key}/{metric}: "
                            f"delta={delta}"
                        )
                merged[key]["merged_sources"] += f"+{source}"
            else:
                row["merged_sources"] = source
                merged[key] = row
    return list(merged.values())


def normalize_replication_rows(
    rows: Iterable[dict[str, str]], seed: int
) -> list[dict[str, Any]]:
    output = []
    for original in rows:
        row: dict[str, Any] = dict(original)
        observed = int(row.get("model_seed", seed))
        if observed != seed:
            raise RuntimeError(f"Row seed {observed} != expected {seed}")
        row["model_seed"] = seed
        output.append(row)
    return output


def validate_rows(rows_by_seed: Mapping[int, list[dict[str, Any]]]) -> None:
    expected_modes = set(MODES)
    for seed in SEEDS:
        rows = rows_by_seed[seed]
        modes = {str(row["mode"]) for row in rows}
        if modes != expected_modes:
            raise RuntimeError(
                f"Seed {seed} modes differ: expected={sorted(expected_modes)}, "
                f"observed={sorted(modes)}"
            )
        keys = [patient_key(row) for row in rows]
        if len(keys) != len(set(keys)):
            raise RuntimeError(f"Seed {seed} has duplicate patient keys")
        for row in rows:
            for metric in METRICS:
                if not math.isfinite(float(row[metric])):
                    raise RuntimeError(f"Seed {seed} has non-finite {metric}")

    reference_keys = {patient_key(row) for row in rows_by_seed[42]}
    for seed in (123, 2026):
        observed = {patient_key(row) for row in rows_by_seed[seed]}
        if observed != reference_keys:
            missing = sorted(reference_keys - observed)[:10]
            extra = sorted(observed - reference_keys)[:10]
            raise RuntimeError(
                f"Seed {seed} patient keys differ from seed42; "
                f"missing={missing}, extra={extra}"
            )


def selected_values(
    rows: Iterable[Mapping[str, Any]],
    mode: str,
    cohort: str,
    condition: str,
    metric: str,
) -> Dict[str, float]:
    selected = {
        str(row["patient_id"]): float(row[metric])
        for row in rows
        if row["mode"] == mode
        and row["cohort"] == cohort
        and row["condition"] == condition
    }
    if not selected:
        raise RuntimeError(f"No rows for {mode}/{cohort}/{condition}/{metric}")
    return selected


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def paired_effect(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    reference: str,
    cohort: str,
    condition: str,
    metric: str,
    resamples: int,
) -> tuple[dict[str, Any], np.ndarray]:
    candidate = selected_values(rows, "full", cohort, condition, metric)
    control = selected_values(rows, reference, cohort, condition, metric)
    if set(candidate) != set(control):
        raise RuntimeError(
            f"Patient mismatch for seed={seed}, reference={reference}, "
            f"endpoint={cohort}/{condition}/{metric}"
        )
    patients = sorted(candidate)
    candidate_values = np.asarray([candidate[p] for p in patients], dtype=float)
    control_values = np.asarray([control[p] for p in patients], dtype=float)
    deltas = candidate_values - control_values
    stable_offset = sum(ord(char) for char in f"{reference}:{cohort}:{condition}:{metric}")
    rng = np.random.default_rng(ANALYSIS_SEED + seed + stable_offset)
    indices = rng.integers(0, len(deltas), size=(resamples, len(deltas)))
    bootstrap = deltas[indices].mean(axis=1)
    low, high = percentile_interval(bootstrap)
    lower_is_better = metric in {"l1", "nmse"}
    improved = deltas < 0.0 if lower_is_better else deltas > 0.0
    mean_reference = float(control_values.mean())
    mean_delta = float(deltas.mean())
    row = {
        "scope": "per_seed_patient_paired",
        "model_seed": seed,
        "candidate_mode": "full",
        "reference_mode": reference,
        "cohort": cohort,
        "condition": condition,
        "metric": metric,
        "candidate_mean": float(candidate_values.mean()),
        "reference_mean": mean_reference,
        "mean_delta_candidate_minus_reference": mean_delta,
        "relative_delta_percent": (
            100.0 * mean_delta / mean_reference
            if mean_reference != 0.0
            else float("nan")
        ),
        "ci95_low": low,
        "ci95_high": high,
        "num_patients": len(patients),
        "candidate_better_patients": int(improved.sum()),
        "candidate_worse_patients": int((~improved & (deltas != 0.0)).sum()),
        "ties": int((deltas == 0.0).sum()),
        "favorable_direction": "negative" if lower_is_better else "positive",
        "bootstrap_unit": "patient",
        "bootstrap_resamples": resamples,
    }
    return row, deltas


def hierarchical_effect(
    deltas_by_seed: Mapping[int, np.ndarray],
    *,
    reference: str,
    cohort: str,
    condition: str,
    metric: str,
    resamples: int,
) -> dict[str, Any]:
    means = np.asarray(
        [deltas_by_seed[seed].mean() for seed in SEEDS], dtype=float
    )
    stable_offset = sum(ord(char) for char in f"{reference}:{cohort}:{condition}:{metric}")
    rng = np.random.default_rng(ANALYSIS_SEED + 50000 + stable_offset)
    bootstrap = np.empty(resamples, dtype=float)
    seeds = np.asarray(SEEDS)
    for index in range(resamples):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        sampled_seed_means = []
        for seed in sampled_seeds:
            values = deltas_by_seed[int(seed)]
            sampled = rng.choice(values, size=len(values), replace=True)
            sampled_seed_means.append(float(sampled.mean()))
        bootstrap[index] = float(np.mean(sampled_seed_means))
    low, high = percentile_interval(bootstrap)
    lower_is_better = metric in {"l1", "nmse"}
    favorable = means < 0.0 if lower_is_better else means > 0.0
    return {
        "scope": "three_seed_hierarchical_descriptive",
        "candidate_mode": "full",
        "reference_mode": reference,
        "cohort": cohort,
        "condition": condition,
        "metric": metric,
        "seed_equal_mean_delta": float(means.mean()),
        "between_seed_sample_sd": float(means.std(ddof=1)),
        "minimum_seed_delta": float(means.min()),
        "maximum_seed_delta": float(means.max()),
        "favorable_seeds": int(favorable.sum()),
        "num_seeds": len(SEEDS),
        "hierarchical_ci95_low": low,
        "hierarchical_ci95_high": high,
        "bootstrap_outer_unit": "seed",
        "bootstrap_inner_unit": "patient",
        "bootstrap_resamples": resamples,
        "inference_status": (
            "exploratory: only three prespecified random seeds; report "
            "per-seed effects and mean+SD as primary replication evidence"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed42_validation_dir", required=True)
    parser.add_argument("--seed42_component_dir", required=True)
    parser.add_argument("--seed123_dir", required=True)
    parser.add_argument("--seed2026_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bootstrap_resamples", type=int, default=10000)
    args = parser.parse_args()
    if args.bootstrap_resamples < 1000:
        raise ValueError("At least 1000 bootstrap resamples are required")

    directories = {
        "seed42_validation": Path(args.seed42_validation_dir).resolve(),
        "seed42_components": Path(args.seed42_component_dir).resolve(),
        "seed123": Path(args.seed123_dir).resolve(),
        "seed2026": Path(args.seed2026_dir).resolve(),
    }
    for path in directories.values():
        if not path.is_dir():
            raise NotADirectoryError(path)

    input_files = {
        "seed42_validation_patient": directories["seed42_validation"]
        / "stagea_full_epoch60_patient_metrics.csv",
        "seed42_validation_audit": directories["seed42_validation"]
        / "stagea_full_epoch60_validation_audit.json",
        "seed42_component_patient": directories["seed42_components"]
        / "stagea_full_epoch60_component_patient_metrics.csv",
        "seed42_component_audit": directories["seed42_components"]
        / "stagea_full_epoch60_component_audit.json",
        "seed123_patient": directories["seed123"]
        / "qmax_full_seed123_epoch60_patient_metrics.csv",
        "seed123_audit": directories["seed123"]
        / "qmax_full_seed123_epoch60_evaluation_audit.json",
        "seed2026_patient": directories["seed2026"]
        / "qmax_full_seed2026_epoch60_patient_metrics.csv",
        "seed2026_audit": directories["seed2026"]
        / "qmax_full_seed2026_epoch60_evaluation_audit.json",
    }
    for key in (
        "seed42_validation_audit",
        "seed42_component_audit",
        "seed123_audit",
        "seed2026_audit",
    ):
        read_passed_audit(input_files[key])

    rows_by_seed = {
        42: merge_seed42(
            read_csv(input_files["seed42_validation_patient"]),
            read_csv(input_files["seed42_component_patient"]),
        ),
        123: normalize_replication_rows(
            read_csv(input_files["seed123_patient"]), 123
        ),
        2026: normalize_replication_rows(
            read_csv(input_files["seed2026_patient"]), 2026
        ),
    }
    validate_rows(rows_by_seed)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "performance_per_seed": output_dir / "three_seed_performance.csv",
        "performance_aggregate": output_dir
        / "three_seed_performance_aggregate.csv",
        "effects_per_seed": output_dir / "three_seed_paired_effects.csv",
        "effects_aggregate": output_dir
        / "three_seed_paired_effects_aggregate.csv",
        "audit": output_dir / "three_seed_summary_audit.json",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise RuntimeError("Refusing to overwrite outputs: " + ", ".join(existing))

    performance = []
    for seed in SEEDS:
        for mode in MODES:
            for cohort, condition in ENDPOINTS:
                for metric in METRICS:
                    values = selected_values(
                        rows_by_seed[seed], mode, cohort, condition, metric
                    )
                    array = np.asarray(list(values.values()), dtype=float)
                    performance.append(
                        {
                            "model_seed": seed,
                            "mode": mode,
                            "cohort": cohort,
                            "condition": condition,
                            "metric": metric,
                            "patient_equal_mean": float(array.mean()),
                            "patient_sd": float(array.std(ddof=1)),
                            "num_patients": len(array),
                        }
                    )

    grouped_performance: Dict[
        tuple[str, str, str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in performance:
        grouped_performance[
            (row["mode"], row["cohort"], row["condition"], row["metric"])
        ].append(row)
    performance_aggregate = []
    for (mode, cohort, condition, metric), rows in sorted(
        grouped_performance.items()
    ):
        if {int(row["model_seed"]) for row in rows} != set(SEEDS):
            raise RuntimeError("Performance group does not contain all seeds")
        values = np.asarray([row["patient_equal_mean"] for row in rows])
        performance_aggregate.append(
            {
                "mode": mode,
                "cohort": cohort,
                "condition": condition,
                "metric": metric,
                "seed_equal_mean": float(values.mean()),
                "between_seed_sample_sd": float(values.std(ddof=1)),
                "minimum_seed_mean": float(values.min()),
                "maximum_seed_mean": float(values.max()),
                "num_seeds": len(SEEDS),
            }
        )

    effects = []
    aggregate_effects = []
    for reference in REFERENCES:
        for cohort, condition in ENDPOINTS:
            for metric in METRICS:
                deltas_by_seed = {}
                for seed in SEEDS:
                    row, deltas = paired_effect(
                        rows_by_seed[seed],
                        seed=seed,
                        reference=reference,
                        cohort=cohort,
                        condition=condition,
                        metric=metric,
                        resamples=args.bootstrap_resamples,
                    )
                    effects.append(row)
                    deltas_by_seed[seed] = deltas
                aggregate_effects.append(
                    hierarchical_effect(
                        deltas_by_seed,
                        reference=reference,
                        cohort=cohort,
                        condition=condition,
                        metric=metric,
                        resamples=args.bootstrap_resamples,
                    )
                )

    write_csv(outputs["performance_per_seed"], performance)
    write_csv(outputs["performance_aggregate"], performance_aggregate)
    write_csv(outputs["effects_per_seed"], effects)
    write_csv(outputs["effects_aggregate"], aggregate_effects)

    audit = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "passed",
        "scope": "locked validation only; held-out test not accessed",
        "model_seeds": list(SEEDS),
        "analysis_seed": ANALYSIS_SEED,
        "strict_patient_key_equality_across_seeds": True,
        "seed42_full_duplicate_tolerance": 1e-10,
        "performance_aggregation": "patient-equal within seed, seed-equal across seeds",
        "primary_replication_reporting": (
            "show every seed plus seed-equal mean and between-seed sample SD"
        ),
        "hierarchical_bootstrap_status": (
            "exploratory because only three prespecified random seeds"
        ),
        "bootstrap_resamples": args.bootstrap_resamples,
        "input_hashes": {
            key: sha256_file(path) for key, path in input_files.items()
        },
        "row_counts_by_seed": {
            str(seed): len(rows_by_seed[seed]) for seed in SEEDS
        },
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    outputs["audit"].write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
