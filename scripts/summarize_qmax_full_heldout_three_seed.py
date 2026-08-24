#!/usr/bin/env python3
from __future__ import annotations

"""Seed-equal summary of the frozen QMax-Full held-out evaluation."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PROTOCOL_VERSION = "QMax-Full-heldout-three-seed-summary-v1"
EVAL_PROTOCOL_VERSION = "QMax-Full-heldout-clean-evaluation-v1"
SEEDS = (42, 123, 2026)
METRICS = ("l1", "nmse", "psnr", "ssim")
ANALYSIS_SEED = 42


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def hierarchical_ci(
    by_seed: dict[int, dict[str, dict[str, float]]], metric: str, resamples: int
) -> tuple[float, float]:
    rng = np.random.default_rng(ANALYSIS_SEED)
    seeds = np.asarray(SEEDS)
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        selected_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        seed_means = []
        for seed_value in selected_seeds:
            values = np.asarray(
                [row[metric] for row in by_seed[int(seed_value)].values()],
                dtype=np.float64,
            )
            sample = rng.choice(values, size=values.size, replace=True)
            seed_means.append(float(sample.mean()))
        draws[index] = float(np.mean(seed_means))
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for seed in SEEDS:
        parser.add_argument(f"--seed{seed}_patient", required=True)
        parser.add_argument(f"--seed{seed}_audit", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bootstrap_resamples", type=int, default=10000)
    args = parser.parse_args()

    if args.bootstrap_resamples < 1000:
        raise ValueError("At least 1000 bootstrap resamples are required")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "per_seed": output_dir / "heldout_three_seed_performance.csv",
        "aggregate": output_dir / "heldout_three_seed_performance_aggregate.csv",
        "patient_seed_mean": output_dir / "heldout_patient_seed_mean.csv",
        "audit": output_dir / "heldout_three_seed_summary_audit.json",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise RuntimeError("Refusing to overwrite held-out summary: " + ", ".join(existing))

    by_seed: dict[int, dict[str, dict[str, float]]] = {}
    manifest_hashes = set()
    checkpoint_hashes = {}
    for seed in SEEDS:
        audit_path = Path(getattr(args, f"seed{seed}_audit")).resolve()
        patient_path = Path(getattr(args, f"seed{seed}_patient")).resolve()
        audit = load_json(audit_path)
        if audit.get("protocol_version") != EVAL_PROTOCOL_VERSION or audit.get("status") != "passed":
            raise RuntimeError(f"Seed {seed} held-out audit did not pass")
        if int(audit.get("model_seed", -1)) != seed:
            raise RuntimeError(f"Seed {seed} audit identity mismatch")
        manifest_hashes.add(str(audit["test_manifest_sha256"]))
        checkpoint_hashes[str(seed)] = audit["checkpoint_audit"]["checkpoint_sha256"]
        rows = load_csv(patient_path)
        selected = {}
        for row in rows:
            patient = str(row["patient_id"])
            if patient in selected:
                raise RuntimeError(f"Duplicate seed {seed} patient: {patient}")
            selected[patient] = {metric: float(row[metric]) for metric in METRICS}
        if len(selected) != 34:
            raise RuntimeError(f"Seed {seed} has {len(selected)} patients, expected 34")
        by_seed[seed] = selected

    if len(manifest_hashes) != 1:
        raise RuntimeError("Seeds used different held-out manifests")
    patient_sets = [set(by_seed[seed]) for seed in SEEDS]
    if not all(values == patient_sets[0] for values in patient_sets[1:]):
        raise RuntimeError("Held-out patient identities differ across seeds")

    per_seed = []
    for seed in SEEDS:
        for metric in METRICS:
            values = np.asarray(
                [row[metric] for row in by_seed[seed].values()], dtype=np.float64
            )
            per_seed.append(
                {
                    "model_seed": seed,
                    "metric": metric,
                    "patient_equal_mean": float(values.mean()),
                    "patient_sample_sd": float(values.std(ddof=1)),
                    "num_patients": values.size,
                }
            )

    aggregate = []
    for metric in METRICS:
        seed_means = np.asarray(
            [
                np.mean([row[metric] for row in by_seed[seed].values()])
                for seed in SEEDS
            ],
            dtype=np.float64,
        )
        low, high = hierarchical_ci(by_seed, metric, args.bootstrap_resamples)
        aggregate.append(
            {
                "metric": metric,
                "seed_equal_mean": float(seed_means.mean()),
                "between_seed_sample_sd": float(seed_means.std(ddof=1)),
                "minimum_seed_mean": float(seed_means.min()),
                "maximum_seed_mean": float(seed_means.max()),
                "num_seeds": len(SEEDS),
                "num_patients_per_seed": 34,
                "hierarchical_ci95_low": low,
                "hierarchical_ci95_high": high,
                "hierarchical_ci_status": "exploratory because only three prespecified random seeds",
            }
        )

    patient_seed_mean = []
    for patient in sorted(patient_sets[0]):
        patient_seed_mean.append(
            {
                "patient_id": patient,
                **{
                    metric: float(
                        np.mean([by_seed[seed][patient][metric] for seed in SEEDS])
                    )
                    for metric in METRICS
                },
                "num_seeds": len(SEEDS),
            }
        )

    write_csv(outputs["per_seed"], per_seed)
    write_csv(outputs["aggregate"], aggregate)
    write_csv(outputs["patient_seed_mean"], patient_seed_mean)
    audit = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "passed",
        "scope": "frozen clean held-out test; no model/seed selection",
        "model_seeds": list(SEEDS),
        "strict_patient_identity_equality": True,
        "num_patients_per_seed": 34,
        "test_manifest_sha256": next(iter(manifest_hashes)),
        "checkpoint_hashes": checkpoint_hashes,
        "primary_reporting": "every seed plus seed-equal mean and between-seed sample SD",
        "best_seed_reporting_forbidden": True,
        "hierarchical_bootstrap_status": "exploratory because only three prespecified random seeds",
        "bootstrap_resamples": args.bootstrap_resamples,
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    outputs["audit"].write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
