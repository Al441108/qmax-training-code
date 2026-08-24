#!/usr/bin/env python3
from __future__ import annotations

"""Exploratory full-clean comparison: PRNF Full versus M2-U Clean at R=8.

This is intentionally separate from the frozen six-model formal evaluator. It
uses the same manifest, checkpoint provenance checks, metric definitions and
patient-level bootstrap helpers, but only loads the two currently available
checkpoints.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_m2_prnf_R8 import (  # noqa: E402
    METRICS,
    PROTOCOL_VERSION,
    ManifestDataset,
    audit_fairness,
    bootstrap_improvement,
    bootstrap_noninferiority,
    build_model,
    evaluate_mode,
    patient_average,
    summary_rows,
    write_csv,
)
from scripts.train_m2_prnf import (  # noqa: E402
    IndexedDataset,
    ShapeBucketBatchSampler,
    locked_code_hashes,
    make_dataset,
    runtime_versions,
    set_seed,
    sha256_file,
)
from src.m2_prnf_corruptions import HardNegativeSampler  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--full_clean_manifest", required=True)
    parser.add_argument("--robustness_manifest", required=True)
    parser.add_argument("--m2u_clean_checkpoint", required=True)
    parser.add_argument("--prnf_full_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--bootstrap_resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_paths = {
        "full_clean": Path(args.full_clean_manifest).resolve(),
        "robustness": Path(args.robustness_manifest).resolve(),
    }
    manifests = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in manifest_paths.items()
    }
    hashes = {name: sha256_file(path) for name, path in manifest_paths.items()}
    for name, manifest in manifests.items():
        if manifest.get("protocol_version") != PROTOCOL_VERSION:
            raise RuntimeError(f"{name}: manifest protocol mismatch")
        if manifest.get("cohort") != name:
            raise RuntimeError(f"{name}: manifest cohort mismatch")
    if int(manifests["full_clean"].get("num_patients", -1)) != 25:
        raise RuntimeError("Full-clean manifest must contain 25 patients")
    if int(manifests["full_clean"].get("num_slices", -1)) != 878:
        raise RuntimeError("Full-clean manifest must contain 878 slices")

    dataset_args = argparse.Namespace(
        metadata_csv=str(Path(args.metadata_csv).resolve()),
        acceleration=8,
        pd_aux_acceleration=2,
    )
    full_dataset = IndexedDataset(make_dataset(dataset_args, "val"))
    clean_dataset = ManifestDataset(full_dataset, manifests["full_clean"])
    clean_loader = DataLoader(
        clean_dataset,
        batch_sampler=ShapeBucketBatchSampler(
            clean_dataset, args.batch_size, False, args.seed
        ),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    negative_sampler = HardNegativeSampler(full_dataset)

    checkpoint_paths = {
        "m2u_clean": Path(args.m2u_clean_checkpoint).resolve(),
        "prnf_full": Path(args.prnf_full_checkpoint).resolve(),
    }
    slice_rows = []
    configs = {}
    checkpoint_audit = {}
    for expected_name, checkpoint_path in checkpoint_paths.items():
        model, observed_name, config, selected_epoch = build_model(
            checkpoint_path,
            device,
            hashes["full_clean"],
            hashes["robustness"],
        )
        if observed_name != expected_name:
            raise RuntimeError(
                f"Expected {expected_name}, checkpoint contains {observed_name}"
            )
        configs[expected_name] = config
        checkpoint_audit[expected_name] = {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "selected_checkpoint_epoch": selected_epoch,
            "training_budget_epochs": 50,
            "seed": int(config["seed"]),
        }
        slice_rows.extend(
            evaluate_mode(
                model,
                expected_name,
                clean_loader,
                full_dataset,
                negative_sampler,
                device,
                "full_clean",
                "correct",
                seed=args.seed,
            )
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fairness = audit_fairness(configs)
    if not fairness["passed"]:
        raise RuntimeError(
            "Cross-checkpoint fairness failed:\n"
            + json.dumps(fairness["mismatches"], indent=2)
        )

    patient_rows = patient_average(slice_rows)
    summaries = summary_rows(patient_rows)
    comparisons = [
        bootstrap_improvement(
            patient_rows,
            "prnf_full",
            "m2u_clean",
            "full_clean",
            "correct",
            metric,
            args.bootstrap_resamples,
            args.seed,
        )
        for metric in METRICS
    ]
    noninferiority = bootstrap_noninferiority(
        patient_rows,
        "prnf_full",
        "m2u_clean",
        "full_clean",
        "correct",
        0.005,
        args.bootstrap_resamples,
        args.seed,
    )
    comparison_by_metric = {row["metric"]: row for row in comparisons}
    point_improvements = {
        metric: comparison_by_metric[metric]["mean_improvement"] > 0
        for metric in METRICS
    }
    ci_superiority = {
        metric: comparison_by_metric[metric]["ci95_low"] > 0
        for metric in METRICS
    }

    if all(ci_superiority.values()):
        interpretation = "PRNF_FULL_CLEAN_SUPERIOR_ACROSS_ALL_METRICS"
    elif all(point_improvements.values()):
        interpretation = "PRNF_FULL_POINT_ESTIMATE_IMPROVEMENT_NOT_UNIFORMLY_CONFIRMED"
    elif noninferiority["passed"]:
        interpretation = "PRNF_FULL_CLEAN_NONINFERIOR_WITH_MIXED_METRIC_RESULTS"
    else:
        interpretation = "PRNF_FULL_CLEAN_NONINFERIORITY_FAILED"

    decision = {
        "protocol_version": PROTOCOL_VERSION,
        "scope": "exploratory two-model full-clean comparison",
        "cohort": {
            "num_patients": manifests["full_clean"]["num_patients"],
            "num_slices": manifests["full_clean"]["num_slices"],
            "patient_level_equal_weighting": True,
        },
        "prnf_full_vs_m2u_clean": {
            "paired_bootstrap": comparisons,
            "l1_noninferiority": noninferiority,
            "all_metric_point_estimates_favour_prnf": all(point_improvements.values()),
            "all_metric_95ci_confirm_prnf_superiority": all(ci_superiority.values()),
            "interpretation": interpretation,
        },
        "warning": (
            "This clean-only analysis does not establish robustness or isolate "
            "the contributions of reliability and need modules."
        ),
    }

    write_csv(output_dir / "clean_pair_per_slice.csv", slice_rows)
    write_csv(output_dir / "clean_pair_patient_level.csv", patient_rows)
    write_csv(output_dir / "clean_pair_summary.csv", summaries)
    write_csv(output_dir / "clean_pair_paired_bootstrap.csv", comparisons)
    (output_dir / "clean_pair_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    audit = {
        "manifests": {
            name: {
                "path": str(path),
                "sha256": hashes[name],
                "num_patients": manifests[name]["num_patients"],
                "num_slices": manifests[name]["num_slices"],
            }
            for name, path in manifest_paths.items()
        },
        "checkpoints": checkpoint_audit,
        "cross_checkpoint_fairness": fairness,
        "code_hashes": locked_code_hashes(),
        "runtime_versions": runtime_versions(),
        "bootstrap_resamples": args.bootstrap_resamples,
        "seed": args.seed,
    }
    (output_dir / "clean_pair_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
