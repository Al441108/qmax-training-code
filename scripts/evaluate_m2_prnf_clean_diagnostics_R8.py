#!/usr/bin/env python3
from __future__ import annotations

"""Full-clean diagnostic evaluation for PRNF gating and auxiliary benefit.

Evaluates the same frozen PRNF Full checkpoint under actual q/n, q=1, n=1,
q=1 and n=1, and target-only inference. M2-U Clean is evaluated as the
primary reference. No checkpoint parameters are changed.
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
    slice_rows = []
    configs = {}
    checkpoint_audit = {}

    m2u_path = Path(args.m2u_clean_checkpoint).resolve()
    m2u_model, observed, m2u_config, selected_epoch = build_model(
        m2u_path, device, hashes["full_clean"], hashes["robustness"]
    )
    if observed != "m2u_clean":
        raise RuntimeError(f"Expected m2u_clean, checkpoint contains {observed}")
    configs["m2u_clean"] = m2u_config
    checkpoint_audit["m2u_clean"] = {
        "path": str(m2u_path),
        "sha256": sha256_file(m2u_path),
        "selected_checkpoint_epoch": selected_epoch,
        "training_budget_epochs": 50,
        "seed": int(m2u_config["seed"]),
    }
    slice_rows.extend(
        evaluate_mode(
            m2u_model, "m2u_clean", clean_loader, full_dataset,
            negative_sampler, device, "full_clean", "correct", seed=args.seed,
        )
    )
    del m2u_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    prnf_path = Path(args.prnf_full_checkpoint).resolve()
    prnf_model, observed, prnf_config, selected_epoch = build_model(
        prnf_path, device, hashes["full_clean"], hashes["robustness"]
    )
    if observed != "prnf_full":
        raise RuntimeError(f"Expected prnf_full, checkpoint contains {observed}")
    configs["prnf_full"] = prnf_config
    checkpoint_audit["prnf_full"] = {
        "path": str(prnf_path),
        "sha256": sha256_file(prnf_path),
        "selected_checkpoint_epoch": selected_epoch,
        "training_budget_epochs": 50,
        "seed": int(prnf_config["seed"]),
    }

    diagnostic_modes = (
        ("prnf_actual", None, None, False),
        ("prnf_q1", 1.0, None, False),
        ("prnf_n1", None, 1.0, False),
        ("prnf_q1_n1", 1.0, 1.0, False),
        ("prnf_target_only", None, None, True),
    )
    for name, q_override, n_override, target_only in diagnostic_modes:
        slice_rows.extend(
            evaluate_mode(
                prnf_model,
                name,
                clean_loader,
                full_dataset,
                negative_sampler,
                device,
                "full_clean",
                "correct",
                reliability_override=q_override,
                need_override=n_override,
                target_only=target_only,
                seed=args.seed,
            )
        )
    del prnf_model
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
    comparison_specs = (
        ("prnf_actual", "m2u_clean", "actual_vs_m2u_clean"),
        ("prnf_q1", "prnf_actual", "q1_vs_actual"),
        ("prnf_n1", "prnf_actual", "n1_vs_actual"),
        ("prnf_q1_n1", "prnf_actual", "q1_n1_vs_actual"),
        ("prnf_q1_n1", "m2u_clean", "q1_n1_vs_m2u_clean"),
        ("prnf_actual", "prnf_target_only", "actual_vs_target_only"),
    )
    comparisons = []
    grouped = {}
    for model_a, model_b, label in comparison_specs:
        grouped[label] = []
        for metric in METRICS:
            row = bootstrap_improvement(
                patient_rows,
                model_a,
                model_b,
                "full_clean",
                "correct",
                metric,
                args.bootstrap_resamples,
                args.seed,
            )
            row["comparison_label"] = label
            comparisons.append(row)
            grouped[label].append(row)

    noninferiority = bootstrap_noninferiority(
        patient_rows,
        "prnf_actual",
        "m2u_clean",
        "full_clean",
        "correct",
        0.005,
        args.bootstrap_resamples,
        args.seed,
    )

    def metric_map(label):
        return {row["metric"]: row for row in grouped[label]}

    q1n1_actual = metric_map("q1_n1_vs_actual")
    q1n1_m2u = metric_map("q1_n1_vs_m2u_clean")
    actual_target = metric_map("actual_vs_target_only")
    gating_suppression_l1 = q1n1_actual["l1"]["mean_improvement"] > 0
    gating_suppression_confirmed_l1 = q1n1_actual["l1"]["ci95_low"] > 0
    fusion_capacity_l1 = q1n1_m2u["l1"]["mean_improvement"] > 0
    fusion_capacity_confirmed_l1 = q1n1_m2u["l1"]["ci95_low"] > 0
    actual_aux_benefit_l1 = actual_target["l1"]["mean_improvement"] > 0

    if gating_suppression_confirmed_l1 and fusion_capacity_l1:
        diagnosis = "GATING_SUPPRESSES_A_CAPABLE_CLEAN_FUSION_PATH"
    elif not fusion_capacity_l1:
        diagnosis = "FUSION_OPERATOR_OR_TRAINING_OBJECTIVE_LIMITS_CLEAN_GAIN"
    elif actual_aux_benefit_l1 and noninferiority["passed"]:
        diagnosis = "AUXILIARY_PATH_BENEFICIAL_BUT_NOT_SUPERIOR_TO_M2U_CLEAN"
    else:
        diagnosis = "NO_CONFIRMED_CLEAN_AUXILIARY_BENEFIT"

    decision = {
        "protocol_version": PROTOCOL_VERSION,
        "scope": "exploratory full-clean gating diagnostics",
        "cohort": {
            "num_patients": manifests["full_clean"]["num_patients"],
            "num_slices": manifests["full_clean"]["num_slices"],
            "patient_level_equal_weighting": True,
        },
        "l1_noninferiority_actual_vs_m2u_clean": noninferiority,
        "diagnostic_flags": {
            "q1_n1_better_than_actual_l1_point_estimate": gating_suppression_l1,
            "q1_n1_better_than_actual_l1_ci_confirmed": gating_suppression_confirmed_l1,
            "q1_n1_better_than_m2u_clean_l1_point_estimate": fusion_capacity_l1,
            "q1_n1_better_than_m2u_clean_l1_ci_confirmed": fusion_capacity_confirmed_l1,
            "actual_prnf_better_than_target_only_l1_point_estimate": actual_aux_benefit_l1,
        },
        "diagnosis": diagnosis,
        "paired_bootstrap": grouped,
        "warning": (
            "Overrides are inference-time mechanism diagnostics, not separately "
            "trained models and not final ablation evidence."
        ),
    }

    write_csv(output_dir / "clean_diagnostic_per_slice.csv", slice_rows)
    write_csv(output_dir / "clean_diagnostic_patient_level.csv", patient_rows)
    write_csv(output_dir / "clean_diagnostic_summary.csv", summaries)
    write_csv(output_dir / "clean_diagnostic_paired_bootstrap.csv", comparisons)
    (output_dir / "clean_diagnostic_decision.json").write_text(
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
        "diagnostic_modes": [mode[0] for mode in diagnostic_modes],
    }
    (output_dir / "clean_diagnostic_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
