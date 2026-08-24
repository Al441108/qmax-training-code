#!/usr/bin/env python3
"""Formal 25-patient robustness confirmation for the selected R=8 model.

This protocol deliberately evaluates only five pre-registered auxiliary-input
conditions on 12 deterministic representative slices per validation patient:

  * correct paired PD;
  * shift8 with reflect padding in the +x direction;
  * same-patient wrong slice;
  * wrong-patient, exact-shape, matched anatomical level;
  * missing PD.

The selected Stage-B model is compared with Stage A, M2-GD v2, M2-U, an
audit-derived constant-q ablation, and q=1. All statistical summaries are
patient-level and include deterministic paired bootstrap confidence intervals.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_m2gd_v21_smoke_audit_R8 import (
    IndexedDataset,
    SelectedIndexDataset,
    ShapeBucketBatchSampler,
    aggregate_patient_level,
    aggregate_summary,
    alternative_batch,
    append_prediction_rows,
    assert_checkpoint_identity,
    batch_ints,
    choose_audit_patients,
    crop_prediction,
    diagnostics_per_sample,
    load_m2gd_v2,
    load_m2gd_v21,
    load_m2u,
    prepare_batch,
    set_seed,
)
from src.auxiliary_corruptions_v21 import (
    HardNegativeSampler,
    translate_nonwrapping,
)
from src.dataset_paired_multicoil_aux_pd_r2 import (
    PairedMulticoilAuxPDToPDFSDataset,
)


METRIC_COLUMNS = (
    "NMSE",
    "PSNR",
    "SSIM",
    "L1",
    "NMSE_central8",
    "PSNR_central8",
    "SSIM_central8",
    "L1_central8",
)
ERROR_METRICS = ("NMSE", "L1", "NMSE_central8", "L1_central8")
CONDITIONS = (
    "correct",
    "shift8_reflect_+x",
    "same_patient_wrong_slice",
    "wrong_patient_matched_level",
    "missing",
)
STAGEB_NAME = "M2GDv21_StageB_actual_q"
REFERENCE_MODELS = {
    "stagea": "M2GDv21_StageA",
    "m2gd_v2": "M2GDv2",
    "m2u": "M2U",
    "constant_q": "M2GDv21_StageB_constant_q",
    "q1": "M2GDv21_StageB_q1",
}


def stable_seed(base_seed: int, *parts: str) -> int:
    payload = "|".join(parts).encode("utf-8")
    offset = int(hashlib.sha256(payload).hexdigest()[:8], 16)
    return int((int(base_seed) + offset) % (2**32 - 1))


def paired_delta_table(patient_df: pd.DataFrame) -> pd.DataFrame:
    """Create one patient-paired row per condition and reference model."""
    candidate = patient_df[patient_df["model"] == STAGEB_NAME][
        ["condition", "patient_id", *METRIC_COLUMNS]
    ].copy()
    rows: List[pd.DataFrame] = []
    for reference_label, reference_model in REFERENCE_MODELS.items():
        reference = patient_df[patient_df["model"] == reference_model][
            ["condition", "patient_id", *METRIC_COLUMNS]
        ].copy()
        merged = candidate.merge(
            reference,
            on=["condition", "patient_id"],
            suffixes=("_stageb", "_reference"),
            validate="one_to_one",
        )
        merged.insert(0, "reference", reference_label)
        merged.insert(1, "reference_model", reference_model)
        for metric in METRIC_COLUMNS:
            stageb_column = f"{metric}_stageb"
            reference_column = f"{metric}_reference"
            improvement_column = f"{metric}_improvement"
            if metric in ERROR_METRICS:
                merged[improvement_column] = (
                    merged[reference_column] - merged[stageb_column]
                )
            else:
                merged[improvement_column] = (
                    merged[stageb_column] - merged[reference_column]
                )
        rows.append(merged)
    if not rows:
        raise RuntimeError("No paired reference comparisons were generated.")
    return pd.concat(rows, ignore_index=True)


def bootstrap_mean_ci(
    values: np.ndarray,
    iterations: int,
    seed: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise RuntimeError("At least two finite patients are required for bootstrap CI.")
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, values.size, size=(iterations, values.size))
    means = values[sampled].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(means, alpha)),
        float(np.quantile(means, 1.0 - alpha)),
    )


def paired_bootstrap_summary(
    paired_df: pd.DataFrame,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    """Summarize paired improvements; positive always favours Stage B."""
    rows: List[Dict[str, Any]] = []
    grouping = paired_df.groupby(["reference", "reference_model", "condition"])
    for (reference, reference_model, condition), group in grouping:
        if int(group["patient_id"].nunique()) != len(group):
            raise RuntimeError(
                f"Duplicate patients in {reference}/{condition} paired comparison."
            )
        for metric in METRIC_COLUMNS:
            values = group[f"{metric}_improvement"].to_numpy(dtype=np.float64)
            lower, upper = bootstrap_mean_ci(
                values,
                iterations=iterations,
                seed=stable_seed(seed, reference, condition, metric),
            )
            reference_values = group[f"{metric}_reference"].to_numpy(
                dtype=np.float64
            )
            stageb_values = group[f"{metric}_stageb"].to_numpy(dtype=np.float64)
            row: Dict[str, Any] = {
                "reference": reference,
                "reference_model": reference_model,
                "condition": condition,
                "metric": metric,
                "num_patients": int(group["patient_id"].nunique()),
                "stageb_mean": float(np.mean(stageb_values)),
                "reference_mean": float(np.mean(reference_values)),
                "paired_improvement_mean": float(np.mean(values)),
                "paired_improvement_median": float(np.median(values)),
                "paired_improvement_q25": float(np.quantile(values, 0.25)),
                "paired_improvement_q75": float(np.quantile(values, 0.75)),
                "paired_mean_ci95_lower": lower,
                "paired_mean_ci95_upper": upper,
                "proportion_stageb_better": float(np.mean(values > 0.0)),
                "proportion_equal": float(np.mean(values == 0.0)),
                "proportion_stageb_worse": float(np.mean(values < 0.0)),
            }
            if metric in ERROR_METRICS:
                row["relative_improvement_percent"] = float(
                    100.0
                    * row["paired_improvement_mean"]
                    / max(abs(row["reference_mean"]), 1e-12)
                )
            else:
                row["relative_improvement_percent"] = float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def summary_value(
    summary_df: pd.DataFrame,
    model: str,
    condition: str,
    column: str,
) -> float:
    row = summary_df[
        (summary_df["model"] == model) & (summary_df["condition"] == condition)
    ]
    if len(row) != 1:
        raise RuntimeError(
            f"Expected one summary row for model={model}, condition={condition}."
        )
    return float(row.iloc[0][column])


def build_mechanism_summary(
    summary_df: pd.DataFrame,
    constant_q: float,
) -> Dict[str, Any]:
    q_values = {
        condition: summary_value(
            summary_df, STAGEB_NAME, condition, "q_hat_mean_mean"
        )
        for condition in CONDITIONS
    }
    l1_ablation: Dict[str, Dict[str, float]] = {}
    for condition in CONDITIONS:
        actual = summary_value(summary_df, STAGEB_NAME, condition, "L1_mean")
        constant = summary_value(
            summary_df,
            REFERENCE_MODELS["constant_q"],
            condition,
            "L1_mean",
        )
        q1 = summary_value(
            summary_df, REFERENCE_MODELS["q1"], condition, "L1_mean"
        )
        l1_ablation[condition] = {
            "actual_q_l1": actual,
            "constant_q_l1": constant,
            "q1_l1": q1,
            "actual_over_constant": actual / max(constant, 1e-12),
            "actual_over_q1": actual / max(q1, 1e-12),
        }
    return {
        "constant_q_frozen_from_reduced_audit": float(constant_q),
        "stageb_q_hat": q_values,
        "q_gaps": {
            "correct_minus_shift8": q_values["correct"]
            - q_values["shift8_reflect_+x"],
            "correct_minus_wrong_slice": q_values["correct"]
            - q_values["same_patient_wrong_slice"],
            "correct_minus_wrong_patient": q_values["correct"]
            - q_values["wrong_patient_matched_level"],
        },
        "missing_effective_q": summary_value(
            summary_df, STAGEB_NAME, "missing", "q_mean_mean"
        ),
        "missing_gated_rms": summary_value(
            summary_df, STAGEB_NAME, "missing", "gated_rms_mean_mean"
        ),
        "l1_ablation": l1_ablation,
    }


def build_confirmation_decision(
    summary_df: pd.DataFrame,
    mechanism: Mapping[str, Any],
    num_patients: int,
    num_slices: int,
    fallback_counts: Mapping[str, int],
) -> Dict[str, Any]:
    criteria: Dict[str, bool] = {
        "all_25_patients_evaluated": int(num_patients) == 25,
        "all_300_slices_evaluated": int(num_slices) == 300,
        "no_hard_negative_fallbacks": sum(fallback_counts.values()) == 0,
        "missing_effective_q_exact_zero": abs(
            float(mechanism["missing_effective_q"])
        )
        <= 1e-10,
        "missing_gated_rms_exact_zero": abs(
            float(mechanism["missing_gated_rms"])
        )
        <= 1e-10,
        "correct_minus_shift8_q_at_least_0p10": float(
            mechanism["q_gaps"]["correct_minus_shift8"]
        )
        >= 0.10,
        "correct_minus_wrong_slice_q_at_least_0p10": float(
            mechanism["q_gaps"]["correct_minus_wrong_slice"]
        )
        >= 0.10,
        "correct_minus_wrong_patient_q_at_least_0p20": float(
            mechanism["q_gaps"]["correct_minus_wrong_patient"]
        )
        >= 0.20,
    }
    for condition in CONDITIONS:
        stageb = summary_value(summary_df, STAGEB_NAME, condition, "L1_mean")
        stagea = summary_value(
            summary_df, REFERENCE_MODELS["stagea"], condition, "L1_mean"
        )
        criteria[f"{condition}_l1_within_1pct_of_stagea"] = (
            stageb / max(stagea, 1e-12) <= 1.01
        )
    for condition in (
        "shift8_reflect_+x",
        "same_patient_wrong_slice",
        "wrong_patient_matched_level",
    ):
        values = mechanism["l1_ablation"][condition]
        criteria[f"{condition}_actual_q_not_worse_than_constant_by_0p5pct"] = (
            float(values["actual_over_constant"]) <= 1.005
        )
        criteria[f"{condition}_actual_q_not_worse_than_q1_by_0p5pct"] = (
            float(values["actual_over_q1"]) <= 1.005
        )
    return {
        "formal_robustness_confirmed": bool(all(criteria.values())),
        "criteria": criteria,
        "failed_criteria": [key for key, value in criteria.items() if not value],
        "interpretation": (
            "This is a 25-patient robustness confirmation of the already "
            "selected Stage-B checkpoint, not a new model-selection stage."
        ),
    }


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Formal 25-patient M2-GD v2.1 robustness confirmation at R=8."
    )
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--m2u_checkpoint", required=True)
    parser.add_argument("--m2gd_v2_checkpoint", required=True)
    parser.add_argument("--stagea_checkpoint", required=True)
    parser.add_argument("--stageb_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_patients", type=int, default=25)
    parser.add_argument("--slices_per_patient", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--bootstrap_iterations", type=int, default=10000)
    parser.add_argument(
        "--constant_q",
        type=float,
        default=0.9214919610725095,
        help=(
            "Fixed constant-q control pre-registered from the preceding "
            "six-patient reduced audit; it is not re-estimated on this cohort."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected_m2u_epoch", type=int, default=50)
    parser.add_argument("--expected_m2gd_v2_epoch", type=int, default=5)
    parser.add_argument("--expected_stagea_epoch", type=int, default=3)
    parser.add_argument("--expected_stageb_epoch", type=int, default=3)
    parser.add_argument(
        "--expected_stageb_sha256",
        default="a917421b98a3c8482c7cd019bba12eaf5568c72b4398ec142c47007fb9213837",
    )
    args = parser.parse_args()

    if args.num_patients != 25 or args.slices_per_patient != 12:
        raise ValueError(
            "The frozen formal protocol requires exactly 25 patients and "
            "12 representative slices per patient."
        )
    if args.bootstrap_iterations < 1000:
        raise ValueError("Use at least 1000 bootstrap iterations.")
    if not 0.0 <= args.constant_q <= 1.0:
        raise ValueError("constant_q must lie within [0,1].")

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Formal robustness evaluation requires a CUDA GPU.")

    base = PairedMulticoilAuxPDToPDFSDataset(
        metadata_csv=args.metadata_csv,
        split="val",
        pdfs_acceleration=8,
        pd_aux_acceleration=2,
        slices_per_patient=None,
        edge_weight=1.0,
    )
    full_dataset = IndexedDataset(base)
    patient_ids, selected_indices, selection_manifest = choose_audit_patients(
        full_dataset,
        num_patients=args.num_patients,
        slices_per_patient=args.slices_per_patient,
    )
    selected = SelectedIndexDataset(full_dataset, selected_indices)
    loader = DataLoader(
        selected,
        batch_sampler=ShapeBucketBatchSampler(
            selected, batch_size=args.batch_size, seed=args.seed
        ),
        num_workers=args.num_workers,
        pin_memory=True,
    )
    negative_sampler = HardNegativeSampler(full_dataset)

    m2u, m2u_identity = load_m2u(Path(args.m2u_checkpoint), device)
    m2gd_v2, m2gd_v2_identity = load_m2gd_v2(
        Path(args.m2gd_v2_checkpoint), device
    )
    stagea, stagea_identity = load_m2gd_v21(
        Path(args.stagea_checkpoint), device
    )
    stageb, stageb_identity = load_m2gd_v21(
        Path(args.stageb_checkpoint), device
    )
    assert_checkpoint_identity(
        m2u_identity, args.expected_m2u_epoch, 8, 2
    )
    assert_checkpoint_identity(
        m2gd_v2_identity, args.expected_m2gd_v2_epoch, 8, 2, "smoke5"
    )
    assert_checkpoint_identity(
        stagea_identity, args.expected_stagea_epoch, 8, 2, "paired3"
    )
    assert_checkpoint_identity(
        stageb_identity, args.expected_stageb_epoch, 8, 2, "fusioncal3"
    )
    if str(stageb_identity["sha256"]) != str(args.expected_stageb_sha256):
        raise RuntimeError(
            "Selected Stage-B SHA-256 mismatch: "
            f"{stageb_identity['sha256']} != {args.expected_stageb_sha256}"
        )

    constant_q = float(args.constant_q)
    print("=" * 96)
    print("M2-GD v2.1 formal robustness confirmation")
    print("GPU:", torch.cuda.get_device_name(0))
    print("Patients:", len(patient_ids))
    print("Slices:", len(selected))
    print("Batches:", len(loader))
    print(f"Pre-registered constant q: {constant_q:.8f}")
    print("=" * 96, flush=True)

    rows: List[Dict[str, Any]] = []
    fallback_counts: Dict[str, int] = defaultdict(int)

    for batch_index, batch in enumerate(loader, start=1):
        kspace, mask, pd_aux, target = prepare_batch(batch, device)
        source_indices = batch_ints(batch, "sample_idx")
        availability_one = torch.ones(
            pd_aux.shape[0], device=device, dtype=pd_aux.dtype
        )
        availability_zero = torch.zeros_like(availability_one)

        wrong_slice_indices: List[int] = []
        wrong_slice_delta: List[float] = []
        rng = random.Random(args.seed + batch_index * 1009)
        for source_index in source_indices:
            candidate = negative_sampler.same_patient_wrong_slice(source_index, rng)
            if candidate is None:
                fallback_counts["wrong_slice_unavailable"] += 1
                raise RuntimeError(
                    f"No same-patient wrong-slice candidate for {source_index}."
                )
            replacement_index, delta_z = candidate
            wrong_slice_indices.append(replacement_index)
            wrong_slice_delta.append(float(delta_z))
        wrong_slice_pd = alternative_batch(
            full_dataset,
            wrong_slice_indices,
            device,
            tuple(int(value) for value in pd_aux.shape[-2:]),
        )

        wrong_patient_indices: List[int] = []
        wrong_patient_delta: List[float] = []
        source_shape = tuple(int(value) for value in pd_aux.shape[-2:])
        for source_index in source_indices:
            candidate = negative_sampler.wrong_patient_matched_level(
                source_index, source_shape
            )
            if candidate is None:
                fallback_counts["wrong_patient_unavailable"] += 1
                raise RuntimeError(
                    "No exact-shape wrong-patient matched-level candidate for "
                    f"source={source_index}, shape={source_shape}."
                )
            replacement_index, delta_z = candidate
            wrong_patient_indices.append(replacement_index)
            wrong_patient_delta.append(float(delta_z))
        wrong_patient_pd = alternative_batch(
            full_dataset,
            wrong_patient_indices,
            device,
            source_shape,
        )

        shifted_pd = torch.stack(
            [
                translate_nonwrapping(image, 0, 8, "reflect")
                for image in pd_aux
            ],
            dim=0,
        )
        conditions: Sequence[
            Tuple[str, torch.Tensor, torch.Tensor, Mapping[str, Any]]
        ] = (
            ("correct", pd_aux, availability_one, {}),
            (
                "shift8_reflect_+x",
                shifted_pd,
                availability_one,
                {
                    "padding_mode": "reflect",
                    "direction": "+x",
                    "dx": 8,
                    "dy": 0,
                    "magnitude_linf": 8,
                },
            ),
            (
                "same_patient_wrong_slice",
                wrong_slice_pd,
                availability_one,
                {
                    "replacement_policy": "same_patient_wrong_slice",
                    "replacement_index": wrong_slice_indices,
                    "delta_z_norm": wrong_slice_delta,
                },
            ),
            (
                "wrong_patient_matched_level",
                wrong_patient_pd,
                availability_one,
                {
                    "replacement_policy": (
                        "wrong_patient_exact_shape_nearest_z_norm"
                    ),
                    "replacement_index": wrong_patient_indices,
                    "delta_z_norm": wrong_patient_delta,
                },
            ),
            (
                "missing",
                torch.zeros_like(pd_aux),
                availability_zero,
                {},
            ),
        )

        for condition, pd_input, availability, metadata in conditions:
            prediction = crop_prediction(
                m2u(kspace, mask, pd_input), target
            )
            append_prediction_rows(
                rows, prediction, target, batch, "M2U", condition, metadata
            )

            prediction, aux = m2gd_v2(
                pdfs_masked_kspace=kspace,
                mask=mask,
                pd_aux_image=pd_input,
                pd_available=availability,
                return_aux=True,
            )
            append_prediction_rows(
                rows,
                crop_prediction(prediction, target),
                target,
                batch,
                "M2GDv2",
                condition,
                metadata,
                diagnostics_per_sample(aux),
            )

            for model, model_name, q_override in (
                (stagea, "M2GDv21_StageA", None),
                (stageb, STAGEB_NAME, None),
                (
                    stageb,
                    "M2GDv21_StageB_constant_q",
                    float(constant_q),
                ),
                (stageb, "M2GDv21_StageB_q1", 1.0),
            ):
                prediction, aux = model(
                    pdfs_masked_kspace=kspace,
                    mask=mask,
                    pd_aux_image=pd_input,
                    pd_available=availability,
                    return_aux=True,
                    q_override=q_override,
                )
                append_prediction_rows(
                    rows,
                    crop_prediction(prediction, target),
                    target,
                    batch,
                    model_name,
                    condition,
                    metadata,
                    diagnostics_per_sample(aux),
                )

        if batch_index == 1 or batch_index % 10 == 0:
            print(
                f"Batch {batch_index:04d}/{len(loader)} complete | rows={len(rows)}",
                flush=True,
            )

    slice_df = pd.DataFrame(rows)
    patient_df = aggregate_patient_level(slice_df)
    summary_df = aggregate_summary(patient_df)
    paired_df = paired_delta_table(patient_df)
    bootstrap_df = paired_bootstrap_summary(
        paired_df,
        iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    mechanism = build_mechanism_summary(summary_df, constant_q)
    decision = build_confirmation_decision(
        summary_df,
        mechanism,
        num_patients=len(patient_ids),
        num_slices=len(selected),
        fallback_counts=fallback_counts,
    )

    selection_manifest.update(
        {
            "full_validation_patients": len(
                {str(record["patient_id"]) for record in full_dataset.records}
            ),
            "full_validation_slices": len(full_dataset),
            "conditions": list(CONDITIONS),
            "seed": args.seed,
        }
    )
    manifest = {
        "protocol": "M2-GD v2.1 formal robustness R8 revision 1",
        "metadata_csv": str(args.metadata_csv),
        "patient_level_aggregation": (
            "mean over 12 fixed slices per patient, then equal-weight patients"
        ),
        "bootstrap": {
            "unit": "patient",
            "iterations": args.bootstrap_iterations,
            "confidence": 0.95,
            "seed": args.seed,
        },
        "checkpoint_identity": {
            "m2u": m2u_identity,
            "m2gd_v2": m2gd_v2_identity,
            "stagea": stagea_identity,
            "stageb_selected": stageb_identity,
        },
        "constant_q": {
            "value": float(constant_q),
            "source": (
                "frozen mean correct-pair q from the preceding six-patient "
                "reduced audit; not estimated on the formal cohort"
            ),
        },
        "fallback_counts": dict(fallback_counts),
    }

    slice_df.to_csv(output_dir / "formal_robustness_per_slice.csv", index=False)
    patient_df.to_csv(
        output_dir / "formal_robustness_patient_level.csv", index=False
    )
    summary_df.to_csv(output_dir / "formal_robustness_summary.csv", index=False)
    paired_df.to_csv(
        output_dir / "formal_robustness_paired_patient_delta.csv", index=False
    )
    bootstrap_df.to_csv(
        output_dir / "formal_robustness_paired_bootstrap_summary.csv",
        index=False,
    )
    (output_dir / "formal_robustness_mechanism_summary.json").write_text(
        json.dumps(mechanism, indent=2), encoding="utf-8"
    )
    (output_dir / "formal_robustness_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    (output_dir / "formal_robustness_patient_selection.json").write_text(
        json.dumps(selection_manifest, indent=2), encoding="utf-8"
    )
    (output_dir / "formal_robustness_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("=" * 96)
    print(json.dumps(decision, indent=2))
    print("Saved formal robustness outputs to:", output_dir)
    print("=" * 96)


if __name__ == "__main__":
    main()
