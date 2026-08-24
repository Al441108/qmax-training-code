#!/usr/bin/env python3
"""Full held-out clean-PD evaluation for the selected R=8 M2-GD v2.1 model.

This script evaluates every validation slice under the correct paired-PD
condition only.  It compares M2-U, M2-GD v2, the audited Stage-A incumbent,
and the selected Stage-B fusion-calibration model.  All inferential summaries
use patients, not slices, as the independent sampling unit.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the already audited checkpoint loaders, image metrics, batching and
# diagnostic extraction rather than maintaining a second implementation.
from evaluate_m2gd_v21_smoke_audit_R8 import (  # noqa: E402
    IndexedDataset,
    ShapeBucketBatchSampler,
    append_prediction_rows,
    assert_checkpoint_identity,
    checkpoint_sha256,
    crop_prediction,
    diagnostics_per_sample,
    load_m2gd_v2,
    load_m2gd_v21,
    load_m2u,
    prepare_batch,
    set_seed,
    ssim_fn,
)
from src.dataset_paired_multicoil_aux_pd_r2 import (  # noqa: E402
    PairedMulticoilAuxPDToPDFSDataset,
)


MODEL_NAMES = (
    "M2U",
    "M2GDv2",
    "M2GDv21_StageA",
    "M2GDv21_StageB",
)
CANDIDATE_MODEL = "M2GDv21_StageB"
REFERENCE_MODELS = (
    "M2U",
    "M2GDv2",
    "M2GDv21_StageA",
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
ERROR_METRICS = {
    "NMSE",
    "L1",
    "NMSE_central8",
    "L1_central8",
}


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def verify_sha256(path: Path, expected: Optional[str], label: str) -> str:
    observed = checkpoint_sha256(path)
    if expected is not None and observed.lower() != expected.lower():
        raise RuntimeError(
            f"{label} SHA-256 mismatch: observed={observed}, expected={expected}."
        )
    return observed


def finite_values(series: pd.Series, label: str) -> np.ndarray:
    values = series.to_numpy(dtype=float)
    if values.size == 0:
        raise RuntimeError(f"No values available for {label}.")
    if not np.all(np.isfinite(values)):
        bad = int(np.sum(~np.isfinite(values)))
        raise RuntimeError(f"{label} contains {bad} non-finite values.")
    return values


def percentile_summary(values: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "median": float(np.median(values)),
        "q1": float(np.quantile(values, 0.25)),
        "q3": float(np.quantile(values, 0.75)),
        "iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def bootstrap_mean_ci(
    values: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> Tuple[float, float]:
    if values.ndim != 1:
        raise ValueError("Bootstrap values must be one-dimensional.")
    if bootstrap_indices.shape[1] != values.size:
        raise ValueError(
            "Bootstrap index width does not match the number of patients."
        )
    bootstrap_means = values[bootstrap_indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    return float(lower), float(upper)


def make_bootstrap_indices(
    num_patients: int,
    num_resamples: int,
    seed: int,
) -> np.ndarray:
    if num_patients < 2:
        raise ValueError("At least two patients are required for bootstrapping.")
    if num_resamples < 1000:
        raise ValueError("Use at least 1000 patient-level bootstrap resamples.")
    rng = np.random.default_rng(seed)
    return rng.integers(
        low=0,
        high=num_patients,
        size=(num_resamples, num_patients),
        endpoint=False,
    )


def aggregate_patient_level(slice_df: pd.DataFrame) -> pd.DataFrame:
    grouping = ["model", "condition", "patient_id"]
    patient_df = (
        slice_df.groupby(grouping, as_index=False)[list(METRIC_COLUMNS)]
        .mean()
    )
    counts = (
        slice_df.groupby(grouping, as_index=False)
        .size()
        .rename(columns={"size": "num_slices"})
    )
    return patient_df.merge(
        counts,
        on=grouping,
        how="left",
        validate="one_to_one",
    )


def equal_weight_model_summary(
    patient_df: pd.DataFrame,
    bootstrap_indices: np.ndarray,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for model_name in MODEL_NAMES:
        model_rows = patient_df[patient_df["model"] == model_name].sort_values(
            "patient_id"
        )
        if model_rows.empty:
            raise RuntimeError(f"No patient-level results for {model_name}.")
        row: Dict[str, Any] = {
            "model": model_name,
            "condition": "correct",
            "num_patients": int(model_rows["patient_id"].nunique()),
            "num_slices": int(model_rows["num_slices"].sum()),
            "aggregation": "equal-weight mean over patient-level slice means",
        }
        for metric in METRIC_COLUMNS:
            values = finite_values(
                model_rows[metric], f"{model_name}/{metric}"
            )
            statistics = percentile_summary(values)
            ci_lower, ci_upper = bootstrap_mean_ci(values, bootstrap_indices)
            for name, value in statistics.items():
                row[f"{metric}_{name}"] = value
            row[f"{metric}_mean_bootstrap_ci95_lower"] = ci_lower
            row[f"{metric}_mean_bootstrap_ci95_upper"] = ci_upper
        rows.append(row)
    return pd.DataFrame(rows)


def paired_patient_delta(
    patient_df: pd.DataFrame,
    reference_model: str,
) -> pd.DataFrame:
    candidate = patient_df[patient_df["model"] == CANDIDATE_MODEL][
        ["patient_id", "num_slices", *METRIC_COLUMNS]
    ].copy()
    reference = patient_df[patient_df["model"] == reference_model][
        ["patient_id", "num_slices", *METRIC_COLUMNS]
    ].copy()
    reference_suffix = reference_model.lower().replace("m2gdv21_", "")
    paired = candidate.merge(
        reference,
        on="patient_id",
        suffixes=("_stageb", f"_{reference_suffix}"),
        validate="one_to_one",
    )
    if not np.array_equal(
        paired["num_slices_stageb"].to_numpy(),
        paired[f"num_slices_{reference_suffix}"].to_numpy(),
    ):
        raise RuntimeError(
            f"Stage B and {reference_model} have different per-patient slice counts."
        )
    paired.insert(0, "candidate_model", CANDIDATE_MODEL)
    paired.insert(1, "reference_model", reference_model)
    paired.insert(2, "condition", "correct")
    for metric in METRIC_COLUMNS:
        candidate_column = f"{metric}_stageb"
        reference_column = f"{metric}_{reference_suffix}"
        if metric in ERROR_METRICS:
            improvement = paired[reference_column] - paired[candidate_column]
            relative = (
                100.0
                * improvement
                / paired[reference_column].clip(lower=1e-12)
            )
            paired[f"{metric}_relative_improvement_pct"] = relative
        else:
            improvement = paired[candidate_column] - paired[reference_column]
        paired[f"{metric}_improvement"] = improvement
    return paired


def paired_comparison_summary(
    paired_frames: Mapping[str, pd.DataFrame],
    bootstrap_indices: np.ndarray,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    equality_tolerance = 1e-12
    for reference_model in REFERENCE_MODELS:
        paired = paired_frames[reference_model].sort_values("patient_id")
        row: Dict[str, Any] = {
            "candidate_model": CANDIDATE_MODEL,
            "reference_model": reference_model,
            "condition": "correct",
            "num_patients": int(paired["patient_id"].nunique()),
            "positive_improvement_definition": (
                "reference-candidate for NMSE/L1; candidate-reference for PSNR/SSIM"
            ),
            "bootstrap_unit": "patient",
        }
        for metric in METRIC_COLUMNS:
            values = finite_values(
                paired[f"{metric}_improvement"],
                f"StageB-vs-{reference_model}/{metric}",
            )
            statistics = percentile_summary(values)
            ci_lower, ci_upper = bootstrap_mean_ci(values, bootstrap_indices)
            for name, value in statistics.items():
                row[f"{metric}_improvement_{name}"] = value
            row[f"{metric}_improvement_mean_bootstrap_ci95_lower"] = ci_lower
            row[f"{metric}_improvement_mean_bootstrap_ci95_upper"] = ci_upper
            row[f"{metric}_patients_improved"] = int(
                np.sum(values > equality_tolerance)
            )
            row[f"{metric}_patients_worse"] = int(
                np.sum(values < -equality_tolerance)
            )
            row[f"{metric}_patients_equal"] = int(
                np.sum(np.abs(values) <= equality_tolerance)
            )
            row[f"{metric}_patients_improved_pct"] = float(
                100.0 * np.mean(values > equality_tolerance)
            )
            if metric in ERROR_METRICS:
                relative_values = finite_values(
                    paired[f"{metric}_relative_improvement_pct"],
                    f"StageB-vs-{reference_model}/{metric}/relative",
                )
                relative_statistics = percentile_summary(relative_values)
                relative_ci_lower, relative_ci_upper = bootstrap_mean_ci(
                    relative_values, bootstrap_indices
                )
                for name, value in relative_statistics.items():
                    row[f"{metric}_relative_improvement_pct_{name}"] = value
                row[
                    f"{metric}_relative_improvement_pct_mean_bootstrap_ci95_lower"
                ] = relative_ci_lower
                row[
                    f"{metric}_relative_improvement_pct_mean_bootstrap_ci95_upper"
                ] = relative_ci_upper
        rows.append(row)
    return pd.DataFrame(rows)


def validate_output_tables(
    slice_df: pd.DataFrame,
    patient_df: pd.DataFrame,
    expected_slices: int,
    expected_patients: int,
) -> None:
    expected_slice_rows = expected_slices * len(MODEL_NAMES)
    if len(slice_df) != expected_slice_rows:
        raise RuntimeError(
            f"Expected {expected_slice_rows} slice rows, found {len(slice_df)}."
        )
    duplicates = slice_df.duplicated(
        subset=["model", "patient_id", "slice_idx"], keep=False
    )
    if bool(duplicates.any()):
        raise RuntimeError(
            f"Found {int(duplicates.sum())} duplicated model/patient/slice rows."
        )
    for model_name in MODEL_NAMES:
        model_slice = slice_df[slice_df["model"] == model_name]
        model_patient = patient_df[patient_df["model"] == model_name]
        if len(model_slice) != expected_slices:
            raise RuntimeError(
                f"{model_name} has {len(model_slice)} slices; expected {expected_slices}."
            )
        if model_patient["patient_id"].nunique() != expected_patients:
            raise RuntimeError(
                f"{model_name} has {model_patient['patient_id'].nunique()} patients; "
                f"expected {expected_patients}."
            )
        for metric in METRIC_COLUMNS:
            finite_values(model_slice[metric], f"{model_name}/slice/{metric}")
            finite_values(model_patient[metric], f"{model_name}/patient/{metric}")


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate M2-U, M2-GD v2, M2-GD v2.1 Stage A and selected "
            "Stage B on every clean held-out R=8 validation slice."
        )
    )
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--m2u_checkpoint", required=True)
    parser.add_argument("--m2gd_v2_checkpoint", required=True)
    parser.add_argument("--stagea_checkpoint", required=True)
    parser.add_argument("--stageb_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap_resamples", type=int, default=10000)
    parser.add_argument("--expected_num_patients", type=int, default=25)
    parser.add_argument("--expected_num_slices", type=int, default=878)
    parser.add_argument("--expected_m2u_epoch", type=int, default=50)
    parser.add_argument("--expected_m2gd_v2_epoch", type=int, default=5)
    parser.add_argument("--expected_stagea_epoch", type=int, default=3)
    parser.add_argument("--expected_stageb_epoch", type=int, default=3)
    parser.add_argument("--expected_m2u_sha256", default=None)
    parser.add_argument("--expected_m2gd_v2_sha256", default=None)
    parser.add_argument("--expected_stagea_sha256", default=None)
    parser.add_argument("--expected_stageb_sha256", default=None)
    args = parser.parse_args()

    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative.")
    if args.expected_num_patients < 1 or args.expected_num_slices < 1:
        raise ValueError("Expected dataset counts must be positive.")
    if ssim_fn is None:
        raise RuntimeError(
            "scikit-image is required because SSIM is a mandatory final metric."
        )

    start_time = time.time()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    completion_path = output_dir / "full_clean_completed.json"
    if completion_path.exists():
        raise RuntimeError(
            f"Completed output already exists: {completion_path}. "
            "Use a new output directory rather than overwriting a final run."
        )

    checkpoint_paths = {
        "M2U": Path(args.m2u_checkpoint),
        "M2GDv2": Path(args.m2gd_v2_checkpoint),
        "M2GDv21_StageA": Path(args.stagea_checkpoint),
        "M2GDv21_StageB": Path(args.stageb_checkpoint),
    }
    expected_hashes = {
        "M2U": args.expected_m2u_sha256,
        "M2GDv2": args.expected_m2gd_v2_sha256,
        "M2GDv21_StageA": args.expected_stagea_sha256,
        "M2GDv21_StageB": args.expected_stageb_sha256,
    }
    verified_hashes: Dict[str, str] = {}
    for model_name, path in checkpoint_paths.items():
        require_file(path, f"{model_name} checkpoint")
        verified_hashes[model_name] = verify_sha256(
            path, expected_hashes[model_name], model_name
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("The full four-model evaluation requires a CUDA GPU.")
    gpu_name = torch.cuda.get_device_name(0)

    dataset_base = PairedMulticoilAuxPDToPDFSDataset(
        metadata_csv=args.metadata_csv,
        split="val",
        pdfs_acceleration=8,
        pd_aux_acceleration=2,
        slices_per_patient=None,
        edge_weight=1.0,
    )
    dataset = IndexedDataset(dataset_base)
    patient_ids = sorted(
        {str(record["patient_id"]) for record in dataset.records}
    )
    if len(dataset) != args.expected_num_slices:
        raise RuntimeError(
            f"Validation split has {len(dataset)} slices; expected "
            f"{args.expected_num_slices}."
        )
    if len(patient_ids) != args.expected_num_patients:
        raise RuntimeError(
            f"Validation split has {len(patient_ids)} patients; expected "
            f"{args.expected_num_patients}."
        )
    loader = DataLoader(
        dataset,
        batch_sampler=ShapeBucketBatchSampler(
            dataset, batch_size=args.batch_size, seed=args.seed
        ),
        num_workers=args.num_workers,
        pin_memory=True,
    )

    m2u, m2u_identity = load_m2u(checkpoint_paths["M2U"], device)
    m2gd_v2, v2_identity = load_m2gd_v2(
        checkpoint_paths["M2GDv2"], device
    )
    stagea, stagea_identity = load_m2gd_v21(
        checkpoint_paths["M2GDv21_StageA"], device
    )
    stageb, stageb_identity = load_m2gd_v21(
        checkpoint_paths["M2GDv21_StageB"], device
    )
    assert_checkpoint_identity(
        m2u_identity,
        expected_epoch=args.expected_m2u_epoch,
        expected_acceleration=8,
        expected_pd_acceleration=2,
    )
    assert_checkpoint_identity(
        v2_identity,
        expected_epoch=args.expected_m2gd_v2_epoch,
        expected_acceleration=8,
        expected_pd_acceleration=2,
        expected_curriculum="smoke5",
    )
    assert_checkpoint_identity(
        stagea_identity,
        expected_epoch=args.expected_stagea_epoch,
        expected_acceleration=8,
        expected_pd_acceleration=2,
        expected_curriculum="paired3",
    )
    assert_checkpoint_identity(
        stageb_identity,
        expected_epoch=args.expected_stageb_epoch,
        expected_acceleration=8,
        expected_pd_acceleration=2,
        expected_curriculum="fusioncal3",
    )

    print("=" * 96)
    print("M2-GD v2.1 full clean held-out validation evaluation")
    print("GPU:", gpu_name)
    print("Patients:", len(patient_ids))
    print("Slices:", len(dataset))
    print("Batches:", len(loader))
    print("Models:", MODEL_NAMES)
    print("=" * 96, flush=True)

    rows: List[Dict[str, Any]] = []
    availability: Optional[torch.Tensor]
    for batch_index, batch in enumerate(loader, start=1):
        kspace, mask, pd_aux, target = prepare_batch(batch, device)
        availability = torch.ones(
            pd_aux.shape[0], device=device, dtype=pd_aux.dtype
        )

        m2u_prediction = crop_prediction(
            m2u(kspace, mask, pd_aux), target
        )
        append_prediction_rows(
            rows,
            m2u_prediction,
            target,
            batch,
            "M2U",
            "correct",
            {},
            None,
        )

        v2_prediction, v2_aux = m2gd_v2(
            pdfs_masked_kspace=kspace,
            mask=mask,
            pd_aux_image=pd_aux,
            pd_available=availability,
            return_aux=True,
        )
        append_prediction_rows(
            rows,
            crop_prediction(v2_prediction, target),
            target,
            batch,
            "M2GDv2",
            "correct",
            {},
            diagnostics_per_sample(v2_aux),
        )

        stagea_prediction, stagea_aux = stagea(
            pdfs_masked_kspace=kspace,
            mask=mask,
            pd_aux_image=pd_aux,
            pd_available=availability,
            return_aux=True,
        )
        append_prediction_rows(
            rows,
            crop_prediction(stagea_prediction, target),
            target,
            batch,
            "M2GDv21_StageA",
            "correct",
            {},
            diagnostics_per_sample(stagea_aux),
        )

        stageb_prediction, stageb_aux = stageb(
            pdfs_masked_kspace=kspace,
            mask=mask,
            pd_aux_image=pd_aux,
            pd_available=availability,
            return_aux=True,
        )
        append_prediction_rows(
            rows,
            crop_prediction(stageb_prediction, target),
            target,
            batch,
            "M2GDv21_StageB",
            "correct",
            {},
            diagnostics_per_sample(stageb_aux),
        )

        if batch_index == 1 or batch_index % 25 == 0:
            print(
                f"Batch {batch_index:04d}/{len(loader)} completed | "
                f"slice rows={len(rows)}",
                flush=True,
            )

    slice_df = pd.DataFrame(rows)
    patient_df = aggregate_patient_level(slice_df)
    validate_output_tables(
        slice_df,
        patient_df,
        expected_slices=args.expected_num_slices,
        expected_patients=args.expected_num_patients,
    )

    ordered_patient_ids = sorted(patient_df["patient_id"].unique().tolist())
    if ordered_patient_ids != patient_ids:
        raise RuntimeError("Patient identities changed during aggregation.")
    bootstrap_indices = make_bootstrap_indices(
        num_patients=len(patient_ids),
        num_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    model_summary_df = equal_weight_model_summary(
        patient_df, bootstrap_indices
    )
    paired_frames = {
        reference: paired_patient_delta(patient_df, reference)
        for reference in REFERENCE_MODELS
    }
    comparison_summary_df = paired_comparison_summary(
        paired_frames, bootstrap_indices
    )

    slice_output = output_dir / "full_clean_per_slice.csv"
    patient_output = output_dir / "full_clean_patient_level.csv"
    model_summary_output = output_dir / "full_clean_equal_weight_summary.csv"
    comparison_summary_output = output_dir / "full_clean_comparison_summary.csv"
    slice_df.to_csv(slice_output, index=False)
    patient_df.to_csv(patient_output, index=False)
    model_summary_df.to_csv(model_summary_output, index=False)
    comparison_summary_df.to_csv(comparison_summary_output, index=False)
    paired_output_paths: Dict[str, str] = {}
    for reference_model, paired_df in paired_frames.items():
        safe_reference = reference_model.lower().replace("m2gdv21_", "")
        path = output_dir / f"stageb_vs_{safe_reference}_patient_delta.csv"
        paired_df.to_csv(path, index=False)
        paired_output_paths[reference_model] = str(path)

    elapsed_seconds = time.time() - start_time
    checkpoint_identities = {
        "M2U": m2u_identity,
        "M2GDv2": v2_identity,
        "M2GDv21_StageA": stagea_identity,
        "M2GDv21_StageB": stageb_identity,
    }
    manifest = {
        "status": "completed",
        "protocol": "full held-out validation; correct paired PD only",
        "metadata_csv": str(args.metadata_csv),
        "split": "val",
        "pdfs_acceleration": 8,
        "pd_aux_acceleration": 2,
        "num_patients": len(patient_ids),
        "num_slices": len(dataset),
        "patient_ids": patient_ids,
        "models": list(MODEL_NAMES),
        "checkpoint_identities": checkpoint_identities,
        "verified_checkpoint_sha256": verified_hashes,
        "metrics": list(METRIC_COLUMNS),
        "metric_definitions": {
            "L1": "per-slice mean absolute error after target-maximum normalization",
            "central8": "remove 8 pixels from every image edge before metric calculation",
        },
        "aggregation": (
            "slice metrics averaged within patient; all patients then receive "
            "equal weight"
        ),
        "bootstrap": {
            "unit": "patient",
            "resamples": args.bootstrap_resamples,
            "seed": args.seed,
            "interval": "percentile 95% CI for the patient-mean statistic",
        },
        "positive_improvement_definition": (
            "reference-candidate for NMSE/L1; candidate-reference for PSNR/SSIM"
        ),
        "gpu": gpu_name,
        "elapsed_seconds": elapsed_seconds,
        "outputs": {
            "per_slice": str(slice_output),
            "patient_level": str(patient_output),
            "equal_weight_summary": str(model_summary_output),
            "comparison_summary": str(comparison_summary_output),
            "paired_patient_delta": paired_output_paths,
        },
    }
    manifest_path = output_dir / "full_clean_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    completion = {
        "status": "completed",
        "num_patients": len(patient_ids),
        "num_slices": len(dataset),
        "candidate_model": CANDIDATE_MODEL,
        "candidate_checkpoint": str(args.stageb_checkpoint),
        "candidate_sha256": verified_hashes[CANDIDATE_MODEL],
        "elapsed_seconds": elapsed_seconds,
        "manifest": str(manifest_path),
    }
    completion_path.write_text(
        json.dumps(completion, indent=2), encoding="utf-8"
    )

    print("=" * 96)
    print(json.dumps(completion, indent=2))
    print("Equal-weight model summary:")
    print(model_summary_df.to_string(index=False))
    print("Stage-B paired comparison summary:")
    print(comparison_summary_df.to_string(index=False))
    print("Saved outputs to:", output_dir)
    print("=" * 96)


if __name__ == "__main__":
    main()
