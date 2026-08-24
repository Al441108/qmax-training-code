#!/usr/bin/env python3
"""Image-space PD helpfulness, overshoot, region and frequency diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage, stats
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze where fixed-checkpoint PD injection helps or harms."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--foreground-fractions",
        default="0.01,0.05,0.10",
        help="Comma-separated target-max thresholds for mask sensitivity.",
    )
    parser.add_argument(
        "--primary-foreground-fraction",
        type=float,
        default=0.05,
        help="Mask used in representative classification maps.",
    )
    parser.add_argument(
        "--active-error-quantile",
        type=float,
        default=0.50,
        help="Within-tissue No-PD error quantile defining active-error pixels.",
    )
    parser.add_argument("--high-error-quantile", type=float, default=0.75)
    parser.add_argument("--edge-quantile", type=float, default=0.75)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def finite_correlation(x: np.ndarray, y: np.ndarray, method: str) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    if method == "pearson":
        return float(np.corrcoef(x, y)[0, 1])
    result = stats.spearmanr(x, y)
    return float(result.statistic)


def safe_mean(array: np.ndarray, mask: np.ndarray) -> float:
    values = array[mask]
    return float(np.mean(values)) if values.size else float("nan")


def safe_rms(array: np.ndarray, mask: np.ndarray) -> float:
    values = array[mask]
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else float("nan")


def frequency_energy_fractions(delta: np.ndarray) -> dict[str, float]:
    centered = delta - float(np.mean(delta))
    power = np.abs(np.fft.fftshift(np.fft.fft2(centered))) ** 2
    height, width = delta.shape
    fy = np.fft.fftshift(np.fft.fftfreq(height))
    fx = np.fft.fftshift(np.fft.fftfreq(width))
    radius = np.sqrt(np.square(fy[:, None]) + np.square(fx[None, :])) / 0.5
    bands = {
        "low_frequency_fraction": radius <= 0.15,
        "mid_frequency_fraction": (radius > 0.15) & (radius <= 0.35),
        "high_frequency_fraction": radius > 0.35,
    }
    total = max(float(np.sum(power)), np.finfo(np.float64).eps)
    return {
        name: float(np.sum(power[mask]) / total)
        for name, mask in bands.items()
    }


def build_masks(
    target: np.ndarray,
    baseline_error: np.ndarray,
    foreground_fraction: float,
    high_error_quantile: float,
    edge_quantile: float,
    active_error_quantile: float,
) -> dict[str, np.ndarray]:
    target_max = max(float(np.max(np.abs(target))), np.finfo(np.float32).eps)
    foreground = np.abs(target) > foreground_fraction * target_max
    if not np.any(foreground):
        foreground = np.ones_like(target, dtype=bool)

    error_threshold = float(
        np.quantile(np.abs(baseline_error)[foreground], high_error_quantile)
    )
    high_error = foreground & (np.abs(baseline_error) >= error_threshold)
    low_error = foreground & ~high_error
    active_error_threshold = float(
        np.quantile(np.abs(baseline_error)[foreground], active_error_quantile)
    )
    active_error = foreground & (
        np.abs(baseline_error) >= active_error_threshold
    )

    normalized_target = target / target_max
    grad_y = ndimage.sobel(normalized_target, axis=0, mode="reflect")
    grad_x = ndimage.sobel(normalized_target, axis=1, mode="reflect")
    gradient = np.hypot(grad_x, grad_y)
    edge_threshold = float(np.quantile(gradient[foreground], edge_quantile))
    edge = foreground & (gradient >= edge_threshold)
    non_edge = foreground & ~edge
    return {
        "foreground": foreground,
        "high_error": high_error,
        "low_error": low_error,
        "active_error": active_error,
        "edge": edge,
        "non_edge": non_edge,
    }


def condition_metrics(
    baseline: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    masks: dict[str, np.ndarray],
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    scale = max(float(np.max(np.abs(target))), np.finfo(np.float32).eps)
    baseline_n = baseline / scale
    prediction_n = prediction / scale
    target_n = target / scale
    desired = target_n - baseline_n
    delta = prediction_n - baseline_n
    baseline_abs_error = np.abs(desired)
    final_abs_error = np.abs(target_n - prediction_n)
    helpfulness = baseline_abs_error - final_abs_error

    foreground = masks["foreground"]
    active_error = masks["active_error"]
    correct_direction = desired * delta > 0
    wrong_direction = desired * delta < 0
    crosses_target = correct_direction & (np.abs(delta) > np.abs(desired))
    harmful_overshoot = crosses_target & (final_abs_error > baseline_abs_error)
    helpful = helpfulness > 0
    harmful = helpfulness < 0

    metrics = {
        "l1": float(np.mean(final_abs_error)),
        "baseline_l1": float(np.mean(baseline_abs_error)),
        "mean_helpfulness": float(np.mean(helpfulness)),
        "foreground_mean_helpfulness": safe_mean(helpfulness, foreground),
        "foreground_coverage": float(np.mean(foreground)),
        "active_error_coverage": float(np.mean(active_error)),
        "active_error_fraction_of_foreground": float(
            np.sum(active_error) / max(np.sum(foreground), 1)
        ),
        "tissue_helpful_fraction": safe_mean(
            helpful.astype(np.float32), foreground
        ),
        "tissue_harmful_fraction": safe_mean(
            harmful.astype(np.float32), foreground
        ),
        "tissue_correct_direction_fraction": safe_mean(
            correct_direction.astype(np.float32), foreground
        ),
        "tissue_wrong_direction_fraction": safe_mean(
            wrong_direction.astype(np.float32), foreground
        ),
        "tissue_target_crossing_fraction": safe_mean(
            crosses_target.astype(np.float32), foreground
        ),
        "tissue_harmful_overshoot_fraction": safe_mean(
            harmful_overshoot.astype(np.float32), foreground
        ),
        "active_helpful_fraction": safe_mean(
            helpful.astype(np.float32), active_error
        ),
        "active_harmful_fraction": safe_mean(
            harmful.astype(np.float32), active_error
        ),
        "active_correct_direction_fraction": safe_mean(
            correct_direction.astype(np.float32), active_error
        ),
        "active_wrong_direction_fraction": safe_mean(
            wrong_direction.astype(np.float32), active_error
        ),
        "active_target_crossing_fraction": safe_mean(
            crosses_target.astype(np.float32), active_error
        ),
        "active_harmful_overshoot_fraction": safe_mean(
            harmful_overshoot.astype(np.float32), active_error
        ),
        "abs_delta_error_pearson": finite_correlation(
            np.abs(delta)[foreground],
            baseline_abs_error[foreground],
            "pearson",
        ),
        "abs_delta_error_spearman": finite_correlation(
            np.abs(delta)[foreground],
            baseline_abs_error[foreground],
            "spearman",
        ),
        "active_abs_delta_error_spearman": finite_correlation(
            np.abs(delta)[active_error],
            baseline_abs_error[active_error],
            "spearman",
        ),
        "delta_rms_foreground": safe_rms(delta, foreground),
        "delta_rms_high_error": safe_rms(delta, masks["high_error"]),
        "delta_rms_low_error": safe_rms(delta, masks["low_error"]),
        "delta_rms_edge": safe_rms(delta, masks["edge"]),
        "delta_rms_non_edge": safe_rms(delta, masks["non_edge"]),
        "positive_helpfulness_mean": safe_mean(
            np.maximum(helpfulness, 0.0), foreground
        ),
        "negative_helpfulness_mean": safe_mean(
            np.minimum(helpfulness, 0.0), foreground
        ),
    }
    metrics.update(frequency_energy_fractions(delta))
    maps = {
        "delta": delta,
        "helpfulness": helpfulness,
        "baseline_abs_error": baseline_abs_error,
        "final_abs_error": final_abs_error,
        "harmful_overshoot": harmful_overshoot & active_error,
        "wrong_direction": wrong_direction & active_error,
        "active_error": active_error,
    }
    return metrics, maps


def robust_limits(image: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(image[np.isfinite(image)], [1.0, 99.5])
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def render_representative(
    output_path: Path,
    patient_id: str,
    slice_idx: int,
    target: np.ndarray,
    no_pd: np.ndarray,
    r2: np.ndarray,
    full: np.ndarray,
    r2_maps: dict[str, np.ndarray],
    full_maps: dict[str, np.ndarray],
) -> None:
    figure, axes = plt.subplots(2, 5, figsize=(17, 7))
    low, high = robust_limits(target)
    scale = max(float(np.max(np.abs(target))), np.finfo(np.float32).eps)

    axes[0, 0].imshow(target, cmap="gray", vmin=low, vmax=high)
    axes[0, 0].set_title("PD-FS target")
    axes[1, 0].imshow(no_pd, cmap="gray", vmin=low, vmax=high)
    axes[1, 0].set_title("No-PD reconstruction")

    axes[0, 1].imshow(r2, cmap="gray", vmin=low, vmax=high)
    axes[0, 1].set_title("R2-ZF PD reconstruction")
    axes[1, 1].imshow(full, cmap="gray", vmin=low, vmax=high)
    axes[1, 1].set_title("Full-PD reconstruction")

    delta_limit = max(
        float(np.percentile(np.abs(r2_maps["delta"]), 99.5)),
        float(np.percentile(np.abs(full_maps["delta"]), 99.5)),
        np.finfo(np.float32).eps,
    )
    axes[0, 2].imshow(
        r2_maps["delta"], cmap="coolwarm", vmin=-delta_limit, vmax=delta_limit
    )
    axes[0, 2].set_title("R2 update vs No-PD")
    axes[1, 2].imshow(
        full_maps["delta"], cmap="coolwarm", vmin=-delta_limit, vmax=delta_limit
    )
    axes[1, 2].set_title("Full-PD update vs No-PD")

    h_limit = max(
        float(np.percentile(np.abs(r2_maps["helpfulness"]), 99.5)),
        float(np.percentile(np.abs(full_maps["helpfulness"]), 99.5)),
        np.finfo(np.float32).eps,
    )
    axes[0, 3].imshow(
        r2_maps["helpfulness"], cmap="RdBu", vmin=-h_limit, vmax=h_limit
    )
    axes[0, 3].set_title("R2 helpfulness (blue=better)")
    axes[1, 3].imshow(
        full_maps["helpfulness"], cmap="RdBu", vmin=-h_limit, vmax=h_limit
    )
    axes[1, 3].set_title("Full helpfulness (blue=better)")

    full_over_r2 = np.abs(r2 / scale - target / scale) - np.abs(
        full / scale - target / scale
    )
    comparison_limit = max(
        float(np.percentile(np.abs(full_over_r2), 99.5)),
        np.finfo(np.float32).eps,
    )
    axes[0, 4].imshow(
        full_over_r2,
        cmap="RdBu",
        vmin=-comparison_limit,
        vmax=comparison_limit,
    )
    axes[0, 4].set_title("Full vs R2 (blue favors Full)")

    class_map = np.zeros_like(target, dtype=np.uint8)
    class_map[r2_maps["wrong_direction"]] = 1
    class_map[r2_maps["harmful_overshoot"]] = 2
    axes[1, 4].imshow(class_map, cmap="viridis", vmin=0, vmax=2)
    axes[1, 4].set_title("R2: 1=wrong direction, 2=harmful overshoot")

    for axis in axes.ravel():
        axis.axis("off")
    figure.suptitle(f"patient={patient_id[:12]}  slice={slice_idx}")
    figure.tight_layout()
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def patient_middle_indices(dataset: Any) -> set[int]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(dataset.records):
        grouped[str(record["patient_id"])].append(index)
    return {indices[len(indices) // 2] for indices in grouped.values()}


def aggregate_patient_rows(slice_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in slice_rows:
        grouped[(row["mask_name"], row["condition"], row["patient_id"])].append(row)
    result = []
    for (mask_name, condition, patient_id), rows in sorted(grouped.items()):
        numeric_keys = [
            key
            for key, value in rows[0].items()
            if key not in {"mask_name", "condition", "patient_id", "slice_idx"}
            and isinstance(value, (int, float))
        ]
        aggregated = {
            "mask_name": mask_name,
            "condition": condition,
            "patient_id": patient_id,
            "num_slices": len(rows),
        }
        for key in numeric_keys:
            values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
            aggregated[key] = float(np.nanmean(values))
        result.append(aggregated)
    return result


def summarize(patient_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"masks": {}}
    comparison_metrics = (
        "l1",
        "mean_helpfulness",
        "foreground_mean_helpfulness",
        "foreground_coverage",
        "active_error_coverage",
        "tissue_helpful_fraction",
        "tissue_harmful_fraction",
        "tissue_wrong_direction_fraction",
        "tissue_target_crossing_fraction",
        "tissue_harmful_overshoot_fraction",
        "active_helpful_fraction",
        "active_harmful_fraction",
        "active_wrong_direction_fraction",
        "active_target_crossing_fraction",
        "active_harmful_overshoot_fraction",
        "abs_delta_error_spearman",
        "active_abs_delta_error_spearman",
        "delta_rms_high_error",
        "delta_rms_low_error",
        "low_frequency_fraction",
        "mid_frequency_fraction",
        "high_frequency_fraction",
    )
    mask_names = sorted({row["mask_name"] for row in patient_rows})
    for mask_name in mask_names:
        mask_rows = [row for row in patient_rows if row["mask_name"] == mask_name]
        mask_summary: dict[str, Any] = {"conditions": {}, "full_minus_r2": {}}
        for condition in ("r2_zf", "full_pd_oracle"):
            rows = [row for row in mask_rows if row["condition"] == condition]
            metric_names = [
                key
                for key in rows[0]
                if key not in {
                    "mask_name", "condition", "patient_id", "num_slices"
                }
            ]
            mask_summary["conditions"][condition] = {
                metric: float(np.nanmean([float(row[metric]) for row in rows]))
                for metric in metric_names
            }
        by_condition = {
            condition: {
                row["patient_id"]: row
                for row in mask_rows
                if row["condition"] == condition
            }
            for condition in ("r2_zf", "full_pd_oracle")
        }
        patients = sorted(
            set(by_condition["r2_zf"]) & set(by_condition["full_pd_oracle"])
        )
        for metric in comparison_metrics:
            differences = [
                float(by_condition["full_pd_oracle"][patient][metric])
                - float(by_condition["r2_zf"][patient][metric])
                for patient in patients
            ]
            mask_summary["full_minus_r2"][metric] = {
                "mean_difference": float(np.nanmean(differences)),
                "positive_patients": int(np.sum(np.asarray(differences) > 0)),
                "negative_patients": int(np.sum(np.asarray(differences) < 0)),
                "num_patients": len(patients),
            }
        summary["masks"][mask_name] = mask_summary
    return summary


def main() -> None:
    args = parse_args()
    foreground_fractions = [
        float(value.strip())
        for value in args.foreground_fractions.split(",")
        if value.strip()
    ]
    if not foreground_fractions:
        raise ValueError("--foreground-fractions is empty")
    if any(not 0 < value < 1 for value in foreground_fractions):
        raise ValueError("Every --foreground-fractions value must lie in (0,1)")
    if args.primary_foreground_fraction not in foreground_fractions:
        raise ValueError(
            "--primary-foreground-fraction must be listed in "
            "--foreground-fractions"
        )
    for value, name in (
        (args.high_error_quantile, "--high-error-quantile"),
        (args.edge_quantile, "--edge-quantile"),
        (args.active_error_quantile, "--active-error-quantile"),
    ):
        if not 0 < value < 1:
            raise ValueError(f"{name} must lie in (0,1)")

    project_root = Path(args.project_root).resolve()
    scripts_dir = project_root / "scripts"
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(scripts_dir))
    from evaluate_pd_oracle_stage2a import (
        condition_pd,
        make_model,
        prepare_common,
        sha256_file,
    )
    from src.dataset_paired_multicoil_aux_pd_r2 import (
        PairedMulticoilAuxPDToPDFSDataset,
    )

    metadata_csv = Path(args.metadata_csv).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    maps_dir = output_dir / "representative_maps"
    for path in (metadata_csv, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    maps_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    if config.get("fusion_design") != "global_direct":
        raise ValueError("This diagnostic is pre-specified for Global-direct")

    patient_ids = config.get("val_patient_ids") if args.split == "val" else None
    dataset = PairedMulticoilAuxPDToPDFSDataset(
        metadata_csv=str(metadata_csv),
        split=args.split,
        pdfs_acceleration=int(config.get("acceleration", 8)),
        pd_aux_acceleration=2,
        patient_ids=patient_ids,
        slices_per_patient=None,
        edge_weight=1.0,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    representative_indices = patient_middle_indices(dataset)

    model = make_model(config, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    slice_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for dataset_index, batch in enumerate(loader):
            if args.max_samples is not None and dataset_index >= args.max_samples:
                break
            kspace, mask, target_tensor = prepare_common(batch, device)
            predictions: dict[str, np.ndarray] = {}
            for condition in ("no_pd", "r2_zf", "full_pd_oracle"):
                pd, availability = condition_pd(batch, condition, device)
                prediction, _ = model(
                    kspace, mask, pd, availability, return_aux=True
                )
                height, width = target_tensor.shape[-2:]
                top = (prediction.shape[-2] - height) // 2
                left = (prediction.shape[-1] - width) // 2
                prediction = prediction[..., top : top + height, left : left + width]
                predictions[condition] = prediction[0].float().cpu().numpy()

            target = target_tensor[0].float().cpu().numpy()
            baseline = predictions["no_pd"]
            baseline_error = target - baseline
            patient_id = str(batch["patient_id"][0])
            slice_idx = int(batch["slice_idx"][0])
            primary_condition_maps = {}
            for foreground_fraction in foreground_fractions:
                masks = build_masks(
                    target,
                    baseline_error,
                    foreground_fraction,
                    args.high_error_quantile,
                    args.edge_quantile,
                    args.active_error_quantile,
                )
                mask_name = f"target_gt_{100 * foreground_fraction:g}pct"
                for condition in ("r2_zf", "full_pd_oracle"):
                    metrics, maps = condition_metrics(
                        baseline, predictions[condition], target, masks
                    )
                    slice_rows.append(
                        {
                            "mask_name": mask_name,
                            "condition": condition,
                            "patient_id": patient_id,
                            "slice_idx": slice_idx,
                            **metrics,
                        }
                    )
                    if foreground_fraction == args.primary_foreground_fraction:
                        primary_condition_maps[condition] = maps

            if dataset_index in representative_indices:
                if set(primary_condition_maps) != {"r2_zf", "full_pd_oracle"}:
                    raise RuntimeError("Primary mask maps were not generated")
                filename = f"{patient_id[:12]}_slice{slice_idx:03d}.png"
                render_representative(
                    maps_dir / filename,
                    patient_id,
                    slice_idx,
                    target,
                    baseline,
                    predictions["r2_zf"],
                    predictions["full_pd_oracle"],
                    primary_condition_maps["r2_zf"],
                    primary_condition_maps["full_pd_oracle"],
                )
            if (dataset_index + 1) % 50 == 0:
                print(f"Analyzed {dataset_index + 1}/{len(dataset)} slices", flush=True)

    patient_rows = aggregate_patient_rows(slice_rows)
    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "split": args.split,
        "num_dataset_slices": len(dataset),
        "num_analyzed_slices": len(slice_rows) // (
            2 * len(foreground_fractions)
        ),
        "region_definitions": {
            "foreground_masks": [
                f"|target| > {value} * max(|target|)"
                for value in foreground_fractions
            ],
            "primary_foreground_fraction": args.primary_foreground_fraction,
            "active_error": (
                f"top {100 * (1 - args.active_error_quantile):.1f}% of No-PD "
                "absolute error within each foreground mask"
            ),
            "high_error": (
                f"top {100 * (1 - args.high_error_quantile):.1f}% of "
                "No-PD absolute error within foreground"
            ),
            "edge": (
                f"top {100 * (1 - args.edge_quantile):.1f}% of target Sobel "
                "gradient within foreground"
            ),
        },
        "overshoot_definitions": {
            "target_crossing": (
                "update has correct sign and magnitude exceeds required correction"
            ),
            "harmful_overshoot": (
                "target crossing where final absolute error exceeds No-PD error"
            ),
        },
        "summary": summarize(patient_rows),
    }
    write_csv(output_dir / "spatial_slice_metrics.csv", slice_rows)
    write_csv(output_dir / "spatial_patient_metrics.csv", patient_rows)
    (output_dir / "spatial_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
