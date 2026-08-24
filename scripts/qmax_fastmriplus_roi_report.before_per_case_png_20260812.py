#!/usr/bin/env python3
from __future__ import annotations

"""Stage 2B: statistics and publication figures from frozen ROI caches only."""

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from skimage.metrics import structural_similarity


PROTOCOL = "QMax-fastMRIplus-ROI-report-v2"
INFERENCE_PROTOCOL = "QMax-fastMRIplus-ROI-inference-cache-v2"
REGIONS = (
    "lesion",
    "perilesional_ring",
    "matched_nonannotated_control",
    "nonlesion_foreground",
)
REGION_LABELS = {
    "lesion": "Lesion box",
    "perilesional_ring": "Perilesional ring",
    "matched_nonannotated_control": "Matched non-annotated control",
    "nonlesion_foreground": "Non-lesion foreground",
}
COLORS = {
    "zero_filled": "#9B9B9B",
    "qmax": "#3E6F9E",
    "lesion": "#B64B4B",
    "perilesional_ring": "#D59545",
    "matched_nonannotated_control": "#668F80",
    "nonlesion_foreground": "#8A78A6",
}


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing empty CSV: {path}")
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_mean(values: Iterable[float]) -> float:
    selected = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(selected)) if selected else float("nan")


def roi_l1(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) == 0:
        return float("nan")
    scale = max(float(np.max(target)), 1e-8)
    return float(np.mean(np.abs(prediction[mask] / scale - target[mask] / scale)))


def local_ssim_map(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    scale = max(float(np.max(target)), 1e-8)
    truth = np.asarray(target, dtype=np.float32) / scale
    pred = np.asarray(prediction, dtype=np.float32) / scale
    _score, value = structural_similarity(
        truth,
        pred,
        data_range=1.0,
        win_size=7,
        gaussian_weights=False,
        full=True,
    )
    return np.asarray(value, dtype=np.float32)


def roi_mean(value: np.ndarray, mask: np.ndarray) -> float:
    return float(value[mask].mean()) if int(mask.sum()) else float("nan")


def bootstrap_mean(values: Sequence[float], seed: int, iterations: int) -> Dict[str, float]:
    data = np.asarray([float(value) for value in values], dtype=np.float64)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    if data.size == 1:
        value = float(data[0])
        return {"mean": value, "ci95_low": float("nan"), "ci95_high": float("nan")}
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, data.size, size=(int(iterations), data.size))
    means = data[indices].mean(axis=1)
    return {
        "mean": float(data.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def rank_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    first = np.asarray(x, dtype=np.float64)
    second = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(first) & np.isfinite(second)
    first, second = first[valid], second[valid]
    if first.size < 3 or np.std(first) <= 0 or np.std(second) <= 0:
        return float("nan")
    return float(np.corrcoef(rank_average(first), rank_average(second))[0, 1])


def bootstrap_spearman(
    x: Sequence[float], y: Sequence[float], seed: int, iterations: int
) -> Dict[str, Any]:
    first = np.asarray(x, dtype=np.float64)
    second = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(first) & np.isfinite(second)
    first, second = first[valid], second[valid]
    estimate = spearman(first, second)
    if first.size < 4:
        return {"n": int(first.size), "rho": estimate, "ci95": [float("nan"), float("nan")]}
    rng = np.random.default_rng(int(seed))
    sampled: List[float] = []
    for _ in range(int(iterations)):
        index = rng.integers(0, first.size, size=first.size)
        value = spearman(first[index], second[index])
        if math.isfinite(value):
            sampled.append(value)
    if len(sampled) < max(100, int(iterations) // 10):
        interval = [float("nan"), float("nan")]
    else:
        interval = [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))]
    return {"n": int(first.size), "rho": estimate, "ci95": interval}


def load_cases(manifest_path: Path) -> Tuple[Dict[str, Any], List[Tuple[Dict[str, Any], Dict[str, np.ndarray]]]]:
    manifest = read_json(manifest_path)
    if manifest.get("protocol_version") != INFERENCE_PROTOCOL or manifest.get("status") != "passed":
        raise RuntimeError(f"Invalid inference manifest: {manifest_path}")
    if "held-out test not accessed" not in str(manifest.get("scope", "")):
        raise RuntimeError("Inference cache scope is not validation-only")
    output = []
    for metadata_name, npz_name in zip(manifest["case_metadata"], manifest["case_caches"]):
        metadata_path = Path(metadata_name)
        npz_path = Path(npz_name)
        metadata = read_json(metadata_path)
        if metadata.get("status") != "passed" or metadata.get("protocol_version") != INFERENCE_PROTOCOL:
            raise RuntimeError(f"Invalid case metadata: {metadata_path}")
        if sha256_file(npz_path) != metadata.get("npz_sha256"):
            raise RuntimeError(f"Case cache hash mismatch: {npz_path}")
        with np.load(npz_path, allow_pickle=False) as handle:
            arrays = {key: np.asarray(handle[key]) for key in handle.files}
        output.append((metadata, arrays))
    if not output:
        raise RuntimeError("Inference manifest contains no cases")
    return manifest, output


def region_energy(metadata: Mapping[str, Any], region: str, name: str) -> float:
    return finite_mean(
        float(row[f"{region}_{name}"])
        for row in metadata["path_records"]
    )


def build_slice_rows(
    cases: Sequence[Tuple[Mapping[str, Any], Mapping[str, np.ndarray]]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    box_rows: List[Dict[str, Any]] = []
    for metadata, arrays in cases:
        target = arrays["target"].astype(np.float32)
        zero_filled = arrays["zero_filled"].astype(np.float32)
        qmax = arrays["qmax_full"].astype(np.float32)
        zf_ssim = local_ssim_map(zero_filled, target)
        qmax_ssim = local_ssim_map(qmax, target)
        base = {
            "patient_id": str(metadata["patient_id"]),
            "slice_idx": int(metadata["slice_idx"]),
            "labels": ";".join(metadata["labels"]),
            "num_boxes": int(metadata["num_boxes"]),
            "slice_q_mean": float(metadata["q_summary"]["q_mean"]),
        }
        for region in REGIONS:
            mask = arrays[f"{region}_mask"].astype(bool)
            if not mask.any():
                continue
            zf_l1 = roi_l1(zero_filled, target, mask)
            qmax_l1 = roi_l1(qmax, target, mask)
            zf_local_ssim = roi_mean(zf_ssim, mask)
            qmax_local_ssim = roi_mean(qmax_ssim, mask)
            qeff_result = metadata["q_eff"].get(region, {})
            rows.append(
                {
                    **base,
                    "region": region,
                    "num_pixels": int(mask.sum()),
                    "zero_filled_l1": zf_l1,
                    "qmax_full_l1": qmax_l1,
                    "delta_l1_qmax_minus_zf": qmax_l1 - zf_l1,
                    "l1_improvement_zf_minus_qmax": zf_l1 - qmax_l1,
                    "zero_filled_roi_mean_local_ssim": zf_local_ssim,
                    "qmax_full_roi_mean_local_ssim": qmax_local_ssim,
                    "delta_local_ssim_qmax_minus_zf": qmax_local_ssim - zf_local_ssim,
                    "q_eff_status": qeff_result.get("status", "missing"),
                    "roi_weighted_q_eff": (
                        float(qeff_result["q_eff"])
                        if qeff_result.get("q_eff") is not None
                        else float("nan")
                    ),
                    "pre_q_direct_energy": region_energy(metadata, region, "pre_q_direct_energy"),
                    "gated_direct_energy": region_energy(metadata, region, "gated_direct_energy"),
                    "correction_energy": region_energy(metadata, region, "correction_energy"),
                    "final_auxiliary_energy": region_energy(metadata, region, "final_auxiliary_energy"),
                }
            )
        labels = list(metadata.get("box_labels", metadata["labels"]))
        boxes = arrays["boxes"].astype(int)
        for box_index, (left, top, right, bottom) in enumerate(boxes):
            mask = np.zeros_like(target, dtype=bool)
            mask[top:bottom, left:right] = True
            box_rows.append(
                {
                    **base,
                    "box_index": box_index,
                    "label": labels[box_index] if box_index < len(labels) else "unspecified",
                    "left": int(left),
                    "top": int(top),
                    "right": int(right),
                    "bottom": int(bottom),
                    "num_pixels": int(mask.sum()),
                    "zero_filled_l1": roi_l1(zero_filled, target, mask),
                    "qmax_full_l1": roi_l1(qmax, target, mask),
                    "zero_filled_roi_mean_local_ssim": roi_mean(zf_ssim, mask),
                    "qmax_full_roi_mean_local_ssim": roi_mean(qmax_ssim, mask),
                }
            )
    return rows, box_rows


def patient_macro(slice_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in slice_rows:
        groups[(str(row["patient_id"]), str(row["region"]))].append(row)
    metric_names = (
        "zero_filled_l1",
        "qmax_full_l1",
        "delta_l1_qmax_minus_zf",
        "l1_improvement_zf_minus_qmax",
        "zero_filled_roi_mean_local_ssim",
        "qmax_full_roi_mean_local_ssim",
        "delta_local_ssim_qmax_minus_zf",
        "slice_q_mean",
        "roi_weighted_q_eff",
        "pre_q_direct_energy",
        "gated_direct_energy",
        "correction_energy",
        "final_auxiliary_energy",
    )
    output: List[Dict[str, Any]] = []
    for (patient, region), values in sorted(groups.items()):
        output.append(
            {
                "patient_id": patient,
                "region": region,
                "num_slices": len(values),
                **{
                    metric: finite_mean(float(row[metric]) for row in values)
                    for metric in metric_names
                },
            }
        )
    return output


def region_summary(
    patient_rows: Sequence[Mapping[str, Any]], seed: int, iterations: int
) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for offset, region in enumerate(REGIONS):
        rows = [row for row in patient_rows if row["region"] == region]
        if not rows:
            continue
        l1_ci = bootstrap_mean(
            [float(row["delta_l1_qmax_minus_zf"]) for row in rows],
            seed + offset,
            iterations,
        )
        ssim_ci = bootstrap_mean(
            [float(row["delta_local_ssim_qmax_minus_zf"]) for row in rows],
            seed + 100 + offset,
            iterations,
        )
        output[region] = {
            "num_patients": len(rows),
            "zero_filled_mean_l1": finite_mean(float(row["zero_filled_l1"]) for row in rows),
            "qmax_full_mean_l1": finite_mean(float(row["qmax_full_l1"]) for row in rows),
            "delta_l1_qmax_minus_zf": l1_ci,
            "zero_filled_mean_roi_local_ssim": finite_mean(
                float(row["zero_filled_roi_mean_local_ssim"]) for row in rows
            ),
            "qmax_full_mean_roi_local_ssim": finite_mean(
                float(row["qmax_full_roi_mean_local_ssim"]) for row in rows
            ),
            "delta_local_ssim_qmax_minus_zf": ssim_ci,
            "patients_qmax_lower_l1": sum(float(row["delta_l1_qmax_minus_zf"]) < 0 for row in rows),
            "mean_slice_q": finite_mean(float(row["slice_q_mean"]) for row in rows),
            "mean_roi_weighted_q_eff": finite_mean(
                float(row["roi_weighted_q_eff"]) for row in rows
            ),
        }
    return output


def differential_rows(patient_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    lookup = {(str(row["patient_id"]), str(row["region"])): row for row in patient_rows}
    patients = sorted({str(row["patient_id"]) for row in patient_rows})
    output = []
    for patient in patients:
        lesion = lookup.get((patient, "lesion"))
        control = lookup.get((patient, "matched_nonannotated_control"))
        if lesion is None or control is None:
            continue
        zf_excess = float(lesion["zero_filled_l1"]) - float(control["zero_filled_l1"])
        qmax_excess = float(lesion["qmax_full_l1"]) - float(control["qmax_full_l1"])
        output.append(
            {
                "patient_id": patient,
                "zero_filled_lesion_excess_l1": zf_excess,
                "qmax_full_lesion_excess_l1": qmax_excess,
                "differential_lesion_benefit": zf_excess - qmax_excess,
                "lesion_l1_improvement": float(lesion["l1_improvement_zf_minus_qmax"]),
                "lesion_roi_weighted_q_eff": float(lesion["roi_weighted_q_eff"]),
                "lesion_slice_q_mean": float(lesion["slice_q_mean"]),
            }
        )
    return output


def representative_cases(slice_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    lesion = [row for row in slice_rows if row["region"] == "lesion"]
    lesion.sort(
        key=lambda row: (
            float(row["l1_improvement_zf_minus_qmax"]),
            str(row["patient_id"]),
            int(row["slice_idx"]),
        )
    )
    if not lesion:
        raise RuntimeError("No lesion rows for representative-case selection")
    indices = [0, len(lesion) // 2, len(lesion) - 1]
    labels = ("minimum", "median", "maximum")
    selected = []
    seen = set()
    for name, index in zip(labels, indices):
        row = lesion[index]
        key = (str(row["patient_id"]), int(row["slice_idx"]))
        if key in seen:
            continue
        seen.add(key)
        selected.append(
            {
                "selection_position": name,
                "selection_rule": (
                    "predeclared minimum/median/maximum lesion-box L1 improvement "
                    "within locked validation FastMRI+ cases"
                ),
                "patient_id": key[0],
                "slice_idx": key[1],
                "lesion_l1_improvement_zf_minus_qmax": float(
                    row["l1_improvement_zf_minus_qmax"]
                ),
            }
        )
    return {"cases": selected}


def target_window(target: np.ndarray) -> Tuple[float, float]:
    positive = target[target > 0]
    low = float(np.percentile(positive, 1)) if positive.size else float(np.min(target))
    high = float(np.percentile(target, 99.5))
    return low, max(high, low + 1e-8)


def crop_bounds(boxes: np.ndarray, shape: Tuple[int, int], margin: int = 16) -> Tuple[int, int, int, int]:
    left = max(0, int(boxes[:, 0].min()) - margin)
    top = max(0, int(boxes[:, 1].min()) - margin)
    right = min(shape[1], int(boxes[:, 2].max()) + margin)
    bottom = min(shape[0], int(boxes[:, 3].max()) + margin)
    return left, top, right, bottom


def draw_boxes(axis, boxes: np.ndarray, crop: Tuple[int, int, int, int] | None = None) -> None:
    offset_x = crop[0] if crop else 0
    offset_y = crop[1] if crop else 0
    for left, top, right, bottom in boxes:
        axis.add_patch(
            Rectangle(
                (left - offset_x, top - offset_y),
                right - left,
                bottom - top,
                fill=False,
                linewidth=1.1,
                edgecolor="#00D6D6",
            )
        )


def save_publication_figure(fig: plt.Figure, stem: Path) -> Dict[str, str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for suffix, kwargs in (
        ("png", {"dpi": 400}),
        ("tiff", {"dpi": 600}),
        ("svg", {}),
        ("pdf", {}),
    ):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, bbox_inches="tight", **kwargs)
        outputs[suffix] = str(path)
    return outputs


def render_representative_plate(
    selected: Mapping[str, Any],
    cases: Sequence[Tuple[Mapping[str, Any], Mapping[str, np.ndarray]]],
    slice_rows: Sequence[Mapping[str, Any]],
    output: Path,
) -> Dict[str, str]:
    case_lookup = {
        (str(metadata["patient_id"]), int(metadata["slice_idx"])): (metadata, arrays)
        for metadata, arrays in cases
    }
    row_lookup = {
        (str(row["patient_id"]), int(row["slice_idx"])): row
        for row in slice_rows
        if row["region"] == "lesion"
    }
    chosen = selected["cases"]
    fig, axes = plt.subplots(len(chosen), 6, figsize=(14.2, 3.05 * len(chosen)), squeeze=False)
    column_titles = (
        "Reference + box",
        "Zero-filled + box",
        "QMax-Full + box",
        "Reference crop",
        "Zero-filled crop",
        "QMax-Full crop",
    )
    for axis, title in zip(axes[0], column_titles):
        axis.set_title(title, fontsize=8, fontweight="bold", pad=5)
    for row_index, selection in enumerate(chosen):
        key = (selection["patient_id"], int(selection["slice_idx"]))
        metadata, arrays = case_lookup[key]
        metrics = row_lookup[key]
        target = arrays["target"]
        zf = arrays["zero_filled"]
        qmax = arrays["qmax_full"]
        boxes = arrays["boxes"].astype(int)
        low, high = target_window(target)
        crop = crop_bounds(boxes, target.shape)
        left, top, right, bottom = crop
        for column, image in enumerate((target, zf, qmax)):
            axes[row_index, column].imshow(image, cmap="gray", vmin=low, vmax=high)
            draw_boxes(axes[row_index, column], boxes)
        for column, image in enumerate((target, zf, qmax), start=3):
            axes[row_index, column].imshow(
                image[top:bottom, left:right], cmap="gray", vmin=low, vmax=high
            )
            draw_boxes(axes[row_index, column], boxes, crop=crop)
        pathology = "; ".join(metadata["labels"])
        qeff = float(metrics["roi_weighted_q_eff"])
        qeff_text = f"{qeff:.2f}" if math.isfinite(qeff) else "not estimable"
        axes[row_index, 0].text(
            -0.08,
            0.5,
            f"{selection['selection_position'].capitalize()} improvement\n"
            f"{pathology}\n{key[0][:8]} · slice {key[1]}",
            transform=axes[row_index, 0].transAxes,
            fontsize=7,
            ha="right",
            va="center",
        )
        axes[row_index, 4].text(
            0.5,
            -0.08,
            f"Lesion L1 {metrics['zero_filled_l1']:.4f}\n"
            f"local SSIM {metrics['zero_filled_roi_mean_local_ssim']:.3f}",
            transform=axes[row_index, 4].transAxes,
            ha="center",
            va="top",
            fontsize=6.5,
        )
        axes[row_index, 5].text(
            0.5,
            -0.08,
            f"Lesion L1 {metrics['qmax_full_l1']:.4f}\n"
            f"local SSIM {metrics['qmax_full_roi_mean_local_ssim']:.3f}\n"
            f"slice q̄ {metrics['slice_q_mean']:.2f}; ROI-weighted q_eff {qeff_text}",
            transform=axes[row_index, 5].transAxes,
            ha="center",
            va="top",
            fontsize=6.5,
        )
        for axis in axes[row_index]:
            axis.axis("off")
    fig.suptitle(
        "Post-hoc lesion-region reconstruction in locked validation cases",
        fontsize=11,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.006,
        "ROI-weighted effective reliability is a post-hoc path-weighted summary, not predicted lesion reliability.",
        ha="center",
        fontsize=6.5,
        color="#444444",
    )
    fig.subplots_adjust(left=0.11, right=0.995, top=0.94, bottom=0.06, wspace=0.06, hspace=0.24)
    outputs = save_publication_figure(fig, output / "representative_lesion_plate")
    plt.close(fig)
    return outputs


def render_error_plate(
    selected: Mapping[str, Any],
    cases: Sequence[Tuple[Mapping[str, Any], Mapping[str, np.ndarray]]],
    output: Path,
) -> Dict[str, str]:
    lookup = {
        (str(metadata["patient_id"]), int(metadata["slice_idx"])): (metadata, arrays)
        for metadata, arrays in cases
    }
    chosen = selected["cases"]
    fig, axes = plt.subplots(len(chosen), 2, figsize=(5.4, 2.45 * len(chosen)), squeeze=False)
    axes[0, 0].set_title("Zero-filled absolute-error crop", fontsize=8, fontweight="bold")
    axes[0, 1].set_title("QMax-Full absolute-error crop", fontsize=8, fontweight="bold")
    for row_index, selection in enumerate(chosen):
        key = (selection["patient_id"], int(selection["slice_idx"]))
        metadata, arrays = lookup[key]
        target = arrays["target"]
        boxes = arrays["boxes"].astype(int)
        left, top, right, bottom = crop_bounds(boxes, target.shape)
        zf_error = np.abs(arrays["zero_filled"] - target)[top:bottom, left:right]
        qmax_error = np.abs(arrays["qmax_full"] - target)[top:bottom, left:right]
        high = max(float(np.percentile(np.concatenate([zf_error.ravel(), qmax_error.ravel()]), 99.5)), 1e-8)
        for column, error in enumerate((zf_error, qmax_error)):
            image = axes[row_index, column].imshow(error, cmap="magma", vmin=0, vmax=high)
            axes[row_index, column].axis("off")
        colorbar = fig.colorbar(image, ax=list(axes[row_index]), fraction=0.025, pad=0.015)
        colorbar.ax.tick_params(labelsize=6)
        axes[row_index, 0].text(
            -0.10,
            0.5,
            f"{selection['selection_position'].capitalize()}\n{'; '.join(metadata['labels'])}",
            transform=axes[row_index, 0].transAxes,
            ha="right",
            va="center",
            fontsize=7,
        )
    fig.suptitle("Lesion-centred absolute-error comparison", fontsize=10, fontweight="bold")
    fig.subplots_adjust(left=0.16, right=0.90, top=0.92, bottom=0.03, wspace=0.08, hspace=0.14)
    outputs = save_publication_figure(fig, output / "representative_lesion_errors")
    plt.close(fig)
    return outputs


def render_region_forest(
    patient_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    differential: Sequence[Mapping[str, Any]],
    differential_ci: Mapping[str, float],
    output: Path,
) -> Dict[str, str]:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.4, 1.0]})
    ordered = [region for region in REGIONS if region in summary]
    y = np.arange(len(ordered))
    for position, region in enumerate(ordered):
        rows = [row for row in patient_rows if row["region"] == region]
        values = np.asarray([float(row["delta_l1_qmax_minus_zf"]) for row in rows])
        jitter = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else np.zeros(len(values))
        axes[0].scatter(values, position + jitter, s=10, alpha=0.45, color=COLORS[region], linewidths=0)
        estimate = summary[region]["delta_l1_qmax_minus_zf"]
        axes[0].errorbar(
            estimate["mean"],
            position,
            xerr=[[estimate["mean"] - estimate["ci95_low"]], [estimate["ci95_high"] - estimate["mean"]]],
            fmt="o",
            markersize=5,
            color=COLORS[region],
            capsize=2,
            linewidth=1.2,
            zorder=5,
        )
    axes[0].axvline(0, color="#333333", linewidth=0.8, linestyle="--")
    axes[0].set_yticks(y, [REGION_LABELS[region] for region in ordered])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("ΔL1 (QMax-Full − zero-filled; lower is better)")
    axes[0].set_title("a  Regional reconstruction error", loc="left", fontweight="bold")

    values = np.asarray([float(row["differential_lesion_benefit"]) for row in differential])
    if len(values):
        jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.zeros(len(values))
        axes[1].scatter(jitter, values, s=14, alpha=0.55, color=COLORS["lesion"], linewidths=0)
        if all(math.isfinite(float(differential_ci[key])) for key in ("mean", "ci95_low", "ci95_high")):
            axes[1].errorbar(
                0,
                differential_ci["mean"],
                yerr=[
                    [differential_ci["mean"] - differential_ci["ci95_low"]],
                    [differential_ci["ci95_high"] - differential_ci["mean"]],
                ],
                fmt="o",
                color="#7E2F2F",
                capsize=3,
                linewidth=1.3,
                markersize=5,
                zorder=5,
            )
    else:
        axes[1].text(
            0.5,
            0.5,
            "Matched control unavailable",
            transform=axes[1].transAxes,
            ha="center",
            va="center",
            color="#555555",
        )
    axes[1].axhline(0, color="#333333", linewidth=0.8, linestyle="--")
    axes[1].set_xlim(-0.35, 0.35)
    axes[1].set_xticks([])
    axes[1].set_ylabel("Differential lesion benefit (positive favours QMax)")
    axes[1].set_title("b  Lesion-specific differential", loc="left", fontweight="bold")
    fig.suptitle(
        "QMax-Full regional performance in FastMRI+ annotated validation cases",
        fontsize=10,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=2.2)
    outputs = save_publication_figure(fig, output / "roi_region_comparison")
    plt.close(fig)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bootstrap_iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.inference_manifest)
    inference, cases = load_cases(manifest_path)
    slice_rows, box_rows = build_slice_rows(cases)
    patient_rows = patient_macro(slice_rows)
    summary = region_summary(patient_rows, args.seed, args.bootstrap_iterations)
    differential = differential_rows(patient_rows)
    differential_ci = bootstrap_mean(
        [float(row["differential_lesion_benefit"]) for row in differential],
        args.seed + 500,
        args.bootstrap_iterations,
    )
    lesion_patients = [row for row in patient_rows if row["region"] == "lesion"]
    if bool(inference["qeff_aggregate_reporting_allowed"]):
        correlation = bootstrap_spearman(
            [float(row["roi_weighted_q_eff"]) for row in lesion_patients],
            [float(row["l1_improvement_zf_minus_qmax"]) for row in lesion_patients],
            args.seed + 1000,
            args.bootstrap_iterations,
        )
    else:
        correlation = {
            "status": "suppressed",
            "reason": "more than 10% of lesion ROIs had near-zero auxiliary energy",
        }
        for value in summary.values():
            value["mean_roi_weighted_q_eff"] = None
    selected = representative_cases(slice_rows)

    write_csv(output / "box_level_descriptive.csv", box_rows)
    write_csv(output / "slice_level_roi_qeff.csv", slice_rows)
    write_csv(output / "patient_level_roi_qeff.csv", patient_rows)
    if differential:
        write_csv(output / "patient_level_lesion_differential.csv", differential)
    atomic_json(output / "representative_cases_locked.json", selected)
    statistics = {
        "protocol_version": PROTOCOL,
        "status": "passed",
        "scope": "post-hoc exploratory; locked validation only; held-out test not accessed",
        "region_summary": summary,
        "differential_lesion_benefit": {
            "definition": (
                "(ZF lesion L1 - QMax lesion L1) - "
                "(ZF matched-control L1 - QMax matched-control L1)"
            ),
            "num_patients": len(differential),
            **differential_ci,
        },
        "exploratory_qeff_vs_lesion_l1_improvement_spearman": correlation,
        "qeff_aggregate_reporting_allowed": bool(
            inference["qeff_aggregate_reporting_allowed"]
        ),
        "statistical_unit": "patient; annotated slices averaged within patient",
        "bootstrap_iterations": int(args.bootstrap_iterations),
        "roi_ssim_definition": (
            "mean of a full-image 7x7 local SSIM map inside each ROI; data range "
            "fixed by target maximum; not SSIM computed on a tiny cropped box"
        ),
    }
    atomic_json(output / "roi_qeff_summary.json", statistics)

    figures_dir = output / "figures"
    figure_outputs = {
        "representative_plate": render_representative_plate(
            selected, cases, slice_rows, figures_dir
        ),
        "representative_errors": render_error_plate(selected, cases, figures_dir),
        "region_comparison": render_region_forest(
            patient_rows, summary, differential, differential_ci, figures_dir
        ),
    }
    audit = {
        "protocol_version": PROTOCOL,
        "status": "passed",
        "scope": statistics["scope"],
        "inference_manifest": str(manifest_path),
        "inference_manifest_sha256": sha256_file(manifest_path),
        "num_slices": len({(row["patient_id"], row["slice_idx"]) for row in slice_rows}),
        "num_patients": len({row["patient_id"] for row in patient_rows}),
        "num_boxes": len(box_rows),
        "figure_contract": {
            "core_conclusion": (
                "QMax-Full reduces reconstruction error in FastMRI+ annotated "
                "lesion and surrounding regions relative to zero-filled input."
            ),
            "archetype": "image plate + quant",
            "backend": "Python/matplotlib only",
            "representative_case_rule": (
                "minimum/median/maximum lesion L1 improvement, locked before drawing"
            ),
            "image_integrity": (
                "one target-derived window per slice; no local contrast adjustment; "
                "row-specific shared error scale"
            ),
        },
        "qeff_language_guard": (
            "Report as ROI-weighted effective reliability, never lesion q, "
            "predicted lesion reliability, or spatial q-map."
        ),
        "outputs": {
            "box_level": str(output / "box_level_descriptive.csv"),
            "slice_level": str(output / "slice_level_roi_qeff.csv"),
            "patient_level": str(output / "patient_level_roi_qeff.csv"),
            "summary": str(output / "roi_qeff_summary.json"),
            "representative_cases": str(output / "representative_cases_locked.json"),
            "figures": figure_outputs,
        },
    }
    atomic_json(output / "roi_qeff_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
