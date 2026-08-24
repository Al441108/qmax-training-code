#!/usr/bin/env python3
from __future__ import annotations

"""Render a cross-model reconstruction and absolute-error image plate.

Figure contract
---------------
Core conclusion:
    Localised reconstruction errors can be compared fairly across model
    families and auxiliary-input conditions.
Evidence:
    Rows are auxiliary conditions; columns are locked models; every row uses a
    shared error scale and every panel uses the same patient and source slice.
Archetype:
    Image plate + quantitative cell annotations.
Integrity:
    No smoothing, local contrast adjustment or per-model error scaling.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import fastmri
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_m2_prnf_final_comparison_R8 import (  # noqa: E402
    BASELINE_NAMES,
    CONDITIONS,
    FUSION_NAMES,
    GAIN_MODEL,
    ManifestDataset,
    condition_pd,
    load_baseline,
    load_fusion,
    metrics,
)
from scripts.train_m2_prnf import (  # noqa: E402
    IndexedDataset,
    make_dataset,
    prepare_batch,
    set_seed,
    sha256_file,
)
from src.fft_utils import center_crop  # noqa: E402

MODEL_ORDER = (
    "zero_filled_exact_manifest",
    "m2u_clean",
    "m2u_augmented",
    "m2u_augcap_mask",
    "global_direct",
    "quality_protected_hybrid_gain",
)
MODEL_LABELS = {
    "zero_filled_exact_manifest": "Zero-filled",
    "m2u_clean": "M2-U\nClean",
    "m2u_augmented": "M2-U\nAugmented",
    "m2u_augcap_mask": "M2-U\nAugCap/Mask",
    "global_direct": "Global-direct",
    "quality_protected_hybrid_gain": "Hybrid-gain",
}
CONDITION_LABELS = {
    "correct": "Correct PD",
    "shift8": "Shift 8 px",
    "wrong_slice": "Wrong slice",
    "wrong_patient": "Wrong patient",
    "missing": "Missing PD",
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: List[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def select_representative(
    rows: List[Mapping[str, str]],
    patient_id: str | None,
    slice_idx: int | None,
) -> Dict[str, Any]:
    """Select the median fifth-arm versus M2-U Augmented robustness case."""
    if patient_id is not None or slice_idx is not None:
        if patient_id is None or slice_idx is None:
            raise ValueError("--patient_id and --slice_idx must be supplied together")
        return {
            "patient_id": str(patient_id),
            "slice_idx": int(slice_idx),
            "selection_rule": "investigator_pre_specified",
        }

    values: Dict[Tuple[str, int, str, str], float] = {}
    for row in rows:
        if (
            row.get("cohort") == "robustness"
            and row.get("condition") in {"shift8", "wrong_slice", "wrong_patient"}
            and row.get("model") in {GAIN_MODEL, "m2u_augmented"}
        ):
            key = (
                str(row["patient_id"]),
                int(row["slice_idx"]),
                str(row["condition"]),
                str(row["model"]),
            )
            values[key] = float(row["l1"])

    scores = []
    sources = sorted({(key[0], key[1]) for key in values})
    for patient, source_slice in sources:
        improvements = []
        complete = True
        for condition in ("shift8", "wrong_slice", "wrong_patient"):
            gain_key = (patient, source_slice, condition, GAIN_MODEL)
            aug_key = (patient, source_slice, condition, "m2u_augmented")
            if gain_key not in values or aug_key not in values:
                complete = False
                break
            improvements.append(values[aug_key] - values[gain_key])
        if complete:
            scores.append(
                {
                    "patient_id": patient,
                    "slice_idx": source_slice,
                    "mean_l1_improvement_vs_m2u_augmented": float(
                        np.mean(improvements)
                    ),
                }
            )
    if not scores:
        raise RuntimeError("No complete robustness slices available for selection")
    median = float(
        np.median(
            [row["mean_l1_improvement_vs_m2u_augmented"] for row in scores]
        )
    )
    selected = min(
        scores,
        key=lambda row: (
            abs(row["mean_l1_improvement_vs_m2u_augmented"] - median),
            row["patient_id"],
            row["slice_idx"],
        ),
    )
    return {
        **selected,
        "cohort_median_l1_improvement": median,
        "selection_rule": (
            "same source slice whose mean fifth-arm minus M2-U-Augmented "
            "improvement across shift8/wrong-slice/wrong-patient is nearest "
            "the cohort median"
        ),
        "cherry_pick_guard": True,
    }


def get_single_batch(dataset, patient_id: str, slice_idx: int, device):
    local_index = None
    for index, record in enumerate(dataset.records):
        if (
            str(record["patient_id"]) == str(patient_id)
            and int(record["slice_idx"]) == int(slice_idx)
        ):
            local_index = index
            break
    if local_index is None:
        raise RuntimeError(
            f"Selected slice not found: patient={patient_id}, slice={slice_idx}"
        )
    loader = DataLoader(Subset(dataset, [local_index]), batch_size=1, num_workers=0)
    batch = next(iter(loader))
    kspace, mask, pd, target, indices = prepare_batch(batch, device)
    return batch, kspace, mask, pd, target, indices


@torch.no_grad()
def predict_model_conditions(
    model,
    model_name,
    batch_data,
    full_dataset,
    condition_lookup,
):
    _batch, kspace, mask, pd, target, indices = batch_data
    output = {}
    for condition in CONDITIONS:
        pd_used, available = condition_pd(
            pd, indices, full_dataset, condition_lookup, condition
        )
        prediction, aux = model(
            kspace,
            mask,
            pd_used,
            available,
            return_aux=True,
        )
        prediction = center_crop(prediction, target.shape[-2], target.shape[-1])
        prediction_2d = prediction[0].detach().float().cpu()
        target_2d = target[0].detach().float().cpu()
        output[condition] = {
            "prediction": prediction_2d.numpy(),
            "target": target_2d.numpy(),
            "q": float(aux["q"][0].detach().float().mean().item()),
            **metrics(prediction_2d, target_2d),
        }
    return output


@torch.no_grad()
def predict_zero_filled(batch_data):
    _batch, kspace, _mask, _pd, target, _indices = batch_data
    prediction = fastmri.rss(
        fastmri.complex_abs(fastmri.ifft2c(kspace)), dim=1
    )
    prediction = center_crop(prediction, target.shape[-2], target.shape[-1])
    prediction_2d = prediction[0].detach().float().cpu()
    target_2d = target[0].detach().float().cpu()
    record = {
        "prediction": prediction_2d.numpy(),
        "target": target_2d.numpy(),
        "q": float("nan"),
        **metrics(prediction_2d, target_2d),
    }
    return {condition: dict(record) for condition in CONDITIONS}


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6.0,
            "axes.linewidth": 0.6,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def save_figure(fig, base: Path) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(
        base.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )


def normalised_arrays(result):
    target = np.asarray(result["target"], dtype=np.float32)
    prediction = np.asarray(result["prediction"], dtype=np.float32)
    scale = max(float(np.max(target)), 1e-8)
    return target / scale, prediction / scale


def render_error_plate(results, output_dir: Path, selection: Mapping[str, Any]):
    configure_matplotlib()
    width_in = 183.0 / 25.4
    height_in = 198.0 / 25.4
    fig = plt.figure(figsize=(width_in, height_in), constrained_layout=False)
    grid = fig.add_gridspec(
        len(CONDITIONS),
        len(MODEL_ORDER) + 1,
        width_ratios=[1, 1, 1, 1, 1, 1, 0.045],
        left=0.115,
        right=0.955,
        bottom=0.045,
        top=0.92,
        wspace=0.08,
        hspace=0.26,
    )
    for row_index, condition in enumerate(CONDITIONS):
        errors = []
        for model_name in MODEL_ORDER:
            target, prediction = normalised_arrays(results[model_name][condition])
            errors.append(np.abs(prediction - target))
        shared_max = max(
            float(np.quantile(np.concatenate([error.ravel() for error in errors]), 0.995)),
            1e-8,
        )
        image = None
        for column_index, (model_name, error) in enumerate(
            zip(MODEL_ORDER, errors)
        ):
            ax = fig.add_subplot(grid[row_index, column_index])
            image = ax.imshow(
                error,
                cmap="magma",
                vmin=0.0,
                vmax=shared_max,
                interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row_index == 0:
                ax.set_title(
                    MODEL_LABELS[model_name],
                    fontsize=6.3,
                    fontweight="semibold",
                    pad=3,
                )
            record = results[model_name][condition]
            text = f"L1 {record['l1']:.4f}"
            if model_name in FUSION_NAMES:
                text += f" | q {record['q']:.2f}"
            ax.set_xlabel(text, fontsize=4.7, labelpad=1.5)
            if column_index == 0:
                ax.set_ylabel(
                    CONDITION_LABELS[condition],
                    fontsize=6.3,
                    fontweight="semibold",
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=18,
                )
        cax = fig.add_subplot(grid[row_index, -1])
        colorbar = fig.colorbar(image, cax=cax)
        colorbar.set_ticks([0.0, shared_max / 2.0, shared_max])
        colorbar.ax.tick_params(labelsize=4.7, width=0.5, length=2)
        colorbar.set_label("Normalised absolute error", fontsize=5.0, labelpad=2)
        colorbar.formatter.set_powerlimits((-2, 2))
        colorbar.update_ticks()

    fig.suptitle(
        "Absolute-error maps across reconstruction models",
        fontsize=9.0,
        fontweight="bold",
        y=0.965,
    )
    fig.text(
        0.012,
        0.972,
        "a",
        fontsize=9,
        fontweight="bold",
        va="top",
    )
    fig.text(
        0.5,
        0.012,
        (
            "One source slice for all rows; error scales are shared within each "
            f"row. Patient {selection['patient_id']}, slice {selection['slice_idx']}."
        ),
        ha="center",
        va="bottom",
        fontsize=4.8,
        color="#444444",
    )
    save_figure(fig, output_dir / "Fig_model_absolute_error_maps")
    plt.close(fig)


def render_reconstruction_plate(
    results, output_dir: Path, selection: Mapping[str, Any]
):
    configure_matplotlib()
    columns = ("target", *MODEL_ORDER)
    labels = {"target": "Reference", **MODEL_LABELS}
    width_in = 183.0 / 25.4
    height_in = 178.0 / 25.4
    fig = plt.figure(figsize=(width_in, height_in), constrained_layout=False)
    grid = fig.add_gridspec(
        len(CONDITIONS),
        len(columns),
        left=0.11,
        right=0.985,
        bottom=0.05,
        top=0.91,
        wspace=0.035,
        hspace=0.18,
    )
    reference, _ = normalised_arrays(results[MODEL_ORDER[0]]["correct"])
    display_max = max(float(np.quantile(reference, 0.995)), 1e-8)
    for row_index, condition in enumerate(CONDITIONS):
        for column_index, name in enumerate(columns):
            ax = fig.add_subplot(grid[row_index, column_index])
            if name == "target":
                image = reference
                record = None
            else:
                _target, image = normalised_arrays(results[name][condition])
                record = results[name][condition]
            ax.imshow(
                image,
                cmap="gray",
                vmin=0.0,
                vmax=display_max,
                interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row_index == 0:
                ax.set_title(
                    labels[name],
                    fontsize=5.7,
                    fontweight="semibold",
                    pad=3,
                )
            if record is not None:
                ax.set_xlabel(
                    f"PSNR {record['psnr']:.2f}\nSSIM {record['ssim']:.3f}",
                    fontsize=4.2,
                    labelpad=1,
                )
            if column_index == 0:
                ax.set_ylabel(
                    CONDITION_LABELS[condition],
                    fontsize=6.0,
                    fontweight="semibold",
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=17,
                )
    fig.suptitle(
        "PD-FS reconstructions across auxiliary-input conditions",
        fontsize=9.0,
        fontweight="bold",
        y=0.96,
    )
    fig.text(0.012, 0.969, "b", fontsize=9, fontweight="bold", va="top")
    fig.text(
        0.5,
        0.012,
        (
            "Global intensity window shared across every panel; no local "
            "contrast adjustment or smoothing."
        ),
        ha="center",
        fontsize=4.8,
        color="#444444",
    )
    save_figure(fig, output_dir / "Fig_model_reconstruction_plate")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--full_clean_manifest", required=True)
    parser.add_argument("--robustness_manifest", required=True)
    parser.add_argument("--condition_manifest", required=True)
    parser.add_argument("--comparison_per_slice", required=True)
    parser.add_argument("--m2u_clean_checkpoint", required=True)
    parser.add_argument("--m2u_augmented_checkpoint", required=True)
    parser.add_argument("--m2u_augcap_mask_checkpoint", required=True)
    parser.add_argument("--global_direct_checkpoint", required=True)
    parser.add_argument("--quality_gain_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--patient_id")
    parser.add_argument("--slice_idx", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    per_slice_path = Path(args.comparison_per_slice).resolve()
    selection = select_representative(
        read_csv(per_slice_path), args.patient_id, args.slice_idx
    )
    (output_dir / "representative_case_selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )

    manifest_paths = {
        "full_clean": Path(args.full_clean_manifest).resolve(),
        "robustness": Path(args.robustness_manifest).resolve(),
    }
    manifests = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in manifest_paths.items()
    }
    manifest_hashes = {
        name: sha256_file(path) for name, path in manifest_paths.items()
    }
    condition_path = Path(args.condition_manifest).resolve()
    condition_manifest = json.loads(condition_path.read_text(encoding="utf-8"))
    condition_hash = sha256_file(condition_path)
    condition_lookup = {
        int(entry["source_index"]): entry
        for entry in condition_manifest["entries"]
    }
    metadata_path = Path(args.metadata_csv).resolve()
    metadata_hash = sha256_file(metadata_path)
    dataset_args = argparse.Namespace(
        metadata_csv=str(metadata_path),
        acceleration=8,
        pd_aux_acceleration=2,
    )
    full_dataset = IndexedDataset(make_dataset(dataset_args, "val"))
    robustness_dataset = ManifestDataset(
        full_dataset, manifests["robustness"]
    )
    batch_data = get_single_batch(
        robustness_dataset,
        selection["patient_id"],
        int(selection["slice_idx"]),
        device,
    )

    checkpoint_paths = {
        "m2u_clean": Path(args.m2u_clean_checkpoint).resolve(),
        "m2u_augmented": Path(args.m2u_augmented_checkpoint).resolve(),
        "m2u_augcap_mask": Path(args.m2u_augcap_mask_checkpoint).resolve(),
        "global_direct": Path(args.global_direct_checkpoint).resolve(),
        GAIN_MODEL: Path(args.quality_gain_checkpoint).resolve(),
    }
    results: Dict[str, Dict[str, Dict[str, Any]]] = {
        "zero_filled_exact_manifest": predict_zero_filled(batch_data)
    }
    for model_name in BASELINE_NAMES:
        model, _config, _summary, _epoch = load_baseline(
            checkpoint_paths[model_name],
            model_name,
            device,
            manifest_hashes["full_clean"],
            manifest_hashes["robustness"],
            metadata_hash,
        )
        results[model_name] = predict_model_conditions(
            model,
            model_name,
            batch_data,
            full_dataset,
            condition_lookup,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    for model_name in FUSION_NAMES:
        model, _config, _summary, _epoch = load_fusion(
            checkpoint_paths[model_name],
            model_name,
            device,
            manifest_hashes["full_clean"],
            manifest_hashes["robustness"],
            condition_hash,
            metadata_hash,
        )
        results[model_name] = predict_model_conditions(
            model,
            model_name,
            batch_data,
            full_dataset,
            condition_lookup,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    panel_rows = []
    arrays = {}
    for condition in CONDITIONS:
        target, _prediction = normalised_arrays(
            results["zero_filled_exact_manifest"][condition]
        )
        arrays[f"{condition}__target"] = target
        for model_name in MODEL_ORDER:
            _target, prediction = normalised_arrays(
                results[model_name][condition]
            )
            arrays[f"{condition}__{model_name}__prediction"] = prediction
            arrays[f"{condition}__{model_name}__absolute_error"] = np.abs(
                prediction - target
            )
            record = results[model_name][condition]
            panel_rows.append(
                {
                    "patient_id": selection["patient_id"],
                    "slice_idx": selection["slice_idx"],
                    "condition": condition,
                    "model": model_name,
                    "nmse": record["nmse"],
                    "psnr": record["psnr"],
                    "ssim": record["ssim"],
                    "l1": record["l1"],
                    "q": record["q"],
                }
            )
    np.savez_compressed(output_dir / "qualitative_source_arrays.npz", **arrays)
    write_csv(output_dir / "qualitative_panel_metrics.csv", panel_rows)

    render_error_plate(results, output_dir, selection)
    render_reconstruction_plate(results, output_dir, selection)
    qa = {
        "backend": "Python/matplotlib",
        "representative_case": selection,
        "source_data": {
            "arrays": "qualitative_source_arrays.npz",
            "metrics": "qualitative_panel_metrics.csv",
        },
        "image_integrity": {
            "crop": "identical dataset/evaluator centre crop",
            "normalisation": "each source slice divided by target maximum",
            "error_scaling": "shared within each condition row; 99.5th percentile",
            "reconstruction_window": "global across all panels; target 99.5th percentile",
            "smoothing": "none",
            "interpolation": "nearest",
            "local_adjustment": "none",
        },
        "exports": [
            "Fig_model_absolute_error_maps.png/pdf/svg/tiff",
            "Fig_model_reconstruction_plate.png/pdf/svg/tiff",
        ],
    }
    (output_dir / "qualitative_figure_QA.json").write_text(
        json.dumps(qa, indent=2), encoding="utf-8"
    )
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
