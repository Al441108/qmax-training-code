#!/usr/bin/env python3
"""Export objective median-effect qualitative examples for formal R=8 robustness.

Cases are selected from the already completed formal cohort. For shift, wrong-
slice and wrong-patient conditions, the selected slice is closest to the median
positive L1 benefit of actual-q over q=1. For missing PD, selection uses the
median positive benefit over M2-U. This rule is deterministic and avoids manual
best-case selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from fastmri.models.varnet import VarNet
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from torch.utils.data import default_collate


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_m2gd_v21_smoke_audit_R8 import (  # noqa: E402
    IndexedDataset,
    alternative_batch,
    assert_checkpoint_identity,
    crop_prediction,
    checkpoint_identity,
    diagnostics_per_sample,
    extract_state_dict,
    load_m2gd_v21,
    load_m2u,
    metric_row,
    model_config,
    prepare_batch,
    set_seed,
    torch_load,
)
from src.auxiliary_corruptions_v21 import translate_nonwrapping  # noqa: E402
from src.dataset_paired_multicoil_aux_pd_r2 import (  # noqa: E402
    PairedMulticoilAuxPDToPDFSDataset,
)


STAGEB = "M2GDv21_StageB_actual_q"
Q1 = "M2GDv21_StageB_q1"
M2U = "M2U"
CONDITIONS = [
    "shift8_reflect_+x",
    "same_patient_wrong_slice",
    "wrong_patient_matched_level",
    "missing",
]
DISPLAY = {
    "shift8_reflect_+x": "Shift 8",
    "same_patient_wrong_slice": "Wrong slice",
    "wrong_patient_matched_level": "Wrong patient",
    "missing": "Missing PD",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_cases(per_slice: pd.DataFrame) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    keys = ["condition", "patient_id", "slice_idx", "source_index"]
    metrics = per_slice.pivot_table(index=keys, columns="model", values="L1", aggfunc="first").reset_index()
    actual_rows = per_slice[per_slice["model"] == STAGEB].copy()
    for condition in CONDITIONS:
        group = metrics[metrics["condition"] == condition].copy()
        reference = M2U if condition == "missing" else Q1
        if STAGEB not in group or reference not in group:
            raise RuntimeError(f"Missing {STAGEB}/{reference} rows for {condition}")
        group["benefit"] = group[reference] - group[STAGEB]
        positive = group[group["benefit"] > 0].copy()
        if positive.empty:
            raise RuntimeError(f"No positive median-effect candidates for {condition}")
        median = float(positive["benefit"].median())
        positive["distance"] = (positive["benefit"] - median).abs()
        chosen = positive.sort_values(
            ["distance", "patient_id", "slice_idx", "source_index"]
        ).iloc[0]
        metadata = actual_rows[
            (actual_rows["condition"] == condition)
            & (actual_rows["slice_idx"].astype(int) == int(chosen["slice_idx"]))
            & (actual_rows["source_index"].astype(int) == int(chosen["source_index"]))
        ]
        if len(metadata) != 1:
            raise RuntimeError(f"Expected one actual-q metadata row for selected {condition} case")
        row = metadata.iloc[0]
        selected.append(
            {
                "condition": condition,
                "patient_id": str(chosen["patient_id"]),
                "slice_idx": int(chosen["slice_idx"]),
                "source_index": int(chosen["source_index"]),
                "reference": reference,
                "selection_benefit_L1": float(chosen["benefit"]),
                "positive_candidate_median_L1_benefit": median,
                "replacement_index": (
                    int(row["replacement_index"])
                    if condition in {"same_patient_wrong_slice", "wrong_patient_matched_level"}
                    else None
                ),
            }
        )
    return selected


def image_array(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().float().cpu().numpy()
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise RuntimeError(f"Expected one 2-D image, got {array.shape}")
    return np.asarray(array, dtype=np.float32)


def robust_limit(array: np.ndarray) -> float:
    value = float(np.percentile(array[np.isfinite(array)], 99.5))
    return max(value, 1e-8)


def clean_single_state(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        while key.startswith("module.") or key.startswith("model."):
            key = key.split(".", 1)[1]
        cleaned[key] = value
    return cleaned


def load_single(path: Path, device: torch.device) -> Tuple[VarNet, Dict[str, Any]]:
    checkpoint = torch_load(path, device)
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("Single VarNet checkpoint must be a mapping.")
    config = model_config(checkpoint)
    model = VarNet(
        num_cascades=int(config.get("num_cascades", 12)),
        sens_chans=int(config.get("sens_chans", 8)),
        sens_pools=int(config.get("sens_pools", 4)),
        chans=int(config.get("chans", 18)),
        pools=int(config.get("pools", 4)),
        mask_center=True,
    ).to(device)
    model.load_state_dict(clean_single_state(extract_state_dict(checkpoint)), strict=True)
    model.eval()
    identity = checkpoint_identity(path, checkpoint)
    for key, expected in (("num_cascades", 12), ("chans", 18), ("pools", 4)):
        observed = identity.get(key)
        if observed is not None and int(observed) != expected:
            raise RuntimeError(f"Single VarNet {key}={observed}; expected {expected}.")
    observed_acceleration = identity.get("acceleration")
    if observed_acceleration is not None and int(observed_acceleration) != 8:
        raise RuntimeError(f"Single VarNet acceleration={observed_acceleration}; expected 8.")
    return model, identity


def save_publication_figure(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def style_image_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("black")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--single_checkpoint", required=True)
    parser.add_argument("--m2u_checkpoint", required=True)
    parser.add_argument("--stageb_checkpoint", required=True)
    parser.add_argument("--formal_results_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected_stageb_sha256", default="a917421b98a3c8482c7cd019bba12eaf5568c72b4398ec142c47007fb9213837")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Qualitative export requires a CUDA GPU.")
    formal_dir = Path(args.formal_results_dir)
    per_slice_path = formal_dir / "formal_robustness_per_slice.csv"
    decision_path = formal_dir / "formal_robustness_decision.json"
    if not per_slice_path.is_file() or not decision_path.is_file():
        raise FileNotFoundError("Formal per-slice results and decision JSON are required.")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if not decision.get("formal_robustness_confirmed", False):
        raise RuntimeError("Formal robustness was not confirmed; qualitative export stopped.")
    cases = select_cases(pd.read_csv(per_slice_path))

    base = PairedMulticoilAuxPDToPDFSDataset(
        metadata_csv=args.metadata_csv,
        split="val",
        pdfs_acceleration=8,
        pd_aux_acceleration=2,
        slices_per_patient=None,
        edge_weight=1.0,
    )
    full_dataset = IndexedDataset(base)
    single, single_identity = load_single(Path(args.single_checkpoint), device)
    m2u, m2u_identity = load_m2u(Path(args.m2u_checkpoint), device)
    stageb_path = Path(args.stageb_checkpoint)
    stageb, stageb_identity = load_m2gd_v21(stageb_path, device)
    assert_checkpoint_identity(m2u_identity, 50, 8, 2)
    assert_checkpoint_identity(stageb_identity, 3, 8, 2, "fusioncal3")
    if sha256_file(stageb_path) != args.expected_stageb_sha256:
        raise RuntimeError("Selected Stage-B SHA-256 mismatch.")

    records: List[Dict[str, Any]] = []
    arrays: Dict[str, np.ndarray] = {}
    for row_index, case in enumerate(cases):
        batch = default_collate([full_dataset[case["source_index"]]])
        kspace, mask, pd_aux, target = prepare_batch(batch, device)
        availability = torch.ones(1, device=device, dtype=pd_aux.dtype)
        condition = case["condition"]
        if condition == "shift8_reflect_+x":
            pd_input = torch.stack([translate_nonwrapping(pd_aux[0], 0, 8, "reflect")])
        elif condition in {"same_patient_wrong_slice", "wrong_patient_matched_level"}:
            pd_input = alternative_batch(
                full_dataset, [int(case["replacement_index"])], device,
                tuple(int(value) for value in pd_aux.shape[-2:])
            )
        elif condition == "missing":
            pd_input = torch.zeros_like(pd_aux)
            availability.zero_()
        else:
            raise RuntimeError(condition)

        pred_single = crop_prediction(single(kspace, mask), target)
        pred_m2u = crop_prediction(m2u(kspace, mask, pd_input), target)
        pred_q1, _ = stageb(
            pdfs_masked_kspace=kspace, mask=mask, pd_aux_image=pd_input,
            pd_available=availability, return_aux=True, q_override=1.0
        )
        pred_actual, aux = stageb(
            pdfs_masked_kspace=kspace, mask=mask, pd_aux_image=pd_input,
            pd_available=availability, return_aux=True, q_override=None
        )
        pred_q1 = crop_prediction(pred_q1, target)
        pred_actual = crop_prediction(pred_actual, target)
        diag = diagnostics_per_sample(aux)[0]
        prefix = f"case{row_index + 1}_{condition}"
        case_arrays = {
            "aux": image_array(pd_input),
            "target": image_array(target),
            "single": image_array(pred_single),
            "m2u": image_array(pred_m2u),
            "q1": image_array(pred_q1),
            "actual": image_array(pred_actual),
        }
        for name, array in case_arrays.items():
            arrays[f"{prefix}_{name}"] = array
        target_array = case_arrays["target"]
        metrics: Dict[str, float] = {}
        for model_name in ("single", "m2u", "q1", "actual"):
            for metric_name, value in metric_row(target_array, case_arrays[model_name]).items():
                metrics[f"{model_name}_{metric_name}"] = float(value)
            error = np.abs(target_array - case_arrays[model_name])
            arrays[f"{prefix}_error_{model_name}"] = error.astype(np.float32)
        records.append({**case, **diag, **metrics, "array_prefix": prefix})

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Figure 2A: reconstruction plate. A dedicated label column prevents text
    # from being drawn over images or clipped by tight bounding boxes.
    recon_names = ["aux", "target", "single", "m2u", "q1", "actual"]
    recon_titles = ["Auxiliary PD", "Target PD-FS", "Single", "M2-U", "$q=1$", "Actual $q$"]
    fig_recon = plt.figure(figsize=(7.2, 4.75), facecolor="white")
    recon_grid = fig_recon.add_gridspec(
        len(records), 1 + len(recon_names),
        width_ratios=[0.82, 1, 1, 1, 1, 1, 1],
        left=0.015, right=0.995, top=0.91, bottom=0.025,
        wspace=0.035, hspace=0.075,
    )
    for row_index, record in enumerate(records):
        label_ax = fig_recon.add_subplot(recon_grid[row_index, 0])
        label_ax.axis("off")
        label_ax.text(
            0.98, 0.55, DISPLAY[record["condition"]], ha="right", va="center",
            fontsize=6.8, fontweight="bold", transform=label_ax.transAxes,
        )
        label_ax.text(
            0.98, 0.34, f"q = {record['q_mean']:.2f}", ha="right", va="center",
            fontsize=5.8, color="#4D4D4D", transform=label_ax.transAxes,
        )
        prefix = record["array_prefix"]
        target = arrays[f"{prefix}_target"]
        target_vmax = robust_limit(target)
        aux = arrays[f"{prefix}_aux"]
        aux_vmax = robust_limit(aux) if np.any(aux) else 1.0
        for column, name in enumerate(recon_names, start=1):
            ax = fig_recon.add_subplot(recon_grid[row_index, column])
            style_image_axis(ax)
            array = arrays[f"{prefix}_{name}"]
            vmax = aux_vmax if name == "aux" else target_vmax
            ax.imshow(array, cmap="gray", vmin=0, vmax=vmax, interpolation="nearest")
            if row_index == 0:
                ax.set_title(recon_titles[column - 1], fontsize=6.3, pad=5)
    fig_recon.text(0.008, 0.965, "a", fontsize=8.5, fontweight="bold", va="top")
    fig_recon.text(
        0.5, 0.965, "Reconstruction comparison under controlled auxiliary-input failure",
        ha="center", va="top", fontsize=8.2, fontweight="bold",
    )
    save_publication_figure(fig_recon, output_dir / "Fig2A_reconstruction_comparison")
    plt.close(fig_recon)

    # Figure 2B: absolute-error plate. All four model error maps within a row
    # share one scale and one explicit colour bar. The colour-bar axis has its
    # own GridSpec column, so tick labels cannot overlap an image panel.
    error_names = ["single", "m2u", "q1", "actual"]
    error_titles = ["Single", "M2-U", "$q=1$", "Actual $q$"]
    fig_error = plt.figure(figsize=(7.2, 4.8), facecolor="white")
    error_grid = fig_error.add_gridspec(
        len(records), 2 + len(error_names),
        width_ratios=[0.94, 1, 1, 1, 1, 0.13],
        left=0.02, right=0.965, top=0.91, bottom=0.03,
        wspace=0.055, hspace=0.09,
    )
    error_scale_rows: List[Dict[str, Any]] = []
    for row_index, record in enumerate(records):
        prefix = record["array_prefix"]
        errors = [arrays[f"{prefix}_error_{name}"] for name in error_names]
        error_vmax = max(
            float(np.percentile(np.concatenate([item.ravel() for item in errors]), 99.5)),
            1e-8,
        )
        ticks = [0.0, error_vmax / 2.0, error_vmax]
        error_scale_rows.append(
            {
                "condition": record["condition"], "vmin": 0.0, "vmax": error_vmax,
                "tick_0": ticks[0], "tick_mid": ticks[1], "tick_max": ticks[2],
                "percentile_rule": 99.5,
            }
        )
        label_ax = fig_error.add_subplot(error_grid[row_index, 0])
        label_ax.axis("off")
        label_ax.text(
            0.98, 0.55, DISPLAY[record["condition"]], ha="right", va="center",
            fontsize=6.8, fontweight="bold", transform=label_ax.transAxes,
        )
        label_ax.text(
            0.98, 0.34, f"q = {record['q_mean']:.2f}", ha="right", va="center",
            fontsize=5.8, color="#4D4D4D", transform=label_ax.transAxes,
        )
        for column, (name, error) in enumerate(zip(error_names, errors), start=1):
            ax = fig_error.add_subplot(error_grid[row_index, column])
            style_image_axis(ax)
            ax.imshow(error, cmap="magma", vmin=0, vmax=error_vmax, interpolation="nearest")
            if row_index == 0:
                ax.set_title(error_titles[column - 1], fontsize=6.5, pad=5)
        color_ax = fig_error.add_subplot(error_grid[row_index, -1])
        colorbar = fig_error.colorbar(
            ScalarMappable(norm=Normalize(vmin=0, vmax=error_vmax), cmap="magma"),
            cax=color_ax, orientation="vertical", ticks=ticks,
        )
        colorbar.ax.tick_params(labelsize=5.0, width=0.6, length=2.2, pad=1.5)
        colorbar.outline.set_linewidth(0.6)
        colorbar.set_label("Absolute error", fontsize=5.4, labelpad=3.0)
    fig_error.text(0.008, 0.965, "b", fontsize=8.5, fontweight="bold", va="top")
    fig_error.text(
        0.5, 0.965, "Absolute-error maps with row-wise shared scales",
        ha="center", va="top", fontsize=8.2, fontweight="bold",
    )
    save_publication_figure(fig_error, output_dir / "Fig2B_absolute_error_maps")
    plt.close(fig_error)

    # Standalone colour bars are saved for slide-layout reuse. They use the
    # exact same scales as Fig. 2B and are never re-normalised independently.
    fig_bars = plt.figure(figsize=(4.8, 1.8), facecolor="white")
    bars_grid = fig_bars.add_gridspec(
        len(records), 2, width_ratios=[1.25, 4.0],
        left=0.02, right=0.98, top=0.94, bottom=0.10,
        wspace=0.08, hspace=0.85,
    )
    for row_index, row in enumerate(error_scale_rows):
        label_ax = fig_bars.add_subplot(bars_grid[row_index, 0])
        label_ax.axis("off")
        label_ax.text(0.98, 0.5, DISPLAY[row["condition"]], transform=label_ax.transAxes,
                      ha="right", va="center", fontsize=6.2, fontweight="bold")
        color_ax = fig_bars.add_subplot(bars_grid[row_index, 1])
        scalar = ScalarMappable(norm=Normalize(vmin=0, vmax=row["vmax"]), cmap="magma")
        colorbar = fig_bars.colorbar(
            scalar, cax=color_ax, orientation="horizontal",
            ticks=[row["tick_0"], row["tick_mid"], row["tick_max"]],
        )
        colorbar.ax.tick_params(labelsize=5.5, pad=1.5)
        colorbar.outline.set_linewidth(0.6)
    save_publication_figure(fig_bars, output_dir / "Fig2B_error_colorbars")
    plt.close(fig_bars)

    np.savez_compressed(output_dir / "Fig2_source_images.npz", **arrays)
    pd.DataFrame(records).to_csv(output_dir / "Fig2_case_metrics.csv", index=False)
    pd.DataFrame(error_scale_rows).to_csv(output_dir / "Fig2_error_scales.csv", index=False)
    manifest = {
        "selection_rule": (
            "Closest-to-median positive slice-level L1 benefit of actual-q over q=1 for shift/wrong-slice/"
            "wrong-patient; actual-q over M2-U for missing PD. Deterministic ties by patient, slice and source index."
        ),
        "cases": records,
        "display": {
            "reconstruction": "linear grayscale, vmin=0, row-specific target 99.5th percentile vmax",
            "auxiliary": "linear grayscale, row-specific auxiliary 99.5th percentile vmax",
            "absolute_error": (
                "magma, shared row-specific 99.5th percentile vmax across Single, M2-U, q=1 and actual-q; "
                "one explicit colour bar per row"
            ),
            "local_adjustments": "none",
            "crop": "none",
        },
        "checkpoint_identity": {
            "single": single_identity, "m2u": m2u_identity, "stageb": stageb_identity
        },
        "error_scales": error_scale_rows,
    }
    (output_dir / "Fig2_selection_and_integrity_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (output_dir / "Fig2_legend.txt").write_text(
        "Fig. 2 | Objective median-effect examples under auxiliary-input failure. a, Reconstructions from "
        "Single PD-FS VarNet, M2-U, Stage-B q=1 and Stage-B actual-q. b, Corresponding absolute-error maps. "
        "Rows show spatial shift, same-patient slice mismatch, wrong-patient mismatch and missing PD. Cases "
        "were selected by a "
        "predefined closest-to-median positive-effect rule from the formal 25-patient cohort, without manual "
        "image selection. Reconstruction images use a shared linear target-derived range within each row. "
        "All four error maps within a row use the same linear colour scale, shown by the adjacent colour bar. "
        "No local image adjustment or cropping was applied.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "completed", "cases": records}, indent=2))


if __name__ == "__main__":
    main()
