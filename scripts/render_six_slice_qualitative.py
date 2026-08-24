#!/usr/bin/env python3
from __future__ import annotations

"""Render six locked validation slices as reconstruction and error plates.

Outputs
-------
Group A uses three slices and compares:
    reference, zero-filled, M2-U Augmented, fifth arm, QMax-Full.

Group B uses three different slices and compares:
    reference, zero-filled, single-contrast VarNet R=8, QMax-Full.

For each of the six slices the script writes one reconstruction figure and one
absolute-error figure (12 slice-level figures total).  It also writes four
three-slice contact sheets for convenient presentation use.

Figure integrity
----------------
* All reconstructions within a slice use one target-derived intensity window.
* All error maps within a slice use one shared 99.5th-percentile error scale.
* No smoothing, local contrast adjustment, or per-model error scaling is used.
* Default case selection is deterministic and metric-blind; explicit cases can
  be locked through --cases_json.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import fastmri
from fastmri.models import VarNet
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_m2_prnf_final_comparison_R8 import (  # noqa: E402
    GAIN_MODEL,
    load_baseline,
    load_fusion,
    metrics,
)
from scripts.evaluate_qmax_counterfactuals import ManifestDataset  # noqa: E402
from scripts.qmax_common import (  # noqa: E402
    IndexedDataset,
    make_dataset,
    prepare_batch,
    set_seed,
    sha256_file,
)
from src.fft_utils import center_crop  # noqa: E402
from src.m2_prnf_qmax_varnet import QMaxAuxPDVarNet  # noqa: E402


GROUP_A_MODELS = (
    "zero_filled",
    "m2u_augmented",
    "quality_protected_hybrid_gain",
    "qmax_full",
)
GROUP_B_MODELS = (
    "zero_filled",
    "single_varnet_r8",
    "qmax_full",
)
MODEL_LABELS = {
    "reference": "Reference",
    "zero_filled": "Zero-filled",
    "m2u_augmented": "M2-U Augmented",
    "quality_protected_hybrid_gain": "Fifth arm",
    "single_varnet_r8": "Single VarNet R=8",
    "qmax_full": "QMax-Full",
}


def torch_load(path: Path, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6.5,
            "axes.linewidth": 0.6,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(
        base.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )


def case_key(case: Mapping[str, Any]) -> str:
    return f"{case['patient_id']}__slice{int(case['slice_idx']):03d}"


def short_case_label(case: Mapping[str, Any]) -> str:
    patient = str(case["patient_id"])
    return f"patient {patient[:10]} · slice {int(case['slice_idx'])}"


def _normalise_case(case_results: Mapping[str, Mapping[str, Any]]):
    target = np.asarray(case_results["reference"]["prediction"], dtype=np.float32)
    scale = max(float(target.max()), 1e-8)
    arrays = {
        name: np.asarray(record["prediction"], dtype=np.float32) / scale
        for name, record in case_results.items()
    }
    return arrays, scale


def _clean_state_dict(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        next_key = str(key)
        if next_key.startswith("module."):
            next_key = next_key[len("module.") :]
        if next_key.startswith("model."):
            next_key = next_key[len("model.") :]
        cleaned[next_key] = value
    return cleaned


def _state_from_checkpoint(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        return checkpoint
    for key in ("model_state_dict", "state_dict", "model"):
        if key in checkpoint:
            return checkpoint[key]
    return checkpoint


def _nested_config_value(config: Mapping[str, Any], key: str, default: Any) -> Any:
    if key in config:
        return config[key]
    for nested_key in ("model_kwargs", "model_config", "args"):
        nested = config.get(nested_key)
        if isinstance(nested, Mapping) and key in nested:
            return nested[key]
    return default


def load_single_varnet(args: argparse.Namespace, device: torch.device) -> VarNet:
    checkpoint = torch_load(Path(args.single_checkpoint), device)
    raw_config = checkpoint.get("config", {}) if isinstance(checkpoint, Mapping) else {}
    if isinstance(raw_config, Mapping):
        config = dict(raw_config)
    elif hasattr(raw_config, "__dict__"):
        config = vars(raw_config)
    else:
        config = {}
    kwargs = {
        "num_cascades": int(
            _nested_config_value(config, "num_cascades", args.single_num_cascades)
        ),
        "sens_chans": int(
            _nested_config_value(config, "sens_chans", args.single_sens_chans)
        ),
        "sens_pools": int(
            _nested_config_value(config, "sens_pools", args.single_sens_pools)
        ),
        "chans": int(_nested_config_value(config, "chans", args.single_chans)),
        "pools": int(_nested_config_value(config, "pools", args.single_pools)),
    }
    model = VarNet(**kwargs).to(device)
    model.load_state_dict(
        _clean_state_dict(_state_from_checkpoint(checkpoint)), strict=True
    )
    model.eval()
    print("Loaded single VarNet strictly:", json.dumps(kwargs, sort_keys=True))
    return model


def load_qmax_full(path: Path, device: torch.device) -> QMaxAuxPDVarNet:
    checkpoint = torch_load(path, device)
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("QMax checkpoint is not a mapping")
    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise RuntimeError("QMax checkpoint lacks config")
    if str(config.get("qmax_variant")) != "qmax_full":
        raise RuntimeError("QMax checkpoint is not qmax_full")
    if int(checkpoint.get("epoch", -1)) != 60:
        raise RuntimeError(
            f"Expected QMax-Full epoch 60, got {checkpoint.get('epoch')}"
        )
    model_kwargs = config.get("model_kwargs")
    if not isinstance(model_kwargs, Mapping):
        raise RuntimeError("QMax checkpoint lacks config.model_kwargs")
    model = QMaxAuxPDVarNet(
        qmax_variant="qmax_full", **dict(model_kwargs)
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


def _record_lookup(dataset: ManifestDataset) -> Dict[Tuple[str, int], int]:
    return {
        (str(record["patient_id"]), int(record["slice_idx"])): index
        for index, record in enumerate(dataset.records)
    }


def validate_explicit_cases(
    cases: Mapping[str, Any], dataset: ManifestDataset
) -> Dict[str, List[Dict[str, Any]]]:
    lookup = _record_lookup(dataset)
    output: Dict[str, List[Dict[str, Any]]] = {}
    seen = set()
    for group in ("group_a", "group_b"):
        values = list(cases.get(group, []))
        if len(values) != 3:
            raise ValueError(f"{group} must contain exactly three cases")
        output[group] = []
        for value in values:
            item = {
                "patient_id": str(value["patient_id"]),
                "slice_idx": int(value["slice_idx"]),
                "selection_rule": "investigator_pre_specified",
            }
            key = (item["patient_id"], item["slice_idx"])
            if key not in lookup:
                raise RuntimeError(f"Case not present in full-clean manifest: {key}")
            if key in seen:
                raise RuntimeError(f"Duplicate case across groups: {key}")
            seen.add(key)
            output[group].append(item)
    return output


def select_metric_blind_cases(
    dataset: ManifestDataset,
) -> Dict[str, List[Dict[str, Any]]]:
    """Select six distinct patients and their central available slice."""
    by_patient: Dict[str, List[int]] = defaultdict(list)
    for record in dataset.records:
        by_patient[str(record["patient_id"])].append(int(record["slice_idx"]))
    patients = sorted(by_patient)
    if len(patients) < 6:
        raise RuntimeError(f"Need at least six patients; found {len(patients)}")
    raw_positions = np.linspace(0, len(patients) - 1, 6)
    positions: List[int] = []
    for value in raw_positions:
        candidate = int(round(float(value)))
        while candidate in positions and candidate + 1 < len(patients):
            candidate += 1
        while candidate in positions and candidate - 1 >= 0:
            candidate -= 1
        positions.append(candidate)
    selected = []
    for position in positions:
        patient = patients[position]
        slices = sorted(set(by_patient[patient]))
        selected.append(
            {
                "patient_id": patient,
                "slice_idx": int(slices[len(slices) // 2]),
                "selection_rule": (
                    "metric-blind: six evenly spaced sorted patients; "
                    "central available full-clean slice"
                ),
            }
        )
    return {"group_a": selected[:3], "group_b": selected[3:]}


def get_batch(
    dataset: ManifestDataset,
    lookup: Mapping[Tuple[str, int], int],
    case: Mapping[str, Any],
):
    index = lookup[(str(case["patient_id"]), int(case["slice_idx"]))]
    loader = DataLoader(Subset(dataset, [index]), batch_size=1, num_workers=0)
    return next(iter(loader))


def _prepared(batch: Mapping[str, Any], device: torch.device):
    return prepare_batch(batch, device)


def _metric_record(
    prediction: torch.Tensor, target: torch.Tensor, q: float = float("nan")
) -> Dict[str, Any]:
    prediction = center_crop(prediction, target.shape[-2], target.shape[-1])
    prediction_2d = prediction[0].detach().float().cpu()
    target_2d = target[0].detach().float().cpu()
    return {
        "prediction": prediction_2d.numpy(),
        "q": float(q),
        **metrics(prediction_2d, target_2d),
    }


@torch.no_grad()
def predict_reference_and_zf(
    batch: Mapping[str, Any], device: torch.device
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    kspace, _mask, _pd, target, _indices = _prepared(batch, device)
    target_2d = target[0].detach().float().cpu()
    reference = {
        "prediction": target_2d.numpy(),
        "q": float("nan"),
        "nmse": 0.0,
        "psnr": float("inf"),
        "ssim": 1.0,
        "l1": 0.0,
    }
    prediction = fastmri.rss(
        fastmri.complex_abs(fastmri.ifft2c(kspace)), dim=1
    )
    return reference, _metric_record(prediction, target)


@torch.no_grad()
def predict_aux_model(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    device: torch.device,
    q_key: str | None,
) -> Dict[str, Any]:
    kspace, mask, pd, target, _indices = _prepared(batch, device)
    available = torch.ones(kspace.shape[0], device=device)
    prediction, auxiliary = model(
        kspace, mask, pd, available, return_aux=True
    )
    q = float("nan")
    if q_key is not None:
        if q_key not in auxiliary:
            raise RuntimeError(
                f"Expected auxiliary key {q_key}; observed={sorted(auxiliary)}"
            )
        q = float(auxiliary[q_key].detach().float().mean().item())
    return _metric_record(prediction, target, q=q)


def _num_low_frequencies(batch: Mapping[str, Any]):
    for key in (
        "pdfs_num_low_frequencies",
        "num_low_frequencies",
        "num_low_freqs",
    ):
        if key in batch:
            value = batch[key]
            if torch.is_tensor(value):
                value = value.detach().cpu().flatten()
                if len(value) == 1:
                    return int(value.item())
            return value
    return None


@torch.no_grad()
def predict_single_model(
    model: VarNet,
    batch: Mapping[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    kspace, mask, _pd, target, _indices = _prepared(batch, device)
    prediction = model(kspace, mask, _num_low_frequencies(batch))
    return _metric_record(prediction, target)


def _metric_text(record: Mapping[str, Any], include_q: bool = False) -> str:
    text = f"L1 {float(record['l1']):.4f}\nSSIM {float(record['ssim']):.3f}"
    if include_q and np.isfinite(float(record.get("q", float("nan")))):
        text += f" · q {float(record['q']):.2f}"
    return text


def render_slice_reconstruction(
    case: Mapping[str, Any],
    case_results: Mapping[str, Mapping[str, Any]],
    model_order: Sequence[str],
    output_base: Path,
) -> None:
    arrays, _scale = _normalise_case(case_results)
    columns = ("reference", *model_order)
    vmax = max(float(np.quantile(arrays["reference"], 0.995)), 1e-8)
    width_in = 183.0 / 25.4
    height_in = 48.0 / 25.4
    fig, axes = plt.subplots(1, len(columns), figsize=(width_in, height_in))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, columns):
        ax.imshow(arrays[name], cmap="gray", vmin=0.0, vmax=vmax, interpolation="nearest")
        ax.set_title(MODEL_LABELS[name], fontsize=6.5, fontweight="semibold", pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if name != "reference":
            ax.set_xlabel(
                _metric_text(
                    case_results[name],
                    include_q=name in {"quality_protected_hybrid_gain", "qmax_full"},
                ),
                fontsize=5.0,
                labelpad=2,
                linespacing=1.2,
            )
    fig.suptitle(
        f"Reconstruction comparison · {short_case_label(case)}",
        fontsize=8.5,
        fontweight="bold",
        y=0.99,
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.82, bottom=0.22, wspace=0.035)
    save_figure(fig, output_base)
    plt.close(fig)


def render_slice_error(
    case: Mapping[str, Any],
    case_results: Mapping[str, Mapping[str, Any]],
    model_order: Sequence[str],
    output_base: Path,
) -> None:
    arrays, _scale = _normalise_case(case_results)
    target = arrays["reference"]
    errors = {name: np.abs(arrays[name] - target) for name in model_order}
    vmax = max(
        float(np.quantile(np.concatenate([value.ravel() for value in errors.values()]), 0.995)),
        1e-8,
    )
    width_in = 183.0 / 25.4
    height_in = 48.0 / 25.4
    fig = plt.figure(figsize=(width_in, height_in))
    grid = fig.add_gridspec(
        1,
        len(model_order) + 1,
        width_ratios=[1] * len(model_order) + [0.045],
        left=0.02,
        right=0.95,
        top=0.82,
        bottom=0.22,
        wspace=0.06,
    )
    image = None
    for index, name in enumerate(model_order):
        ax = fig.add_subplot(grid[0, index])
        image = ax.imshow(
            errors[name], cmap="magma", vmin=0.0, vmax=vmax, interpolation="nearest"
        )
        ax.set_title(MODEL_LABELS[name], fontsize=6.5, fontweight="semibold", pad=3)
        ax.set_xlabel(
            f"L1 {float(case_results[name]['l1']):.4f}", fontsize=5.1, labelpad=2
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    cax = fig.add_subplot(grid[0, -1])
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_label("Normalised absolute error", fontsize=5.2, labelpad=3)
    colorbar.ax.tick_params(labelsize=4.8, width=0.5, length=2)
    colorbar.set_ticks([0.0, vmax / 2.0, vmax])
    fig.suptitle(
        f"Absolute-error comparison · {short_case_label(case)}",
        fontsize=8.5,
        fontweight="bold",
        y=0.99,
    )
    save_figure(fig, output_base)
    plt.close(fig)


def render_group_reconstruction(
    group_name: str,
    cases: Sequence[Mapping[str, Any]],
    results: Mapping[str, Mapping[str, Mapping[str, Any]]],
    model_order: Sequence[str],
    output_base: Path,
) -> None:
    columns = ("reference", *model_order)
    width_in = 183.0 / 25.4
    height_in = 150.0 / 25.4
    fig = plt.figure(figsize=(width_in, height_in))
    grid = fig.add_gridspec(
        len(cases), len(columns), left=0.12, right=0.99, top=0.91, bottom=0.055,
        wspace=0.035, hspace=0.30
    )
    for row, case in enumerate(cases):
        case_results = results[case_key(case)]
        arrays, _scale = _normalise_case(case_results)
        vmax = max(float(np.quantile(arrays["reference"], 0.995)), 1e-8)
        for column, name in enumerate(columns):
            ax = fig.add_subplot(grid[row, column])
            ax.imshow(arrays[name], cmap="gray", vmin=0.0, vmax=vmax, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(MODEL_LABELS[name], fontsize=6.2, fontweight="semibold", pad=3)
            if name != "reference":
                ax.set_xlabel(
                    _metric_text(case_results[name], include_q=False),
                    fontsize=4.5, labelpad=1.5, linespacing=1.15,
                )
            if column == 0:
                ax.set_ylabel(
                    f"{group_name}{row + 1}\n{str(case['patient_id'])[:8]}\nslice {int(case['slice_idx'])}",
                    fontsize=5.4, rotation=0, ha="right", va="center", labelpad=13,
                )
    fig.suptitle(
        f"{group_name}: reconstruction comparison across three locked slices",
        fontsize=9.0, fontweight="bold", y=0.975,
    )
    fig.text(
        0.5, 0.012,
        "One target-derived intensity window per slice; no local contrast adjustment.",
        ha="center", fontsize=4.8, color="#444444",
    )
    save_figure(fig, output_base)
    plt.close(fig)


def render_group_error(
    group_name: str,
    cases: Sequence[Mapping[str, Any]],
    results: Mapping[str, Mapping[str, Mapping[str, Any]]],
    model_order: Sequence[str],
    output_base: Path,
) -> None:
    width_in = 183.0 / 25.4
    height_in = 150.0 / 25.4
    fig = plt.figure(figsize=(width_in, height_in))
    grid = fig.add_gridspec(
        len(cases), len(model_order) + 1,
        width_ratios=[1] * len(model_order) + [0.045],
        left=0.12, right=0.95, top=0.91, bottom=0.055,
        wspace=0.06, hspace=0.30,
    )
    for row, case in enumerate(cases):
        case_results = results[case_key(case)]
        arrays, _scale = _normalise_case(case_results)
        target = arrays["reference"]
        errors = {name: np.abs(arrays[name] - target) for name in model_order}
        vmax = max(
            float(np.quantile(np.concatenate([e.ravel() for e in errors.values()]), 0.995)),
            1e-8,
        )
        image = None
        for column, name in enumerate(model_order):
            ax = fig.add_subplot(grid[row, column])
            image = ax.imshow(
                errors[name], cmap="magma", vmin=0.0, vmax=vmax, interpolation="nearest"
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(MODEL_LABELS[name], fontsize=6.2, fontweight="semibold", pad=3)
            ax.set_xlabel(
                f"L1 {float(case_results[name]['l1']):.4f}", fontsize=4.7, labelpad=1.5
            )
            if column == 0:
                ax.set_ylabel(
                    f"{group_name}{row + 1}\n{str(case['patient_id'])[:8]}\nslice {int(case['slice_idx'])}",
                    fontsize=5.4, rotation=0, ha="right", va="center", labelpad=13,
                )
        cax = fig.add_subplot(grid[row, -1])
        colorbar = fig.colorbar(image, cax=cax)
        colorbar.set_ticks([0.0, vmax / 2.0, vmax])
        colorbar.ax.tick_params(labelsize=4.5, width=0.5, length=2)
        colorbar.set_label("Normalised error", fontsize=4.8, labelpad=2)
    fig.suptitle(
        f"{group_name}: absolute-error comparison across three locked slices",
        fontsize=9.0, fontweight="bold", y=0.975,
    )
    fig.text(
        0.5, 0.012,
        "Error scale shared across models within each slice; colour bars are row-specific.",
        ha="center", fontsize=4.8, color="#444444",
    )
    save_figure(fig, output_base)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--full_clean_manifest", required=True)
    parser.add_argument("--robustness_manifest", required=True)
    parser.add_argument("--condition_manifest", required=True)
    parser.add_argument("--m2u_augmented_checkpoint", required=True)
    parser.add_argument("--quality_gain_checkpoint", required=True)
    parser.add_argument("--qmax_full_checkpoint", required=True)
    parser.add_argument("--single_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--cases_json",
        help=(
            "Optional JSON with group_a and group_b, each containing exactly "
            "three {patient_id, slice_idx} objects."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--single_num_cascades", type=int, default=12)
    parser.add_argument("--single_chans", type=int, default=18)
    parser.add_argument("--single_pools", type=int, default=4)
    parser.add_argument("--single_sens_chans", type=int, default=8)
    parser.add_argument("--single_sens_pools", type=int, default=4)
    args = parser.parse_args()

    # The historical final-comparison loaders deliberately accept only the
    # validation-selected checkpoints.  Fail here with a concise message
    # instead of reaching the loader after dataset construction.
    for label, value in (
        ("M2-U Augmented", args.m2u_augmented_checkpoint),
        ("fifth arm", args.quality_gain_checkpoint),
    ):
        if Path(value).name != "model_best.pt":
            raise RuntimeError(
                f"{label} must use model_best.pt because the frozen historical "
                f"evaluator rejects other checkpoint roles; got {value}"
            )

    set_seed(args.seed)
    configure_matplotlib()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir).resolve()
    slice_dir = output_dir / "slice_level_figures"
    group_dir = output_dir / "group_contact_sheets"
    slice_dir.mkdir(parents=True, exist_ok=True)
    group_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = Path(args.metadata_csv).resolve()
    full_clean_path = Path(args.full_clean_manifest).resolve()
    robustness_path = Path(args.robustness_manifest).resolve()
    condition_path = Path(args.condition_manifest).resolve()
    full_clean_manifest = read_json(full_clean_path)

    full_dataset = IndexedDataset(
        make_dataset(
            metadata_csv=str(metadata_path),
            split="val",
            acceleration=8,
            pd_aux_acceleration=2,
        )
    )
    clean_dataset = ManifestDataset(full_dataset, full_clean_manifest)
    lookup = _record_lookup(clean_dataset)

    if args.cases_json:
        selections = validate_explicit_cases(
            read_json(Path(args.cases_json).resolve()), clean_dataset
        )
    else:
        selections = select_metric_blind_cases(clean_dataset)
    (output_dir / "selected_six_cases.json").write_text(
        json.dumps(selections, indent=2), encoding="utf-8"
    )
    all_cases = list(selections["group_a"]) + list(selections["group_b"])

    results: Dict[str, Dict[str, Dict[str, Any]]] = {
        case_key(case): {} for case in all_cases
    }
    for case in all_cases:
        batch = get_batch(clean_dataset, lookup, case)
        reference, zf = predict_reference_and_zf(batch, device)
        results[case_key(case)]["reference"] = reference
        results[case_key(case)]["zero_filled"] = zf

    full_clean_hash = sha256_file(full_clean_path)
    robustness_hash = sha256_file(robustness_path)
    condition_hash = sha256_file(condition_path)
    metadata_hash = sha256_file(metadata_path)

    model, _config, _summary, _epoch = load_baseline(
        Path(args.m2u_augmented_checkpoint).resolve(),
        "m2u_augmented",
        device,
        full_clean_hash,
        robustness_hash,
        metadata_hash,
    )
    for case in selections["group_a"]:
        results[case_key(case)]["m2u_augmented"] = predict_aux_model(
            model, get_batch(clean_dataset, lookup, case), device, q_key=None
        )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model, _config, _summary, _epoch = load_fusion(
        Path(args.quality_gain_checkpoint).resolve(),
        GAIN_MODEL,
        device,
        full_clean_hash,
        robustness_hash,
        condition_hash,
        metadata_hash,
    )
    for case in selections["group_a"]:
        results[case_key(case)]["quality_protected_hybrid_gain"] = predict_aux_model(
            model, get_batch(clean_dataset, lookup, case), device, q_key="q"
        )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    qmax = load_qmax_full(Path(args.qmax_full_checkpoint).resolve(), device)
    for case in all_cases:
        results[case_key(case)]["qmax_full"] = predict_aux_model(
            qmax, get_batch(clean_dataset, lookup, case), device, q_key="q_hat"
        )
    del qmax
    if device.type == "cuda":
        torch.cuda.empty_cache()

    single = load_single_varnet(args, device)
    for case in selections["group_b"]:
        results[case_key(case)]["single_varnet_r8"] = predict_single_model(
            single, get_batch(clean_dataset, lookup, case), device
        )
    del single
    if device.type == "cuda":
        torch.cuda.empty_cache()

    metrics_rows: List[Dict[str, Any]] = []
    arrays: Dict[str, np.ndarray] = {}
    for group_name, cases, model_order in (
        ("A", selections["group_a"], GROUP_A_MODELS),
        ("B", selections["group_b"], GROUP_B_MODELS),
    ):
        for index, case in enumerate(cases, start=1):
            key = case_key(case)
            case_results = results[key]
            stem = f"{group_name}{index}_{str(case['patient_id'])[:10]}_slice{int(case['slice_idx']):03d}"
            render_slice_reconstruction(
                case, case_results, model_order, slice_dir / f"{stem}_reconstruction"
            )
            render_slice_error(
                case, case_results, model_order, slice_dir / f"{stem}_absolute_error"
            )
            for model_name in ("reference", *model_order):
                record = case_results[model_name]
                arrays[f"{stem}__{model_name}"] = np.asarray(
                    record["prediction"], dtype=np.float32
                )
                metrics_rows.append(
                    {
                        "group": group_name,
                        "case_index": index,
                        "patient_id": case["patient_id"],
                        "slice_idx": int(case["slice_idx"]),
                        "selection_rule": case["selection_rule"],
                        "model": model_name,
                        "l1": record["l1"],
                        "nmse": record["nmse"],
                        "psnr": record["psnr"],
                        "ssim": record["ssim"],
                        "q": record["q"],
                    }
                )
        render_group_reconstruction(
            group_name,
            cases,
            results,
            model_order,
            group_dir / f"Group_{group_name}_three_slice_reconstruction_plate",
        )
        render_group_error(
            group_name,
            cases,
            results,
            model_order,
            group_dir / f"Group_{group_name}_three_slice_absolute_error_plate",
        )

    write_csv(output_dir / "six_slice_panel_metrics.csv", metrics_rows)
    np.savez_compressed(output_dir / "six_slice_source_arrays.npz", **arrays)
    audit = {
        "protocol_version": "QMax-final-qualitative-validation-v1",
        "status": "passed",
        "scope": "locked validation only; held-out test not accessed",
        "scientific_role": "metric-blind qualitative validation; not model or epoch selection",
        "backend": "Python/matplotlib",
        "device": str(device),
        "groups": {
            "A": ["reference", *GROUP_A_MODELS],
            "B": ["reference", *GROUP_B_MODELS],
        },
        "num_distinct_slices": 6,
        "num_slice_level_figure_pairs": 6,
        "num_slice_level_figures": 12,
        "selected_cases": selections,
        "image_integrity": {
            "crop": "same evaluator centre crop for every model",
            "reconstruction_window": "target-derived 99.5th percentile, shared within slice",
            "error_scale": "99.5th percentile across all compared models, shared within slice",
            "normalisation": "each slice divided by its target maximum",
            "smoothing": "none",
            "interpolation": "nearest",
            "local_adjustment": "none",
        },
        "source_data": {
            "metrics": "six_slice_panel_metrics.csv",
            "arrays": "six_slice_source_arrays.npz",
        },
        "checkpoint_sha256": {
            "m2u_augmented": sha256_file(Path(args.m2u_augmented_checkpoint)),
            "quality_protected_hybrid_gain": sha256_file(Path(args.quality_gain_checkpoint)),
            "qmax_full": sha256_file(Path(args.qmax_full_checkpoint)),
            "single_varnet_r8": sha256_file(Path(args.single_checkpoint)),
        },
    }
    (output_dir / "six_slice_qualitative_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
