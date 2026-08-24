#!/usr/bin/env python3
"""Unified R=8 robustness evaluation for M0, M1, M2-U and M2-GD.

The evaluation uses the complete validation set and identical PD-FS k-space,
mask, target and auxiliary-PD corruption for every auxiliary model.

Models
------
M0    Single-contrast PD-FS VarNet.
M1    Direct image-domain auxiliary concatenation.
M2-U  Ungated multi-scale separate feature fusion.
M2-GD Availability-constrained global/local disagreement-gated fusion.

Auxiliary conditions
--------------------
correct, shift_2px, shift_4px, shift_8px, missing, wrong_patient.
Wrong-patient PD is selected deterministically from a different patient with
the same acquisition shape.

Primary statistical unit
------------------------
Metrics are first averaged across slices within each patient. Summary means,
medians and IQRs are then calculated across patients with equal patient weight.
Positive paired delta always means that the candidate is better.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler

try:
    from skimage.metrics import structural_similarity as ssim_fn
except Exception:
    ssim_fn = None

from fastmri.models.varnet import VarNet
from src.auxiliary_varnet import AuxPDVarNet
from src.dataset_paired_multicoil_aux_pd_r2 import (
    PairedMulticoilAuxPDToPDFSDataset,
)
from src.m2gd_auxiliary_varnet import M2GDAuxPDVarNet
from src.m2u_auxiliary_varnet_optimized import M2UAuxPDVarNet


METRICS = ("NMSE", "PSNR", "SSIM", "L1")
LOWER_IS_BETTER = {"NMSE", "L1"}
AUXILIARY_CONDITIONS = (
    "correct",
    "shift_2px",
    "shift_4px",
    "shift_8px",
    "missing",
    "wrong_patient",
)
MODEL_NAMES = ("M0_single", "M1_auxconcat", "M2U_ungated", "M2GD_gated")


class ShapeBucketBatchSampler(Sampler[List[int]]):
    """Batch slices with identical k-space tensor shape."""

    def __init__(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)

        buckets: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
        for index, record in enumerate(dataset.records):
            with h5py.File(record["pdfs_path"], "r") as hf:
                shape_key = tuple(int(value) for value in hf["kspace"].shape[1:])
            buckets[shape_key].append(index)

        self.buckets = dict(buckets)
        self.num_batches = sum(
            math.ceil(len(indices) / self.batch_size)
            for indices in self.buckets.values()
        )
        print(
            f"ShapeBucketBatchSampler: {len(self.buckets)} shape buckets, "
            f"{self.num_batches} batches, batch_size={self.batch_size}, "
            f"shuffle={self.shuffle}"
        )

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        batches: List[List[int]] = []
        for indices in self.buckets.values():
            indices = list(indices)
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batches.append(indices[start : start + self.batch_size])
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return self.num_batches


class IndexedDataset:
    """Add dataset index while preserving records for shape bucketing."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.records = dataset.records

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        item = self.dataset[index]
        item["sample_idx"] = int(index)
        return item


def torch_load_checkpoint(path: Path):
    """Load on CPU so optimizer/RNG entries do not occupy GPU memory."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_state_dict(checkpoint, path: Path):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "model", "state_dict", "net", "network"):
            state = checkpoint.get(key)
            if isinstance(state, dict):
                return state
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint
    raise RuntimeError(
        f"Cannot find model state dict in {path}. "
        f"Checkpoint type={type(checkpoint)}"
    )


def strip_module_prefix(state: Mapping[str, torch.Tensor]):
    if state and all(key.startswith("module.") for key in state):
        return {key[len("module.") :]: value for key, value in state.items()}
    return dict(state)


def checkpoint_config(checkpoint) -> dict:
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("config"), dict):
        return checkpoint["config"]
    return {}


def load_single(path: Path, device: torch.device) -> VarNet:
    checkpoint = torch_load_checkpoint(path)
    config = checkpoint_config(checkpoint)
    model = VarNet(
        num_cascades=int(config.get("num_cascades", 12)),
        sens_chans=int(config.get("sens_chans", 8)),
        sens_pools=int(config.get("sens_pools", 4)),
        chans=int(config.get("chans", 18)),
        pools=int(config.get("pools", 4)),
        mask_center=True,
    )
    model.load_state_dict(
        strip_module_prefix(extract_state_dict(checkpoint, path)), strict=True
    )
    model.to(device).eval()
    print(
        f"Loaded M0: {path} | epoch={checkpoint.get('epoch') if isinstance(checkpoint, dict) else None} "
        f"| best_epoch={checkpoint.get('best_epoch') if isinstance(checkpoint, dict) else None}"
    )
    return model


def load_m1(path: Path, device: torch.device) -> AuxPDVarNet:
    checkpoint = torch_load_checkpoint(path)
    config = checkpoint_config(checkpoint)
    model = AuxPDVarNet(
        num_cascades=int(config.get("num_cascades", 12)),
        chans=int(config.get("chans", 18)),
        pools=int(config.get("pools", 4)),
        sens_chans=int(config.get("sens_chans", 8)),
        sens_pools=int(config.get("sens_pools", 4)),
        mask_center=True,
    )
    model.load_state_dict(
        strip_module_prefix(extract_state_dict(checkpoint, path)), strict=True
    )
    model.to(device).eval()
    print(
        f"Loaded M1: {path} | epoch={checkpoint.get('epoch') if isinstance(checkpoint, dict) else None} "
        f"| best_epoch={checkpoint.get('best_epoch') if isinstance(checkpoint, dict) else None}"
    )
    return model


def load_m2u(path: Path, device: torch.device) -> M2UAuxPDVarNet:
    checkpoint = torch_load_checkpoint(path)
    config = checkpoint_config(checkpoint)
    model = M2UAuxPDVarNet(
        num_cascades=int(config.get("num_cascades", 12)),
        sens_chans=int(config.get("sens_chans", 8)),
        sens_pools=int(config.get("sens_pools", 4)),
        chans=int(config.get("chans", 18)),
        pools=int(config.get("pools", 4)),
        mask_center=True,
        initial_aux_alpha=float(config.get("initial_aux_alpha", 0.1)),
    )
    model.load_state_dict(
        strip_module_prefix(extract_state_dict(checkpoint, path)), strict=True
    )
    model.to(device).eval()
    print(
        f"Loaded M2-U: {path} | epoch={checkpoint.get('epoch') if isinstance(checkpoint, dict) else None} "
        f"| best_epoch={checkpoint.get('best_epoch') if isinstance(checkpoint, dict) else None}"
    )
    return model


def load_m2gd(path: Path, device: torch.device) -> M2GDAuxPDVarNet:
    checkpoint = torch_load_checkpoint(path)
    config = checkpoint_config(checkpoint)
    budgets = config.get("contribution_budgets", None)
    model = M2GDAuxPDVarNet(
        num_cascades=int(config.get("num_cascades", 12)),
        sens_chans=int(config.get("sens_chans", 8)),
        sens_pools=int(config.get("sens_pools", 4)),
        chans=int(config.get("chans", 18)),
        pools=int(config.get("pools", 4)),
        mask_center=True,
        initial_q=float(config.get("initial_q", 0.8)),
        initial_local_gate=float(config.get("initial_local_gate", 0.35)),
        contribution_budgets=budgets,
    )
    model.load_state_dict(
        strip_module_prefix(extract_state_dict(checkpoint, path)), strict=True
    )
    model.to(device).eval()
    print(
        f"Loaded M2-GD: {path} | epoch={checkpoint.get('epoch') if isinstance(checkpoint, dict) else None} "
        f"| best_epoch={checkpoint.get('best_epoch') if isinstance(checkpoint, dict) else None}"
    )
    return model


def prepare_batch(batch: dict, device: torch.device):
    kspace = batch["pdfs_masked_kspace"].to(device, non_blocking=True)
    if torch.is_complex(kspace):
        kspace = torch.view_as_real(kspace).float()
    else:
        kspace = kspace.float()

    mask = batch["mask"].to(device, non_blocking=True)
    if mask.ndim == 1:
        mask = mask[None, None, None, :, None]
    elif mask.ndim == 2:
        mask = mask[:, None, None, :, None]
    elif mask.ndim == 3:
        if mask.shape[1] == 1:
            mask = mask[:, :, None, :, None]
        else:
            mask = mask[:, None, None, :, None]
    elif mask.ndim == 4:
        mask = mask[..., None]
    elif mask.ndim != 5:
        raise RuntimeError(f"Unexpected mask shape: {tuple(mask.shape)}")
    mask = mask.bool()

    pd_aux = batch["pd_aux_image"].to(device, non_blocking=True).float()
    target = batch["pdfs_target_raw"].to(device, non_blocking=True).float()
    if pd_aux.ndim == 4 and pd_aux.shape[1] == 1:
        pd_aux = pd_aux[:, 0]
    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]
    if pd_aux.ndim != 3 or target.ndim != 3:
        raise RuntimeError(
            f"Expected PD and target [B,H,W], got {tuple(pd_aux.shape)} and "
            f"{tuple(target.shape)}"
        )
    return kspace, mask, pd_aux, target


def center_crop_tensor(x: torch.Tensor, crop_h: int, crop_w: int) -> torch.Tensor:
    height, width = x.shape[-2:]
    if (height, width) == (crop_h, crop_w):
        return x
    if height < crop_h or width < crop_w:
        raise RuntimeError(
            f"Cannot crop tensor from {(height, width)} to {(crop_h, crop_w)}"
        )
    top = (height - crop_h) // 2
    left = (width - crop_w) // 2
    return x[..., top : top + crop_h, left : left + crop_w]


def center_crop_np(x: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    height, width = x.shape[-2:]
    target_h, target_w = shape
    if (height, width) == shape:
        return x
    if height < target_h or width < target_w:
        raise RuntimeError(
            f"Cannot crop array from {(height, width)} to {(target_h, target_w)}"
        )
    top = (height - target_h) // 2
    left = (width - target_w) // 2
    return x[..., top : top + target_h, left : left + target_w]


def get_batch_value(batch: dict, key: str, index: int):
    value = batch[key]
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    return value[index]


def translate_zero_pad(
    image: torch.Tensor,
    shift_y: int,
    shift_x: int,
) -> torch.Tensor:
    """Translate [B,H,W] without circular wrap-around."""
    output = torch.zeros_like(image)
    height, width = image.shape[-2:]

    src_y0 = max(0, -shift_y)
    src_y1 = min(height, height - shift_y)
    dst_y0 = max(0, shift_y)
    dst_y1 = min(height, height + shift_y)
    src_x0 = max(0, -shift_x)
    src_x1 = min(width, width - shift_x)
    dst_x0 = max(0, shift_x)
    dst_x1 = min(width, width + shift_x)

    if src_y1 > src_y0 and src_x1 > src_x0:
        output[..., dst_y0:dst_y1, dst_x0:dst_x1] = image[
            ..., src_y0:src_y1, src_x0:src_x1
        ]
    return output


def shifted_pd(pd_aux: torch.Tensor, pixels: int, mode: str) -> torch.Tensor:
    if mode == "x":
        return translate_zero_pad(pd_aux, shift_y=0, shift_x=pixels)
    if mode == "y":
        return translate_zero_pad(pd_aux, shift_y=pixels, shift_x=0)
    if mode == "diagonal":
        return translate_zero_pad(pd_aux, shift_y=pixels, shift_x=pixels)
    raise ValueError(f"Unsupported shift mode: {mode}")


def record_shape_key(record: dict) -> Tuple[int, ...]:
    with h5py.File(record["pdfs_path"], "r") as hf:
        return tuple(int(value) for value in hf["kspace"].shape[1:])


def build_shape_matched_wrong_patient_map(dataset) -> Dict[int, int]:
    """Choose a deterministic different-patient sample in the same shape bucket."""
    grouped: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
    patient_ids = [str(record["patient_id"]) for record in dataset.records]
    for index, record in enumerate(dataset.records):
        grouped[record_shape_key(record)].append(index)

    mapping: Dict[int, int] = {}
    for indices in grouped.values():
        if len({patient_ids[index] for index in indices}) < 2:
            raise RuntimeError(
                "A shape bucket contains fewer than two patients; cannot create "
                "shape-matched wrong-patient auxiliary input."
            )
        count = len(indices)
        for local_position, index in enumerate(indices):
            replacement: Optional[int] = None
            start_offset = max(1, count // 3)
            for offset in range(start_offset, start_offset + count):
                candidate = indices[(local_position + offset) % count]
                if patient_ids[candidate] != patient_ids[index]:
                    replacement = candidate
                    break
            if replacement is None:
                raise RuntimeError(
                    f"No wrong-patient replacement for index={index}, "
                    f"patient={patient_ids[index]}"
                )
            mapping[index] = replacement
    return mapping


def sample_pd_by_indices(
    dataset,
    indices: Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    images: List[torch.Tensor] = []
    for index in indices:
        item = dataset[int(index)]
        pd_image = item["pd_aux_image"]
        if not torch.is_tensor(pd_image):
            pd_image = torch.as_tensor(pd_image)
        pd_image = pd_image.float()
        if pd_image.ndim == 3 and pd_image.shape[0] == 1:
            pd_image = pd_image[0]
        if pd_image.ndim != 2:
            raise RuntimeError(
                f"Unexpected wrong-patient PD shape at index={index}: "
                f"{tuple(pd_image.shape)}"
            )
        images.append(pd_image)
    return torch.stack(images, dim=0).to(device, non_blocking=True)


def nmse(target: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(np.linalg.norm(target) ** 2)
    if denominator <= 0:
        return float("nan")
    return float(np.linalg.norm(target - prediction) ** 2 / denominator)


def psnr(target: np.ndarray, prediction: np.ndarray) -> float:
    mse = float(np.mean((target - prediction) ** 2))
    if mse == 0:
        return float("inf")
    data_range = float(target.max() - target.min())
    if data_range <= 0:
        data_range = float(np.max(np.abs(target)))
    if data_range <= 0:
        return float("nan")
    return float(20 * np.log10(data_range) - 10 * np.log10(mse))


def ssim_metric(target: np.ndarray, prediction: np.ndarray) -> float:
    if ssim_fn is None:
        return float("nan")
    data_range = float(target.max() - target.min())
    if data_range <= 0:
        data_range = float(np.max(np.abs(target)))
    if data_range <= 0:
        return float("nan")
    return float(ssim_fn(target, prediction, data_range=data_range))


def l1_metric(target: np.ndarray, prediction: np.ndarray) -> float:
    scale = float(np.max(target))
    if scale < 1e-12:
        scale = 1.0
    return float(np.mean(np.abs(prediction / scale - target / scale)))


def prediction_rows(
    prediction: torch.Tensor,
    target: torch.Tensor,
    batch: dict,
    model_name: str,
    condition: str,
) -> List[dict]:
    prediction_np = prediction.detach().cpu().float().numpy()
    target_np = target.detach().cpu().float().numpy()
    rows: List[dict] = []

    for sample_index in range(prediction_np.shape[0]):
        pred = prediction_np[sample_index]
        truth = target_np[sample_index]
        if pred.shape != truth.shape:
            shape = (
                min(pred.shape[-2], truth.shape[-2]),
                min(pred.shape[-1], truth.shape[-1]),
            )
            pred = center_crop_np(pred, shape)
            truth = center_crop_np(truth, shape)

        slice_index = int(get_batch_value(batch, "slice_idx", sample_index))
        rows.append(
            {
                "model": model_name,
                "condition": condition,
                "patient_id": str(
                    get_batch_value(batch, "patient_id", sample_index)
                ),
                "slice_idx": slice_index,
                "sample_idx": int(
                    get_batch_value(batch, "sample_idx", sample_index)
                ),
                "first_slice_proxy": bool(
                    get_batch_value(batch, "is_edge", sample_index)
                ),
                "R": 8,
                "pd_aux_R": 2,
                "NMSE": nmse(truth, pred),
                "PSNR": psnr(truth, pred),
                "SSIM": ssim_metric(truth, pred),
                "L1": l1_metric(truth, pred),
            }
        )
    return rows


def m2gd_gate_rows(
    aux: Mapping[str, torch.Tensor],
    batch: dict,
    condition: str,
    scale_names: Sequence[str],
) -> List[dict]:
    q_hat = aux["q_hat"].detach().cpu()
    q = aux["q"].detach().cpu()
    rms = aux["gated_aux_to_target_rms"].detach().cpu()
    g_ch = aux["g_ch_mean"].detach().cpu()
    g_sp = aux["g_sp_mean"].detach().cpu()
    w_mean = aux["w_mean"].detach().cpu()

    rows: List[dict] = []
    for sample_index in range(q_hat.shape[0]):
        row = {
            "condition": condition,
            "patient_id": str(get_batch_value(batch, "patient_id", sample_index)),
            "slice_idx": int(get_batch_value(batch, "slice_idx", sample_index)),
            "sample_idx": int(get_batch_value(batch, "sample_idx", sample_index)),
            "q_hat_mean": float(q_hat[sample_index].mean()),
            "q_mean": float(q[sample_index].mean()),
            "gated_rms_mean": float(rms[sample_index].mean()),
        }
        for scale_index, scale_name in enumerate(scale_names):
            safe_name = scale_name.replace("/", "_")
            row[f"gated_rms_{safe_name}"] = float(
                rms[sample_index, :, scale_index].mean()
            )
            row[f"g_ch_{safe_name}"] = float(
                g_ch[sample_index, :, scale_index].mean()
            )
            row[f"g_sp_{safe_name}"] = float(
                g_sp[sample_index, :, scale_index].mean()
            )
            row[f"w_{safe_name}"] = float(
                w_mean[sample_index, :, scale_index].mean()
            )
        rows.append(row)
    return rows


def update_cascade_accumulator(
    accumulator: dict,
    aux: Mapping[str, torch.Tensor],
    condition: str,
) -> None:
    q_hat = aux["q_hat"].detach().double().cpu()
    q = aux["q"].detach().double().cpu()
    rms = aux["gated_aux_to_target_rms"].detach().double().cpu()

    if condition not in accumulator:
        accumulator[condition] = {
            "count": 0,
            "q_hat_sum": torch.zeros(q_hat.shape[1], dtype=torch.float64),
            "q_sum": torch.zeros(q.shape[1], dtype=torch.float64),
            "rms_sum": torch.zeros(
                rms.shape[1], rms.shape[2], dtype=torch.float64
            ),
        }
    item = accumulator[condition]
    item["count"] += int(q_hat.shape[0])
    item["q_hat_sum"] += q_hat.sum(dim=0)
    item["q_sum"] += q.sum(dim=0)
    item["rms_sum"] += rms.sum(dim=0)


def summarise_patient_metrics(patient_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for (model, condition), group in patient_df.groupby(["model", "condition"]):
        row = {
            "model": model,
            "condition": condition,
            "R": 8,
            "pd_aux_R": 2,
            "n_patients": int(group["patient_id"].nunique()),
        }
        for metric in METRICS:
            values = group[metric].dropna().to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values)) if len(values) else float("nan")
            row[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
            row[f"{metric}_median"] = float(np.median(values)) if len(values) else float("nan")
            row[f"{metric}_iqr_low"] = float(np.percentile(values, 25)) if len(values) else float("nan")
            row[f"{metric}_iqr_high"] = float(np.percentile(values, 75)) if len(values) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["model", "condition"])


def positive_improvement(
    candidate: pd.Series,
    reference: pd.Series,
    metric: str,
) -> pd.Series:
    if metric in LOWER_IS_BETTER:
        return reference - candidate
    return candidate - reference


def paired_comparison(
    patient_df: pd.DataFrame,
    candidate_model: str,
    reference_model: str,
    conditions: Sequence[str],
    reference_condition: Optional[str] = None,
) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for condition in conditions:
        candidate = patient_df[
            (patient_df["model"] == candidate_model)
            & (patient_df["condition"] == condition)
        ].copy()
        ref_condition = reference_condition or condition
        reference = patient_df[
            (patient_df["model"] == reference_model)
            & (patient_df["condition"] == ref_condition)
        ].copy()
        merged = candidate.merge(
            reference,
            on="patient_id",
            suffixes=("_candidate", "_reference"),
            how="inner",
            validate="one_to_one",
        )
        if merged.empty:
            raise RuntimeError(
                f"No paired patients for {candidate_model}/{condition} versus "
                f"{reference_model}/{ref_condition}"
            )
        output = pd.DataFrame(
            {
                "patient_id": merged["patient_id"],
                "candidate_model": candidate_model,
                "candidate_condition": condition,
                "reference_model": reference_model,
                "reference_condition": ref_condition,
            }
        )
        for metric in METRICS:
            output[f"{metric}_candidate"] = merged[f"{metric}_candidate"]
            output[f"{metric}_reference"] = merged[f"{metric}_reference"]
            output[f"{metric}_improvement"] = positive_improvement(
                merged[f"{metric}_candidate"],
                merged[f"{metric}_reference"],
                metric,
            )
        rows.append(output)
    return pd.concat(rows, ignore_index=True)


def summarise_paired_delta(delta_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    keys = [
        "candidate_model",
        "candidate_condition",
        "reference_model",
        "reference_condition",
    ]
    for group_key, group in delta_df.groupby(keys):
        row = dict(zip(keys, group_key))
        row["n_patients"] = int(group["patient_id"].nunique())
        for metric in METRICS:
            values = group[f"{metric}_improvement"].dropna().to_numpy(dtype=float)
            row[f"{metric}_improvement_mean"] = float(np.mean(values)) if len(values) else float("nan")
            row[f"{metric}_improvement_median"] = float(np.median(values)) if len(values) else float("nan")
            row[f"{metric}_improvement_iqr_low"] = float(np.percentile(values, 25)) if len(values) else float("nan")
            row[f"{metric}_improvement_iqr_high"] = float(np.percentile(values, 75)) if len(values) else float("nan")
            row[f"{metric}_pct_candidate_worse"] = (
                float(np.mean(values < 0) * 100.0) if len(values) else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys)


def summarise_gate_patient(gate_patient_df: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        column
        for column in gate_patient_df.columns
        if column not in {"patient_id", "condition"}
    ]
    rows: List[dict] = []
    for condition, group in gate_patient_df.groupby("condition"):
        row = {
            "condition": condition,
            "n_patients": int(group["patient_id"].nunique()),
        }
        for column in numeric_columns:
            values = group[column].dropna().to_numpy(dtype=float)
            row[f"{column}_mean"] = float(np.mean(values)) if len(values) else float("nan")
            row[f"{column}_median"] = float(np.median(values)) if len(values) else float("nan")
            row[f"{column}_iqr_low"] = float(np.percentile(values, 25)) if len(values) else float("nan")
            row[f"{column}_iqr_high"] = float(np.percentile(values, 75)) if len(values) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("condition")


def cascade_summary_rows(accumulator: dict, scale_names: Sequence[str]) -> List[dict]:
    rows: List[dict] = []
    for condition, item in sorted(accumulator.items()):
        count = int(item["count"])
        if count <= 0:
            continue
        q_hat_mean = item["q_hat_sum"] / count
        q_mean = item["q_sum"] / count
        rms_mean = item["rms_sum"] / count
        for cascade_index in range(q_hat_mean.shape[0]):
            row = {
                "condition": condition,
                "cascade": cascade_index + 1,
                "q_hat_mean": float(q_hat_mean[cascade_index]),
                "q_mean": float(q_mean[cascade_index]),
                "n_slices": count,
            }
            for scale_index, scale_name in enumerate(scale_names):
                safe_name = scale_name.replace("/", "_")
                row[f"gated_rms_{safe_name}"] = float(
                    rms_mean[cascade_index, scale_index]
                )
            rows.append(row)
    return rows


def validate_completeness(
    metrics_df: pd.DataFrame,
    n_slices: int,
    n_patients: int,
) -> None:
    expected = {
        ("M0_single", "no_auxiliary"),
        *{
            (model, condition)
            for model in ("M1_auxconcat", "M2U_ungated", "M2GD_gated")
            for condition in AUXILIARY_CONDITIONS
        },
    }
    observed = set(
        zip(metrics_df["model"].astype(str), metrics_df["condition"].astype(str))
    )
    missing = expected - observed
    if missing:
        raise RuntimeError(f"Missing model-condition outputs: {sorted(missing)}")

    for model, condition in sorted(expected):
        group = metrics_df[
            (metrics_df["model"] == model)
            & (metrics_df["condition"] == condition)
        ]
        if len(group) != n_slices:
            raise RuntimeError(
                f"Incomplete slice count for {model}/{condition}: "
                f"{len(group)} versus expected {n_slices}"
            )
        if group["patient_id"].nunique() != n_patients:
            raise RuntimeError(
                f"Incomplete patient count for {model}/{condition}: "
                f"{group['patient_id'].nunique()} versus expected {n_patients}"
            )


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified M0/M1/M2-U/M2-GD R=8 full-validation comparison."
    )
    parser.add_argument("--metadata_csv", type=Path, required=True)
    parser.add_argument("--single_checkpoint", type=Path, required=True)
    parser.add_argument("--m1_checkpoint", type=Path, required=True)
    parser.add_argument("--m2u_checkpoint", type=Path, required=True)
    parser.add_argument("--m2gd_checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--shift_pixels", type=int, nargs="+", default=[2, 4, 8]
    )
    parser.add_argument(
        "--shift_mode",
        choices=("x", "y", "diagonal"),
        default="diagonal",
        help="Use diagonal to match the previous M1/M2-U robustness protocol.",
    )
    args = parser.parse_args()

    if sorted(args.shift_pixels) != [2, 4, 8]:
        raise ValueError(
            "This controlled comparison requires --shift_pixels 2 4 8."
        )
    for path in (
        args.metadata_csv,
        args.single_checkpoint,
        args.m1_checkpoint,
        args.m2u_checkpoint,
        args.m2gd_checkpoint,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 88)
    print("Unified M0 / M1 / M2-U / M2-GD R=8 robustness evaluation")
    print("=" * 88)
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Metadata:", args.metadata_csv)
    print("Shift mode:", args.shift_mode)
    print("Output:", args.output_dir)
    print("=" * 88)

    base_dataset = PairedMulticoilAuxPDToPDFSDataset(
        metadata_csv=str(args.metadata_csv),
        split="val",
        pdfs_acceleration=8,
        pd_aux_acceleration=2,
    )
    dataset = IndexedDataset(base_dataset)
    n_slices = len(dataset)
    n_patients = len({str(record["patient_id"]) for record in dataset.records})
    print(f"Validation set: {n_patients} patients, {n_slices} slices")

    wrong_patient_map = build_shape_matched_wrong_patient_map(dataset)
    print(
        "Shape-matched wrong-patient map: PASSED | entries=",
        len(wrong_patient_map),
    )

    sampler = ShapeBucketBatchSampler(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        seed=42,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    single_model = load_single(args.single_checkpoint, device)
    m1_model = load_m1(args.m1_checkpoint, device)
    m2u_model = load_m2u(args.m2u_checkpoint, device)
    m2gd_model = load_m2gd(args.m2gd_checkpoint, device)

    all_metric_rows: List[dict] = []
    all_gate_rows: List[dict] = []
    cascade_accumulator: dict = {}
    scale_names = list(m2gd_model.fusion_scale_names)

    for batch_index, batch in enumerate(loader, start=1):
        kspace, mask, pd_aux, target = prepare_batch(batch, device)
        crop_h, crop_w = target.shape[-2:]
        dataset_indices = [
            int(value) for value in batch["sample_idx"].detach().cpu().tolist()
        ]

        single_prediction = center_crop_tensor(
            single_model(kspace, mask), crop_h, crop_w
        )
        all_metric_rows.extend(
            prediction_rows(
                single_prediction,
                target,
                batch,
                "M0_single",
                "no_auxiliary",
            )
        )
        del single_prediction

        wrong_indices = [wrong_patient_map[index] for index in dataset_indices]
        wrong_pd = sample_pd_by_indices(dataset, wrong_indices, device)
        if wrong_pd.shape != pd_aux.shape:
            raise RuntimeError(
                f"Wrong-patient PD shape mismatch: {tuple(wrong_pd.shape)} "
                f"versus {tuple(pd_aux.shape)}"
            )

        condition_inputs: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {
            "correct": (
                pd_aux,
                torch.ones(pd_aux.shape[0], device=device, dtype=pd_aux.dtype),
            ),
            "missing": (
                torch.zeros_like(pd_aux),
                torch.zeros(pd_aux.shape[0], device=device, dtype=pd_aux.dtype),
            ),
            "wrong_patient": (
                wrong_pd,
                torch.ones(pd_aux.shape[0], device=device, dtype=pd_aux.dtype),
            ),
        }
        for pixels in args.shift_pixels:
            condition_inputs[f"shift_{pixels}px"] = (
                shifted_pd(pd_aux, pixels=pixels, mode=args.shift_mode),
                torch.ones(pd_aux.shape[0], device=device, dtype=pd_aux.dtype),
            )

        # Preserve a fixed, explicit order in the outputs.
        ordered_inputs = [
            (condition, condition_inputs[condition])
            for condition in AUXILIARY_CONDITIONS
        ]

        for condition, (condition_pd, availability) in ordered_inputs:
            m1_prediction = center_crop_tensor(
                m1_model(
                    pdfs_masked_kspace=kspace,
                    mask=mask,
                    pd_aux_image=condition_pd,
                ),
                crop_h,
                crop_w,
            )
            all_metric_rows.extend(
                prediction_rows(
                    m1_prediction,
                    target,
                    batch,
                    "M1_auxconcat",
                    condition,
                )
            )
            del m1_prediction

            m2u_prediction = center_crop_tensor(
                m2u_model(
                    pdfs_masked_kspace=kspace,
                    mask=mask,
                    pd_aux_image=condition_pd,
                ),
                crop_h,
                crop_w,
            )
            all_metric_rows.extend(
                prediction_rows(
                    m2u_prediction,
                    target,
                    batch,
                    "M2U_ungated",
                    condition,
                )
            )
            del m2u_prediction

            m2gd_prediction, m2gd_aux = m2gd_model(
                pdfs_masked_kspace=kspace,
                mask=mask,
                pd_aux_image=condition_pd,
                pd_available=availability,
                return_aux=True,
            )
            m2gd_prediction = center_crop_tensor(
                m2gd_prediction, crop_h, crop_w
            )
            all_metric_rows.extend(
                prediction_rows(
                    m2gd_prediction,
                    target,
                    batch,
                    "M2GD_gated",
                    condition,
                )
            )
            all_gate_rows.extend(
                m2gd_gate_rows(
                    m2gd_aux,
                    batch,
                    condition,
                    scale_names,
                )
            )
            update_cascade_accumulator(
                cascade_accumulator,
                m2gd_aux,
                condition,
            )
            del m2gd_prediction, m2gd_aux

        del wrong_pd, condition_inputs, ordered_inputs
        if batch_index == 1 or batch_index % 25 == 0 or batch_index == len(loader):
            print(f"Batch {batch_index}/{len(loader)}", flush=True)

    metrics_df = pd.DataFrame(all_metric_rows)
    validate_completeness(metrics_df, n_slices=n_slices, n_patients=n_patients)

    # Patient-level unit: average all slices within each patient first.
    patient_df = (
        metrics_df.groupby(
            ["patient_id", "model", "condition", "R", "pd_aux_R"],
            as_index=False,
        )[list(METRICS)]
        .mean()
        .sort_values(["model", "condition", "patient_id"])
    )
    patient_summary_df = summarise_patient_metrics(patient_df)

    vs_single_frames = []
    for model in ("M1_auxconcat", "M2U_ungated", "M2GD_gated"):
        vs_single_frames.append(
            paired_comparison(
                patient_df,
                candidate_model=model,
                reference_model="M0_single",
                conditions=AUXILIARY_CONDITIONS,
                reference_condition="no_auxiliary",
            )
        )
    vs_single_df = pd.concat(vs_single_frames, ignore_index=True)
    vs_single_summary_df = summarise_paired_delta(vs_single_df)

    m2gd_vs_m2u_df = paired_comparison(
        patient_df,
        candidate_model="M2GD_gated",
        reference_model="M2U_ungated",
        conditions=AUXILIARY_CONDITIONS,
    )
    m2gd_vs_m2u_summary_df = summarise_paired_delta(m2gd_vs_m2u_df)

    m2gd_vs_m1_df = paired_comparison(
        patient_df,
        candidate_model="M2GD_gated",
        reference_model="M1_auxconcat",
        conditions=AUXILIARY_CONDITIONS,
    )
    m2gd_vs_m1_summary_df = summarise_paired_delta(m2gd_vs_m1_df)

    gate_df = pd.DataFrame(all_gate_rows).sort_values(
        ["condition", "patient_id", "slice_idx"]
    )
    gate_numeric_columns = [
        column
        for column in gate_df.columns
        if column not in {"condition", "patient_id", "slice_idx", "sample_idx"}
    ]
    gate_patient_df = (
        gate_df.groupby(["patient_id", "condition"], as_index=False)[
            gate_numeric_columns
        ]
        .mean()
        .sort_values(["condition", "patient_id"])
    )
    gate_summary_df = summarise_gate_patient(gate_patient_df)
    gate_cascade_df = pd.DataFrame(
        cascade_summary_rows(cascade_accumulator, scale_names)
    ).sort_values(["condition", "cascade"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "per_slice_metrics": args.output_dir / "unified_R8_per_slice_metrics.csv",
        "patient_level_metrics": args.output_dir / "unified_R8_patient_level_metrics.csv",
        "patient_summary": args.output_dir / "unified_R8_patient_summary.csv",
        "delta_vs_single": args.output_dir / "unified_R8_patient_delta_vs_single.csv",
        "delta_summary_vs_single": args.output_dir / "unified_R8_patient_delta_summary_vs_single.csv",
        "m2gd_vs_m2u": args.output_dir / "m2gd_vs_m2u_patient_delta.csv",
        "m2gd_vs_m2u_summary": args.output_dir / "m2gd_vs_m2u_patient_delta_summary.csv",
        "m2gd_vs_m1": args.output_dir / "m2gd_vs_m1_patient_delta.csv",
        "m2gd_vs_m1_summary": args.output_dir / "m2gd_vs_m1_patient_delta_summary.csv",
        "m2gd_gate_per_slice": args.output_dir / "m2gd_gate_per_slice.csv",
        "m2gd_gate_patient": args.output_dir / "m2gd_gate_patient_level.csv",
        "m2gd_gate_summary": args.output_dir / "m2gd_gate_summary.csv",
        "m2gd_gate_cascade": args.output_dir / "m2gd_gate_cascade_summary.csv",
        "manifest": args.output_dir / "evaluation_manifest.json",
    }

    metrics_df.to_csv(output_paths["per_slice_metrics"], index=False)
    patient_df.to_csv(output_paths["patient_level_metrics"], index=False)
    patient_summary_df.to_csv(output_paths["patient_summary"], index=False)
    vs_single_df.to_csv(output_paths["delta_vs_single"], index=False)
    vs_single_summary_df.to_csv(
        output_paths["delta_summary_vs_single"], index=False
    )
    m2gd_vs_m2u_df.to_csv(output_paths["m2gd_vs_m2u"], index=False)
    m2gd_vs_m2u_summary_df.to_csv(
        output_paths["m2gd_vs_m2u_summary"], index=False
    )
    m2gd_vs_m1_df.to_csv(output_paths["m2gd_vs_m1"], index=False)
    m2gd_vs_m1_summary_df.to_csv(
        output_paths["m2gd_vs_m1_summary"], index=False
    )
    gate_df.to_csv(output_paths["m2gd_gate_per_slice"], index=False)
    gate_patient_df.to_csv(output_paths["m2gd_gate_patient"], index=False)
    gate_summary_df.to_csv(output_paths["m2gd_gate_summary"], index=False)
    gate_cascade_df.to_csv(output_paths["m2gd_gate_cascade"], index=False)

    manifest = {
        "metadata_csv": str(args.metadata_csv.resolve()),
        "checkpoints": {
            "M0_single": str(args.single_checkpoint.resolve()),
            "M1_auxconcat": str(args.m1_checkpoint.resolve()),
            "M2U_ungated": str(args.m2u_checkpoint.resolve()),
            "M2GD_gated": str(args.m2gd_checkpoint.resolve()),
        },
        "R": 8,
        "pd_aux_R": 2,
        "n_patients": n_patients,
        "n_slices": n_slices,
        "conditions": list(AUXILIARY_CONDITIONS),
        "shift_mode": args.shift_mode,
        "patient_aggregation": (
            "mean metrics across slices within each patient, followed by "
            "equal-weight statistics across patients"
        ),
        "delta_sign": "positive always means candidate better",
        "wrong_patient_protocol": (
            "deterministic different-patient auxiliary selected within the "
            "same k-space shape bucket"
        ),
        "first_slice_proxy_note": (
            "The dataset is_edge field is retained only as first_slice_proxy "
            "and is not reported as formal edge-slice performance."
        ),
        "outputs": {key: str(path.resolve()) for key, path in output_paths.items() if key != "manifest"},
    }
    with open(output_paths["manifest"], "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    print("=" * 88)
    print("PATIENT-LEVEL SUMMARY")
    print("=" * 88)
    print(patient_summary_df.to_string(index=False))
    print("=" * 88)
    print("M2-GD VERSUS M2-U: positive improvement means M2-GD is better")
    print("=" * 88)
    print(m2gd_vs_m2u_summary_df.to_string(index=False))
    print("=" * 88)
    print("M2-GD GATE SUMMARY")
    print("=" * 88)
    print(gate_summary_df.to_string(index=False))
    print("=" * 88)
    print("Saved outputs:")
    for key, path in output_paths.items():
        print(f"  {key}: {path}")
    print("=" * 88)


if __name__ == "__main__":
    main()
