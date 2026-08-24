#!/usr/bin/env python3
"""Full-validation protocol audit for R=8 M2-GD before 50-epoch continuation.

Purpose
-------
This single job combines the checks needed to decide whether the current
M2-GD checkpoint can be continued from epoch 15 to epoch 50.

Models
------
M0    Single-contrast PD-FS VarNet (target-only reference; evaluated once).
M2-U  Ungated multi-scale PD fusion (robustness reference).
M2-GD Reliability-supervised global/local disagreement-gated fusion.

Stage A: direction-averaged zero-padded shifts
------------------------------------------------
For 2, 4 and 8 pixels, evaluate (+x), (-x), (+y), (-y).  Patient-level
metrics and gates are averaged across the four cardinal directions, while
per-direction variability is retained.

Stage B: 8-pixel shortcut controls
----------------------------------
* zero-padded cardinal shift: content moves, black border appears;
* reflect-padded cardinal shift: content moves, no black border;
* border-only cardinal control: content does not move, black border appears;
* wrong-patient PD: strong anatomical mismatch, no padding border;
* missing PD: hard availability=0 target-only-pathway reference.

Outputs include full-image and central-ROI metrics, q_hat/effective q,
per-scale and per-cascade gated RMS, channel/spatial gates, effective w,
protocol metadata, and paired M2-GD-vs-M2-U deltas.  Positive paired metric
improvement always means M2-GD is better.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

try:
    from skimage.metrics import structural_similarity as ssim_fn
except Exception:
    ssim_fn = None

from fastmri.models.varnet import VarNet
from src.dataset_paired_multicoil_aux_pd_r2 import (
    PairedMulticoilAuxPDToPDFSDataset,
)
from src.m2gd_auxiliary_varnet import M2GDAuxPDVarNet
from src.m2u_auxiliary_varnet_optimized import M2UAuxPDVarNet


CARDINAL_DIRECTIONS: Mapping[str, Tuple[int, int]] = {
    "pos_x": (0, 1),
    "neg_x": (0, -1),
    "pos_y": (1, 0),
    "neg_y": (-1, 0),
}
SHIFT_MAGNITUDES = (2, 4, 8)
METRICS = ("NMSE", "PSNR", "SSIM", "L1")
CENTRAL_METRICS = tuple(f"{metric}_central8" for metric in METRICS)
ALL_METRICS = METRICS + CENTRAL_METRICS
LOWER_IS_BETTER = {"NMSE", "L1", "NMSE_central8", "L1_central8"}
AUX_MODELS = ("M2U_ungated", "M2GD_gated")


class ShapeBucketBatchSampler(Sampler[List[int]]):
    """Batch slices with identical multicoil k-space tensor shape."""

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
    """Add a stable sample index while preserving records for bucketing."""

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
    raise RuntimeError(f"Cannot find model state dict in {path}")


def strip_module_prefix(state: Mapping[str, torch.Tensor]):
    if state and all(key.startswith("module.") for key in state):
        return {key[len("module.") :]: value for key, value in state.items()}
    return dict(state)


def checkpoint_config(checkpoint) -> dict:
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("config"), dict):
        return checkpoint["config"]
    return {}


def load_single(path: Path, device: torch.device) -> Tuple[VarNet, dict]:
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
        f"Loaded M0: {path} | epoch={checkpoint.get('epoch')} "
        f"| best_epoch={checkpoint.get('best_epoch')}"
    )
    return model, config


def load_m2u(path: Path, device: torch.device) -> Tuple[M2UAuxPDVarNet, dict]:
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
        f"Loaded M2-U: {path} | epoch={checkpoint.get('epoch')} "
        f"| best_epoch={checkpoint.get('best_epoch')}"
    )
    return model, config


def load_m2gd(path: Path, device: torch.device) -> Tuple[M2GDAuxPDVarNet, dict]:
    checkpoint = torch_load_checkpoint(path)
    config = checkpoint_config(checkpoint)
    model = M2GDAuxPDVarNet(
        num_cascades=int(config.get("num_cascades", 12)),
        sens_chans=int(config.get("sens_chans", 8)),
        sens_pools=int(config.get("sens_pools", 4)),
        chans=int(config.get("chans", 18)),
        pools=int(config.get("pools", 4)),
        mask_center=True,
        initial_q=float(config.get("initial_q", 0.8)),
        initial_local_gate=float(config.get("initial_local_gate", 0.35)),
        contribution_budgets=config.get("contribution_budgets", None),
    )
    model.load_state_dict(
        strip_module_prefix(extract_state_dict(checkpoint, path)), strict=True
    )
    model.to(device).eval()
    print(
        f"Loaded M2-GD: {path} | epoch={checkpoint.get('epoch')} "
        f"| best_epoch={checkpoint.get('best_epoch')}"
    )
    return model, config


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
    """Translate [B,H,W] without circular wrap-around, filling with zeros."""
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


def translate_reflect_pad(
    image: torch.Tensor,
    shift_y: int,
    shift_x: int,
) -> torch.Tensor:
    """Translate [B,H,W] with reflected boundary values and no black strip."""
    if image.ndim != 3:
        raise RuntimeError(f"Expected [B,H,W], got {tuple(image.shape)}")
    height, width = image.shape[-2:]
    pad = max(abs(int(shift_y)), abs(int(shift_x)))
    if pad == 0:
        return image.clone()
    if pad >= height or pad >= width:
        raise ValueError(
            f"Reflect padding {pad} must be smaller than image {(height, width)}"
        )
    padded = F.pad(
        image.unsqueeze(1),
        (pad, pad, pad, pad),
        mode="reflect",
    )[:, 0]
    start_y = pad - int(shift_y)
    start_x = pad - int(shift_x)
    return padded[..., start_y : start_y + height, start_x : start_x + width]


def border_only_zero(
    image: torch.Tensor,
    magnitude: int,
    direction: str,
) -> torch.Tensor:
    """Insert the zero strip caused by a shift without moving image content."""
    output = image.clone()
    magnitude = int(magnitude)
    if direction == "pos_x":
        output[..., :, :magnitude] = 0
    elif direction == "neg_x":
        output[..., :, -magnitude:] = 0
    elif direction == "pos_y":
        output[..., :magnitude, :] = 0
    elif direction == "neg_y":
        output[..., -magnitude:, :] = 0
    else:
        raise ValueError(f"Unsupported cardinal direction: {direction}")
    return output


def apply_cardinal_transform(
    pd_aux: torch.Tensor,
    family: str,
    magnitude: int,
    direction: str,
) -> torch.Tensor:
    dy_unit, dx_unit = CARDINAL_DIRECTIONS[direction]
    shift_y = int(magnitude) * dy_unit
    shift_x = int(magnitude) * dx_unit
    if family == "shift_zero":
        return translate_zero_pad(pd_aux, shift_y=shift_y, shift_x=shift_x)
    if family == "shift_reflect":
        return translate_reflect_pad(pd_aux, shift_y=shift_y, shift_x=shift_x)
    if family == "border_only":
        return border_only_zero(pd_aux, magnitude=magnitude, direction=direction)
    raise ValueError(f"Unsupported transform family: {family}")


def record_shape_key(record: dict) -> Tuple[int, ...]:
    with h5py.File(record["pdfs_path"], "r") as hf:
        return tuple(int(value) for value in hf["kspace"].shape[1:])


def record_slice_value(record: dict, fallback_index: int) -> float:
    """Return a deterministic slice-order value from heterogeneous record schemas."""
    for key in ("slice_idx", "slice_index", "slice"):
        if key in record:
            try:
                return float(record[key])
            except (TypeError, ValueError):
                pass
    return float(fallback_index)


def normalised_slice_positions(dataset) -> Dict[int, float]:
    """Map each sample to its relative through-plane position within a patient."""
    by_patient: Dict[str, List[int]] = defaultdict(list)
    for index, record in enumerate(dataset.records):
        by_patient[str(record["patient_id"])].append(index)

    positions: Dict[int, float] = {}
    for indices in by_patient.values():
        ordered = sorted(
            indices,
            key=lambda idx: (
                record_slice_value(dataset.records[idx], idx),
                idx,
            ),
        )
        denominator = max(len(ordered) - 1, 1)
        for rank, index in enumerate(ordered):
            positions[index] = float(rank) / float(denominator)
    return positions


def shape_distance(
    source_shape: Tuple[int, ...],
    candidate_shape: Tuple[int, ...],
) -> Tuple[int, int, int]:
    """Rank shape mismatch, prioritising spatial H/W agreement."""
    source_hw = source_shape[-2:] if len(source_shape) >= 2 else source_shape
    candidate_hw = (
        candidate_shape[-2:] if len(candidate_shape) >= 2 else candidate_shape
    )

    spatial_exact = int(tuple(source_hw) != tuple(candidate_hw))
    spatial_distance = sum(
        abs(int(a) - int(b)) for a, b in zip(source_hw, candidate_hw)
    )

    source_prefix = source_shape[:-2]
    candidate_prefix = candidate_shape[:-2]
    prefix_length = max(len(source_prefix), len(candidate_prefix))
    source_prefix = (0,) * (prefix_length - len(source_prefix)) + source_prefix
    candidate_prefix = (0,) * (prefix_length - len(candidate_prefix)) + candidate_prefix
    prefix_distance = sum(
        abs(int(a) - int(b))
        for a, b in zip(source_prefix, candidate_prefix)
    )
    return spatial_exact, spatial_distance, prefix_distance


def build_shape_matched_wrong_patient_map(
    dataset,
) -> Tuple[Dict[int, int], Dict[str, int]]:
    """Create deterministic wrong-patient matches with a safe fallback.

    Exact PDFS k-space shape matches are preferred. If an exact shape bucket
    contains only the source patient, a different patient with the nearest
    spatial shape is selected. Within either candidate set, the closest
    normalised through-plane slice position is used.
    """
    records = dataset.records
    patient_ids = [str(record["patient_id"]) for record in records]
    shape_keys = [record_shape_key(record) for record in records]
    relative_positions = normalised_slice_positions(dataset)

    grouped: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
    patients: Dict[str, List[int]] = defaultdict(list)
    for index, (shape_key, patient_id) in enumerate(zip(shape_keys, patient_ids)):
        grouped[shape_key].append(index)
        patients[patient_id].append(index)

    if len(patients) < 2:
        raise RuntimeError(
            "Wrong-patient control requires at least two validation patients."
        )

    mapping: Dict[int, int] = {}
    exact_count = 0
    fallback_count = 0
    all_indices = list(range(len(records)))

    for index in all_indices:
        source_patient = patient_ids[index]
        source_shape = shape_keys[index]
        source_position = relative_positions[index]

        exact_candidates = [
            candidate
            for candidate in grouped[source_shape]
            if patient_ids[candidate] != source_patient
        ]

        if exact_candidates:
            exact_count += 1
            replacement = min(
                exact_candidates,
                key=lambda candidate: (
                    abs(relative_positions[candidate] - source_position),
                    patient_ids[candidate],
                    candidate,
                ),
            )
        else:
            fallback_count += 1
            fallback_candidates = [
                candidate
                for candidate in all_indices
                if patient_ids[candidate] != source_patient
            ]
            if not fallback_candidates:
                raise RuntimeError(
                    f"No different-patient replacement exists for index={index}"
                )
            replacement = min(
                fallback_candidates,
                key=lambda candidate: (
                    *shape_distance(source_shape, shape_keys[candidate]),
                    abs(relative_positions[candidate] - source_position),
                    patient_ids[candidate],
                    candidate,
                ),
            )

        if patient_ids[replacement] == source_patient:
            raise RuntimeError(
                f"Internal error: source and replacement patient are identical "
                f"for index={index}"
            )
        mapping[index] = replacement

    if len(mapping) != len(records):
        raise RuntimeError(
            f"Wrong-patient map incomplete: {len(mapping)} of {len(records)} samples"
        )

    audit = {
        "total_samples": len(records),
        "exact_shape_matches": exact_count,
        "nearest_shape_fallbacks": fallback_count,
    }
    return mapping, audit


def center_crop_or_pad_2d(
    image: torch.Tensor,
    target_h: int,
    target_w: int,
) -> Tuple[torch.Tensor, bool]:
    """Centre-crop or zero-pad a 2-D image; never interpolate or resize."""
    if image.ndim != 2:
        raise ValueError(f"Expected 2-D image, got shape={tuple(image.shape)}")

    original_shape = tuple(int(value) for value in image.shape)
    height, width = original_shape

    if height > target_h:
        top = (height - target_h) // 2
        image = image[top : top + target_h, :]
    if width > target_w:
        left = (width - target_w) // 2
        image = image[:, left : left + target_w]

    height, width = image.shape
    pad_top = max((target_h - height) // 2, 0)
    pad_bottom = max(target_h - height - pad_top, 0)
    pad_left = max((target_w - width) // 2, 0)
    pad_right = max(target_w - width - pad_left, 0)

    if pad_top or pad_bottom or pad_left or pad_right:
        image = F.pad(
            image.unsqueeze(0).unsqueeze(0),
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=0.0,
        )[0, 0]

    if tuple(image.shape) != (target_h, target_w):
        raise RuntimeError(
            f"Could not align wrong-patient PD from {original_shape} "
            f"to {(target_h, target_w)}; got {tuple(image.shape)}"
        )
    return image, original_shape != (target_h, target_w)


def sample_pd_by_indices(
    dataset,
    indices: Sequence[int],
    device: torch.device,
    target_hw: Tuple[int, int],
) -> Tuple[torch.Tensor, int]:
    """Load wrong-patient RSS and align every image to the current source batch."""
    images: List[torch.Tensor] = []
    adjusted_count = 0
    target_h, target_w = int(target_hw[0]), int(target_hw[1])

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

        pd_image, adjusted = center_crop_or_pad_2d(
            pd_image,
            target_h=target_h,
            target_w=target_w,
        )
        adjusted_count += int(adjusted)
        images.append(pd_image)

    return (
        torch.stack(images, dim=0).to(device, non_blocking=True),
        adjusted_count,
    )


def condition_records() -> List[dict]:
    records: List[dict] = [
        {
            "condition": "correct",
            "condition_group": "correct",
            "family": "correct",
            "magnitude": 0,
            "direction": "none",
            "padding": "none",
            "content_moved": False,
            "black_border": False,
            "availability": 1,
            "q_target": 1.0,
        },
        {
            "condition": "missing",
            "condition_group": "missing",
            "family": "missing",
            "magnitude": 0,
            "direction": "none",
            "padding": "none",
            "content_moved": False,
            "black_border": False,
            "availability": 0,
            "q_target": 0.0,
        },
        {
            "condition": "wrong_patient",
            "condition_group": "wrong_patient",
            "family": "wrong_patient",
            "magnitude": 0,
            "direction": "none",
            "padding": "none",
            "content_moved": True,
            "black_border": False,
            "availability": 1,
            "q_target": float("nan"),
        },
    ]

    q_targets = {2: 0.65, 4: 0.35, 8: 0.05}
    for magnitude in SHIFT_MAGNITUDES:
        for direction in CARDINAL_DIRECTIONS:
            records.append(
                {
                    "condition": f"shift_zero_{magnitude}px_{direction}",
                    "condition_group": f"shift_zero_{magnitude}px",
                    "family": "shift_zero",
                    "magnitude": magnitude,
                    "direction": direction,
                    "padding": "zero",
                    "content_moved": True,
                    "black_border": True,
                    "availability": 1,
                    "q_target": q_targets[magnitude],
                }
            )

    for direction in CARDINAL_DIRECTIONS:
        records.append(
            {
                "condition": f"shift_reflect_8px_{direction}",
                "condition_group": "shift_reflect_8px",
                "family": "shift_reflect",
                "magnitude": 8,
                "direction": direction,
                "padding": "reflect",
                "content_moved": True,
                "black_border": False,
                "availability": 1,
                "q_target": float("nan"),
            }
        )
        records.append(
            {
                "condition": f"border_only_8px_{direction}",
                "condition_group": "border_only_8px",
                "family": "border_only",
                "magnitude": 8,
                "direction": direction,
                "padding": "zero",
                "content_moved": False,
                "black_border": True,
                "availability": 1,
                "q_target": float("nan"),
            }
        )
    return records


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


def metric_bundle(target: np.ndarray, prediction: np.ndarray) -> dict:
    return {
        "NMSE": nmse(target, prediction),
        "PSNR": psnr(target, prediction),
        "SSIM": ssim_metric(target, prediction),
        "L1": l1_metric(target, prediction),
    }


def central_roi(array: np.ndarray, border: int = 8) -> np.ndarray:
    if array.shape[-2] <= 2 * border or array.shape[-1] <= 2 * border:
        raise RuntimeError(
            f"Cannot remove {border}-pixel border from shape {array.shape[-2:]}"
        )
    return array[..., border:-border, border:-border]


def prediction_rows(
    prediction: torch.Tensor,
    target: torch.Tensor,
    batch: dict,
    model_name: str,
    meta: Mapping[str, object],
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

        full_metrics = metric_bundle(truth, pred)
        central_metrics = metric_bundle(central_roi(truth), central_roi(pred))
        row = {
            "model": model_name,
            **meta,
            "patient_id": str(get_batch_value(batch, "patient_id", sample_index)),
            "slice_idx": int(get_batch_value(batch, "slice_idx", sample_index)),
            "sample_idx": int(get_batch_value(batch, "sample_idx", sample_index)),
            "R": 8,
            "pd_aux_R": 2,
            **full_metrics,
        }
        for metric, value in central_metrics.items():
            row[f"{metric}_central8"] = value
        rows.append(row)
    return rows


def gate_rows(
    aux: Mapping[str, torch.Tensor],
    batch: dict,
    meta: Mapping[str, object],
    scale_names: Sequence[str],
) -> List[dict]:
    q_hat = aux["q_hat"].detach().cpu()
    q = aux["q"].detach().cpu()
    rms = aux["gated_aux_to_target_rms"].detach().cpu()
    g_ch = aux["g_ch_mean"].detach().cpu()
    g_sp = aux["g_sp_mean"].detach().cpu()
    w_mean = aux["w_mean"].detach().cpu()

    q_target = float(meta["q_target"])
    rows: List[dict] = []
    for sample_index in range(q_hat.shape[0]):
        if math.isfinite(q_target):
            target = torch.full_like(q_hat[sample_index], q_target)
            q_bce = float(
                F.binary_cross_entropy(
                    q_hat[sample_index].clamp(1e-7, 1 - 1e-7),
                    target,
                    reduction="mean",
                )
            )
        else:
            q_bce = float("nan")

        row = {
            **meta,
            "patient_id": str(get_batch_value(batch, "patient_id", sample_index)),
            "slice_idx": int(get_batch_value(batch, "slice_idx", sample_index)),
            "sample_idx": int(get_batch_value(batch, "sample_idx", sample_index)),
            "q_hat_mean": float(q_hat[sample_index].mean()),
            "q_mean": float(q[sample_index].mean()),
            "q_bce_to_protocol_target": q_bce,
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
    tensors = {
        "q_hat": aux["q_hat"].detach().double().cpu(),
        "q": aux["q"].detach().double().cpu(),
        "rms": aux["gated_aux_to_target_rms"].detach().double().cpu(),
        "g_ch": aux["g_ch_mean"].detach().double().cpu(),
        "g_sp": aux["g_sp_mean"].detach().double().cpu(),
        "w": aux["w_mean"].detach().double().cpu(),
    }
    if condition not in accumulator:
        accumulator[condition] = {
            "count": 0,
            "q_hat_sum": torch.zeros(tensors["q_hat"].shape[1], dtype=torch.float64),
            "q_sum": torch.zeros(tensors["q"].shape[1], dtype=torch.float64),
            "rms_sum": torch.zeros(
                tensors["rms"].shape[1], tensors["rms"].shape[2], dtype=torch.float64
            ),
            "g_ch_sum": torch.zeros_like(tensors["rms"].sum(dim=0)),
            "g_sp_sum": torch.zeros_like(tensors["rms"].sum(dim=0)),
            "w_sum": torch.zeros_like(tensors["rms"].sum(dim=0)),
        }
    item = accumulator[condition]
    item["count"] += int(tensors["q_hat"].shape[0])
    item["q_hat_sum"] += tensors["q_hat"].sum(dim=0)
    item["q_sum"] += tensors["q"].sum(dim=0)
    item["rms_sum"] += tensors["rms"].sum(dim=0)
    item["g_ch_sum"] += tensors["g_ch"].sum(dim=0)
    item["g_sp_sum"] += tensors["g_sp"].sum(dim=0)
    item["w_sum"] += tensors["w"].sum(dim=0)


def cascade_summary_rows(
    accumulator: dict,
    scale_names: Sequence[str],
    meta_by_condition: Mapping[str, Mapping[str, object]],
) -> List[dict]:
    rows: List[dict] = []
    for condition, item in accumulator.items():
        count = max(1, int(item["count"]))
        q_hat = item["q_hat_sum"] / count
        q = item["q_sum"] / count
        rms = item["rms_sum"] / count
        g_ch = item["g_ch_sum"] / count
        g_sp = item["g_sp_sum"] / count
        w = item["w_sum"] / count
        meta = meta_by_condition[condition]
        for cascade_index in range(q_hat.shape[0]):
            row = {
                **meta,
                "cascade": cascade_index + 1,
                "q_hat_mean": float(q_hat[cascade_index]),
                "q_mean": float(q[cascade_index]),
                "n_slices": count,
            }
            for scale_index, scale_name in enumerate(scale_names):
                safe_name = scale_name.replace("/", "_")
                row[f"gated_rms_{safe_name}"] = float(rms[cascade_index, scale_index])
                row[f"g_ch_{safe_name}"] = float(g_ch[cascade_index, scale_index])
                row[f"g_sp_{safe_name}"] = float(g_sp[cascade_index, scale_index])
                row[f"w_{safe_name}"] = float(w[cascade_index, scale_index])
            rows.append(row)
    return rows


def summarise_across_patients(
    patient_df: pd.DataFrame,
    group_columns: Sequence[str],
    value_columns: Sequence[str],
) -> pd.DataFrame:
    rows: List[dict] = []
    for group_key, group in patient_df.groupby(list(group_columns), dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = dict(zip(group_columns, group_key))
        row["n_patients"] = int(group["patient_id"].nunique())
        for column in value_columns:
            values = group[column].dropna().to_numpy(dtype=float)
            row[f"{column}_mean"] = float(np.mean(values)) if len(values) else float("nan")
            row[f"{column}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
            row[f"{column}_median"] = float(np.median(values)) if len(values) else float("nan")
            row[f"{column}_iqr_low"] = (
                float(np.percentile(values, 25)) if len(values) else float("nan")
            )
            row[f"{column}_iqr_high"] = (
                float(np.percentile(values, 75)) if len(values) else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(list(group_columns))


def positive_improvement(candidate: pd.Series, reference: pd.Series, metric: str):
    if metric in LOWER_IS_BETTER:
        return reference - candidate
    return candidate - reference


def paired_m2gd_vs_m2u(patient_group_df: pd.DataFrame) -> pd.DataFrame:
    candidate = patient_group_df[patient_group_df["model"] == "M2GD_gated"]
    reference = patient_group_df[patient_group_df["model"] == "M2U_ungated"]
    merged = candidate.merge(
        reference,
        on=["patient_id", "condition_group"],
        suffixes=("_m2gd", "_m2u"),
        how="inner",
        validate="one_to_one",
    )
    rows = pd.DataFrame(
        {
            "patient_id": merged["patient_id"],
            "condition_group": merged["condition_group"],
        }
    )
    for metric in ALL_METRICS:
        rows[f"{metric}_m2gd"] = merged[f"{metric}_m2gd"]
        rows[f"{metric}_m2u"] = merged[f"{metric}_m2u"]
        rows[f"{metric}_improvement"] = positive_improvement(
            merged[f"{metric}_m2gd"], merged[f"{metric}_m2u"], metric
        )
    return rows


def summarise_paired(delta_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for condition, group in delta_df.groupby("condition_group"):
        row = {
            "condition_group": condition,
            "n_patients": int(group["patient_id"].nunique()),
        }
        for metric in ALL_METRICS:
            values = group[f"{metric}_improvement"].dropna().to_numpy(dtype=float)
            row[f"{metric}_improvement_mean"] = float(np.mean(values)) if len(values) else float("nan")
            row[f"{metric}_improvement_median"] = float(np.median(values)) if len(values) else float("nan")
            row[f"{metric}_pct_m2gd_worse"] = (
                float(np.mean(values < 0) * 100.0) if len(values) else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("condition_group")


def direction_variability(
    patient_direction_df: pd.DataFrame,
    model_column: bool,
    value_columns: Sequence[str],
) -> pd.DataFrame:
    directional = patient_direction_df[
        patient_direction_df["direction"].isin(CARDINAL_DIRECTIONS.keys())
    ].copy()
    group_cols = ["patient_id", "condition_group"]
    if model_column:
        group_cols.insert(1, "model")
    rows: List[dict] = []
    for key, group in directional.groupby(group_cols):
        if group["direction"].nunique() != 4:
            continue
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        for column in value_columns:
            values = group[column].dropna().to_numpy(dtype=float)
            row[f"{column}_direction_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
            row[f"{column}_direction_range"] = (
                float(np.max(values) - np.min(values)) if len(values) else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def protocol_audit_summary(
    gate_group_summary: pd.DataFrame,
    paired_summary: pd.DataFrame,
) -> dict:
    def gate_value(group: str, column: str) -> float:
        rows = gate_group_summary[gate_group_summary["condition_group"] == group]
        if rows.empty:
            return float("nan")
        return float(rows.iloc[0][column])

    q_clean = gate_value("correct", "q_hat_mean_mean")
    q_zero8 = gate_value("shift_zero_8px", "q_hat_mean_mean")
    q_reflect8 = gate_value("shift_reflect_8px", "q_hat_mean_mean")
    q_border8 = gate_value("border_only_8px", "q_hat_mean_mean")
    q_wrong = gate_value("wrong_patient", "q_hat_mean_mean")
    rms_missing = gate_value("missing", "gated_rms_mean_mean")

    paired_lookup = {
        str(row["condition_group"]): row
        for _, row in paired_summary.iterrows()
    }

    return {
        "q_hat": {
            "clean": q_clean,
            "zero_shift_8px": q_zero8,
            "reflect_shift_8px": q_reflect8,
            "border_only_8px": q_border8,
            "wrong_patient": q_wrong,
        },
        "q_drop_from_clean": {
            "zero_shift_8px": q_clean - q_zero8,
            "reflect_shift_8px": q_clean - q_reflect8,
            "border_only_8px": q_clean - q_border8,
            "wrong_patient": q_clean - q_wrong,
        },
        "padding_shortcut_contrast": {
            "zero_minus_reflect_q_drop": (q_clean - q_zero8) - (q_clean - q_reflect8),
            "border_vs_reflect_q_drop_ratio": (
                (q_clean - q_border8) / max(abs(q_clean - q_reflect8), 1e-12)
            ),
        },
        "missing_gated_rms": rms_missing,
        "m2gd_vs_m2u_L1_improvement": {
            condition: float(row["L1_improvement_mean"])
            for condition, row in paired_lookup.items()
        },
        "interpretation_note": (
            "No hard Go/No-go threshold is imposed automatically. Continue to "
            "50 epochs only if reflect-shift and wrong-patient inputs reduce q/RMS, "
            "border-only does not cause a comparable reduction, missing RMS is zero, "
            "and M2-GD robustness is superior to M2-U without unacceptable clean loss."
        ),
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="M2-GD R=8 direction and padding-shortcut protocol audit."
    )
    parser.add_argument("--metadata_csv", type=Path, required=True)
    parser.add_argument("--single_checkpoint", type=Path, required=True)
    parser.add_argument("--m2u_checkpoint", type=Path, required=True)
    parser.add_argument("--m2gd_checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    for path in (
        args.metadata_csv,
        args.single_checkpoint,
        args.m2u_checkpoint,
        args.m2gd_checkpoint,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 96)
    print("M2-GD R=8 full-validation direction / padding-shortcut audit")
    print("=" * 96)
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Metadata:", args.metadata_csv)
    print("Output:", args.output_dir)
    print("Stage A: zero-pad cardinal shifts at 2/4/8 px")
    print("Stage B: zero/reflect/border-only 8 px + wrong patient + missing")
    print("=" * 96)

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

    wrong_patient_map, wrong_patient_map_audit = build_shape_matched_wrong_patient_map(dataset)
    print(
        "Wrong-patient map: PASSED | "
        f"entries={len(wrong_patient_map)} | "
        f"exact_shape={wrong_patient_map_audit['exact_shape_matches']} | "
        f"nearest_shape_fallback="
        f"{wrong_patient_map_audit['nearest_shape_fallbacks']}"
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

    single_model, single_config = load_single(args.single_checkpoint, device)
    m2u_model, m2u_config = load_m2u(args.m2u_checkpoint, device)
    m2gd_model, m2gd_config = load_m2gd(args.m2gd_checkpoint, device)

    condition_meta = condition_records()
    meta_by_condition = {str(item["condition"]): item for item in condition_meta}
    scale_names = list(m2gd_model.fusion_scale_names)

    all_metric_rows: List[dict] = []
    all_gate_rows: List[dict] = []
    cascade_accumulator: dict = {}
    shape_adjustment_count = 0
    wrong_patient_shape_adjustment_count = 0
    total_shape_checks = 0

    m0_meta = {
        "condition": "no_auxiliary",
        "condition_group": "no_auxiliary",
        "family": "no_auxiliary",
        "magnitude": 0,
        "direction": "none",
        "padding": "none",
        "content_moved": False,
        "black_border": False,
        "availability": 0,
        "q_target": float("nan"),
    }

    for batch_index, batch in enumerate(loader, start=1):
        kspace, mask, pd_aux, target = prepare_batch(batch, device)
        crop_h, crop_w = target.shape[-2:]
        kspace_hw = (int(kspace.shape[-3]), int(kspace.shape[-2]))
        pd_hw = (int(pd_aux.shape[-2]), int(pd_aux.shape[-1]))
        total_shape_checks += int(pd_aux.shape[0])
        if pd_hw != kspace_hw:
            shape_adjustment_count += int(pd_aux.shape[0])

        dataset_indices = [
            int(value) for value in batch["sample_idx"].detach().cpu().tolist()
        ]
        wrong_indices = [wrong_patient_map[index] for index in dataset_indices]
        wrong_pd, wrong_adjusted = sample_pd_by_indices(
            dataset,
            wrong_indices,
            device,
            target_hw=(int(pd_aux.shape[-2]), int(pd_aux.shape[-1])),
        )
        wrong_patient_shape_adjustment_count += int(wrong_adjusted)
        if wrong_pd.shape != pd_aux.shape:
            raise RuntimeError(
                f"Wrong-patient PD shape mismatch: {tuple(wrong_pd.shape)} "
                f"versus {tuple(pd_aux.shape)}"
            )

        single_prediction = center_crop_tensor(
            single_model(kspace, mask), crop_h, crop_w
        )
        all_metric_rows.extend(
            prediction_rows(
                single_prediction,
                target,
                batch,
                "M0_single",
                m0_meta,
            )
        )
        del single_prediction

        for meta in condition_meta:
            family = str(meta["family"])
            if family == "correct":
                condition_pd = pd_aux
            elif family == "missing":
                condition_pd = torch.zeros_like(pd_aux)
            elif family == "wrong_patient":
                condition_pd = wrong_pd
            else:
                condition_pd = apply_cardinal_transform(
                    pd_aux,
                    family=family,
                    magnitude=int(meta["magnitude"]),
                    direction=str(meta["direction"]),
                )

            availability = torch.full(
                (pd_aux.shape[0],),
                float(meta["availability"]),
                device=device,
                dtype=pd_aux.dtype,
            )

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
                    meta,
                )
            )
            del m2u_prediction

            m2gd_prediction, aux = m2gd_model(
                pdfs_masked_kspace=kspace,
                mask=mask,
                pd_aux_image=condition_pd,
                pd_available=availability,
                return_aux=True,
            )
            m2gd_prediction = center_crop_tensor(m2gd_prediction, crop_h, crop_w)
            all_metric_rows.extend(
                prediction_rows(
                    m2gd_prediction,
                    target,
                    batch,
                    "M2GD_gated",
                    meta,
                )
            )
            all_gate_rows.extend(gate_rows(aux, batch, meta, scale_names))
            update_cascade_accumulator(
                cascade_accumulator,
                aux,
                condition=str(meta["condition"]),
            )
            del m2gd_prediction, aux, condition_pd, availability

        del wrong_pd
        if batch_index == 1 or batch_index % 10 == 0 or batch_index == len(loader):
            print(f"Batch {batch_index}/{len(loader)}", flush=True)

    metrics_df = pd.DataFrame(all_metric_rows)
    gate_df = pd.DataFrame(all_gate_rows)

    expected_metric_rows = n_slices * (1 + len(AUX_MODELS) * len(condition_meta))
    if len(metrics_df) != expected_metric_rows:
        raise RuntimeError(
            f"Metric row count {len(metrics_df)} != expected {expected_metric_rows}"
        )
    expected_gate_rows = n_slices * len(condition_meta)
    if len(gate_df) != expected_gate_rows:
        raise RuntimeError(
            f"Gate row count {len(gate_df)} != expected {expected_gate_rows}"
        )

    metadata_columns = [
        "condition",
        "condition_group",
        "family",
        "magnitude",
        "direction",
        "padding",
        "content_moved",
        "black_border",
        "availability",
        "q_target",
    ]

    # Average slices within each patient and direction first.
    patient_direction_df = (
        metrics_df.groupby(
            ["patient_id", "model", *metadata_columns],
            as_index=False,
            dropna=False,
        )[list(ALL_METRICS)]
        .mean()
        .sort_values(["model", "condition_group", "direction", "patient_id"])
    )

    # Direction-averaged patient-level unit for cardinal conditions.
    patient_group_df = (
        patient_direction_df.groupby(
            ["patient_id", "model", "condition_group"],
            as_index=False,
            dropna=False,
        )[list(ALL_METRICS)]
        .mean()
        .sort_values(["model", "condition_group", "patient_id"])
    )
    patient_group_summary_df = summarise_across_patients(
        patient_group_df,
        group_columns=["model", "condition_group"],
        value_columns=ALL_METRICS,
    )

    metric_direction_variability_df = direction_variability(
        patient_direction_df,
        model_column=True,
        value_columns=ALL_METRICS,
    )
    metric_direction_variability_summary_df = summarise_across_patients(
        metric_direction_variability_df,
        group_columns=["model", "condition_group"],
        value_columns=[
            column
            for column in metric_direction_variability_df.columns
            if column.endswith("_direction_std") or column.endswith("_direction_range")
        ],
    )

    paired_df = paired_m2gd_vs_m2u(patient_group_df)
    paired_summary_df = summarise_paired(paired_df)

    gate_value_columns = [
        column
        for column in gate_df.columns
        if column
        not in {
            "condition",
            "condition_group",
            "family",
            "magnitude",
            "direction",
            "padding",
            "content_moved",
            "black_border",
            "availability",
            "q_target",
            "patient_id",
            "slice_idx",
            "sample_idx",
        }
    ]
    gate_patient_direction_df = (
        gate_df.groupby(
            ["patient_id", *metadata_columns],
            as_index=False,
            dropna=False,
        )[gate_value_columns]
        .mean()
        .sort_values(["condition_group", "direction", "patient_id"])
    )
    gate_patient_group_df = (
        gate_patient_direction_df.groupby(
            ["patient_id", "condition_group"],
            as_index=False,
            dropna=False,
        )[gate_value_columns]
        .mean()
        .sort_values(["condition_group", "patient_id"])
    )
    gate_group_summary_df = summarise_across_patients(
        gate_patient_group_df,
        group_columns=["condition_group"],
        value_columns=gate_value_columns,
    )
    gate_direction_variability_df = direction_variability(
        gate_patient_direction_df,
        model_column=False,
        value_columns=[
            "q_hat_mean",
            "q_mean",
            "gated_rms_mean",
            *[
                column
                for column in gate_value_columns
                if column.startswith("gated_rms_H_")
            ],
        ],
    )
    gate_direction_variability_summary_df = summarise_across_patients(
        gate_direction_variability_df,
        group_columns=["condition_group"],
        value_columns=[
            column
            for column in gate_direction_variability_df.columns
            if column.endswith("_direction_std") or column.endswith("_direction_range")
        ],
    )

    cascade_df = pd.DataFrame(
        cascade_summary_rows(cascade_accumulator, scale_names, meta_by_condition)
    ).sort_values(["condition_group", "direction", "cascade"])

    audit_summary = protocol_audit_summary(
        gate_group_summary_df,
        paired_summary_df,
    )

    training_protocol = {
        "pixel_unit": (
            "integer pixels in the dataset PD R=2 zero-filled RSS tensor, "
            "before M2GDAuxPDVarNet._prepare_pd and before PD-encoder normalisation"
        ),
        "training_shift_padding": "zero padding; no circular roll or reflection",
        "training_shift_magnitudes": m2gd_config.get("shift_magnitudes"),
        "training_condition_probabilities": {
            key: m2gd_config.get(key)
            for key in (
                "prob_clean",
                "prob_missing",
                "prob_shift",
                "prob_blur",
                "prob_noise",
            )
        },
        "training_reliability_targets": {
            key: m2gd_config.get(key)
            for key in (
                "reliability_clean",
                "reliability_missing",
                "reliability_shift_2",
                "reliability_shift_4",
                "reliability_shift_8",
                "reliability_blur_light",
                "reliability_blur_severe",
                "reliability_noise_light",
                "reliability_noise_severe",
            )
        },
        "evaluation_main_shift_directions": list(CARDINAL_DIRECTIONS),
        "evaluation_diagonal_shift": False,
        "q_aggregation": "mean over all 12 cascades per slice, then patient aggregation",
        "rms_aggregation": (
            "raw [slice,cascade,scale] retained; overall means and per-scale/"
            "per-cascade summaries also reported"
        ),
        "severity_supervision_note": (
            "q_hat is a severity-supervised reliability score, not a calibrated probability"
        ),
        "shape_preparation": {
            "samples_checked": total_shape_checks,
            "samples_requiring_model_center_crop_or_pad": shape_adjustment_count,
            "wrong_patient_samples_center_cropped_or_padded": wrong_patient_shape_adjustment_count,
            "corruption_order": (
                "corruption applied to dataset PD auxiliary before model _prepare_pd; "
                "model then center-crops/pads to target k-space H/W and normalises per sample"
            ),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "per_slice_metrics": args.output_dir / "protocol_audit_per_slice_metrics.csv",
        "patient_direction_metrics": args.output_dir / "protocol_audit_patient_direction_metrics.csv",
        "patient_direction_averaged": args.output_dir / "protocol_audit_patient_direction_averaged.csv",
        "patient_summary": args.output_dir / "protocol_audit_patient_summary.csv",
        "metric_direction_variability": args.output_dir / "protocol_audit_metric_direction_variability.csv",
        "metric_direction_variability_summary": args.output_dir / "protocol_audit_metric_direction_variability_summary.csv",
        "m2gd_vs_m2u": args.output_dir / "protocol_audit_m2gd_vs_m2u_patient_delta.csv",
        "m2gd_vs_m2u_summary": args.output_dir / "protocol_audit_m2gd_vs_m2u_summary.csv",
        "gate_per_slice": args.output_dir / "protocol_audit_gate_per_slice.csv",
        "gate_patient_direction": args.output_dir / "protocol_audit_gate_patient_direction.csv",
        "gate_patient_direction_averaged": args.output_dir / "protocol_audit_gate_patient_direction_averaged.csv",
        "gate_summary": args.output_dir / "protocol_audit_gate_summary.csv",
        "gate_direction_variability": args.output_dir / "protocol_audit_gate_direction_variability.csv",
        "gate_direction_variability_summary": args.output_dir / "protocol_audit_gate_direction_variability_summary.csv",
        "gate_cascade": args.output_dir / "protocol_audit_gate_cascade_summary.csv",
        "audit_summary": args.output_dir / "protocol_audit_decision_summary.json",
        "manifest": args.output_dir / "protocol_audit_manifest.json",
    }

    metrics_df.to_csv(output_paths["per_slice_metrics"], index=False)
    patient_direction_df.to_csv(output_paths["patient_direction_metrics"], index=False)
    patient_group_df.to_csv(output_paths["patient_direction_averaged"], index=False)
    patient_group_summary_df.to_csv(output_paths["patient_summary"], index=False)
    metric_direction_variability_df.to_csv(
        output_paths["metric_direction_variability"], index=False
    )
    metric_direction_variability_summary_df.to_csv(
        output_paths["metric_direction_variability_summary"], index=False
    )
    paired_df.to_csv(output_paths["m2gd_vs_m2u"], index=False)
    paired_summary_df.to_csv(output_paths["m2gd_vs_m2u_summary"], index=False)
    gate_df.to_csv(output_paths["gate_per_slice"], index=False)
    gate_patient_direction_df.to_csv(output_paths["gate_patient_direction"], index=False)
    gate_patient_group_df.to_csv(
        output_paths["gate_patient_direction_averaged"], index=False
    )
    gate_group_summary_df.to_csv(output_paths["gate_summary"], index=False)
    gate_direction_variability_df.to_csv(
        output_paths["gate_direction_variability"], index=False
    )
    gate_direction_variability_summary_df.to_csv(
        output_paths["gate_direction_variability_summary"], index=False
    )
    cascade_df.to_csv(output_paths["gate_cascade"], index=False)
    with open(output_paths["audit_summary"], "w", encoding="utf-8") as file:
        json.dump(audit_summary, file, indent=2)

    manifest = {
        "metadata_csv": str(args.metadata_csv.resolve()),
        "checkpoints": {
            "M0_single": str(args.single_checkpoint.resolve()),
            "M2U_ungated": str(args.m2u_checkpoint.resolve()),
            "M2GD_gated": str(args.m2gd_checkpoint.resolve()),
        },
        "n_patients": n_patients,
        "n_slices": n_slices,
        "R": 8,
        "pd_aux_R": 2,
        "central_roi": "outer 8 pixels removed on every side",
        "condition_matrix": condition_meta,
        "training_protocol": training_protocol,
        "patient_aggregation": (
            "mean slices within each patient/direction, then mean four cardinal "
            "directions within each patient/condition group, then equal-weight "
            "statistics across patients"
        ),
        "delta_sign": "positive means M2-GD better than M2-U",
        "wrong_patient_protocol": {
            "selection": (
                "different-patient PD; exact k-space shape preferred, "
                "nearest-shape fallback used when needed"
            ),
            "slice_matching": "nearest normalised through-plane position",
            "shape_alignment": (
                "centre crop or zero pad RSS to source batch H/W; "
                "no interpolation or resize"
            ),
            "mapping_audit": wrong_patient_map_audit,
            "runtime_shape_adjustments": wrong_patient_shape_adjustment_count,
        },
        "outputs": {
            key: str(path.resolve())
            for key, path in output_paths.items()
            if key != "manifest"
        },
    }
    with open(output_paths["manifest"], "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    print("=" * 96)
    print("DIRECTION-AVERAGED PATIENT SUMMARY")
    print("=" * 96)
    print(patient_group_summary_df.to_string(index=False))
    print("=" * 96)
    print("M2-GD VERSUS M2-U: positive means M2-GD better")
    print("=" * 96)
    print(paired_summary_df.to_string(index=False))
    print("=" * 96)
    print("M2-GD GATE SUMMARY")
    print("=" * 96)
    print(gate_group_summary_df.to_string(index=False))
    print("=" * 96)
    print("AUDIT SUMMARY")
    print(json.dumps(audit_summary, indent=2))
    print("=" * 96)
    print("Saved outputs:")
    for key, path in output_paths.items():
        print(f"  {key}: {path}")
    print("=" * 96)


if __name__ == "__main__":
    main()