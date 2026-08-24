#!/usr/bin/env python3
"""Reduced protocol audit for the 5-epoch M2-GD v2 smoke checkpoint.

The audit is deliberately small enough to run before a 15-epoch pilot while
still testing the failure mode found in M2-GD v1:
  * clean paired PD;
  * shift8 across zero/reflect/replicate padding and all eight directions;
  * border-only8 (zero border shortcut control);
  * same-patient wrong slice;
  * wrong-patient matched anatomical level;
  * missing PD with exact hard availability masking.

M2-U is evaluated under every condition as the pretrained reconstruction
reference, allowing patient-paired robustness comparisons.
All reconstruction summaries are patient-level means. Positive improvement
means M2-GD v2 is better than M2-U for error metrics and quality metrics alike.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from skimage.metrics import structural_similarity as ssim_fn
except Exception:
    ssim_fn = None

from src.auxiliary_corruptions_v2 import (
    HardNegativeSampler,
    border_only,
    load_pd_auxiliary,
    translate_nonwrapping,
)
from src.dataset_paired_multicoil_aux_pd_r2 import (
    PairedMulticoilAuxPDToPDFSDataset,
)
from src.fft_utils import center_crop
from src.m2gd_v2_auxiliary_varnet import M2GDv2AuxPDVarNet
from src.m2u_auxiliary_varnet_optimized import M2UAuxPDVarNet


METRICS = ("NMSE", "PSNR", "SSIM", "L1")
SCALE_NAMES = ("H/2", "H/4", "H/8", "H/16")
DIRECTIONS: Tuple[Tuple[str, int, int, str], ...] = (
    ("+x", 0, +1, "cardinal"),
    ("-x", 0, -1, "cardinal"),
    ("+y", +1, 0, "cardinal"),
    ("-y", -1, 0, "cardinal"),
    ("+x+y", +1, +1, "diagonal"),
    ("+x-y", -1, +1, "diagonal"),
    ("-x+y", +1, -1, "diagonal"),
    ("-x-y", -1, -1, "diagonal"),
)
PADDING_MODES = ("zero", "reflect", "replicate")


class IndexedDataset:
    def __init__(self, dataset):
        self.dataset = dataset
        self.records = dataset.records
        self.patient_rows = dataset.patient_rows

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        item = self.dataset[index]
        item["sample_idx"] = int(index)
        return item


class SelectedIndexDataset:
    """View of an IndexedDataset that retains full-dataset source indices."""

    def __init__(self, full_dataset: IndexedDataset, selected_indices: Sequence[int]):
        self.full_dataset = full_dataset
        self.selected_indices = [int(index) for index in selected_indices]
        self.records = [full_dataset.records[index] for index in self.selected_indices]

    def __len__(self):
        return len(self.selected_indices)

    def __getitem__(self, local_index: int):
        source_index = self.selected_indices[int(local_index)]
        return self.full_dataset[source_index]


class ShapeBucketBatchSampler(Sampler[List[int]]):
    def __init__(self, dataset, batch_size: int, seed: int = 42):
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        buckets: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
        for local_index, record in enumerate(dataset.records):
            with h5py.File(record["pdfs_path"], "r") as hf:
                key = tuple(int(value) for value in hf["kspace"].shape[1:])
            buckets[key].append(local_index)
        self.buckets = dict(buckets)
        self.num_batches = sum(
            math.ceil(len(indices) / self.batch_size)
            for indices in self.buckets.values()
        )

    def __iter__(self):
        for key in sorted(self.buckets, key=str):
            indices = self.buckets[key]
            for start in range(0, len(indices), self.batch_size):
                yield indices[start : start + self.batch_size]

    def __len__(self):
        return self.num_batches


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def extract_state_dict(checkpoint: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    for key in ("model_state_dict", "model", "state_dict", "net", "network"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value  # type: ignore[return-value]
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint  # type: ignore[return-value]
    raise RuntimeError("Checkpoint contains no model state dict.")


def strip_module_prefix(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    state = dict(state)
    if state and all(key.startswith("module.") for key in state):
        return {key[len("module.") :]: value for key, value in state.items()}
    return state


def model_config(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    value = checkpoint.get("config", {})
    return value if isinstance(value, Mapping) else {}


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_identity(
    path: Path,
    checkpoint: Mapping[str, Any],
) -> Dict[str, Any]:
    config = model_config(checkpoint)
    return {
        "path": str(path),
        "sha256": checkpoint_sha256(path),
        "size_bytes": int(path.stat().st_size),
        "epoch": checkpoint.get("epoch"),
        "best_epoch": checkpoint.get("best_epoch"),
        "best_val": checkpoint.get("best_val"),
        "acceleration": config.get("acceleration", config.get("pdfs_acceleration")),
        "pd_aux_acceleration": config.get("pd_aux_acceleration"),
        "curriculum": config.get("curriculum"),
        "num_cascades": config.get("num_cascades"),
        "chans": config.get("chans"),
        "pools": config.get("pools"),
    }


def assert_checkpoint_identity(
    identity: Mapping[str, Any],
    expected_epoch: int,
    expected_acceleration: int,
    expected_pd_acceleration: int,
    expected_curriculum: Optional[str] = None,
) -> None:
    if int(identity.get("epoch", -1)) != int(expected_epoch):
        raise RuntimeError(
            f"Checkpoint {identity.get('path')} has epoch={identity.get('epoch')}; "
            f"expected {expected_epoch}."
        )
    for key, expected in (
        ("acceleration", expected_acceleration),
        ("pd_aux_acceleration", expected_pd_acceleration),
    ):
        actual = identity.get(key)
        if actual is None:
            raise RuntimeError(
                f"Checkpoint {identity.get('path')} does not record {key}."
            )
        if int(actual) != int(expected):
            raise RuntimeError(
                f"Checkpoint {identity.get('path')} has {key}={actual}; "
                f"expected {expected}."
            )
    if expected_curriculum is not None:
        actual = identity.get("curriculum")
        if str(actual) != str(expected_curriculum):
            raise RuntimeError(
                f"Checkpoint {identity.get('path')} has curriculum={actual}; "
                f"expected {expected_curriculum}."
            )


def load_m2u(
    path: Path,
    device: torch.device,
) -> Tuple[M2UAuxPDVarNet, Dict[str, Any]]:
    checkpoint = torch_load(path, device)
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("M2-U checkpoint must be a mapping.")
    config = model_config(checkpoint)
    model = M2UAuxPDVarNet(
        num_cascades=int(config.get("num_cascades", 12)),
        sens_chans=int(config.get("sens_chans", 8)),
        sens_pools=int(config.get("sens_pools", 4)),
        chans=int(config.get("chans", 18)),
        pools=int(config.get("pools", 4)),
        mask_center=True,
        initial_aux_alpha=float(config.get("initial_aux_alpha", 0.1)),
    ).to(device)
    state = strip_module_prefix(extract_state_dict(checkpoint))
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, checkpoint_identity(path, checkpoint)


def load_m2gd_v2(
    path: Path,
    device: torch.device,
) -> Tuple[M2GDv2AuxPDVarNet, Dict[str, Any]]:
    checkpoint = torch_load(path, device)
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("M2-GD v2 checkpoint must be a mapping.")
    config = model_config(checkpoint)
    model = M2GDv2AuxPDVarNet(
        num_cascades=int(config.get("num_cascades", 12)),
        sens_chans=int(config.get("sens_chans", 8)),
        sens_pools=int(config.get("sens_pools", 4)),
        chans=int(config.get("chans", 18)),
        pools=int(config.get("pools", 4)),
        mask_center=True,
        initial_aux_alpha=float(config.get("initial_aux_alpha", 0.1)),
        initial_gate_probability=float(config.get("initial_gate_probability", 0.99)),
    ).to(device)
    state = strip_module_prefix(extract_state_dict(checkpoint))
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, checkpoint_identity(path, checkpoint)


def prepare_batch(batch: Mapping[str, Any], device: torch.device):
    kspace = batch["pdfs_masked_kspace"].to(device, non_blocking=True)
    if not torch.is_complex(kspace):
        raise TypeError(f"Expected complex PDFS k-space, got {kspace.dtype}.")
    kspace = torch.view_as_real(kspace).float()

    mask = batch["mask"].to(device, non_blocking=True).bool()
    if mask.ndim == 2:
        mask = mask[:, None, None, :, None]
    elif mask.ndim != 5:
        raise RuntimeError(f"Unexpected mask shape: {tuple(mask.shape)}")

    pd_aux = batch["pd_aux_image"].to(device, non_blocking=True).float()
    target = batch["pdfs_target_raw"].to(device, non_blocking=True).float()
    if pd_aux.ndim == 4 and pd_aux.shape[1] == 1:
        pd_aux = pd_aux[:, 0]
    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]
    if pd_aux.ndim != 3 or target.ndim != 3:
        raise RuntimeError(
            f"Expected [B,H,W] images, got PD={tuple(pd_aux.shape)}, "
            f"target={tuple(target.shape)}."
        )
    return kspace, mask, pd_aux, target


def crop_prediction(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return center_crop(
        prediction,
        crop_h=int(target.shape[-2]),
        crop_w=int(target.shape[-1]),
    )


def central_crop_np(array: np.ndarray, margin: int = 8) -> np.ndarray:
    if min(array.shape[-2:]) <= 2 * margin:
        raise RuntimeError(
            f"Central-{margin} crop invalid for shape {array.shape[-2:]}"
        )
    return array[margin:-margin, margin:-margin]


def nmse(target: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(np.linalg.norm(target) ** 2)
    if denominator <= 0:
        return float("nan")
    return float(np.linalg.norm(target - prediction) ** 2 / denominator)


def psnr(target: np.ndarray, prediction: np.ndarray) -> float:
    mse = float(np.mean((target - prediction) ** 2))
    if mse <= 0:
        return float("inf")
    data_range = float(target.max() - target.min())
    if data_range <= 0:
        data_range = float(np.max(np.abs(target)))
    if data_range <= 0:
        return float("nan")
    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(mse))


def ssim(target: np.ndarray, prediction: np.ndarray) -> float:
    if ssim_fn is None:
        return float("nan")
    data_range = float(target.max() - target.min())
    if data_range <= 0:
        data_range = float(np.max(np.abs(target)))
    if data_range <= 0:
        return float("nan")
    return float(ssim_fn(target, prediction, data_range=data_range))


def l1(target: np.ndarray, prediction: np.ndarray) -> float:
    scale = max(float(np.max(target)), 1e-8)
    return float(np.mean(np.abs(target / scale - prediction / scale)))


def metric_row(target: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    central_target = central_crop_np(target, 8)
    central_prediction = central_crop_np(prediction, 8)
    return {
        "NMSE": nmse(target, prediction),
        "PSNR": psnr(target, prediction),
        "SSIM": ssim(target, prediction),
        "L1": l1(target, prediction),
        "NMSE_central8": nmse(central_target, central_prediction),
        "PSNR_central8": psnr(central_target, central_prediction),
        "SSIM_central8": ssim(central_target, central_prediction),
        "L1_central8": l1(central_target, central_prediction),
    }


def batch_strings(batch: Mapping[str, Any], key: str) -> List[str]:
    return [str(value) for value in batch[key]]


def batch_ints(batch: Mapping[str, Any], key: str) -> List[int]:
    value = batch[key]
    if torch.is_tensor(value):
        return [int(item) for item in value.detach().cpu().tolist()]
    return [int(item) for item in value]


def diagnostics_per_sample(aux: Mapping[str, Any]) -> List[Dict[str, float]]:
    q_hat = aux["q_hat"].detach().float().cpu()
    q = aux["q"].detach().float().cpu()
    gated = aux["gated_aux_to_target_rms"].detach().float().cpu()
    ungated = aux["ungated_aux_to_target_rms"].detach().float().cpu()
    channel = aux["channel_gate_mean"].detach().float().cpu()
    spatial = aux["spatial_gate_mean"].detach().float().cpu()
    effective = aux["effective_weight_mean"].detach().float().cpu()
    alpha = aux["alpha"].detach().float().cpu()

    if q_hat.ndim != 3 or q_hat.shape[-1] != len(SCALE_NAMES):
        raise RuntimeError(f"Unexpected q_hat shape: {tuple(q_hat.shape)}")

    records: List[Dict[str, float]] = []
    for sample in range(q_hat.shape[0]):
        record: Dict[str, float] = {
            "q_hat_mean": float(q_hat[sample].mean()),
            "q_mean": float(q[sample].mean()),
            "gated_rms_mean": float(gated[sample].mean()),
            "ungated_rms_mean": float(ungated[sample].mean()),
            "channel_gate_mean": float(channel[sample].mean()),
            "spatial_gate_mean": float(spatial[sample].mean()),
            "effective_weight_mean": float(effective[sample].mean()),
            "alpha_mean": float(alpha[sample].mean()),
        }
        for scale_index, scale_name in enumerate(SCALE_NAMES):
            suffix = scale_name.replace("/", "_")
            record[f"q_hat_{suffix}"] = float(q_hat[sample, :, scale_index].mean())
            record[f"q_{suffix}"] = float(q[sample, :, scale_index].mean())
            record[f"gated_rms_{suffix}"] = float(
                gated[sample, :, scale_index].mean()
            )
        for cascade_index in range(q_hat.shape[1]):
            suffix = f"c{cascade_index + 1:02d}"
            record[f"q_hat_{suffix}"] = float(q_hat[sample, cascade_index].mean())
            record[f"q_{suffix}"] = float(q[sample, cascade_index].mean())
            record[f"gated_rms_{suffix}"] = float(
                gated[sample, cascade_index].mean()
            )
        records.append(record)
    return records


def append_prediction_rows(
    rows: List[Dict[str, Any]],
    prediction: torch.Tensor,
    target: torch.Tensor,
    batch: Mapping[str, Any],
    model_name: str,
    condition: str,
    metadata: Mapping[str, Any],
    diagnostics: Optional[List[Dict[str, float]]] = None,
) -> None:
    prediction_np = prediction.detach().float().cpu().numpy()
    target_np = target.detach().float().cpu().numpy()
    patient_ids = batch_strings(batch, "patient_id")
    slice_indices = batch_ints(batch, "slice_idx")
    source_indices = batch_ints(batch, "sample_idx")

    for sample in range(prediction_np.shape[0]):
        sample_metadata: Dict[str, Any] = {}
        for key, value in metadata.items():
            if (
                isinstance(value, (list, tuple, np.ndarray))
                and not isinstance(value, str)
                and len(value) == prediction_np.shape[0]
            ):
                sample_metadata[key] = value[sample]
            else:
                sample_metadata[key] = value
        row: Dict[str, Any] = {
            "model": model_name,
            "condition": condition,
            "patient_id": patient_ids[sample],
            "slice_idx": slice_indices[sample],
            "source_index": source_indices[sample],
            **sample_metadata,
            **metric_row(target_np[sample], prediction_np[sample]),
        }
        if diagnostics is not None:
            row.update(diagnostics[sample])
        rows.append(row)


def alternative_batch(
    dataset: IndexedDataset,
    indices: Sequence[int],
    device: torch.device,
    expected_shape: Tuple[int, int],
) -> torch.Tensor:
    images = [load_pd_auxiliary(dataset, index, device) for index in indices]
    for index, image in zip(indices, images):
        if tuple(image.shape) != tuple(expected_shape):
            raise RuntimeError(
                f"Hard-negative PD shape mismatch at index={index}: "
                f"{tuple(image.shape)} vs {expected_shape}."
            )
    return torch.stack(images, dim=0)


def choose_audit_patients(
    dataset: IndexedDataset,
    num_patients: int,
    slices_per_patient: int,
) -> Tuple[List[str], List[int], Dict[str, Any]]:
    """Select deterministic eligible patients with broad shape coverage.

    A patient is eligible only when every source H/W shape has at least one
    different-patient candidate in the full validation set. This keeps the
    wrong-patient audit exact-shape and avoids a size-based shortcut.
    """
    patient_to_indices: Dict[str, List[int]] = defaultdict(list)
    patient_to_shapes: Dict[str, set[Tuple[int, int]]] = defaultdict(set)
    shape_to_patients: Dict[Tuple[int, int], set[str]] = defaultdict(set)
    shape_cache: Dict[str, Tuple[int, int]] = {}
    for index, record in enumerate(dataset.records):
        patient_id = str(record["patient_id"])
        patient_to_indices[patient_id].append(index)
        path = str(record["pdfs_path"])
        if path not in shape_cache:
            with h5py.File(path, "r") as hf:
                source = (
                    hf["reconstruction_rss"]
                    if "reconstruction_rss" in hf
                    else hf["kspace"]
                )
                shape_cache[path] = tuple(
                    int(value) for value in source.shape[-2:]
                )
        shape = shape_cache[path]
        patient_to_shapes[patient_id].add(shape)
        shape_to_patients[shape].add(patient_id)

    eligible = [
        patient
        for patient in sorted(patient_to_indices)
        if all(len(shape_to_patients[shape] - {patient}) >= 1
               for shape in patient_to_shapes[patient])
        and len(patient_to_indices[patient]) >= 2
    ]
    excluded = sorted(set(patient_to_indices) - set(eligible))

    chosen: List[str] = []
    covered_shapes: set[Tuple[int, int]] = set()
    remaining = list(eligible)
    while remaining and len(chosen) < num_patients:
        best = sorted(
            remaining,
            key=lambda patient: (
                -len(patient_to_shapes[patient] - covered_shapes),
                -len(patient_to_indices[patient]),
                patient,
            ),
        )[0]
        chosen.append(best)
        covered_shapes.update(patient_to_shapes[best])
        remaining.remove(best)

    if len(chosen) != num_patients:
        raise RuntimeError(
            f"Requested {num_patients} eligible audit patients but found "
            f"{len(chosen)}. Eligible={eligible}"
        )

    selected_indices: List[int] = []
    selected_by_patient: Dict[str, List[int]] = {}
    for patient in chosen:
        ordered = sorted(
            patient_to_indices[patient],
            key=lambda index: int(dataset.records[index]["slice_idx"]),
        )
        count = min(int(slices_per_patient), len(ordered))
        positions = np.linspace(0, len(ordered) - 1, num=count)
        positions = sorted({int(round(value)) for value in positions})
        # Fill rare duplicates deterministically.
        if len(positions) < count:
            for position in range(len(ordered)):
                if position not in positions:
                    positions.append(position)
                if len(positions) == count:
                    break
            positions.sort()
        chosen_indices = [ordered[position] for position in positions]
        selected_indices.extend(chosen_indices)
        selected_by_patient[patient] = [
            int(dataset.records[index]["slice_idx"])
            for index in chosen_indices
        ]

    manifest = {
        "patient_ids": chosen,
        "num_patients": len(chosen),
        "slices_per_patient_requested": int(slices_per_patient),
        "num_slices": len(selected_indices),
        "selected_slice_indices": selected_by_patient,
        "covered_shapes": [list(shape) for shape in sorted(covered_shapes)],
        "eligible_patient_count": len(eligible),
        "excluded_patients_without_exact_shape_hard_negative": excluded,
        "selection": (
            "eligible exact-shape hard-negative patients; greedy shape coverage; "
            "evenly spaced representative slices"
        ),
    }
    return chosen, selected_indices, manifest


def aggregate_patient_level(slice_df: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        column
        for column in slice_df.columns
        if column in {
            "NMSE", "PSNR", "SSIM", "L1",
            "NMSE_central8", "PSNR_central8", "SSIM_central8", "L1_central8",
            "q_hat_mean", "q_mean", "gated_rms_mean", "ungated_rms_mean",
            "channel_gate_mean", "spatial_gate_mean", "effective_weight_mean",
            "alpha_mean",
        }
        or column.startswith((
            "q_hat_H_", "q_H_", "gated_rms_H_",
            "q_hat_c", "q_c", "gated_rms_c",
        ))
    ]
    grouping = ["model", "condition", "patient_id"]
    result = slice_df.groupby(grouping, as_index=False)[numeric_columns].mean()
    counts = (
        slice_df.groupby(grouping, as_index=False)
        .size()
        .rename(columns={"size": "num_slices"})
    )
    return result.merge(counts, on=grouping, how="left")


def aggregate_summary(patient_df: pd.DataFrame) -> pd.DataFrame:
    numeric = patient_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric = [column for column in numeric if column != "num_slices"]
    rows: List[Dict[str, Any]] = []
    for (model_name, condition), group in patient_df.groupby(["model", "condition"]):
        row: Dict[str, Any] = {
            "model": model_name,
            "condition": condition,
            "num_patients": int(group["patient_id"].nunique()),
            "num_slices": int(group["num_slices"].sum()),
        }
        for column in numeric:
            values = group[column].dropna().to_numpy(dtype=float)
            row[f"{column}_mean"] = float(np.mean(values)) if len(values) else float("nan")
            row[f"{column}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def paired_model_deltas(patient_df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "NMSE", "PSNR", "SSIM", "L1",
        "NMSE_central8", "PSNR_central8", "SSIM_central8", "L1_central8",
    ]
    v2 = patient_df[patient_df["model"] == "M2GDv2"][
        ["condition", "patient_id", *metric_columns]
    ].copy()
    m2u = patient_df[patient_df["model"] == "M2U"][
        ["condition", "patient_id", *metric_columns]
    ].copy()
    merged = v2.merge(
        m2u,
        on=["condition", "patient_id"],
        suffixes=("_m2gd_v2", "_m2u"),
        validate="one_to_one",
    )
    for metric in metric_columns:
        if metric.startswith(("NMSE", "L1")):
            merged[f"{metric}_improvement"] = (
                merged[f"{metric}_m2u"] - merged[f"{metric}_m2gd_v2"]
            )
        else:
            merged[f"{metric}_improvement"] = (
                merged[f"{metric}_m2gd_v2"] - merged[f"{metric}_m2u"]
            )
    return merged


def paired_delta_summary(delta_df: pd.DataFrame) -> pd.DataFrame:
    improvement_columns = [
        column for column in delta_df.columns if column.endswith("_improvement")
    ]
    rows: List[Dict[str, Any]] = []
    for condition, group in delta_df.groupby("condition"):
        row: Dict[str, Any] = {
            "condition": condition,
            "num_patients": int(group["patient_id"].nunique()),
        }
        for column in improvement_columns:
            values = group[column].to_numpy(dtype=float)
            row[f"{column}_mean"] = float(np.mean(values))
            row[f"{column}_median"] = float(np.median(values))
            row[f"{column}_pct_m2gd_v2_worse"] = float(np.mean(values < 0) * 100.0)
        rows.append(row)
    return pd.DataFrame(rows)


def direction_averaged_patient(patient_df: pd.DataFrame) -> pd.DataFrame:
    shift = patient_df[
        patient_df["condition"].str.startswith("shift8_", na=False)
    ].copy()
    if shift.empty:
        return pd.DataFrame()
    parsed = shift["condition"].str.extract(
        r"shift8_(zero|reflect|replicate)_(.+)"
    )
    shift["padding_mode"] = parsed[0]
    shift["direction"] = parsed[1]
    direction_class = {
        name: family for name, _, _, family in DIRECTIONS
    }
    shift["direction_class"] = shift["direction"].map(direction_class)
    numeric = shift.select_dtypes(include=[np.number]).columns.tolist()
    numeric = [column for column in numeric if column != "num_slices"]
    grouping = ["model", "patient_id", "padding_mode", "direction_class"]
    grouped = shift.groupby(grouping, as_index=False)[numeric].mean()
    slice_counts = (
        shift.groupby(grouping, as_index=False)["num_slices"]
        .max()
    )
    grouped = grouped.merge(slice_counts, on=grouping, how="left")
    grouped["condition"] = (
        "shift8_" + grouped["padding_mode"] + "_" + grouped["direction_class"] + "_mean"
    )
    return grouped


def model_summary_value(
    summary: pd.DataFrame,
    model_name: str,
    condition: str,
    column: str,
) -> float:
    row = summary[
        (summary["model"] == model_name) & (summary["condition"] == condition)
    ]
    if len(row) != 1:
        raise RuntimeError(
            f"Missing unique {model_name} summary for condition={condition}."
        )
    return float(row.iloc[0][column])


def direction_model_mean(
    direction_patient: pd.DataFrame,
    model_name: str,
    padding_mode: str,
    column: str,
) -> float:
    rows = direction_patient[
        (direction_patient["model"] == model_name)
        & (direction_patient["padding_mode"] == padding_mode)
    ]
    if rows.empty:
        raise RuntimeError(
            f"Missing direction-averaged {model_name}/{padding_mode} results."
        )
    return float(rows[column].mean())


def build_decision_summary(
    summary: pd.DataFrame,
    direction_patient: pd.DataFrame,
    robustness_tolerance_relative: float,
) -> Dict[str, Any]:
    q_clean = model_summary_value(summary, "M2GDv2", "correct", "q_hat_mean_mean")
    q_border = model_summary_value(
        summary, "M2GDv2", "border_only8_zero", "q_hat_mean_mean"
    )
    q_wrong_slice = model_summary_value(
        summary, "M2GDv2", "same_patient_wrong_slice", "q_hat_mean_mean"
    )
    q_wrong = model_summary_value(
        summary, "M2GDv2", "wrong_patient_matched_level", "q_hat_mean_mean"
    )
    q_missing = model_summary_value(summary, "M2GDv2", "missing", "q_hat_mean_mean")
    missing_rms = model_summary_value(
        summary, "M2GDv2", "missing", "gated_rms_mean_mean"
    )

    shift_padding_q: Dict[str, float] = {}
    for padding in PADDING_MODES:
        rows = direction_patient[
            (direction_patient["model"] == "M2GDv2")
            & (direction_patient["condition"].str.startswith(
                f"shift8_{padding}_", na=False
            ))
        ]
        if rows.empty:
            raise RuntimeError(f"No direction-averaged shift result for {padding}.")
        shift_padding_q[padding] = float(rows["q_hat_mean"].mean())

    q_shift_mean = float(np.mean(list(shift_padding_q.values())))
    padding_range = float(
        max(shift_padding_q.values()) - min(shift_padding_q.values())
    )

    m2u_l1 = model_summary_value(summary, "M2U", "correct", "L1_mean")
    v2_l1 = model_summary_value(summary, "M2GDv2", "correct", "L1_mean")
    clean_ratio = v2_l1 / max(m2u_l1, 1e-12)

    robustness_conditions = (
        "missing",
        "border_only8_zero",
        "same_patient_wrong_slice",
        "wrong_patient_matched_level",
    )
    reconstruction_ratios: Dict[str, float] = {}
    for condition in robustness_conditions:
        m2u_value = model_summary_value(summary, "M2U", condition, "L1_mean")
        v2_value = model_summary_value(summary, "M2GDv2", condition, "L1_mean")
        reconstruction_ratios[condition] = v2_value / max(m2u_value, 1e-12)
    for padding in PADDING_MODES:
        m2u_value = direction_model_mean(
            direction_patient, "M2U", padding, "L1"
        )
        v2_value = direction_model_mean(
            direction_patient, "M2GDv2", padding, "L1"
        )
        reconstruction_ratios[f"shift8_{padding}"] = (
            v2_value / max(m2u_value, 1e-12)
        )

    max_allowed_ratio = 1.0 + float(robustness_tolerance_relative)

    criteria = {
        "q_correct_minus_shift8_at_least_0p10": q_clean - q_shift_mean >= 0.10,
        "q_correct_minus_wrong_at_least_0p20": q_clean - q_wrong >= 0.20,
        "q_correct_minus_wrong_slice_at_least_0p10": (
            q_clean - q_wrong_slice >= 0.10
        ),
        "q_border_minus_shift8_at_least_0p10": q_border - q_shift_mean >= 0.10,
        "shift8_padding_q_range_at_most_0p05": padding_range <= 0.05,
        "missing_gated_rms_exact_zero": abs(missing_rms) <= 1e-10,
        "missing_q_hat_at_most_0p10": q_missing <= 0.10,
        "clean_l1_within_2_percent_of_m2u": clean_ratio <= 1.02,
    }
    for condition, ratio in reconstruction_ratios.items():
        criteria[f"{condition}_l1_not_worse_than_m2u"] = (
            ratio <= max_allowed_ratio
        )
    return {
        "q_correct": q_clean,
        "q_border_only8_zero": q_border,
        "q_same_patient_wrong_slice": q_wrong_slice,
        "q_wrong_patient": q_wrong,
        "q_missing_hat": q_missing,
        "q_shift8_padding_means": shift_padding_q,
        "q_shift8_all_padding_mean": q_shift_mean,
        "q_correct_minus_shift8": q_clean - q_shift_mean,
        "q_correct_minus_wrong_patient": q_clean - q_wrong,
        "q_border_minus_shift8": q_border - q_shift_mean,
        "shift8_padding_q_range": padding_range,
        "missing_gated_rms": missing_rms,
        "m2u_clean_patient_l1": m2u_l1,
        "m2gd_v2_clean_patient_l1": v2_l1,
        "clean_l1_ratio_v2_over_m2u": clean_ratio,
        "robustness_tolerance_relative": float(robustness_tolerance_relative),
        "m2gd_v2_over_m2u_l1_ratios": reconstruction_ratios,
        "criteria": criteria,
        "go_to_15_epoch_pilot": bool(all(criteria.values())),
        "interpretation": (
            "This automated flag is a pre-registered engineering screen. "
            "Inspect patient-level and per-scale results before proceeding."
        ),
    }


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reduced M2-GD v2 mechanism audit after the 5-epoch smoke run."
    )
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--m2u_checkpoint", required=True)
    parser.add_argument("--m2gd_v2_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_audit_patients", type=int, default=6)
    parser.add_argument("--slices_per_patient", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected_m2u_epoch", type=int, default=50)
    parser.add_argument("--expected_m2gd_v2_epoch", type=int, default=5)
    parser.add_argument("--robustness_tolerance_relative", type=float, default=0.01)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("The reduced audit must run on a CUDA GPU.")

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
        args.num_audit_patients,
        args.slices_per_patient,
    )
    selection_manifest["full_validation_patients"] = len(
        {str(record["patient_id"]) for record in full_dataset.records}
    )
    selection_manifest["full_validation_slices"] = len(full_dataset)
    (output_dir / "audit_patient_selection.json").write_text(
        json.dumps(selection_manifest, indent=2), encoding="utf-8"
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
    m2gd, m2gd_identity = load_m2gd_v2(
        Path(args.m2gd_v2_checkpoint), device
    )
    assert_checkpoint_identity(
        m2u_identity,
        expected_epoch=args.expected_m2u_epoch,
        expected_acceleration=8,
        expected_pd_acceleration=2,
    )
    assert_checkpoint_identity(
        m2gd_identity,
        expected_epoch=args.expected_m2gd_v2_epoch,
        expected_acceleration=8,
        expected_pd_acceleration=2,
        expected_curriculum="smoke5",
    )

    print("=" * 96)
    print("M2-GD v2 reduced smoke audit")
    print("GPU:", torch.cuda.get_device_name(0))
    print("Audit patients:", patient_ids)
    print("Audit slices:", len(selected))
    print("Batches:", len(loader))
    print("=" * 96, flush=True)

    rows: List[Dict[str, Any]] = []
    fallback_counts = defaultdict(int)

    for batch_index, batch in enumerate(loader, start=1):
        kspace, mask, pd_aux, target = prepare_batch(batch, device)
        source_indices = batch_ints(batch, "sample_idx")
        availability_one = torch.ones(
            pd_aux.shape[0], device=device, dtype=pd_aux.dtype
        )
        availability_zero = torch.zeros_like(availability_one)

        def evaluate_m2u(
            condition: str,
            pd_input: torch.Tensor,
            metadata: Mapping[str, Any],
        ) -> None:
            prediction = crop_prediction(
                m2u(kspace, mask, pd_input),
                target,
            )
            append_prediction_rows(
                rows,
                prediction,
                target,
                batch,
                "M2U",
                condition,
                metadata,
                None,
            )

        def evaluate_v2(
            condition: str,
            pd_input: torch.Tensor,
            availability: torch.Tensor,
            metadata: Mapping[str, Any],
        ) -> None:
            prediction, aux = m2gd(
                pdfs_masked_kspace=kspace,
                mask=mask,
                pd_aux_image=pd_input,
                pd_available=availability,
                return_aux=True,
            )
            prediction = crop_prediction(prediction, target)
            append_prediction_rows(
                rows,
                prediction,
                target,
                batch,
                "M2GDv2",
                condition,
                metadata,
                diagnostics_per_sample(aux),
            )

        def evaluate_pair(
            condition: str,
            pd_input: torch.Tensor,
            availability: torch.Tensor,
            metadata: Mapping[str, Any],
        ) -> None:
            evaluate_m2u(condition, pd_input, metadata)
            evaluate_v2(condition, pd_input, availability, metadata)

        evaluate_pair("correct", pd_aux, availability_one, {})
        evaluate_pair(
            "missing", torch.zeros_like(pd_aux), availability_zero, {}
        )

        border = torch.stack(
            [border_only(image, 8, "zero") for image in pd_aux], dim=0
        )
        evaluate_pair(
            "border_only8_zero",
            border,
            availability_one,
            {
                "padding_mode": "zero",
                "magnitude_linf": 8,
                "reliability_target": 0.85,
            },
        )

        wrong_slice_indices: List[int] = []
        wrong_slice_delta: List[float] = []
        rng = random.Random(args.seed + batch_index * 1009)
        for source_index in source_indices:
            candidate = negative_sampler.same_patient_wrong_slice(source_index, rng)
            if candidate is None:
                fallback_counts["wrong_slice_unavailable"] += 1
                raise RuntimeError(
                    f"No same-patient wrong-slice candidate for source={source_index}."
                )
            replacement_index, delta_z = candidate
            wrong_slice_indices.append(replacement_index)
            wrong_slice_delta.append(delta_z)
        wrong_slice_pd = alternative_batch(
            full_dataset,
            wrong_slice_indices,
            device,
            tuple(int(value) for value in pd_aux.shape[-2:]),
        )
        evaluate_pair(
            "same_patient_wrong_slice",
            wrong_slice_pd,
            availability_one,
            {
                "replacement_policy": "same_patient_wrong_slice",
                "replacement_index": wrong_slice_indices,
                "delta_z_norm": wrong_slice_delta,
                "delta_z_norm_batch_mean": float(np.mean(wrong_slice_delta)),
            },
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
            wrong_patient_delta.append(delta_z)
        wrong_patient_pd = alternative_batch(
            full_dataset,
            wrong_patient_indices,
            device,
            source_shape,
        )
        evaluate_pair(
            "wrong_patient_matched_level",
            wrong_patient_pd,
            availability_one,
            {
                "replacement_policy": "wrong_patient_exact_shape_nearest_z_norm",
                "replacement_index": wrong_patient_indices,
                "delta_z_norm": wrong_patient_delta,
                "delta_z_norm_batch_mean": float(np.mean(wrong_patient_delta)),
            },
        )

        for padding_mode in PADDING_MODES:
            for direction_name, unit_dy, unit_dx, direction_class in DIRECTIONS:
                dy = 8 * unit_dy
                dx = 8 * unit_dx
                shifted = torch.stack(
                    [
                        translate_nonwrapping(
                            image, dy, dx, padding_mode
                        )
                        for image in pd_aux
                    ],
                    dim=0,
                )
                condition = f"shift8_{padding_mode}_{direction_name}"
                evaluate_pair(
                    condition,
                    shifted,
                    availability_one,
                    {
                        "padding_mode": padding_mode,
                        "direction": direction_name,
                        "direction_class": direction_class,
                        "dx": dx,
                        "dy": dy,
                        "magnitude_linf": 8,
                        "magnitude_l2": float(math.hypot(dx, dy)),
                    },
                )

        if batch_index == 1 or batch_index % 10 == 0:
            print(
                f"Batch {batch_index:04d}/{len(loader)} completed | "
                f"rows={len(rows)}",
                flush=True,
            )

    slice_df = pd.DataFrame(rows)
    patient_df = aggregate_patient_level(slice_df)
    summary_df = aggregate_summary(patient_df)
    delta_df = paired_model_deltas(patient_df)
    delta_summary_df = paired_delta_summary(delta_df)

    direction_df = direction_averaged_patient(patient_df)
    direction_summary = (
        aggregate_summary(direction_df)
        if not direction_df.empty
        else pd.DataFrame()
    )

    decision = build_decision_summary(
        summary_df,
        direction_df,
        robustness_tolerance_relative=args.robustness_tolerance_relative,
    )
    decision["audit_patient_ids"] = patient_ids
    decision["fallback_counts"] = dict(fallback_counts)
    decision["checkpoint"] = str(args.m2gd_v2_checkpoint)
    decision["reference_checkpoint"] = str(args.m2u_checkpoint)
    decision["checkpoint_identity"] = {
        "m2u": m2u_identity,
        "m2gd_v2": m2gd_identity,
    }

    slice_df.to_csv(output_dir / "m2gd_v2_smoke_audit_per_slice.csv", index=False)
    patient_df.to_csv(output_dir / "m2gd_v2_smoke_audit_patient_level.csv", index=False)
    summary_df.to_csv(output_dir / "m2gd_v2_smoke_audit_summary.csv", index=False)
    delta_df.to_csv(
        output_dir / "m2gd_v2_vs_m2u_patient_delta.csv",
        index=False,
    )
    delta_summary_df.to_csv(
        output_dir / "m2gd_v2_vs_m2u_delta_summary.csv",
        index=False,
    )
    direction_df.to_csv(
        output_dir / "m2gd_v2_smoke_audit_direction_averaged_patient.csv",
        index=False,
    )
    direction_summary.to_csv(
        output_dir / "m2gd_v2_smoke_audit_direction_summary.csv",
        index=False,
    )
    (output_dir / "m2gd_v2_smoke_audit_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    manifest = {
        "metadata_csv": str(args.metadata_csv),
        "m2u_checkpoint": str(args.m2u_checkpoint),
        "m2gd_v2_checkpoint": str(args.m2gd_v2_checkpoint),
        "checkpoint_identity": {
            "m2u": m2u_identity,
            "m2gd_v2": m2gd_identity,
        },
        "num_audit_patients": args.num_audit_patients,
        "slices_per_patient_requested": args.slices_per_patient,
        "num_audit_slices": len(selected),
        "shift_definition": "L_inf=8 pixels; cardinal and diagonal reported separately",
        "padding_modes": list(PADDING_MODES),
        "directions": [name for name, _, _, _ in DIRECTIONS],
        "central_roi": "remove 8 pixels from each image edge",
        "patient_level_aggregation": "mean over slices within each patient, then equal-weight patient summary",
        "go_no_go_guardrails": decision["criteria"],
    }
    (output_dir / "m2gd_v2_smoke_audit_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("=" * 96)
    print(json.dumps(decision, indent=2))
    print("Saved audit outputs to:", output_dir)
    print("=" * 96)


if __name__ == "__main__":
    main()
