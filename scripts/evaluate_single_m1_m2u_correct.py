#!/usr/bin/env python3
"""
Unified correct-PD validation evaluation:
  - Single PD-FS VarNet
  - M1 direct auxiliary PD R=2 -> PD-FS VarNet
  - M2-U multi-scale auxiliary PD R=2 -> PD-FS VarNet

Evaluates R = 4, 6, 8 on the same validation dataset, masks, targets,
metrics, and patient/slice pairing.

Outputs:
  per_slice_metrics.csv
  patient_level_metrics.csv
  slice_level_summary.csv
  patient_level_summary.csv
  paired_delta_vs_single_per_slice.csv
  paired_delta_vs_single_patient_level.csv
  paired_delta_summary_slice_level.csv
  paired_delta_summary_patient_level.csv
  evaluation_config.json
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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
from src.m2u_auxiliary_varnet_optimized import M2UAuxPDVarNet
from src.dataset_paired_multicoil_aux_pd_r2 import (
    PairedMulticoilAuxPDToPDFSDataset,
)

METRICS = ["NMSE", "PSNR", "SSIM", "L1"]
HIGHER_IS_BETTER = {
    "NMSE": False,
    "PSNR": True,
    "SSIM": True,
    "L1": False,
}


class ShapeBucketBatchSampler(Sampler[List[int]]):
    """Avoid collating slices with different k-space spatial shapes."""

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

        buckets: Dict[Tuple[int, int], List[int]] = {}
        for idx, record in enumerate(dataset.records):
            pdfs_path = record["pdfs_path"]
            with h5py.File(pdfs_path, "r") as hf:
                shape = hf["kspace"].shape
                key = (int(shape[-2]), int(shape[-1]))
            buckets.setdefault(key, []).append(idx)

        self.buckets = buckets
        self.num_batches = sum(
            math.ceil(len(indices) / self.batch_size)
            for indices in buckets.values()
        )

        print(
            f"ShapeBucketBatchSampler: {len(self.buckets)} buckets, "
            f"{self.num_batches} batches, batch_size={self.batch_size}"
        )

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        all_batches = []

        for indices in self.buckets.values():
            indices = list(indices)
            if self.shuffle:
                rng.shuffle(indices)

            for start in range(0, len(indices), self.batch_size):
                all_batches.append(indices[start:start + self.batch_size])

        if self.shuffle:
            rng.shuffle(all_batches)

        yield from all_batches

    def __len__(self):
        return self.num_batches


def center_crop_tensor(
    x: torch.Tensor,
    crop_h: int,
    crop_w: int,
) -> torch.Tensor:
    h, w = x.shape[-2:]
    if (h, w) == (crop_h, crop_w):
        return x
    if h < crop_h or w < crop_w:
        raise RuntimeError(
            f"Cannot crop tensor from {(h, w)} to {(crop_h, crop_w)}"
        )
    top = (h - crop_h) // 2
    left = (w - crop_w) // 2
    return x[..., top:top + crop_h, left:left + crop_w]


def center_crop_np(
    x: np.ndarray,
    shape: Tuple[int, int],
) -> np.ndarray:
    h, w = x.shape[-2:]
    th, tw = shape
    if (h, w) == (th, tw):
        return x
    if h < th or w < tw:
        raise RuntimeError(
            f"Cannot crop array from {(h, w)} to {(th, tw)}"
        )
    top = (h - th) // 2
    left = (w - tw) // 2
    return x[..., top:top + th, left:left + tw]


def nmse(target: np.ndarray, pred: np.ndarray) -> float:
    denom = np.linalg.norm(target) ** 2
    if denom <= 0:
        return float("nan")
    return float(np.linalg.norm(target - pred) ** 2 / denom)


def psnr(target: np.ndarray, pred: np.ndarray) -> float:
    mse = np.mean((target - pred) ** 2)
    if mse <= 0:
        return float("inf")
    data_range = float(target.max() - target.min())
    if data_range <= 0:
        data_range = float(np.max(np.abs(target)))
    if data_range <= 0:
        return float("nan")
    return float(20 * np.log10(data_range) - 10 * np.log10(mse))


def ssim_metric(target: np.ndarray, pred: np.ndarray) -> float:
    if ssim_fn is None:
        return float("nan")
    data_range = float(target.max() - target.min())
    if data_range <= 0:
        data_range = float(np.max(np.abs(target)))
    if data_range <= 0:
        return float("nan")
    return float(ssim_fn(target, pred, data_range=data_range))


def l1_metric(target: np.ndarray, pred: np.ndarray) -> float:
    scale = float(np.max(target))
    if scale < 1e-12:
        scale = 1.0
    return float(np.mean(np.abs(pred / scale - target / scale)))


def torch_load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
):
    try:
        return torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def extract_state_dict(ckpt, checkpoint_path: Path):
    if isinstance(ckpt, dict):
        for key in [
            "model_state_dict",
            "model",
            "state_dict",
            "net",
            "network",
        ]:
            state = ckpt.get(key)
            if isinstance(state, dict):
                return state

    if (
        isinstance(ckpt, dict)
        and len(ckpt) > 0
        and all(torch.is_tensor(v) for v in ckpt.values())
    ):
        return ckpt

    raise RuntimeError(
        f"Cannot find state dict in {checkpoint_path}; "
        f"keys={list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}"
    )


def clean_state_dict(state):
    cleaned = {}
    for key, value in state.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module."):]
        if new_key.startswith("model."):
            new_key = new_key[len("model."):]
        cleaned[new_key] = value
    return cleaned


def load_single_model(
    checkpoint_path: Path,
    device: torch.device,
) -> VarNet:
    ckpt = torch_load_checkpoint(checkpoint_path, device)
    config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    model = VarNet(
        num_cascades=int(config.get("num_cascades", 12)),
        sens_chans=int(config.get("sens_chans", 8)),
        sens_pools=int(config.get("sens_pools", 4)),
        chans=int(config.get("chans", 18)),
        pools=int(config.get("pools", 4)),
        mask_center=True,
    ).to(device)

    state = clean_state_dict(extract_state_dict(ckpt, checkpoint_path))
    model.load_state_dict(state, strict=True)
    model.eval()

    print(
        f"Loaded single: {checkpoint_path} | "
        f"epoch={ckpt.get('epoch') if isinstance(ckpt, dict) else None} | "
        f"best_epoch={ckpt.get('best_epoch') if isinstance(ckpt, dict) else None}"
    )
    return model


def load_m1_model(
    checkpoint_path: Path,
    device: torch.device,
) -> AuxPDVarNet:
    ckpt = torch_load_checkpoint(checkpoint_path, device)
    config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    model = AuxPDVarNet(
        num_cascades=int(config.get("num_cascades", 12)),
        sens_chans=int(config.get("sens_chans", 8)),
        sens_pools=int(config.get("sens_pools", 4)),
        chans=int(config.get("chans", 18)),
        pools=int(config.get("pools", 4)),
        mask_center=True,
    ).to(device)

    state = clean_state_dict(extract_state_dict(ckpt, checkpoint_path))
    model.load_state_dict(state, strict=True)
    model.eval()

    print(
        f"Loaded M1: {checkpoint_path} | "
        f"epoch={ckpt.get('epoch') if isinstance(ckpt, dict) else None} | "
        f"best_epoch={ckpt.get('best_epoch') if isinstance(ckpt, dict) else None}"
    )
    return model


def load_m2u_model(
    checkpoint_path: Path,
    device: torch.device,
) -> M2UAuxPDVarNet:
    ckpt = torch_load_checkpoint(checkpoint_path, device)
    config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    model = M2UAuxPDVarNet(
        num_cascades=int(config.get("num_cascades", 12)),
        sens_chans=int(config.get("sens_chans", 8)),
        sens_pools=int(config.get("sens_pools", 4)),
        chans=int(config.get("chans", 18)),
        pools=int(config.get("pools", 4)),
        mask_center=True,
        initial_aux_alpha=float(config.get("initial_aux_alpha", 0.1)),
    ).to(device)

    state = clean_state_dict(extract_state_dict(ckpt, checkpoint_path))
    model.load_state_dict(state, strict=True)
    model.eval()

    print(
        f"Loaded M2-U: {checkpoint_path} | "
        f"epoch={ckpt.get('epoch') if isinstance(ckpt, dict) else None} | "
        f"best_epoch={ckpt.get('best_epoch') if isinstance(ckpt, dict) else None}"
    )
    return model


def prepare_batch(batch: Dict, device: torch.device):
    pdfs_kspace = batch["pdfs_masked_kspace"].to(
        device,
        non_blocking=True,
    )

    if torch.is_complex(pdfs_kspace):
        pdfs_kspace = torch.view_as_real(pdfs_kspace).float()
    else:
        pdfs_kspace = pdfs_kspace.float()

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

    pd_aux = batch["pd_aux_image"].to(
        device,
        non_blocking=True,
    ).float()

    target = batch["pdfs_target_raw"].to(
        device,
        non_blocking=True,
    ).float()

    if pd_aux.ndim == 4 and pd_aux.shape[1] == 1:
        pd_aux = pd_aux[:, 0]

    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]

    if pd_aux.ndim != 3:
        raise RuntimeError(
            f"Expected PD auxiliary [B,H,W], got {tuple(pd_aux.shape)}"
        )

    if target.ndim != 3:
        raise RuntimeError(
            f"Expected target [B,H,W], got {tuple(target.shape)}"
        )

    return pdfs_kspace, mask, pd_aux, target


def get_batch_value(batch: Dict, key: str, index: int):
    value = batch[key]
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    return value[index]


def rows_for_prediction(
    pred: torch.Tensor,
    target: torch.Tensor,
    batch: Dict,
    model_name: str,
    acceleration: int,
    checkpoint_path: Path,
) -> List[Dict]:
    rows = []
    pred_np = pred.detach().cpu().float().numpy()
    target_np = target.detach().cpu().float().numpy()

    for i in range(pred_np.shape[0]):
        p = pred_np[i]
        t = target_np[i]

        if p.shape != t.shape:
            h = min(p.shape[-2], t.shape[-2])
            w = min(p.shape[-1], t.shape[-1])
            p = center_crop_np(p, (h, w))
            t = center_crop_np(t, (h, w))

        rows.append(
            {
                "model": model_name,
                "condition": (
                    "no_auxiliary"
                    if model_name == "single"
                    else "correct_pdR2_aux"
                ),
                "checkpoint": str(checkpoint_path),
                "patient_id": str(
                    get_batch_value(batch, "patient_id", i)
                ),
                "slice_idx": int(
                    get_batch_value(batch, "slice_idx", i)
                ),
                "num_slices": int(
                    get_batch_value(batch, "num_slices", i)
                ),
                "contrast": "PD-FS",
                "R": int(acceleration),
                "pd_aux_R": 2,
                "pd_flip_lr": bool(
                    get_batch_value(batch, "pd_flip_lr", i)
                ),
                "is_edge": bool(
                    get_batch_value(batch, "is_edge", i)
                ),
                "NMSE": nmse(t, p),
                "PSNR": psnr(t, p),
                "SSIM": ssim_metric(t, p),
                "L1": l1_metric(t, p),
            }
        )

    return rows


def summarise_grouped(
    df: pd.DataFrame,
    group_cols: List[str],
    n_unit_col: str,
) -> pd.DataFrame:
    rows = []

    for keys, group in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = {
            col: value
            for col, value in zip(group_cols, keys)
        }
        row["n"] = int(group[n_unit_col].nunique())

        for metric in METRICS:
            vals = group[metric].dropna().to_numpy()
            row[f"{metric}_mean"] = (
                float(np.mean(vals)) if len(vals) else float("nan")
            )
            row[f"{metric}_std"] = (
                float(np.std(vals, ddof=1))
                if len(vals) > 1
                else float("nan")
            )
            row[f"{metric}_median"] = (
                float(np.median(vals)) if len(vals) else float("nan")
            )
            row[f"{metric}_iqr_low"] = (
                float(np.percentile(vals, 25))
                if len(vals)
                else float("nan")
            )
            row[f"{metric}_iqr_high"] = (
                float(np.percentile(vals, 75))
                if len(vals)
                else float("nan")
            )

        rows.append(row)

    return pd.DataFrame(rows).sort_values(group_cols)


def compute_paired_delta(
    source_df: pd.DataFrame,
    key_cols: List[str],
) -> pd.DataFrame:
    base = source_df[source_df["model"] == "single"].copy()
    base = base[key_cols + METRICS].rename(
        columns={metric: f"{metric}_single" for metric in METRICS}
    )

    result = []

    for model_name in ["m1", "m2u"]:
        current = source_df[source_df["model"] == model_name].copy()
        merged = current.merge(
            base,
            on=key_cols,
            how="inner",
            validate="one_to_one",
        )

        for metric in METRICS:
            if HIGHER_IS_BETTER[metric]:
                delta = (
                    merged[metric]
                    - merged[f"{metric}_single"]
                )
            else:
                delta = (
                    merged[f"{metric}_single"]
                    - merged[metric]
                )

            merged[f"{metric}_delta_vs_single"] = delta
            merged[f"{metric}_worse_than_single"] = delta < 0

        result.append(merged)

    return pd.concat(result, ignore_index=True)


def summarise_delta(
    delta_df: pd.DataFrame,
    unit_cols: List[str],
) -> pd.DataFrame:
    rows = []

    for (model, R), group in delta_df.groupby(["model", "R"]):
        row = {
            "model": model,
            "R": int(R),
            "n": int(group[unit_cols].drop_duplicates().shape[0]),
        }

        for metric in METRICS:
            vals = group[
                f"{metric}_delta_vs_single"
            ].dropna().to_numpy()

            worse = group[
                f"{metric}_worse_than_single"
            ].dropna().to_numpy()

            row[f"{metric}_delta_mean"] = (
                float(np.mean(vals)) if len(vals) else float("nan")
            )
            row[f"{metric}_delta_std"] = (
                float(np.std(vals, ddof=1))
                if len(vals) > 1
                else float("nan")
            )
            row[f"{metric}_delta_median"] = (
                float(np.median(vals)) if len(vals) else float("nan")
            )
            row[f"{metric}_delta_iqr_low"] = (
                float(np.percentile(vals, 25))
                if len(vals)
                else float("nan")
            )
            row[f"{metric}_delta_iqr_high"] = (
                float(np.percentile(vals, 75))
                if len(vals)
                else float("nan")
            )
            row[f"{metric}_pct_worse_than_single"] = (
                float(np.mean(worse) * 100.0)
                if len(worse)
                else float("nan")
            )

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["R", "model"])


@torch.no_grad()
def evaluate_one_R(
    acceleration: int,
    metadata_csv: Path,
    single_checkpoint: Path,
    m1_checkpoint: Path,
    m2u_checkpoint: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> pd.DataFrame:
    print("=" * 88)
    print(f"Evaluating R={acceleration}")
    print("=" * 88)

    dataset = PairedMulticoilAuxPDToPDFSDataset(
        metadata_csv=str(metadata_csv),
        split="val",
        pdfs_acceleration=int(acceleration),
        pd_aux_acceleration=2,
    )

    sampler = ShapeBucketBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=42,
    )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    single_model = load_single_model(single_checkpoint, device)
    m1_model = load_m1_model(m1_checkpoint, device)
    m2u_model = load_m2u_model(m2u_checkpoint, device)

    rows = []

    for batch_idx, batch in enumerate(loader, start=1):
        pdfs_kspace, mask, pd_aux, target = prepare_batch(
            batch,
            device,
        )

        single_pred = single_model(pdfs_kspace, mask)
        m1_pred = m1_model(
            pdfs_masked_kspace=pdfs_kspace,
            mask=mask,
            pd_aux_image=pd_aux,
        )
        m2u_pred = m2u_model(
            pdfs_masked_kspace=pdfs_kspace,
            mask=mask,
            pd_aux_image=pd_aux,
        )

        crop_h, crop_w = target.shape[-2:]
        single_pred = center_crop_tensor(single_pred, crop_h, crop_w)
        m1_pred = center_crop_tensor(m1_pred, crop_h, crop_w)
        m2u_pred = center_crop_tensor(m2u_pred, crop_h, crop_w)

        rows.extend(
            rows_for_prediction(
                single_pred,
                target,
                batch,
                "single",
                acceleration,
                single_checkpoint,
            )
        )
        rows.extend(
            rows_for_prediction(
                m1_pred,
                target,
                batch,
                "m1",
                acceleration,
                m1_checkpoint,
            )
        )
        rows.extend(
            rows_for_prediction(
                m2u_pred,
                target,
                batch,
                "m2u",
                acceleration,
                m2u_checkpoint,
            )
        )

        if (
            batch_idx == 1
            or batch_idx % 25 == 0
            or batch_idx == len(loader)
        ):
            print(
                f"R={acceleration} | "
                f"batch {batch_idx}/{len(loader)}"
            )

    del single_model, m1_model, m2u_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Unified Single vs M1 vs M2-U correct-PD evaluation."
    )

    parser.add_argument(
        "--metadata_csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )

    for R in [4, 6, 8]:
        parser.add_argument(
            f"--single_R{R}",
            type=Path,
            required=True,
        )
        parser.add_argument(
            f"--m1_R{R}",
            type=Path,
            required=True,
        )
        parser.add_argument(
            f"--m2u_R{R}",
            type=Path,
            required=True,
        )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 88)
    print("Unified correct-PD evaluation")
    print("=" * 88)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Metadata: {args.metadata_csv}")
    print(f"Output: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 88)

    checkpoint_plan = {
        4: {
            "single": args.single_R4,
            "m1": args.m1_R4,
            "m2u": args.m2u_R4,
        },
        6: {
            "single": args.single_R6,
            "m1": args.m1_R6,
            "m2u": args.m2u_R6,
        },
        8: {
            "single": args.single_R8,
            "m1": args.m1_R8,
            "m2u": args.m2u_R8,
        },
    }

    for R, paths in checkpoint_plan.items():
        for name, path in paths.items():
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing {name} checkpoint for R={R}: {path}"
                )

    all_slice_frames = []

    for R in [4, 6, 8]:
        frame = evaluate_one_R(
            acceleration=R,
            metadata_csv=args.metadata_csv,
            single_checkpoint=checkpoint_plan[R]["single"],
            m1_checkpoint=checkpoint_plan[R]["m1"],
            m2u_checkpoint=checkpoint_plan[R]["m2u"],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        all_slice_frames.append(frame)

    per_slice = (
        pd.concat(all_slice_frames, ignore_index=True)
        .sort_values(
            ["R", "patient_id", "slice_idx", "model"]
        )
        .reset_index(drop=True)
    )

    # Equal-weight patient/volume aggregation:
    # each patient's metric is the mean across its slices.
    patient_level = (
        per_slice
        .groupby(
            [
                "model",
                "condition",
                "patient_id",
                "R",
                "pd_aux_R",
            ],
            as_index=False,
        )[METRICS]
        .mean()
        .sort_values(["R", "patient_id", "model"])
        .reset_index(drop=True)
    )

    slice_summary = summarise_grouped(
        per_slice,
        group_cols=["model", "condition", "R", "pd_aux_R"],
        n_unit_col="slice_idx",
    )

    patient_summary = summarise_grouped(
        patient_level,
        group_cols=["model", "condition", "R", "pd_aux_R"],
        n_unit_col="patient_id",
    )

    slice_delta = compute_paired_delta(
        per_slice,
        key_cols=["patient_id", "slice_idx", "R"],
    )

    patient_delta = compute_paired_delta(
        patient_level,
        key_cols=["patient_id", "R"],
    )

    slice_delta_summary = summarise_delta(
        slice_delta,
        unit_cols=["patient_id", "slice_idx", "R"],
    )

    patient_delta_summary = summarise_delta(
        patient_delta,
        unit_cols=["patient_id", "R"],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "per_slice": args.output_dir / "per_slice_metrics.csv",
        "patient_level": args.output_dir / "patient_level_metrics.csv",
        "slice_summary": args.output_dir / "slice_level_summary.csv",
        "patient_summary": args.output_dir / "patient_level_summary.csv",
        "slice_delta": args.output_dir / "paired_delta_vs_single_per_slice.csv",
        "patient_delta": args.output_dir / "paired_delta_vs_single_patient_level.csv",
        "slice_delta_summary": args.output_dir / "paired_delta_summary_slice_level.csv",
        "patient_delta_summary": args.output_dir / "paired_delta_summary_patient_level.csv",
        "config": args.output_dir / "evaluation_config.json",
    }

    per_slice.to_csv(paths["per_slice"], index=False)
    patient_level.to_csv(paths["patient_level"], index=False)
    slice_summary.to_csv(paths["slice_summary"], index=False)
    patient_summary.to_csv(paths["patient_summary"], index=False)
    slice_delta.to_csv(paths["slice_delta"], index=False)
    patient_delta.to_csv(paths["patient_delta"], index=False)
    slice_delta_summary.to_csv(
        paths["slice_delta_summary"],
        index=False,
    )
    patient_delta_summary.to_csv(
        paths["patient_delta_summary"],
        index=False,
    )

    config_payload = {
        "metadata_csv": str(args.metadata_csv),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "patient_aggregation": "mean across slices, then equal-weight across patients",
        "positive_delta_definition": {
            "NMSE": "single - model",
            "PSNR": "model - single",
            "SSIM": "model - single",
            "L1": "single - model",
        },
        "checkpoints": {
            str(R): {
                name: str(path)
                for name, path in checkpoint_plan[R].items()
            }
            for R in checkpoint_plan
        },
    }

    with open(paths["config"], "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)

    print("=" * 88)
    print("Patient-level summary")
    print("=" * 88)
    print(patient_summary.to_string(index=False))
    print("=" * 88)
    print("Patient-level paired delta vs single")
    print("Positive delta means better than single for every metric.")
    print("=" * 88)
    print(patient_delta_summary.to_string(index=False))
    print("=" * 88)

    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
