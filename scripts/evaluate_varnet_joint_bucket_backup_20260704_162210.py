#!/usr/bin/env python3
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

from src.dataset_paired_multicoil import PairedMulticoilDataset
from src.joint_varnet import JointVarNet


class ShapeBucketBatchSampler(Sampler[List[int]]):
    def __init__(self, dataset, batch_size: int, shuffle: bool = False, seed: int = 42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        buckets: Dict[Tuple[int, int], List[int]] = {}
        for idx, record in enumerate(dataset.records):
            pd_path = record["pd_path"]
            with h5py.File(pd_path, "r") as hf:
                shape = hf["kspace"].shape
                key = (int(shape[-3]), int(shape[-2]))
            buckets.setdefault(key, []).append(idx)

        self.buckets = buckets

        self.num_batches = 0
        for _, indices in self.buckets.items():
            self.num_batches += math.ceil(len(indices) / self.batch_size)

        print(
            f"ShapeBucketBatchSampler: {len(self.buckets)} shape buckets, "
            f"{self.num_batches} batches, batch_size={self.batch_size}, shuffle={self.shuffle}"
        )

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        all_batches = []

        for _, indices in self.buckets.items():
            indices = list(indices)
            if self.shuffle:
                rng.shuffle(indices)

            for start in range(0, len(indices), self.batch_size):
                all_batches.append(indices[start:start + self.batch_size])

        if self.shuffle:
            rng.shuffle(all_batches)

        for batch in all_batches:
            yield batch

    def __len__(self):
        return self.num_batches


def center_crop_np(x: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    h, w = x.shape[-2:]
    th, tw = shape
    if h == th and w == tw:
        return x
    top = (h - th) // 2
    left = (w - tw) // 2
    return x[..., top:top + th, left:left + tw]


def to_numpy_2d(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().float().numpy()
    return x


def nmse(target: np.ndarray, pred: np.ndarray) -> float:
    denom = np.linalg.norm(target) ** 2
    if denom == 0:
        return float("nan")
    return float((np.linalg.norm(target - pred) ** 2) / denom)


def psnr(target: np.ndarray, pred: np.ndarray) -> float:
    mse = np.mean((target - pred) ** 2)
    if mse == 0:
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
    return float(np.mean(np.abs(target - pred)))


def prepare_batch(batch: Dict, device: torch.device):
    pd_kspace = batch["pd_masked_kspace"].to(device)
    pdfs_kspace = batch["pdfs_masked_kspace"].to(device)
    mask = batch["mask"].to(device)

    if torch.is_complex(pd_kspace):
        pd_kspace = torch.view_as_real(pd_kspace)
    if torch.is_complex(pdfs_kspace):
        pdfs_kspace = torch.view_as_real(pdfs_kspace)

    # fastMRI VarNet / SensitivityModel expects mask shape:
    # [B, 1, 1, W, 1]
    # Dataset/DataLoader may return [B, W], [B, 1, W], [B, 1, 1, W],
    # or already [B, 1, 1, W, 1]. Normalise it here.
    if mask.ndim == 1:
        mask = mask[None, None, None, :, None]
    elif mask.ndim == 2:
        # Usually [B, W]
        mask = mask[:, None, None, :, None]
    elif mask.ndim == 3:
        # Usually [B, 1, W] or [B, H, W]; use phase-encode dimension.
        if mask.shape[1] == 1:
            mask = mask[:, :, None, :, None]
        else:
            mask = mask[:, None, None, :, None]
    elif mask.ndim == 4:
        # Usually [B, 1, 1, W]
        mask = mask[..., None]
    elif mask.ndim == 5:
        pass
    else:
        raise RuntimeError(f"Unexpected mask shape: {tuple(mask.shape)}")

    mask = mask.bool()

    pd_target = batch["pd_target_raw"].to(device)
    pdfs_target = batch["pdfs_target_raw"].to(device)

    if pd_target.ndim == 4 and pd_target.shape[1] == 1:
        pd_target = pd_target[:, 0]
    if pdfs_target.ndim == 4 and pdfs_target.shape[1] == 1:
        pdfs_target = pdfs_target[:, 0]

    return pd_kspace, pdfs_kspace, mask, pd_target, pdfs_target


def metric_rows_for_contrast(
    pred: torch.Tensor,
    target: torch.Tensor,
    batch: Dict,
    contrast: str,
    acceleration: int,
) -> List[Dict]:
    rows = []

    pred_np = to_numpy_2d(pred)
    target_np = to_numpy_2d(target)

    batch_size = pred_np.shape[0]

    for i in range(batch_size):
        p = pred_np[i]
        t = target_np[i]

        # Align shapes by centre crop if needed.
        if p.shape != t.shape:
            h = min(p.shape[-2], t.shape[-2])
            w = min(p.shape[-1], t.shape[-1])
            p = center_crop_np(p, (h, w))
            t = center_crop_np(t, (h, w))

        rows.append({
            "patient_id": batch["patient_id"][i],
            "slice_idx": int(batch["slice_idx"][i]),
            "contrast": contrast,
            "R": int(acceleration),
            "NMSE": nmse(t, p),
            "PSNR": psnr(t, p),
            "SSIM": ssim_metric(t, p),
            "L1": l1_metric(t, p),
            "is_edge": bool(batch["is_edge"][i]),
        })

    return rows


def summarise_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (contrast, R), g in df.groupby(["contrast", "R"]):
        row = {
            "model": "joint",
            "contrast": contrast,
            "R": int(R),
            "n_slices": int(len(g)),
        }

        for metric in ["NMSE", "PSNR", "SSIM", "L1"]:
            vals = g[metric].dropna().values
            row[f"{metric}_mean"] = float(np.mean(vals))
            row[f"{metric}_median"] = float(np.median(vals))
            row[f"{metric}_iqr_low"] = float(np.percentile(vals, 25))
            row[f"{metric}_iqr_high"] = float(np.percentile(vals, 75))

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["R", "contrast"])


def patient_level_metrics(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["NMSE", "PSNR", "SSIM", "L1"]
    return (
        df.groupby(["patient_id", "contrast", "R"], as_index=False)[metric_cols]
        .median()
        .sort_values(["R", "contrast", "patient_id"])
    )


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> JointVarNet:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})

    model = JointVarNet(
        num_cascades=int(config.get("num_cascades", 12)),
        chans=int(config.get("chans", 18)),
        pools=int(config.get("pools", 4)),
        sens_chans=int(config.get("sens_chans", 8)),
        sens_pools=int(config.get("sens_pools", 4)),
        cross_fusion=str(config.get("cross_fusion", "concat")),
    ).to(device)

    state = ckpt.get("model_state_dict", ckpt.get("model", None))
    if state is None:
        raise RuntimeError(f"Cannot find model state dict in {checkpoint_path}")

    model.load_state_dict(state, strict=True)
    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Checkpoint epoch: {ckpt.get('epoch')}, best_epoch: {ckpt.get('best_epoch')}")
    print(f"Checkpoint config: acceleration={config.get('acceleration')}, cross_fusion={config.get('cross_fusion')}")

    return model


@torch.no_grad()
def evaluate_one_R(
    acceleration: int,
    checkpoint_path: Path,
    metadata_csv: Path,
    output_dir: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
):
    print("=" * 80)
    print(f"Evaluating joint VarNet R={acceleration}")
    print("=" * 80)

    dataset = PairedMulticoilDataset(
        metadata_csv=str(metadata_csv),
        split="val",
        acceleration=acceleration,
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
        pin_memory=True,
    )

    model = load_model_from_checkpoint(checkpoint_path, device)

    all_rows = []

    for batch_idx, batch in enumerate(loader, start=1):
        pd_kspace, pdfs_kspace, mask, pd_target, pdfs_target = prepare_batch(batch, device)

        pd_pred, pdfs_pred = model(pd_kspace, pdfs_kspace, mask)

        all_rows.extend(
            metric_rows_for_contrast(
                pred=pd_pred,
                target=pd_target,
                batch=batch,
                contrast="PD",
                acceleration=acceleration,
            )
        )

        all_rows.extend(
            metric_rows_for_contrast(
                pred=pdfs_pred,
                target=pdfs_target,
                batch=batch,
                contrast="PD-FS",
                acceleration=acceleration,
            )
        )

        if batch_idx == 1 or batch_idx % 25 == 0:
            print(f"R={acceleration} | batch {batch_idx}/{len(loader)}")

    df = pd.DataFrame(all_rows)
    summary_df = summarise_metrics(df)
    patient_df = patient_level_metrics(df)

    output_dir.mkdir(parents=True, exist_ok=True)

    per_slice_path = output_dir / f"joint_R{acceleration}_val_per_slice_metrics.csv"
    patient_path = output_dir / f"joint_R{acceleration}_val_patient_level_metrics.csv"
    summary_path = output_dir / f"joint_R{acceleration}_val_summary.csv"
    summary_json_path = output_dir / f"joint_R{acceleration}_val_summary.json"

    df.to_csv(per_slice_path, index=False)
    patient_df.to_csv(patient_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    with open(summary_json_path, "w") as f:
        json.dump(summary_df.to_dict(orient="records"), f, indent=2)

    print(f"Saved per-slice metrics: {per_slice_path}")
    print(f"Saved patient-level metrics: {patient_path}")
    print(f"Saved summary: {summary_path}")

    print(summary_df.to_string(index=False))

    return summary_df


def main():
    parser = argparse.ArgumentParser()

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
        default=4,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--R4_checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--R6_checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--R8_checkpoint",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("Joint VarNet formal validation evaluation")
    print("=" * 80)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Metadata: {args.metadata_csv}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 80)

    checkpoints = {
        4: args.R4_checkpoint,
        6: args.R6_checkpoint,
        8: args.R8_checkpoint,
    }

    all_summaries = []

    for R, ckpt_path in checkpoints.items():
        summary = evaluate_one_R(
            acceleration=R,
            checkpoint_path=ckpt_path,
            metadata_csv=args.metadata_csv,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        all_summaries.append(summary)

    combined = pd.concat(all_summaries, ignore_index=True)
    combined = combined.sort_values(["R", "contrast"])

    combined_path = args.output_dir / "joint_varnet_val_slice_level_summary_R4_R6_R8.csv"
    combined.to_csv(combined_path, index=False)

    print("=" * 80)
    print("Combined joint summary")
    print("=" * 80)
    print(combined.to_string(index=False))
    print(f"Saved combined summary: {combined_path}")


if __name__ == "__main__":
    main()
