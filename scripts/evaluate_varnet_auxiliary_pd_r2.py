#!/usr/bin/env python3
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

from src.dataset_paired_multicoil_aux_pd_r2 import PairedMulticoilAuxPDToPDFSDataset
from src.auxiliary_varnet import AuxPDVarNet


class ShapeBucketBatchSampler(Sampler[List[int]]):
    def __init__(self, dataset, batch_size: int, shuffle: bool = False, seed: int = 42):
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
        self.num_batches = sum(math.ceil(len(indices) / self.batch_size) for indices in self.buckets.values())

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
        yield from all_batches

    def __len__(self):
        return self.num_batches


def center_crop_tensor(x: torch.Tensor, crop_h: int, crop_w: int) -> torch.Tensor:
    h, w = x.shape[-2:]
    if h == crop_h and w == crop_w:
        return x
    if h < crop_h or w < crop_w:
        raise RuntimeError(f"Cannot crop tensor from {(h, w)} to {(crop_h, crop_w)}")
    top = (h - crop_h) // 2
    left = (w - crop_w) // 2
    return x[..., top:top + crop_h, left:left + crop_w]


def center_crop_np(x: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    h, w = x.shape[-2:]
    th, tw = shape
    if h == th and w == tw:
        return x
    if h < th or w < tw:
        raise RuntimeError(f"Cannot crop array of shape {(h, w)} to {(th, tw)}")
    top = (h - th) // 2
    left = (w - tw) // 2
    return x[..., top:top + th, left:left + tw]


def to_numpy_2d(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().float().numpy()


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
    scale = float(np.max(target))
    if scale < 1e-12:
        scale = 1.0
    return float(np.mean(np.abs(pred / scale - target / scale)))


def prepare_batch(batch: Dict, device: torch.device):
    pdfs_kspace = batch["pdfs_masked_kspace"].to(device, non_blocking=True)
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
    elif mask.ndim == 5:
        pass
    else:
        raise RuntimeError(f"Unexpected mask shape: {tuple(mask.shape)}")
    mask = mask.bool()

    pd_aux = batch["pd_aux_image"].to(device, non_blocking=True).float()
    pdfs_target = batch["pdfs_target_raw"].to(device, non_blocking=True).float()

    if pd_aux.ndim == 4 and pd_aux.shape[1] == 1:
        pd_aux = pd_aux[:, 0]
    if pdfs_target.ndim == 4 and pdfs_target.shape[1] == 1:
        pdfs_target = pdfs_target[:, 0]

    if pd_aux.ndim != 3:
        raise RuntimeError(f"Expected PD auxiliary [B,H,W], got {tuple(pd_aux.shape)}")
    if pdfs_target.ndim != 3:
        raise RuntimeError(f"Expected PDFS target [B,H,W], got {tuple(pdfs_target.shape)}")
    if pd_aux.shape[-2:] != pdfs_target.shape[-2:]:
        raise RuntimeError(
            f"PD auxiliary / PDFS target shape mismatch: {tuple(pd_aux.shape)} vs {tuple(pdfs_target.shape)}"
        )

    return pdfs_kspace, mask, pd_aux, pdfs_target


def get_batch_value(batch: Dict, key: str, index: int):
    value = batch[key]
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    return value[index]


def metric_rows_for_pdfs(pred, target, batch, acceleration, pd_aux_acceleration, checkpoint_path):
    rows = []
    pred_np = to_numpy_2d(pred)
    target_np = to_numpy_2d(target)
    batch_size = pred_np.shape[0]

    for i in range(batch_size):
        p = pred_np[i]
        t = target_np[i]
        if p.shape != t.shape:
            h = min(p.shape[-2], t.shape[-2])
            w = min(p.shape[-1], t.shape[-1])
            p = center_crop_np(p, (h, w))
            t = center_crop_np(t, (h, w))

        rows.append({
            "model": "auxiliary_pdR2_to_pdfs_varnet",
            "checkpoint": str(checkpoint_path),
            "patient_id": str(get_batch_value(batch, "patient_id", i)),
            "slice_idx": int(get_batch_value(batch, "slice_idx", i)),
            "contrast": "PD-FS",
            "R": int(acceleration),
            "pd_aux_R": int(pd_aux_acceleration),
            "auxiliary_input": "zero-filled RSS PD image from undersampled PD k-space",
            "NMSE": nmse(t, p),
            "PSNR": psnr(t, p),
            "SSIM": ssim_metric(t, p),
            "L1": l1_metric(t, p),
            "is_edge": bool(get_batch_value(batch, "is_edge", i)),
        })
    return rows


def summarise_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (contrast, R), g in df.groupby(["contrast", "R"]):
        row = {
            "model": "auxiliary_pdR2_to_pdfs_varnet",
            "contrast": contrast,
            "R": int(R),
            "pd_aux_R": int(g["pd_aux_R"].iloc[0]),
            "n_slices": int(len(g)),
            "auxiliary_input": "zero-filled RSS PD image from undersampled PD k-space",
        }
        for metric in ["NMSE", "PSNR", "SSIM", "L1"]:
            vals = g[metric].dropna().values
            row[f"{metric}_mean"] = float(np.mean(vals)) if len(vals) else float("nan")
            row[f"{metric}_median"] = float(np.median(vals)) if len(vals) else float("nan")
            row[f"{metric}_iqr_low"] = float(np.percentile(vals, 25)) if len(vals) else float("nan")
            row[f"{metric}_iqr_high"] = float(np.percentile(vals, 75)) if len(vals) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["R", "contrast"])


def patient_level_metrics(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["NMSE", "PSNR", "SSIM", "L1"]
    return (
        df.groupby(["patient_id", "contrast", "R", "pd_aux_R"], as_index=False)[metric_cols]
        .median()
        .sort_values(["R", "contrast", "patient_id"])
    )


def torch_load_checkpoint(checkpoint_path: Path, device: torch.device):
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def extract_state_dict(ckpt, checkpoint_path: Path):
    if isinstance(ckpt, dict):
        for key in ["model_state_dict", "model", "state_dict", "net", "network"]:
            state = ckpt.get(key)
            if isinstance(state, dict):
                return state

    if isinstance(ckpt, dict) and all(torch.is_tensor(v) for v in ckpt.values()):
        return ckpt

    raise RuntimeError(
        f"Cannot find model state dict in {checkpoint_path}. "
        f"Keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}"
    )


def strip_module_prefix_if_needed(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state:
        return state
    if all(k.startswith("module.") for k in state.keys()):
        return {k[len("module."):]: v for k, v in state.items()}
    return state


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> AuxPDVarNet:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    ckpt = torch_load_checkpoint(checkpoint_path, device)
    config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    model = AuxPDVarNet(
        num_cascades=int(config.get("num_cascades", 12)),
        chans=int(config.get("chans", 18)),
        pools=int(config.get("pools", 4)),
        sens_chans=int(config.get("sens_chans", 8)),
        sens_pools=int(config.get("sens_pools", 4)),
        mask_center=True,
    ).to(device)

    state = strip_module_prefix_if_needed(extract_state_dict(ckpt, checkpoint_path))
    model.load_state_dict(state, strict=True)
    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")
    if isinstance(ckpt, dict):
        print(f"Checkpoint epoch: {ckpt.get('epoch')}, best_epoch: {ckpt.get('best_epoch')}")
        print(
            "Checkpoint config: "
            f"acceleration={config.get('acceleration')}, "
            f"pd_aux_acceleration={config.get('pd_aux_acceleration')}, "
            f"num_cascades={config.get('num_cascades')}, "
            f"chans={config.get('chans')}, "
            f"sens_chans={config.get('sens_chans')}, "
            f"batch_size={config.get('batch_size')}, "
            f"learning_rate={config.get('learning_rate')}"
        )
    return model


@torch.no_grad()
def evaluate_one_R(acceleration, pd_aux_acceleration, checkpoint_path, metadata_csv, output_dir, batch_size, num_workers, device):
    print("=" * 80)
    print(f"Evaluating AuxPDVarNet PD R={pd_aux_acceleration} -> PD-FS R={acceleration}")
    print("=" * 80)

    dataset = PairedMulticoilAuxPDToPDFSDataset(
        metadata_csv=str(metadata_csv),
        split="val",
        pdfs_acceleration=int(acceleration),
        pd_aux_acceleration=int(pd_aux_acceleration),
    )

    sampler = ShapeBucketBatchSampler(dataset=dataset, batch_size=batch_size, shuffle=False, seed=42)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers, pin_memory=(device.type == "cuda"))

    model = load_model_from_checkpoint(checkpoint_path, device)
    all_rows = []

    for batch_idx, batch in enumerate(loader, start=1):
        pdfs_kspace, mask, pd_aux, pdfs_target = prepare_batch(batch, device)
        pdfs_pred = model(pdfs_masked_kspace=pdfs_kspace, mask=mask, pd_aux_image=pd_aux)

        pdfs_pred = center_crop_tensor(
            pdfs_pred,
            crop_h=pdfs_target.shape[-2],
            crop_w=pdfs_target.shape[-1],
        )

        all_rows.extend(metric_rows_for_pdfs(
            pred=pdfs_pred,
            target=pdfs_target,
            batch=batch,
            acceleration=acceleration,
            pd_aux_acceleration=pd_aux_acceleration,
            checkpoint_path=checkpoint_path,
        ))

        if batch_idx == 1 or batch_idx % 25 == 0 or batch_idx == len(loader):
            print(f"R={acceleration} | batch {batch_idx}/{len(loader)}")

    df = pd.DataFrame(all_rows)
    summary_df = summarise_metrics(df)
    patient_df = patient_level_metrics(df)

    output_dir.mkdir(parents=True, exist_ok=True)
    per_slice_path = output_dir / f"aux_pdR{pd_aux_acceleration}_to_pdfs_R{acceleration}_val_per_slice_metrics.csv"
    patient_path = output_dir / f"aux_pdR{pd_aux_acceleration}_to_pdfs_R{acceleration}_val_patient_level_metrics.csv"
    summary_path = output_dir / f"aux_pdR{pd_aux_acceleration}_to_pdfs_R{acceleration}_val_summary.csv"
    summary_json_path = output_dir / f"aux_pdR{pd_aux_acceleration}_to_pdfs_R{acceleration}_val_summary.json"

    df.to_csv(per_slice_path, index=False)
    patient_df.to_csv(patient_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_df.to_dict(orient="records"), f, indent=2)

    print(f"Saved per-slice metrics: {per_slice_path}")
    print(f"Saved patient-level metrics: {patient_path}")
    print(f"Saved summary: {summary_path}")
    print(summary_df.to_string(index=False))

    return df, summary_df, patient_df


def build_checkpoint_plan(args):
    plan = {}

    if args.checkpoint is not None or args.acceleration is not None:
        if args.checkpoint is None or args.acceleration is None:
            raise SystemExit("Use --checkpoint and --acceleration together, or use --R4_checkpoint/--R6_checkpoint/--R8_checkpoint.")
        plan[int(args.acceleration)] = Path(args.checkpoint)
        return plan

    if args.R4_checkpoint is not None:
        plan[4] = Path(args.R4_checkpoint)
    if args.R6_checkpoint is not None:
        plan[6] = Path(args.R6_checkpoint)
    if args.R8_checkpoint is not None:
        plan[8] = Path(args.R8_checkpoint)

    if not plan:
        raise SystemExit("No checkpoint supplied.")

    return plan


def main():
    parser = argparse.ArgumentParser(description="Evaluate AuxPDVarNet with PD R=2 zero-filled auxiliary input.")
    parser.add_argument("--metadata_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--pd_aux_acceleration", type=int, choices=[2, 4, 6, 8], default=2)

    parser.add_argument("--acceleration", type=int, choices=[4, 6, 8], default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)

    parser.add_argument("--R4_checkpoint", type=Path, default=None)
    parser.add_argument("--R6_checkpoint", type=Path, default=None)
    parser.add_argument("--R8_checkpoint", type=Path, default=None)

    args = parser.parse_args()

    checkpoint_plan = build_checkpoint_plan(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("AuxPDVarNet PD R=2 -> PD-FS formal validation evaluation")
    print("=" * 80)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Metadata: {args.metadata_csv}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"PD auxiliary acceleration: {args.pd_aux_acceleration}")
    print(f"Checkpoint plan: {checkpoint_plan}")
    print("=" * 80)

    all_slice_dfs = []
    all_summaries = []
    all_patient_dfs = []

    for R in sorted(checkpoint_plan):
        per_slice_df, summary_df, patient_df = evaluate_one_R(
            acceleration=R,
            pd_aux_acceleration=args.pd_aux_acceleration,
            checkpoint_path=checkpoint_plan[R],
            metadata_csv=args.metadata_csv,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        all_slice_dfs.append(per_slice_df)
        all_summaries.append(summary_df)
        all_patient_dfs.append(patient_df)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    combined_summary = pd.concat(all_summaries, ignore_index=True).sort_values(["R", "contrast"])
    combined_summary_path = args.output_dir / "aux_pdR2_to_pdfs_val_slice_level_summary_R4_R6_R8.csv"
    combined_summary.to_csv(combined_summary_path, index=False)

    combined_slice = pd.concat(all_slice_dfs, ignore_index=True).sort_values(["R", "contrast", "patient_id", "slice_idx"])
    combined_slice_path = args.output_dir / "aux_pdR2_to_pdfs_val_per_slice_metrics_R4_R6_R8.csv"
    combined_slice.to_csv(combined_slice_path, index=False)

    combined_patient = pd.concat(all_patient_dfs, ignore_index=True).sort_values(["R", "contrast", "patient_id"])
    combined_patient_path = args.output_dir / "aux_pdR2_to_pdfs_val_patient_level_metrics_R4_R6_R8.csv"
    combined_patient.to_csv(combined_patient_path, index=False)

    print("=" * 80)
    print("Combined AuxPDVarNet PD R=2 -> PD-FS summary")
    print("=" * 80)
    print(combined_summary.to_string(index=False))
    print(f"Saved combined summary: {combined_summary_path}")
    print(f"Saved combined per-slice metrics: {combined_slice_path}")
    print(f"Saved combined patient-level metrics: {combined_patient_path}")


if __name__ == "__main__":
    main()
