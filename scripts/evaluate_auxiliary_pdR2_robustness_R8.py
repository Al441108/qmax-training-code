#!/usr/bin/env python3
"""
R=8 robustness evaluation for a PD R=2 auxiliary PD-FS reconstruction model.

Evaluated conditions:
  1. Single PD-FS VarNet baseline
  2. Auxiliary model with correct PD R=2 input
  3. Auxiliary model with zero / missing PD input
  4. Auxiliary model with translated PD input, using non-wrapping zero padding
  5. Auxiliary model with wrong-patient PD input, mapped over the whole validation set

Main safety outputs:
  - per-condition per-slice metrics
  - patient-level medians
  - paired delta versus the single baseline
  - percentage of slices worse than the single baseline
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
from src.dataset_paired_multicoil_aux_pd_r2 import PairedMulticoilAuxPDToPDFSDataset
from src.auxiliary_varnet import AuxPDVarNet


METRICS = ["NMSE", "PSNR", "SSIM", "L1"]
HIGHER_IS_BETTER = {
    "NMSE": False,
    "PSNR": True,
    "SSIM": True,
    "L1": False,
}


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
        self.num_batches = sum(math.ceil(len(v) / self.batch_size) for v in buckets.values())

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


class IndexedDataset:
    """Thin wrapper that adds sample_idx while preserving dataset.records for the bucket sampler."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.records = dataset.records

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        item["sample_idx"] = int(idx)
        return item


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
        raise RuntimeError(f"Cannot crop array from {(h, w)} to {(th, tw)}")
    top = (h - th) // 2
    left = (w - tw) // 2
    return x[..., top:top + th, left:left + tw]


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


def strip_module_prefix_if_needed(state):
    if not state:
        return state
    if all(k.startswith("module.") for k in state.keys()):
        return {k[len("module."):]: v for k, v in state.items()}
    return state


def warn_checkpoint_config(ckpt, checkpoint_path: Path, expected_pdfs_R: int = 8, expected_pd_aux_R: int = 2):
    if not isinstance(ckpt, dict):
        print(f"WARNING: {checkpoint_path} has no readable checkpoint dictionary/config.")
        return

    config = ckpt.get("config", {})
    print(f"Checkpoint config for {checkpoint_path}: {json.dumps(config, indent=2, default=str)}")

    pdfs_R = config.get("pdfs_acceleration", config.get("acceleration", None))
    pd_aux_R = config.get("pd_aux_acceleration", None)
    if pdfs_R is not None and int(pdfs_R) != expected_pdfs_R:
        print(f"WARNING: checkpoint PDFS acceleration appears to be {pdfs_R}, expected {expected_pdfs_R}.")
    if pd_aux_R is not None and int(pd_aux_R) != expected_pd_aux_R:
        print(f"WARNING: checkpoint PD auxiliary acceleration appears to be {pd_aux_R}, expected {expected_pd_aux_R}.")
    if pd_aux_R is None:
        print("WARNING: checkpoint config has no pd_aux_acceleration field; confirm this is really a PD R=2 auxiliary model.")


def load_aux_model(checkpoint_path: Path, device: torch.device) -> AuxPDVarNet:
    ckpt = torch_load_checkpoint(checkpoint_path, device)
    config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    warn_checkpoint_config(ckpt, checkpoint_path)

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

    print(f"Loaded auxiliary checkpoint: {checkpoint_path}")
    if isinstance(ckpt, dict):
        print(f"Aux checkpoint epoch={ckpt.get('epoch')}, best_epoch={ckpt.get('best_epoch')}")
    return model


def load_single_model(checkpoint_path: Path, device: torch.device) -> VarNet:
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

    state = strip_module_prefix_if_needed(extract_state_dict(ckpt, checkpoint_path))
    model.load_state_dict(state, strict=True)
    model.eval()

    print(f"Loaded single checkpoint: {checkpoint_path}")
    if isinstance(ckpt, dict):
        print(f"Single checkpoint epoch={ckpt.get('epoch')}, best_epoch={ckpt.get('best_epoch')}")
    return model


def find_single_R8_checkpoint() -> Path:
    base = Path("outputs/varnet_single")
    files = []
    for pattern in ["*.pt", "*.pth", "*.ckpt"]:
        files.extend(base.rglob(pattern))

    candidates = []
    for p in files:
        s = str(p).lower()
        if "r8" not in s:
            continue
        if "pdfs" not in s:
            continue
        if "smoke" in s:
            continue
        if p.name != "model_best.pt" and "best" not in p.name.lower():
            continue
        candidates.append(p)

    if not candidates:
        raise FileNotFoundError("Cannot auto-find single PD-FS R8 checkpoint under outputs/varnet_single")

    candidates = sorted(
        candidates,
        key=lambda p: (
            0 if "ep30" in str(p).lower() else 1,
            0 if "bs8" in str(p).lower() else 1,
            0 if p.name == "model_best.pt" else 1,
            len(str(p)),
            str(p),
        )
    )
    return candidates[0]


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

    if pd_aux.shape[-2:] != pdfs_target.shape[-2:]:
        raise RuntimeError(
            f"PD auxiliary / PDFS target mismatch: {tuple(pd_aux.shape)} vs {tuple(pdfs_target.shape)}"
        )

    return pdfs_kspace, mask, pd_aux, pdfs_target


def get_batch_value(batch: Dict, key: str, index: int):
    value = batch[key]
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    return value[index]


def translate_zero_pad(x: torch.Tensor, shift_y: int, shift_x: int) -> torch.Tensor:
    """Translate [B,H,W] or [B,1,H,W] without wrap-around; empty regions are zero-filled."""
    out = torch.zeros_like(x)
    h, w = x.shape[-2:]

    src_y0 = max(0, -shift_y)
    src_y1 = min(h, h - shift_y)
    dst_y0 = max(0, shift_y)
    dst_y1 = min(h, h + shift_y)

    src_x0 = max(0, -shift_x)
    src_x1 = min(w, w - shift_x)
    dst_x0 = max(0, shift_x)
    dst_x1 = min(w, w + shift_x)

    if src_y1 > src_y0 and src_x1 > src_x0:
        out[..., dst_y0:dst_y1, dst_x0:dst_x1] = x[..., src_y0:src_y1, src_x0:src_x1]
    return out


def build_wrong_patient_index(dataset) -> Dict[int, int]:
    """Map each sample index to a deterministic sample from a different patient over the whole validation set."""
    patient_ids = [str(r["patient_id"]) for r in dataset.records]
    n = len(patient_ids)
    mapping = {}

    unique_patients = sorted(set(patient_ids))
    if len(unique_patients) < 2:
        raise RuntimeError("Wrong-patient robustness needs at least two patients in the validation set.")

    for i, pid in enumerate(patient_ids):
        replacement = None
        # Use a large-ish offset to avoid simply choosing an adjacent slice.
        for offset in range(max(1, n // 3), n + max(1, n // 3)):
            j = (i + offset) % n
            if patient_ids[j] != pid:
                replacement = j
                break
        if replacement is None:
            raise RuntimeError(f"Could not find wrong-patient replacement for index {i}, patient={pid}")
        mapping[i] = replacement
    return mapping


def sample_pd_aux_by_indices(dataset, indices: Sequence[int], device: torch.device) -> torch.Tensor:
    pd_list = []
    for idx in indices:
        item = dataset[int(idx)]
        pd = item["pd_aux_image"]
        if not torch.is_tensor(pd):
            pd = torch.as_tensor(pd)
        pd = pd.float()
        if pd.ndim == 3 and pd.shape[0] == 1:
            pd = pd[0]
        elif pd.ndim != 2:
            raise RuntimeError(f"Unexpected wrong-patient PD aux shape at idx={idx}: {tuple(pd.shape)}")
        pd_list.append(pd)
    return torch.stack(pd_list, dim=0).to(device, non_blocking=True)


def rows_for_prediction(pred, target, batch, model_name, condition):
    rows = []
    pred_np = pred.detach().cpu().float().numpy()
    target_np = target.detach().cpu().float().numpy()
    b = pred_np.shape[0]

    for i in range(b):
        p = pred_np[i]
        t = target_np[i]

        if p.shape != t.shape:
            h = min(p.shape[-2], t.shape[-2])
            w = min(p.shape[-1], t.shape[-1])
            p = center_crop_np(p, (h, w))
            t = center_crop_np(t, (h, w))

        rows.append({
            "model": model_name,
            "condition": condition,
            "patient_id": str(get_batch_value(batch, "patient_id", i)),
            "slice_idx": int(get_batch_value(batch, "slice_idx", i)),
            "contrast": "PD-FS",
            "R": 8,
            "pd_aux_R": 2,
            "NMSE": nmse(t, p),
            "PSNR": psnr(t, p),
            "SSIM": ssim_metric(t, p),
            "L1": l1_metric(t, p),
            "is_edge": bool(get_batch_value(batch, "is_edge", i)),
        })

    return rows


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (model, condition), g in df.groupby(["model", "condition"]):
        row = {
            "model": model,
            "condition": condition,
            "R": 8,
            "pd_aux_R": 2,
            "n_slices": int(len(g)),
            "n_patients": int(g["patient_id"].nunique()),
        }

        for metric in METRICS:
            vals = g[metric].dropna().values
            row[f"{metric}_mean"] = float(np.mean(vals)) if len(vals) else float("nan")
            row[f"{metric}_median"] = float(np.median(vals)) if len(vals) else float("nan")
            row[f"{metric}_iqr_low"] = float(np.percentile(vals, 25)) if len(vals) else float("nan")
            row[f"{metric}_iqr_high"] = float(np.percentile(vals, 75)) if len(vals) else float("nan")

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["model", "condition"])


def compute_paired_delta(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = df[df["condition"] == "no_auxiliary"].copy()
    base = base[["patient_id", "slice_idx", "R", "NMSE", "PSNR", "SSIM", "L1"]].rename(
        columns={m: f"{m}_single" for m in METRICS}
    )

    rows = []
    for condition in sorted(c for c in df["condition"].unique() if c != "no_auxiliary"):
        cur = df[df["condition"] == condition].copy()
        merged = cur.merge(base, on=["patient_id", "slice_idx", "R"], how="inner", validate="one_to_one")
        merged["baseline_condition"] = "no_auxiliary"

        for metric in METRICS:
            if HIGHER_IS_BETTER[metric]:
                merged[f"{metric}_delta_vs_single"] = merged[metric] - merged[f"{metric}_single"]
            else:
                merged[f"{metric}_delta_vs_single"] = merged[f"{metric}_single"] - merged[metric]
            merged[f"{metric}_worse_than_single"] = merged[f"{metric}_delta_vs_single"] < 0

        rows.append(merged)

    delta_df = pd.concat(rows, ignore_index=True)

    summary_rows = []
    for (model, condition), g in delta_df.groupby(["model", "condition"]):
        row = {
            "model": model,
            "condition": condition,
            "baseline_condition": "no_auxiliary",
            "R": 8,
            "pd_aux_R": 2,
            "n_paired_slices": int(len(g)),
            "n_patients": int(g["patient_id"].nunique()),
        }
        for metric in METRICS:
            vals = g[f"{metric}_delta_vs_single"].dropna().values
            worse = g[f"{metric}_worse_than_single"].dropna().values
            row[f"{metric}_delta_mean"] = float(np.mean(vals)) if len(vals) else float("nan")
            row[f"{metric}_delta_median"] = float(np.median(vals)) if len(vals) else float("nan")
            row[f"{metric}_delta_iqr_low"] = float(np.percentile(vals, 25)) if len(vals) else float("nan")
            row[f"{metric}_delta_iqr_high"] = float(np.percentile(vals, 75)) if len(vals) else float("nan")
            row[f"{metric}_pct_worse_than_single"] = float(np.mean(worse) * 100.0) if len(worse) else float("nan")
        summary_rows.append(row)

    delta_summary = pd.DataFrame(summary_rows).sort_values(["model", "condition"])
    return delta_df, delta_summary


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="R=8 robustness test for PD R=2 auxiliary model.")
    parser.add_argument("--metadata_csv", type=Path, required=True)
    parser.add_argument("--aux_checkpoint", type=Path, required=True)
    parser.add_argument("--single_checkpoint", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--shift_pixels", type=int, nargs="+", default=[2, 4, 8])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.single_checkpoint is None:
        args.single_checkpoint = find_single_R8_checkpoint()

    print("=" * 80)
    print("AuxPDVarNet PD R=2 -> PD-FS R=8 robustness evaluation")
    print("=" * 80)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Metadata: {args.metadata_csv}")
    print(f"Aux checkpoint: {args.aux_checkpoint}")
    print(f"Single checkpoint: {args.single_checkpoint}")
    print(f"Output: {args.output_dir}")
    print(f"Shift pixels: {args.shift_pixels}")
    print("=" * 80)

    base_dataset = PairedMulticoilAuxPDToPDFSDataset(
        metadata_csv=str(args.metadata_csv),
        split="val",
        pdfs_acceleration=8,
        pd_aux_acceleration=2,
    )
    dataset = IndexedDataset(base_dataset)

    wrong_patient_map = build_wrong_patient_index(dataset)
    print(f"Built whole-validation wrong-patient map for {len(wrong_patient_map)} slices.")

    sampler = ShapeBucketBatchSampler(dataset, batch_size=args.batch_size, shuffle=False, seed=42)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    aux_model = load_aux_model(args.aux_checkpoint, device)
    single_model = load_single_model(args.single_checkpoint, device)

    all_rows = []

    for batch_idx, batch in enumerate(loader, start=1):
        pdfs_kspace, mask, pd_aux, pdfs_target = prepare_batch(batch, device)

        dataset_indices = [int(x) for x in batch["sample_idx"]] if "sample_idx" in batch else None
        if dataset_indices is None:
            # Fallback for datasets without sample_idx: infer from loader order.
            # This assumes ShapeBucketBatchSampler uses the same index order within each batch.
            # Prefer adding sample_idx to the dataset if this warning appears.
            raise RuntimeError(
                "Batch has no sample_idx. Add sample_idx=idx to dataset __getitem__ output, "
                "or modify this script to receive batch indices from the sampler."
            )

        single_pred = single_model(pdfs_kspace, mask)
        single_pred = center_crop_tensor(single_pred, pdfs_target.shape[-2], pdfs_target.shape[-1])
        all_rows.extend(rows_for_prediction(single_pred, pdfs_target, batch, "single_pdfs_varnet", "no_auxiliary"))

        pred_correct = aux_model(pdfs_masked_kspace=pdfs_kspace, mask=mask, pd_aux_image=pd_aux)
        pred_correct = center_crop_tensor(pred_correct, pdfs_target.shape[-2], pdfs_target.shape[-1])
        all_rows.extend(rows_for_prediction(pred_correct, pdfs_target, batch, "aux_pdR2_to_pdfs_varnet", "correct_pdR2_aux"))

        pd_zero = torch.zeros_like(pd_aux)
        pred_zero = aux_model(pdfs_masked_kspace=pdfs_kspace, mask=mask, pd_aux_image=pd_zero)
        pred_zero = center_crop_tensor(pred_zero, pdfs_target.shape[-2], pdfs_target.shape[-1])
        all_rows.extend(rows_for_prediction(pred_zero, pdfs_target, batch, "aux_pdR2_to_pdfs_varnet", "zero_missing_pd_aux"))

        for shift in args.shift_pixels:
            pd_shift = translate_zero_pad(pd_aux, shift_y=int(shift), shift_x=int(shift))
            pred_shift = aux_model(pdfs_masked_kspace=pdfs_kspace, mask=mask, pd_aux_image=pd_shift)
            pred_shift = center_crop_tensor(pred_shift, pdfs_target.shape[-2], pdfs_target.shape[-1])
            all_rows.extend(
                rows_for_prediction(
                    pred_shift,
                    pdfs_target,
                    batch,
                    "aux_pdR2_to_pdfs_varnet",
                    f"shifted_pd_aux_{int(shift)}px_zero_pad",
                )
            )

        wrong_indices = [wrong_patient_map[i] for i in dataset_indices]
        pd_wrong = sample_pd_aux_by_indices(dataset, wrong_indices, device)
        if pd_wrong.shape[-2:] != pd_aux.shape[-2:]:
            raise RuntimeError(f"Wrong-patient PD shape mismatch: {tuple(pd_wrong.shape)} vs {tuple(pd_aux.shape)}")
        pred_wrong = aux_model(pdfs_masked_kspace=pdfs_kspace, mask=mask, pd_aux_image=pd_wrong)
        pred_wrong = center_crop_tensor(pred_wrong, pdfs_target.shape[-2], pdfs_target.shape[-1])
        all_rows.extend(rows_for_prediction(pred_wrong, pdfs_target, batch, "aux_pdR2_to_pdfs_varnet", "wrong_patient_pd_aux"))

        if batch_idx == 1 or batch_idx % 25 == 0 or batch_idx == len(loader):
            print(f"Batch {batch_idx}/{len(loader)}")

    df = pd.DataFrame(all_rows)
    summary = summarise(df)
    delta_df, delta_summary = compute_paired_delta(df)

    patient_df = (
        df.groupby(["patient_id", "model", "condition", "R", "pd_aux_R"], as_index=False)[METRICS]
        .median()
        .sort_values(["condition", "patient_id"])
    )

    patient_delta_df = (
        delta_df.groupby(["patient_id", "model", "condition", "baseline_condition", "R", "pd_aux_R"], as_index=False)[
            [f"{m}_delta_vs_single" for m in METRICS]
        ]
        .median()
        .sort_values(["condition", "patient_id"])
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_slice_path = args.output_dir / "aux_pdR2_R8_robustness_per_slice_metrics.csv"
    summary_path = args.output_dir / "aux_pdR2_R8_robustness_summary.csv"
    patient_path = args.output_dir / "aux_pdR2_R8_robustness_patient_level_metrics.csv"
    delta_path = args.output_dir / "aux_pdR2_R8_robustness_paired_delta_vs_single.csv"
    delta_summary_path = args.output_dir / "aux_pdR2_R8_robustness_paired_delta_summary.csv"
    patient_delta_path = args.output_dir / "aux_pdR2_R8_robustness_patient_level_delta_vs_single.csv"
    json_path = args.output_dir / "aux_pdR2_R8_robustness_summary.json"

    df.to_csv(per_slice_path, index=False)
    summary.to_csv(summary_path, index=False)
    patient_df.to_csv(patient_path, index=False)
    delta_df.to_csv(delta_path, index=False)
    delta_summary.to_csv(delta_summary_path, index=False)
    patient_delta_df.to_csv(patient_delta_path, index=False)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "condition_summary": summary.to_dict(orient="records"),
                "paired_delta_summary": delta_summary.to_dict(orient="records"),
            },
            f,
            indent=2,
        )

    print("=" * 80)
    print("Robustness summary")
    print("=" * 80)
    print(summary.to_string(index=False))
    print("=" * 80)
    print("Paired delta vs single baseline")
    print("Positive delta means better than single for all metrics; error metrics are inverted as single - aux.")
    print("=" * 80)
    print(delta_summary.to_string(index=False))
    print(f"Saved per-slice: {per_slice_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved patient-level: {patient_path}")
    print(f"Saved paired deltas: {delta_path}")
    print(f"Saved paired delta summary: {delta_summary_path}")
    print(f"Saved patient-level deltas: {patient_delta_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
