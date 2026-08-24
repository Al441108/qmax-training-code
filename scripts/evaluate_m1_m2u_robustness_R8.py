#!/usr/bin/env python3
"""
Unified R=8 robustness evaluation:
  - Single PD-FS VarNet baseline
  - M1 direct auxiliary PD R=2 -> PD-FS VarNet
  - M2-U ungated multi-scale auxiliary PD R=2 -> PD-FS VarNet

Conditions:
  1. no auxiliary (single baseline)
  2. correct PD R=2 auxiliary
  3. zero / missing PD auxiliary
  4. shifted PD auxiliary: 2, 4, 8 px with zero padding
  5. wrong-patient PD auxiliary

All models use the same validation set, fixed masks, targets, and metrics.
Positive paired delta means better than the single baseline for every metric.
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
from src.dataset_paired_multicoil_aux_pd_r2 import (
    PairedMulticoilAuxPDToPDFSDataset,
)
from src.auxiliary_varnet import AuxPDVarNet
from src.m2u_auxiliary_varnet_optimized import M2UAuxPDVarNet


METRICS = ["NMSE", "PSNR", "SSIM", "L1"]
HIGHER_IS_BETTER = {
    "NMSE": False,
    "PSNR": True,
    "SSIM": True,
    "L1": False,
}


class ShapeBucketBatchSampler(Sampler[List[int]]):
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
            with h5py.File(record["pdfs_path"], "r") as hf:
                shape = hf["kspace"].shape
                key = (int(shape[-2]), int(shape[-1]))
            buckets.setdefault(key, []).append(idx)

        self.buckets = buckets
        self.num_batches = sum(
            math.ceil(len(v) / self.batch_size)
            for v in buckets.values()
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
                all_batches.append(
                    indices[start:start + self.batch_size]
                )

        if self.shuffle:
            rng.shuffle(all_batches)

        yield from all_batches

    def __len__(self):
        return self.num_batches


class IndexedDataset:
    def __init__(self, dataset):
        self.dataset = dataset
        self.records = dataset.records

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        item["sample_idx"] = int(idx)
        return item


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
            f"Cannot crop tensor from {(h, w)} "
            f"to {(crop_h, crop_w)}"
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
    return float(
        20 * np.log10(data_range)
        - 10 * np.log10(mse)
    )


def ssim_metric(target: np.ndarray, pred: np.ndarray) -> float:
    if ssim_fn is None:
        return float("nan")
    data_range = float(target.max() - target.min())
    if data_range <= 0:
        data_range = float(np.max(np.abs(target)))
    if data_range <= 0:
        return float("nan")
    return float(
        ssim_fn(target, pred, data_range=data_range)
    )


def l1_metric(target: np.ndarray, pred: np.ndarray) -> float:
    scale = float(np.max(target))
    if scale < 1e-12:
        scale = 1.0
    return float(
        np.mean(np.abs(pred / scale - target / scale))
    )


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
        return torch.load(
            checkpoint_path,
            map_location=device,
        )


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
        f"Cannot find state dict in {checkpoint_path}"
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

    state = clean_state_dict(
        extract_state_dict(ckpt, checkpoint_path)
    )
    model.load_state_dict(state, strict=True)
    model.eval()

    print(
        f"Loaded single: {checkpoint_path} | "
        f"epoch={ckpt.get('epoch')} | "
        f"best_epoch={ckpt.get('best_epoch')}"
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

    state = clean_state_dict(
        extract_state_dict(ckpt, checkpoint_path)
    )
    model.load_state_dict(state, strict=True)
    model.eval()

    print(
        f"Loaded M1: {checkpoint_path} | "
        f"epoch={ckpt.get('epoch')} | "
        f"best_epoch={ckpt.get('best_epoch')}"
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
        initial_aux_alpha=float(
            config.get("initial_aux_alpha", 0.1)
        ),
    ).to(device)

    state = clean_state_dict(
        extract_state_dict(ckpt, checkpoint_path)
    )
    model.load_state_dict(state, strict=True)
    model.eval()

    print(
        f"Loaded M2-U: {checkpoint_path} | "
        f"epoch={ckpt.get('epoch')} | "
        f"best_epoch={ckpt.get('best_epoch')}"
    )
    print(
        "M2-U fusion diagnostics:",
        json.dumps(model.fusion_diagnostics(), indent=2),
    )
    return model


def prepare_batch(
    batch: Dict,
    device: torch.device,
):
    pdfs_kspace = batch["pdfs_masked_kspace"].to(
        device,
        non_blocking=True,
    )

    if torch.is_complex(pdfs_kspace):
        pdfs_kspace = torch.view_as_real(
            pdfs_kspace
        ).float()
    else:
        pdfs_kspace = pdfs_kspace.float()

    mask = batch["mask"].to(
        device,
        non_blocking=True,
    )

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
        raise RuntimeError(
            f"Unexpected mask shape: {tuple(mask.shape)}"
        )

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
            f"Expected PD auxiliary [B,H,W], "
            f"got {tuple(pd_aux.shape)}"
        )

    if target.ndim != 3:
        raise RuntimeError(
            f"Expected target [B,H,W], got {tuple(target.shape)}"
        )

    return pdfs_kspace, mask, pd_aux, target


def get_batch_value(
    batch: Dict,
    key: str,
    index: int,
):
    value = batch[key]
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    return value[index]


def translate_zero_pad(
    x: torch.Tensor,
    shift_y: int,
    shift_x: int,
) -> torch.Tensor:
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
        out[
            ...,
            dst_y0:dst_y1,
            dst_x0:dst_x1,
        ] = x[
            ...,
            src_y0:src_y1,
            src_x0:src_x1,
        ]

    return out


def build_wrong_patient_index(
    dataset,
) -> Dict[int, int]:
    patient_ids = [
        str(record["patient_id"])
        for record in dataset.records
    ]
    n = len(patient_ids)

    if len(set(patient_ids)) < 2:
        raise RuntimeError(
            "Wrong-patient test needs at least two patients."
        )

    mapping = {}
    start_offset = max(1, n // 3)

    for i, patient_id in enumerate(patient_ids):
        replacement = None
        for offset in range(
            start_offset,
            n + start_offset,
        ):
            j = (i + offset) % n
            if patient_ids[j] != patient_id:
                replacement = j
                break

        if replacement is None:
            raise RuntimeError(
                f"No wrong-patient replacement for index {i}"
            )

        mapping[i] = replacement

    return mapping


def sample_pd_aux_by_indices(
    dataset,
    indices: Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    pd_list = []

    for idx in indices:
        item = dataset[int(idx)]
        pd_aux = item["pd_aux_image"]

        if not torch.is_tensor(pd_aux):
            pd_aux = torch.as_tensor(pd_aux)

        pd_aux = pd_aux.float()

        if pd_aux.ndim == 3 and pd_aux.shape[0] == 1:
            pd_aux = pd_aux[0]
        elif pd_aux.ndim != 2:
            raise RuntimeError(
                f"Unexpected PD aux shape at idx={idx}: "
                f"{tuple(pd_aux.shape)}"
            )

        pd_list.append(pd_aux)

    return torch.stack(
        pd_list,
        dim=0,
    ).to(
        device,
        non_blocking=True,
    )


def rows_for_prediction(
    pred: torch.Tensor,
    target: torch.Tensor,
    batch: Dict,
    model_name: str,
    condition: str,
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
                "condition": condition,
                "checkpoint": str(checkpoint_path),
                "patient_id": str(
                    get_batch_value(
                        batch,
                        "patient_id",
                        i,
                    )
                ),
                "slice_idx": int(
                    get_batch_value(
                        batch,
                        "slice_idx",
                        i,
                    )
                ),
                "R": 8,
                "pd_aux_R": 2,
                "NMSE": nmse(t, p),
                "PSNR": psnr(t, p),
                "SSIM": ssim_metric(t, p),
                "L1": l1_metric(t, p),
                "pd_flip_lr": bool(
                    get_batch_value(
                        batch,
                        "pd_flip_lr",
                        i,
                    )
                ),
                "is_edge": bool(
                    get_batch_value(
                        batch,
                        "is_edge",
                        i,
                    )
                ),
            }
        )

    return rows


def aggregate_patient_level(
    per_slice: pd.DataFrame,
) -> pd.DataFrame:
    return (
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
        .sort_values(
            ["model", "condition", "patient_id"]
        )
        .reset_index(drop=True)
    )


def summarise(
    df: pd.DataFrame,
    unit: str,
) -> pd.DataFrame:
    rows = []

    for (model, condition), group in df.groupby(
        ["model", "condition"]
    ):
        row = {
            "model": model,
            "condition": condition,
            "R": 8,
            "pd_aux_R": 2,
            "aggregation_level": unit,
            "n": int(len(group)),
        }

        for metric in METRICS:
            values = group[metric].dropna().to_numpy()
            row[f"{metric}_mean"] = (
                float(np.mean(values))
                if len(values)
                else float("nan")
            )
            row[f"{metric}_std"] = (
                float(np.std(values, ddof=1))
                if len(values) > 1
                else float("nan")
            )
            row[f"{metric}_median"] = (
                float(np.median(values))
                if len(values)
                else float("nan")
            )
            row[f"{metric}_iqr_low"] = (
                float(np.percentile(values, 25))
                if len(values)
                else float("nan")
            )
            row[f"{metric}_iqr_high"] = (
                float(np.percentile(values, 75))
                if len(values)
                else float("nan")
            )

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["model", "condition"]
    )


def compute_paired_delta(
    source_df: pd.DataFrame,
    key_cols: List[str],
) -> pd.DataFrame:
    baseline = (
        source_df[
            (source_df["model"] == "single")
            & (source_df["condition"] == "no_auxiliary")
        ][key_cols + METRICS]
        .rename(
            columns={
                metric: f"{metric}_single"
                for metric in METRICS
            }
        )
    )

    output = []

    for model_name in ["m1", "m2u"]:
        model_df = source_df[
            source_df["model"] == model_name
        ]

        for condition in sorted(
            model_df["condition"].unique()
        ):
            current = model_df[
                model_df["condition"] == condition
            ].copy()

            merged = current.merge(
                baseline,
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

                merged[
                    f"{metric}_delta_vs_single"
                ] = delta
                merged[
                    f"{metric}_worse_than_single"
                ] = delta < 0

            output.append(merged)

    return pd.concat(output, ignore_index=True)


def summarise_delta(
    delta_df: pd.DataFrame,
    aggregation_level: str,
) -> pd.DataFrame:
    rows = []

    for (model, condition), group in delta_df.groupby(
        ["model", "condition"]
    ):
        row = {
            "model": model,
            "condition": condition,
            "baseline": "single_no_auxiliary",
            "R": 8,
            "pd_aux_R": 2,
            "aggregation_level": aggregation_level,
            "n": int(len(group)),
        }

        for metric in METRICS:
            values = group[
                f"{metric}_delta_vs_single"
            ].dropna().to_numpy()

            worse = group[
                f"{metric}_worse_than_single"
            ].dropna().to_numpy()

            row[f"{metric}_delta_mean"] = (
                float(np.mean(values))
                if len(values)
                else float("nan")
            )
            row[f"{metric}_delta_std"] = (
                float(np.std(values, ddof=1))
                if len(values) > 1
                else float("nan")
            )
            row[f"{metric}_delta_median"] = (
                float(np.median(values))
                if len(values)
                else float("nan")
            )
            row[
                f"{metric}_pct_worse_than_single"
            ] = (
                float(np.mean(worse) * 100.0)
                if len(worse)
                else float("nan")
            )

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["condition", "model"]
    )


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Unified R=8 robustness comparison for "
            "Single, M1 and M2-U."
        )
    )
    parser.add_argument(
        "--metadata_csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--single_checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--m1_checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--m2u_checkpoint",
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
    parser.add_argument(
        "--shift_pixels",
        type=int,
        nargs="+",
        default=[2, 4, 8],
    )
    args = parser.parse_args()

    for path in [
        args.metadata_csv,
        args.single_checkpoint,
        args.m1_checkpoint,
        args.m2u_checkpoint,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 88)
    print("Unified R=8 robustness: Single vs M1 vs M2-U")
    print("=" * 88)
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Metadata:", args.metadata_csv)
    print("Single:", args.single_checkpoint)
    print("M1:", args.m1_checkpoint)
    print("M2-U:", args.m2u_checkpoint)
    print("Shifts:", args.shift_pixels)
    print("Output:", args.output_dir)
    print("=" * 88)

    base_dataset = PairedMulticoilAuxPDToPDFSDataset(
        metadata_csv=str(args.metadata_csv),
        split="val",
        pdfs_acceleration=8,
        pd_aux_acceleration=2,
    )
    dataset = IndexedDataset(base_dataset)

    wrong_patient_map = build_wrong_patient_index(
        dataset
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

    single_model = load_single_model(
        args.single_checkpoint,
        device,
    )
    m1_model = load_m1_model(
        args.m1_checkpoint,
        device,
    )
    m2u_model = load_m2u_model(
        args.m2u_checkpoint,
        device,
    )

    rows = []

    for batch_idx, batch in enumerate(
        loader,
        start=1,
    ):
        pdfs_kspace, mask, pd_aux, target = (
            prepare_batch(batch, device)
        )

        dataset_indices = [
            int(x) for x in batch["sample_idx"]
        ]

        single_pred = single_model(
            pdfs_kspace,
            mask,
        )
        single_pred = center_crop_tensor(
            single_pred,
            target.shape[-2],
            target.shape[-1],
        )
        rows.extend(
            rows_for_prediction(
                single_pred,
                target,
                batch,
                "single",
                "no_auxiliary",
                args.single_checkpoint,
            )
        )

        conditions = {
            "correct_pdR2_aux": pd_aux,
            "zero_missing_pd_aux": torch.zeros_like(
                pd_aux
            ),
        }

        for shift in args.shift_pixels:
            conditions[
                f"shifted_pd_aux_{int(shift)}px_zero_pad"
            ] = translate_zero_pad(
                pd_aux,
                shift_y=int(shift),
                shift_x=int(shift),
            )

        wrong_indices = [
            wrong_patient_map[index]
            for index in dataset_indices
        ]
        pd_wrong = sample_pd_aux_by_indices(
            dataset,
            wrong_indices,
            device,
        )

        if pd_wrong.shape[-2:] != pd_aux.shape[-2:]:
            raise RuntimeError(
                f"Wrong-patient shape mismatch: "
                f"{tuple(pd_wrong.shape)} vs "
                f"{tuple(pd_aux.shape)}"
            )

        conditions[
            "wrong_patient_pd_aux"
        ] = pd_wrong

        for condition, auxiliary in conditions.items():
            m1_pred = m1_model(
                pdfs_masked_kspace=pdfs_kspace,
                mask=mask,
                pd_aux_image=auxiliary,
            )
            m1_pred = center_crop_tensor(
                m1_pred,
                target.shape[-2],
                target.shape[-1],
            )
            rows.extend(
                rows_for_prediction(
                    m1_pred,
                    target,
                    batch,
                    "m1",
                    condition,
                    args.m1_checkpoint,
                )
            )

            m2u_pred = m2u_model(
                pdfs_masked_kspace=pdfs_kspace,
                mask=mask,
                pd_aux_image=auxiliary,
            )
            m2u_pred = center_crop_tensor(
                m2u_pred,
                target.shape[-2],
                target.shape[-1],
            )
            rows.extend(
                rows_for_prediction(
                    m2u_pred,
                    target,
                    batch,
                    "m2u",
                    condition,
                    args.m2u_checkpoint,
                )
            )

        if (
            batch_idx == 1
            or batch_idx % 25 == 0
            or batch_idx == len(loader)
        ):
            print(
                f"Batch {batch_idx}/{len(loader)}"
            )

    per_slice = (
        pd.DataFrame(rows)
        .sort_values(
            [
                "patient_id",
                "slice_idx",
                "condition",
                "model",
            ]
        )
        .reset_index(drop=True)
    )

    patient_level = aggregate_patient_level(
        per_slice
    )

    slice_summary = summarise(
        per_slice,
        unit="slice",
    )
    patient_summary = summarise(
        patient_level,
        unit="patient",
    )

    slice_delta = compute_paired_delta(
        per_slice,
        key_cols=[
            "patient_id",
            "slice_idx",
            "R",
        ],
    )
    patient_delta = compute_paired_delta(
        patient_level,
        key_cols=[
            "patient_id",
            "R",
        ],
    )

    slice_delta_summary = summarise_delta(
        slice_delta,
        aggregation_level="slice",
    )
    patient_delta_summary = summarise_delta(
        patient_delta,
        aggregation_level="patient",
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = {
        "per_slice_metrics.csv": per_slice,
        "patient_level_metrics.csv": patient_level,
        "slice_level_summary.csv": slice_summary,
        "patient_level_summary.csv": patient_summary,
        "paired_delta_vs_single_per_slice.csv": slice_delta,
        "paired_delta_vs_single_patient_level.csv": patient_delta,
        "paired_delta_summary_slice_level.csv": slice_delta_summary,
        "paired_delta_summary_patient_level.csv": patient_delta_summary,
    }

    for filename, dataframe in outputs.items():
        dataframe.to_csv(
            args.output_dir / filename,
            index=False,
        )

    config = {
        "metadata_csv": str(args.metadata_csv),
        "single_checkpoint": str(
            args.single_checkpoint
        ),
        "m1_checkpoint": str(
            args.m1_checkpoint
        ),
        "m2u_checkpoint": str(
            args.m2u_checkpoint
        ),
        "R": 8,
        "pd_aux_R": 2,
        "shift_pixels": [
            int(x) for x in args.shift_pixels
        ],
        "patient_aggregation": (
            "mean across slices, equal patient weight"
        ),
        "positive_delta": (
            "better than single for all metrics"
        ),
    }

    with open(
        args.output_dir / "evaluation_config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(config, file, indent=2)

    print("=" * 88)
    print("Patient-level robustness summary")
    print("=" * 88)
    print(patient_summary.to_string(index=False))
    print("=" * 88)
    print("Patient-level paired delta vs single")
    print("Positive delta = better than single")
    print("=" * 88)
    print(patient_delta_summary.to_string(index=False))
    print("=" * 88)
    print("Saved to:", args.output_dir)


if __name__ == "__main__":
    main()
