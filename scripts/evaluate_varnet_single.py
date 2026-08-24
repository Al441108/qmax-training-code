#!/usr/bin/env python3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim_fn

import fastmri
from fastmri.models import VarNet
from src.dataset_paired_multicoil import PairedMulticoilDataset


# -------------------------
# Metrics
# -------------------------

def nmse(pred, target):
    pred = np.asarray(pred)
    target = np.asarray(target)
    denom = np.sum(np.abs(target) ** 2)
    if denom < 1e-12:
        return np.nan
    return float(np.sum(np.abs(pred - target) ** 2) / denom)


def psnr(pred, target):
    pred = np.asarray(pred)
    target = np.asarray(target)
    mse = np.mean((pred - target) ** 2)
    if mse < 1e-12:
        return np.inf

    data_range = float(target.max() - target.min())
    if data_range < 1e-12:
        data_range = float(target.max())
    if data_range < 1e-12:
        return np.nan

    return float(20 * np.log10(data_range) - 10 * np.log10(mse))


def ssim_metric(pred, target):
    pred = np.asarray(pred)
    target = np.asarray(target)

    data_range = float(target.max() - target.min())
    if data_range < 1e-12:
        data_range = float(target.max())
    if data_range < 1e-12:
        return np.nan

    return float(ssim_fn(target, pred, data_range=data_range))


def l1_scaled(pred, target):
    scale = float(np.max(target))
    if scale < 1e-12:
        scale = 1.0
    return float(np.mean(np.abs(pred / scale - target / scale)))


# -------------------------
# Tensor/image utilities
# -------------------------

def ensure_fastmri_complex_dim(kspace):
    """
    Official fastMRI VarNet expects:
        [B, coils, H, W, 2]

    Handles:
    - complex tensor [B, coils, H, W]
    - split-complex tensor [B, coils, H, W, 2]
    - possible [B, H, W, coils, 2]
    """
    if not torch.is_tensor(kspace):
        kspace = torch.as_tensor(kspace)

    if torch.is_complex(kspace):
        kspace = torch.view_as_real(kspace)

    if kspace.ndim == 5 and kspace.shape[-1] == 2:
        # If it looks like [B, H, W, coils, 2], move coil dim to dim=1.
        if kspace.shape[1] > 32 and kspace.shape[-2] <= 64:
            kspace = kspace.permute(0, 3, 1, 2, 4).contiguous()
        return kspace.float()

    if kspace.ndim == 4:
        print(
            "Warning: kspace has no complex dim and is not complex dtype. "
            "Adding zero imaginary channel. Please verify dataset preprocessing."
        )
        zero_imag = torch.zeros_like(kspace)
        kspace = torch.stack([kspace, zero_imag], dim=-1)
        return kspace.float()

    raise ValueError(
        f"Cannot convert kspace to fastMRI format. "
        f"shape={tuple(kspace.shape)}, dtype={kspace.dtype}"
    )


def to_numpy_image(x):
    """
    Convert tensor/array to 2D float image.

    Handles:
    - [B, H, W]
    - [B, 1, H, W]
    - [H, W]
    - complex tensors
    - split-complex tensors [..., 2]
    - [coils, H, W] using RSS-like reduction
    """
    if torch.is_tensor(x):
        x = x.detach().cpu()

        if torch.is_complex(x):
            x = torch.abs(x)

        if x.ndim >= 1 and x.shape[-1] == 2:
            x = torch.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2)

        x = x.squeeze().numpy()

    x = np.asarray(x)

    if np.iscomplexobj(x):
        x = np.abs(x)

    x = np.squeeze(x)

    if x.ndim == 3:
        # [coils, H, W] -> RSS-like magnitude
        if x.shape[0] <= 64:
            x = np.sqrt(np.sum(np.abs(x) ** 2, axis=0))
        else:
            x = x[0]

    if x.ndim != 2:
        raise ValueError(f"Expected 2D image after conversion, got shape {x.shape}")

    return x.astype(np.float32)


def center_crop_np(x, shape):
    """
    Center crop a 2D numpy array to target shape.
    """
    x = np.asarray(x)
    h, w = x.shape
    th, tw = shape

    if h < th or w < tw:
        raise ValueError(
            f"Cannot center crop shape {x.shape} to larger target shape {shape}"
        )

    top = (h - th) // 2
    left = (w - tw) // 2

    return x[top:top + th, left:left + tw]


def match_to_target_shape(image, target_shape):
    """
    Match a 2D image to target shape by center crop.
    """
    image = np.asarray(image)

    if image.shape == target_shape:
        return image

    if image.shape[0] >= target_shape[0] and image.shape[1] >= target_shape[1]:
        return center_crop_np(image, target_shape)

    raise ValueError(
        f"Image shape {image.shape} cannot be matched to target shape {target_shape}"
    )


def zero_filled_from_masked_kspace(masked_kspace):
    """
    Generate true zero-filled reconstruction from masked k-space.

    Input:
        masked_kspace: [B, coils, H, W, 2]
    Output:
        RSS image as 2D numpy array
    """
    with torch.no_grad():
        image = fastmri.ifft2c(masked_kspace)
        image_abs = fastmri.complex_abs(image)
        image_rss = fastmri.rss(image_abs, dim=1)

    return to_numpy_image(image_rss)


def fix_fastmri_mask_shape(mask, masked_kspace):
    """
    Official fastMRI VarNet expects mask:
        [B, 1, 1, PE, 1]

    In this project, raw mask can be [1, 372].
    The PE dimension is inferred from the mask itself.
    """
    if mask is None:
        raise ValueError("Mask is None. VarNet requires a sampling mask.")

    if not torch.is_tensor(mask):
        mask = torch.as_tensor(mask)

    mask = mask.to(device=masked_kspace.device)

    B = masked_kspace.shape[0]

    if mask.ndim == 5:
        if mask.shape[0] == 1 and B > 1:
            mask = mask.expand(B, -1, -1, -1, -1)
        return mask.bool()

    if mask.ndim == 1:
        pe = mask.shape[0]
        mask = mask.view(1, 1, 1, pe, 1)

    elif mask.ndim == 2:
        if mask.shape[0] == 1:
            pe = mask.shape[1]
            mask = mask.view(1, 1, 1, pe, 1)

        elif mask.shape[0] == B:
            pe = mask.shape[1]
            mask = mask.view(B, 1, 1, pe, 1)

        else:
            row_profile = mask[:, 0]
            col_profile = mask[0, :]

            if row_profile.numel() <= col_profile.numel():
                pe = col_profile.numel()
                mask = col_profile.view(1, 1, 1, pe, 1)
            else:
                pe = row_profile.numel()
                mask = row_profile.view(1, 1, 1, pe, 1)

    elif mask.ndim == 3:
        if mask.shape[0] == B:
            if mask.shape[1] == 1:
                pe = mask.shape[2]
                mask = mask[:, 0, :].view(B, 1, 1, pe, 1)
            else:
                pe = mask.shape[-1]
                mask = mask[:, 0, :].view(B, 1, 1, pe, 1)
        else:
            pe = mask.shape[-1]
            mask = mask[0, 0, :].view(1, 1, 1, pe, 1)

    else:
        raise ValueError(
            f"Unsupported mask shape {tuple(mask.shape)} for masked_kspace shape "
            f"{tuple(masked_kspace.shape)}"
        )

    if mask.shape[0] == 1 and B > 1:
        mask = mask.expand(B, -1, -1, -1, -1)

    return mask.bool()


# -------------------------
# PNG saving
# -------------------------

def normalise_for_display(x, vmax=None):
    x = np.asarray(x).astype(np.float32)
    if vmax is None:
        vmax = np.percentile(x, 99.5)
        if vmax <= 0:
            vmax = float(x.max()) if x.max() > 0 else 1.0
    return x, vmax


def save_qualitative_png(zf, pred, target, out_path, title):
    zf = to_numpy_image(zf)
    pred = to_numpy_image(pred)
    target = to_numpy_image(target)

    err = np.abs(pred - target)

    _, vmax = normalise_for_display(target)

    fig, axes = plt.subplots(1, 4, figsize=(15, 4))

    axes[0].imshow(zf, cmap="gray", vmin=0, vmax=vmax)
    axes[0].set_title("Zero-filled")

    axes[1].imshow(pred, cmap="gray", vmin=0, vmax=vmax)
    axes[1].set_title("VarNet")

    axes[2].imshow(target, cmap="gray", vmin=0, vmax=vmax)
    axes[2].set_title("Target")

    err_vmax = np.percentile(err, 99.5)
    if err_vmax <= 0:
        err_vmax = float(err.max()) if err.max() > 0 else 1.0

    im_err = axes[3].imshow(err, cmap="magma", vmin=0, vmax=err_vmax)
    axes[3].set_title("Absolute error")

    cbar = fig.colorbar(im_err, ax=axes[3], fraction=0.046, pad=0.04)
    cbar.set_label("|Prediction - Target|", rotation=270, labelpad=14)

    for ax in axes:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# -------------------------
# Batch helpers
# -------------------------

def get_batch_value(batch, key):
    if key not in batch:
        raise KeyError(
            f"Missing key '{key}' in batch. Available keys: {list(batch.keys())}"
        )
    return batch[key]


def get_optional_batch_value(batch, possible_keys, default=None):
    for key in possible_keys:
        if key in batch:
            return batch[key]
    return default


def parse_patient_id(patient_id, batch_idx):
    if patient_id is None:
        return f"unknown_{batch_idx}"

    if isinstance(patient_id, (list, tuple)):
        return str(patient_id[0])

    if torch.is_tensor(patient_id):
        return str(patient_id.detach().cpu().flatten()[0].item())

    return str(patient_id)


def parse_slice_idx(slice_idx, batch_idx):
    if slice_idx is None:
        return int(batch_idx)

    if torch.is_tensor(slice_idx):
        return int(slice_idx.detach().cpu().flatten()[0].item())

    if isinstance(slice_idx, (list, tuple)):
        return int(slice_idx[0])

    return int(slice_idx)


# -------------------------
# Model / dataset
# -------------------------

def load_model(args, device):
    model = VarNet(
        num_cascades=args.num_cascades,
        sens_chans=args.sens_chans,
        sens_pools=args.sens_pools,
        chans=args.chans,
        pools=args.pools,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("model."):
            k = k[len("model."):]
        cleaned[k] = v

    model.load_state_dict(cleaned, strict=True)
    model.eval()

    print(
        f"Loaded checkpoint with VarNet config: "
        f"num_cascades={args.num_cascades}, chans={args.chans}, pools={args.pools}, "
        f"sens_chans={args.sens_chans}, sens_pools={args.sens_pools}"
    )

    return model


def build_dataset(args):
    df = pd.read_csv(args.metadata_csv)

    if args.split == "val":
        possible_split_names = ["val", "validation"]
    else:
        possible_split_names = [args.split]

    split_df = df[df["split"].isin(possible_split_names)].copy()

    if len(split_df) == 0:
        raise ValueError(
            f"No rows found for split={args.split}. "
            f"Available split values: {sorted(df['split'].unique().tolist())}"
        )

    if "patient_id" not in split_df.columns:
        raise ValueError(
            f"'patient_id' column not found in {args.metadata_csv}. "
            f"Available columns: {list(split_df.columns)}"
        )

    patient_ids = sorted(split_df["patient_id"].astype(str).unique().tolist())

    dataset = PairedMulticoilDataset(
        metadata_csv=args.metadata_csv,
        split=args.split,
        acceleration=args.acceleration,
        patient_ids=patient_ids,
        slices_per_patient=None,
        edge_weight=1.0,
    )

    print(
        f"Loaded {args.split} dataset with {len(patient_ids)} patients "
        f"and {len(dataset)} slices"
    )

    return dataset


# -------------------------
# Summary / JSON
# -------------------------

def summarise_metrics(df):
    rows = []
    metric_cols = ["NMSE", "PSNR", "SSIM", "L1"]

    groupings = [
        ["contrast"],
        ["contrast", "is_edge"],
    ]

    for group_cols in groupings:
        grouped = df.groupby(group_cols)

        for name, g in grouped:
            if not isinstance(name, tuple):
                name = (name,)

            row = {}
            for c, v in zip(group_cols, name):
                if isinstance(v, np.bool_):
                    v = bool(v)
                row[c] = v

            row["n_slices"] = int(len(g))
            row["n_patients"] = int(g["patient_id"].nunique())

            for metric in metric_cols:
                values = g[metric].dropna().values
                if len(values) == 0:
                    row[f"{metric}_median"] = None
                    row[f"{metric}_iqr_low"] = None
                    row[f"{metric}_iqr_high"] = None
                else:
                    row[f"{metric}_median"] = float(np.median(values))
                    row[f"{metric}_iqr_low"] = float(np.percentile(values, 25))
                    row[f"{metric}_iqr_high"] = float(np.percentile(values, 75))

            rows.append(row)

    return rows


def safe_json_dump(obj, path):
    def convert(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            val = float(o)
            if np.isnan(val) or np.isinf(val):
                return None
            return val
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, float):
            if np.isnan(o) or np.isinf(o):
                return None
            return o
        return o

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=convert)


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--metadata_csv", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--contrast", type=str, choices=["pd", "pdfs"], required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--acceleration", type=int, default=4)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)

    # Matched to current trained single-contrast VarNet baseline.
    parser.add_argument("--num_cascades", type=int, default=6)
    parser.add_argument("--sens_chans", type=int, default=8)
    parser.add_argument("--sens_pools", type=int, default=4)
    parser.add_argument("--chans", type=int, default=12)
    parser.add_argument("--pools", type=int, default=4)

    parser.add_argument("--save_png_every", type=int, default=80)
    parser.add_argument("--max_png", type=int, default=12)
    parser.add_argument("--debug_shapes", action="store_true")

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    png_dir = out_dir / "qualitative_png"
    png_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = build_dataset(args)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = load_model(args, device)

    contrast = args.contrast.lower()
    masked_key = f"{contrast}_masked_kspace"
    target_key = f"{contrast}_target_raw"

    rows = []
    png_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            masked_kspace = get_batch_value(batch, masked_key).to(device)
            masked_kspace = ensure_fastmri_complex_dim(masked_kspace)

            target = get_batch_value(batch, target_key).to(device)

            mask = get_optional_batch_value(
                batch,
                [
                    f"{contrast}_mask",
                    "mask",
                    "sampling_mask",
                    "undersampling_mask",
                ],
                default=None,
            )

            if mask is None:
                raise ValueError(
                    f"No mask found in batch. Available keys: {list(batch.keys())}"
                )

            if args.debug_shapes and batch_idx == 0:
                print("Batch keys:", list(batch.keys()))
                print("masked_kspace shape after complex fix:", tuple(masked_kspace.shape))
                print("masked_kspace dtype:", masked_kspace.dtype)
                print("target shape:", tuple(target.shape))
                print("target dtype:", target.dtype)
                print("raw mask shape:", tuple(mask.shape))

            mask = fix_fastmri_mask_shape(mask, masked_kspace)

            if args.debug_shapes and batch_idx == 0:
                print("fixed mask shape:", tuple(mask.shape))
                print("fixed mask dtype:", mask.dtype)

            num_low_frequencies = get_optional_batch_value(
                batch,
                [
                    f"{contrast}_num_low_frequencies",
                    "num_low_frequencies",
                    "num_low_freqs",
                ],
                default=None,
            )

            if torch.is_tensor(num_low_frequencies):
                num_low_frequencies = num_low_frequencies.detach().cpu().flatten()
                if len(num_low_frequencies) == 1:
                    num_low_frequencies = int(num_low_frequencies.item())

            pred = model(masked_kspace, mask, num_low_frequencies)

            pred_np = to_numpy_image(pred)
            target_np = to_numpy_image(target)

            # Correct alignment: center crop VarNet output to target size.
            if pred_np.shape != target_np.shape:
                pred_np = match_to_target_shape(pred_np, target_np.shape)

            # True zero-filled reconstruction from masked k-space.
            zf = get_optional_batch_value(
                batch,
                [
                    f"{contrast}_zero_filled",
                    f"{contrast}_zf",
                    "zero_filled",
                    "zf",
                ],
                default=None,
            )

            if zf is None:
                zf_np = zero_filled_from_masked_kspace(masked_kspace)
            else:
                zf_np = to_numpy_image(zf)

            if zf_np.shape != target_np.shape:
                zf_np = match_to_target_shape(zf_np, target_np.shape)

            patient_id = get_optional_batch_value(
                batch,
                ["patient_id", "fname", "file_id"],
                default=None,
            )

            slice_idx = get_optional_batch_value(
                batch,
                ["slice_idx", "slice_num", "slice"],
                default=None,
            )

            patient_id_val = parse_patient_id(patient_id, batch_idx)
            slice_idx_val = parse_slice_idx(slice_idx, batch_idx)
            is_edge = slice_idx_val == 0

            row = {
                "batch_idx": int(batch_idx),
                "patient_id": patient_id_val,
                "slice_idx": int(slice_idx_val),
                "contrast": contrast,
                "split": args.split,
                "acceleration": int(args.acceleration),
                "is_edge": bool(is_edge),
                "NMSE": nmse(pred_np, target_np),
                "PSNR": psnr(pred_np, target_np),
                "SSIM": ssim_metric(pred_np, target_np),
                "L1": l1_scaled(pred_np, target_np),
            }

            rows.append(row)

            # Save a controlled subset of qualitative examples.
            # Do not force saving all edge slices.
            if args.save_png_every > 0 and png_count < args.max_png:
                should_save = batch_idx % args.save_png_every == 0

                if should_save:
                    png_name = (
                        f"{contrast}_batch{batch_idx:05d}_"
                        f"patient_{patient_id_val}_slice{slice_idx_val:03d}_"
                        f"{'edge' if is_edge else 'central'}.png"
                    )

                    save_qualitative_png(
                        zf_np,
                        pred_np,
                        target_np,
                        png_dir / png_name,
                        title=(
                            f"{contrast.upper()} | patient {patient_id_val} | "
                            f"slice {slice_idx_val} | "
                            f"{'edge' if is_edge else 'central'}"
                        ),
                    )

                    png_count += 1

            if (batch_idx + 1) % 50 == 0:
                print(f"Evaluated {batch_idx + 1}/{len(loader)} slices")

    df = pd.DataFrame(rows)

    per_slice_csv = out_dir / f"{contrast}_{args.split}_per_slice_metrics.csv"
    df.to_csv(per_slice_csv, index=False)

    patient_df = (
        df.groupby(["patient_id", "contrast", "is_edge"])[
            ["NMSE", "PSNR", "SSIM", "L1"]
        ]
        .median()
        .reset_index()
    )

    patient_csv = out_dir / f"{contrast}_{args.split}_patient_level_metrics.csv"
    patient_df.to_csv(patient_csv, index=False)

    summary = {
        "metadata_csv": args.metadata_csv,
        "checkpoint": args.checkpoint,
        "output_dir": str(out_dir),
        "contrast": contrast,
        "split": args.split,
        "acceleration": int(args.acceleration),
        "n_slices": int(len(df)),
        "n_patients": int(df["patient_id"].nunique()),
        "slice_level_summary": summarise_metrics(df),
        "patient_level_summary": summarise_metrics(patient_df),
    }

    summary_json = out_dir / f"{contrast}_{args.split}_summary.json"
    safe_json_dump(summary, summary_json)

    print("Evaluation complete.")
    print(f"Per-slice CSV: {per_slice_csv}")
    print(f"Patient-level CSV: {patient_csv}")
    print(f"Summary JSON: {summary_json}")
    print(f"Qualitative PNG dir: {png_dir}")


if __name__ == "__main__":
    main()