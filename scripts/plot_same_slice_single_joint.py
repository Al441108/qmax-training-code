#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data._utils.collate import default_collate

try:
    from skimage.metrics import structural_similarity as ssim_fn
except Exception:
    ssim_fn = None

from src.dataset_paired_multicoil import PairedMulticoilDataset
from src.joint_varnet_revised import JointVarNet

# Most likely import for your single VarNet.
# If this fails, check your old evaluate_varnet_single.py import line.
try:
    from fastmri.models.varnet import VarNet
except Exception:
    from src.varnet import VarNet


def normalise_mask(mask: torch.Tensor) -> torch.Tensor:
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
    return mask.bool()


def prepare_batch(batch: Dict, device: torch.device):
    pd_kspace = batch["pd_masked_kspace"].to(device)
    pdfs_kspace = batch["pdfs_masked_kspace"].to(device)
    mask = batch["mask"].to(device)

    if torch.is_complex(pd_kspace):
        pd_kspace = torch.view_as_real(pd_kspace)
    if torch.is_complex(pdfs_kspace):
        pdfs_kspace = torch.view_as_real(pdfs_kspace)

    mask = normalise_mask(mask)

    pd_target = batch["pd_target_raw"].to(device)
    pdfs_target = batch["pdfs_target_raw"].to(device)

    if pd_target.ndim == 4 and pd_target.shape[1] == 1:
        pd_target = pd_target[:, 0]
    if pdfs_target.ndim == 4 and pdfs_target.shape[1] == 1:
        pdfs_target = pdfs_target[:, 0]

    return pd_kspace, pdfs_kspace, mask, pd_target, pdfs_target


def extract_state_dict(ckpt):
    for key in ["model_state_dict", "state_dict", "model"]:
        if isinstance(ckpt, dict) and key in ckpt:
            return ckpt[key]
    if isinstance(ckpt, dict):
        # Some checkpoints are already raw state_dict-like.
        if all(isinstance(k, str) for k in ckpt.keys()):
            return ckpt
    raise RuntimeError("Could not find model state dict in checkpoint.")


def load_single_model(checkpoint_path: Path, device: torch.device, args):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    model = VarNet(
        num_cascades=int(config.get("num_cascades", args.num_cascades)),
        sens_chans=int(config.get("sens_chans", args.sens_chans)),
        sens_pools=int(config.get("sens_pools", args.sens_pools)),
        chans=int(config.get("chans", args.chans)),
        pools=int(config.get("pools", args.pools)),
    ).to(device)

    state = extract_state_dict(ckpt)

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        print("Strict loading failed for single model. Retrying with strict=False.")
        print(e)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)

    model.eval()
    print(f"Loaded single checkpoint: {checkpoint_path}")
    if isinstance(ckpt, dict):
        print(f"Single checkpoint epoch: {ckpt.get('epoch')}, best_epoch: {ckpt.get('best_epoch')}")
    return model


def load_joint_model(checkpoint_path: Path, device: torch.device, args):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    model = JointVarNet(
        num_cascades=int(config.get("num_cascades", args.num_cascades)),
        chans=int(config.get("chans", args.chans)),
        pools=int(config.get("pools", args.pools)),
        sens_chans=int(config.get("sens_chans", args.sens_chans)),
        sens_pools=int(config.get("sens_pools", args.sens_pools)),
        cross_fusion=str(config.get("cross_fusion", args.cross_fusion)),
    ).to(device)

    state = extract_state_dict(ckpt)

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        print("Strict loading failed for joint model. Retrying with strict=False.")
        print(e)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)

    model.eval()
    print(f"Loaded joint checkpoint: {checkpoint_path}")
    if isinstance(ckpt, dict):
        print(f"Joint checkpoint epoch: {ckpt.get('epoch')}, best_epoch: {ckpt.get('best_epoch')}")
    return model


def call_single_model(model, kspace, mask):
    try:
        out = model(kspace, mask)
    except TypeError:
        out = model(kspace, mask, None)

    if isinstance(out, (tuple, list)):
        out = out[0]

    return out


def center_crop_np(x: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    h, w = x.shape[-2:]
    th, tw = shape
    if h == th and w == tw:
        return x
    top = (h - th) // 2
    left = (w - tw) // 2
    return x[..., top:top + th, left:left + tw]


def align_arrays(gt, single, joint):
    h = min(gt.shape[-2], single.shape[-2], joint.shape[-2])
    w = min(gt.shape[-1], single.shape[-1], joint.shape[-1])
    gt = center_crop_np(gt, (h, w))
    single = center_crop_np(single, (h, w))
    joint = center_crop_np(joint, (h, w))
    return gt, single, joint


def nmse(target, pred):
    denom = np.linalg.norm(target) ** 2
    if denom == 0:
        return float("nan")
    return float((np.linalg.norm(target - pred) ** 2) / denom)


def psnr(target, pred):
    mse = np.mean((target - pred) ** 2)
    if mse == 0:
        return float("inf")
    data_range = float(target.max() - target.min())
    if data_range <= 0:
        data_range = float(np.max(np.abs(target)))
    if data_range <= 0:
        return float("nan")
    return float(20 * np.log10(data_range) - 10 * np.log10(mse))


def ssim_metric(target, pred):
    if ssim_fn is None:
        return float("nan")
    data_range = float(target.max() - target.min())
    if data_range <= 0:
        data_range = float(np.max(np.abs(target)))
    if data_range <= 0:
        return float("nan")
    return float(ssim_fn(target, pred, data_range=data_range))


def l1_metric(target, pred):
    scale = float(np.max(target))
    if scale < 1e-12:
        scale = 1.0
    return float(np.mean(np.abs(pred / scale - target / scale)))


def compute_metrics(gt, pred):
    return {
        "NMSE": nmse(gt, pred),
        "PSNR": psnr(gt, pred),
        "SSIM": ssim_metric(gt, pred),
        "L1": l1_metric(gt, pred),
    }


def find_dataset_index(dataset, patient_id: str, slice_idx: int):
    matches = []
    for i, rec in enumerate(dataset.records):
        pid = str(rec.get("patient_id", ""))
        sidx = int(rec.get("slice_idx", -999))
        if pid.startswith(patient_id) and sidx == slice_idx:
            matches.append(i)

    if len(matches) == 0:
        raise RuntimeError(
            f"No matching slice found for patient_id prefix={patient_id}, slice_idx={slice_idx}"
        )
    if len(matches) > 1:
        print("Warning: multiple patient_id prefix matches found. Using first match.")
        print(matches[:10])
    return matches[0]


def default_single_checkpoint(contrast: str, R: int):
    c = "pd" if contrast == "PD" else "pdfs"
    return Path(f"outputs/varnet_single/{c}_R{R}_c12_ch18_ep30_bs8/model_best.pt")


def default_joint_checkpoint(R: int):
    if R == 4:
        return Path("outputs/varnet_joint_revised/joint_R4_jvn_adaptive_lr1e4_pilot_ep5_bs4/model_best.pt")
    if R == 6:
        return Path("outputs/varnet_joint_revised/joint_R6_jvn_adaptive_lr1e4_ep30_bs4/model_best.pt")
    if R == 8:
        return Path("outputs/varnet_joint_revised/joint_R8_jvn_adaptive_lr1e4_ep30_bs4/model_best.pt")
    raise ValueError(f"Unsupported R: {R}")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--metadata_csv", type=Path, required=True)
    parser.add_argument("--patient_id", type=str, required=True, help="Full patient_id or prefix.")
    parser.add_argument("--slice_idx", type=int, required=True)
    parser.add_argument("--contrast", type=str, choices=["PD", "PD-FS"], required=True)
    parser.add_argument("--R", type=int, choices=[4, 6, 8], required=True)

    parser.add_argument("--single_checkpoint", type=Path, default=None)
    parser.add_argument("--joint_checkpoint", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/figures/single_vs_joint_same_slice"))

    parser.add_argument("--num_cascades", type=int, default=12)
    parser.add_argument("--chans", type=int, default=18)
    parser.add_argument("--sens_chans", type=int, default=8)
    parser.add_argument("--pools", type=int, default=4)
    parser.add_argument("--sens_pools", type=int, default=4)
    parser.add_argument("--cross_fusion", type=str, default="concat")

    args = parser.parse_args()

    if args.single_checkpoint is None:
        args.single_checkpoint = default_single_checkpoint(args.contrast, args.R)
    if args.joint_checkpoint is None:
        args.joint_checkpoint = default_joint_checkpoint(args.R)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("Same-slice single vs joint reconstruction figure")
    print("=" * 80)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"R: {args.R}")
    print(f"Contrast: {args.contrast}")
    print(f"Patient prefix: {args.patient_id}")
    print(f"Slice index: {args.slice_idx}")
    print(f"Single checkpoint: {args.single_checkpoint}")
    print(f"Joint checkpoint: {args.joint_checkpoint}")
    print("=" * 80)

    dataset = PairedMulticoilDataset(
        metadata_csv=str(args.metadata_csv),
        split="val",
        acceleration=args.R,
    )

    ds_idx = find_dataset_index(dataset, args.patient_id, args.slice_idx)
    print(f"Dataset index: {ds_idx}")

    sample = dataset[ds_idx]
    batch = default_collate([sample])

    pd_kspace, pdfs_kspace, mask, pd_target, pdfs_target = prepare_batch(batch, device)

    single_model = load_single_model(args.single_checkpoint, device, args)
    joint_model = load_joint_model(args.joint_checkpoint, device, args)

    if args.contrast == "PD":
        single_kspace = pd_kspace
        gt_tensor = pd_target
    else:
        single_kspace = pdfs_kspace
        gt_tensor = pdfs_target

    single_pred = call_single_model(single_model, single_kspace, mask)

    pd_joint_pred, pdfs_joint_pred = joint_model(pd_kspace, pdfs_kspace, mask)
    joint_pred = pd_joint_pred if args.contrast == "PD" else pdfs_joint_pred

    gt = gt_tensor[0].detach().cpu().float().numpy()
    single = single_pred[0].detach().cpu().float().numpy()
    joint = joint_pred[0].detach().cpu().float().numpy()

    gt, single, joint = align_arrays(gt, single, joint)

    m_single = compute_metrics(gt, single)
    m_joint = compute_metrics(gt, joint)

    residual_single = np.abs(single - gt)
    residual_joint = np.abs(joint - gt)

    vmax_img = np.percentile(gt, 99.5)
    if vmax_img <= 0:
        vmax_img = gt.max()

    vmax_res = np.percentile(np.concatenate([residual_single.ravel(), residual_joint.ravel()]), 99.5)
    if vmax_res <= 0:
        vmax_res = max(residual_single.max(), residual_joint.max())

    args.output_dir.mkdir(parents=True, exist_ok=True)

    safe_pid = str(batch["patient_id"][0])[:12]
    safe_contrast = args.contrast.replace("-", "")
    out_png = args.output_dir / f"same_slice_R{args.R}_{safe_contrast}_patient{safe_pid}_slice{args.slice_idx}.png"
    out_csv = args.output_dir / f"same_slice_R{args.R}_{safe_contrast}_patient{safe_pid}_slice{args.slice_idx}_metrics.csv"
    out_npz = args.output_dir / f"same_slice_R{args.R}_{safe_contrast}_patient{safe_pid}_slice{args.slice_idx}_arrays.npz"

    fig, axes = plt.subplots(1, 5, figsize=(24, 5.2))

    short_pid = str(batch["patient_id"][0])[:12]

    axes[0].imshow(gt, cmap="gray", vmin=0, vmax=vmax_img)
    axes[0].set_title("Ground truth", fontsize=13)

    axes[1].imshow(single, cmap="gray", vmin=0, vmax=vmax_img)
    axes[1].set_title(
        "Single VarNet\n"
        f"PSNR {m_single['PSNR']:.2f} | SSIM {m_single['SSIM']:.3f}",
        fontsize=12
    )

    axes[2].imshow(joint, cmap="gray", vmin=0, vmax=vmax_img)
    axes[2].set_title(
        "Joint VarNet\n"
        f"PSNR {m_joint['PSNR']:.2f} | SSIM {m_joint['SSIM']:.3f}",
        fontsize=12
    )

    axes[3].imshow(residual_single, cmap="magma", vmin=0, vmax=vmax_res)
    axes[3].set_title(
        "|Single - GT|\n"
        f"NMSE {m_single['NMSE']:.4f} | L1 {m_single['L1']:.4f}",
        fontsize=12
    )

    axes[4].imshow(residual_joint, cmap="magma", vmin=0, vmax=vmax_res)
    axes[4].set_title(
        "|Joint - GT|\n"
        f"NMSE {m_joint['NMSE']:.4f} | L1 {m_joint['L1']:.4f}",
        fontsize=12
    )

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"{args.contrast} same-slice comparison | R={args.R} | "
        f"patient={short_pid} | slice={args.slice_idx}",
        fontsize=14,
        y=1.02
    )

    plt.subplots_adjust(wspace=0.05, top=0.80)
    plt.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close(fig)

    metrics_df = pd.DataFrame([
        {
            "model": "single",
            "patient_id": batch["patient_id"][0],
            "slice_idx": args.slice_idx,
            "contrast": args.contrast,
            "R": args.R,
            **m_single,
        },
        {
            "model": "joint",
            "patient_id": batch["patient_id"][0],
            "slice_idx": args.slice_idx,
            "contrast": args.contrast,
            "R": args.R,
            **m_joint,
        },
    ])

    metrics_df.to_csv(out_csv, index=False)
    np.savez_compressed(out_npz, gt=gt, single=single, joint=joint,
                        residual_single=residual_single, residual_joint=residual_joint)

    print("=" * 80)
    print("Metrics")
    print("=" * 80)
    print(metrics_df.to_string(index=False))
    print("=" * 80)
    print(f"Saved figure: {out_png}")
    print(f"Saved metrics: {out_csv}")
    print(f"Saved arrays: {out_npz}")


if __name__ == "__main__":
    main()
