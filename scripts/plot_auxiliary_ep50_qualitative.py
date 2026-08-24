#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_metric

sys.path.append(str(Path(__file__).resolve().parents[1]))

import fastmri
from fastmri.models.varnet import VarNet

from src.dataset_paired_multicoil import PairedMulticoilDataset
from src.joint_varnet_revised import JointVarNet
from src.auxiliary_varnet import AuxPDVarNet


def load_checkpoint(path, device):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict):
        config = ckpt.get("config", {})
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    else:
        config = {}
        state = ckpt

    clean_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        clean_state[k] = v

    print(f"Loaded checkpoint: {path}")
    if isinstance(ckpt, dict):
        print(f"  epoch={ckpt.get('epoch')} | best_epoch={ckpt.get('best_epoch')}")
    return clean_state, config


def center_crop_tensor(x, target_shape):
    th, tw = int(target_shape[-2]), int(target_shape[-1])
    h, w = x.shape[-2], x.shape[-1]

    if h == th and w == tw:
        return x
    if h < th or w < tw:
        raise RuntimeError(f"Cannot crop tensor from {(h, w)} to {(th, tw)}")

    top = (h - th) // 2
    left = (w - tw) // 2
    return x[..., top:top + th, left:left + tw]


def to_numpy(x):
    return x.detach().cpu().float().numpy()


def compute_metrics(gt, pred):
    gt = gt.astype(np.float64)
    pred = pred.astype(np.float64)

    mse = np.mean((pred - gt) ** 2)
    nmse = np.sum((pred - gt) ** 2) / (np.sum(gt ** 2) + 1e-12)

    data_range = float(gt.max() - gt.min())
    if data_range <= 1e-12:
        data_range = float(gt.max())
    if data_range <= 1e-12:
        data_range = 1.0

    psnr = 20.0 * np.log10(data_range / np.sqrt(mse + 1e-12))
    ssim = ssim_metric(gt, pred, data_range=data_range)

    scale = float(np.max(gt))
    if scale < 1e-8:
        scale = 1.0
    l1 = np.mean(np.abs(pred / scale - gt / scale))

    return {
        "NMSE": float(nmse),
        "PSNR": float(psnr),
        "SSIM": float(ssim),
        "L1": float(l1),
    }


def robust_vmax(arrays, percentile=99.5):
    values = np.concatenate([a.ravel() for a in arrays])
    vmax = float(np.percentile(values, percentile))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.max(values))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    return vmax


def prepare_sample(sample, device):
    pd_kspace = sample["pd_masked_kspace"].unsqueeze(0).to(device)
    pdfs_kspace = sample["pdfs_masked_kspace"].unsqueeze(0).to(device)

    if torch.is_complex(pd_kspace):
        pd_kspace = torch.view_as_real(pd_kspace).float()
    else:
        pd_kspace = pd_kspace.float()

    if torch.is_complex(pdfs_kspace):
        pdfs_kspace = torch.view_as_real(pdfs_kspace).float()
    else:
        pdfs_kspace = pdfs_kspace.float()

    mask = sample["mask"].unsqueeze(0).to(device).bool()
    mask = mask[:, None, None, :, None]

    pd_target = sample["pd_target_raw"].unsqueeze(0).to(device).float()
    pdfs_target = sample["pdfs_target_raw"].unsqueeze(0).to(device).float()

    if pd_target.ndim == 4 and pd_target.shape[1] == 1:
        pd_target = pd_target[:, 0]
    if pdfs_target.ndim == 4 and pdfs_target.shape[1] == 1:
        pdfs_target = pdfs_target[:, 0]

    return pd_kspace, pdfs_kspace, mask, pd_target, pdfs_target


def find_single_checkpoint(R):
    base = Path("outputs/varnet_single")
    files = []
    for pattern in ["*.pt", "*.pth", "*.ckpt"]:
        files.extend(base.rglob(pattern))

    candidates = []
    for p in files:
        s = str(p).lower()
        if f"r{R}" not in s:
            continue
        if "pdfs" not in s:
            continue
        if "smoke" in s:
            continue
        if p.name != "model_best.pt" and "best" not in p.name.lower():
            continue
        candidates.append(p)

    if not candidates:
        raise FileNotFoundError(f"Cannot find single PD-FS checkpoint for R={R}")

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
    return str(candidates[0])


def default_joint_checkpoint(R):
    paths = {
        4: "outputs/varnet_joint_revised/joint_R4_jvn_adaptive_lr1e4_pilot_ep5_bs4/model_best.pt",
        6: "outputs/varnet_joint_revised/joint_R6_jvn_adaptive_lr1e4_ep30_bs4/model_best.pt",
        8: "outputs/varnet_joint_revised/joint_R8_jvn_adaptive_lr1e4_ep30_bs4/model_best.pt",
    }
    return paths[int(R)]


def default_aux_ep50_checkpoint(R):
    # These model_best.pt files are updated by the ep50 resume training.
    paths = {
        4: "outputs/varnet_auxiliary/aux_pdfs_R4_ep30_bs8_lr1e4/model_best.pt",
        6: "outputs/varnet_auxiliary/aux_pdfs_R6_ep30_bs8_lr1e4/model_best.pt",
        8: "outputs/varnet_auxiliary/aux_pdfs_R8_ep30_bs8_lr1e4/model_best.pt",
    }
    return paths[int(R)]


def make_single_model(config, args, device):
    model = VarNet(
        num_cascades=int(config.get("num_cascades", args.num_cascades)),
        sens_chans=int(config.get("sens_chans", args.sens_chans)),
        sens_pools=int(config.get("sens_pools", args.sens_pools)),
        chans=int(config.get("chans", args.chans)),
        pools=int(config.get("pools", args.pools)),
        mask_center=True,
    )
    return model.to(device)


def make_joint_model(config, args, device):
    model = JointVarNet(
        num_cascades=int(config.get("num_cascades", args.num_cascades)),
        sens_chans=int(config.get("sens_chans", args.sens_chans)),
        sens_pools=int(config.get("sens_pools", args.sens_pools)),
        chans=int(config.get("chans", args.chans)),
        pools=int(config.get("pools", args.pools)),
        mask_center=True,
        cross_fusion=config.get("cross_fusion", "concat"),
    )
    return model.to(device)


def make_aux_model(config, args, device):
    model = AuxPDVarNet(
        num_cascades=int(config.get("num_cascades", args.num_cascades)),
        sens_chans=int(config.get("sens_chans", args.sens_chans)),
        sens_pools=int(config.get("sens_pools", args.sens_pools)),
        chans=int(config.get("chans", args.chans)),
        pools=int(config.get("pools", args.pools)),
        mask_center=True,
    )
    return model.to(device)


def save_metrics_csv(path, metrics):
    rows = [
        {"model": "single", **metrics["single"]},
        {"model": "symmetric_joint", **metrics["joint"]},
        {"model": "auxiliary_ep50", **metrics["aux"]},
    ]
    fieldnames = ["model", "NMSE", "PSNR", "SSIM", "L1"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--patient_id", required=True)
    parser.add_argument("--slice_idx", type=int, required=True)
    parser.add_argument("--R", type=int, choices=[4, 6, 8], required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--single_checkpoint", default=None)
    parser.add_argument("--joint_checkpoint", default=None)
    parser.add_argument("--aux_checkpoint", default=None)

    parser.add_argument("--num_cascades", type=int, default=12)
    parser.add_argument("--chans", type=int, default=18)
    parser.add_argument("--sens_chans", type=int, default=8)
    parser.add_argument("--pools", type=int, default=4)
    parser.add_argument("--sens_pools", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=300)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    if args.single_checkpoint is None:
        args.single_checkpoint = find_single_checkpoint(args.R)
    if args.joint_checkpoint is None:
        args.joint_checkpoint = default_joint_checkpoint(args.R)
    if args.aux_checkpoint is None:
        args.aux_checkpoint = default_aux_ep50_checkpoint(args.R)

    print("Single checkpoint:", args.single_checkpoint)
    print("Joint checkpoint:", args.joint_checkpoint)
    print("Aux ep50 checkpoint:", args.aux_checkpoint)

    dataset = PairedMulticoilDataset(
        metadata_csv=args.metadata_csv,
        split=args.split,
        acceleration=args.R,
        patient_ids=[args.patient_id],
        slices_per_patient=None,
        edge_weight=1.0,
    )

    target_index = None
    for i, record in enumerate(dataset.records):
        if str(record["patient_id"]) == str(args.patient_id) and int(record["slice_idx"]) == int(args.slice_idx):
            target_index = i
            break

    if target_index is None:
        available = sorted({int(r["slice_idx"]) for r in dataset.records})
        raise RuntimeError(
            f"Could not find patient={args.patient_id}, slice={args.slice_idx}. "
            f"Available slices: {available[:50]}"
        )

    sample = dataset[target_index]
    pd_kspace, pdfs_kspace, mask, pd_target, pdfs_target = prepare_sample(sample, device)

    single_state, single_config = load_checkpoint(args.single_checkpoint, device)
    joint_state, joint_config = load_checkpoint(args.joint_checkpoint, device)
    aux_state, aux_config = load_checkpoint(args.aux_checkpoint, device)

    single_model = make_single_model(single_config, args, device)
    joint_model = make_joint_model(joint_config, args, device)
    aux_model = make_aux_model(aux_config, args, device)

    single_model.load_state_dict(single_state, strict=True)
    joint_model.load_state_dict(joint_state, strict=True)
    aux_model.load_state_dict(aux_state, strict=True)

    single_model.eval()
    joint_model.eval()
    aux_model.eval()

    single_pred = single_model(pdfs_kspace, mask)

    _, joint_pdfs_pred = joint_model(
        pd_masked_kspace=pd_kspace,
        pdfs_masked_kspace=pdfs_kspace,
        mask=mask,
    )

    aux_pred = aux_model(
        pdfs_masked_kspace=pdfs_kspace,
        mask=mask,
        pd_aux_image=pd_target,
    )

    single_pred = center_crop_tensor(single_pred, pdfs_target.shape[-2:])
    joint_pdfs_pred = center_crop_tensor(joint_pdfs_pred, pdfs_target.shape[-2:])
    aux_pred = center_crop_tensor(aux_pred, pdfs_target.shape[-2:])
    pd_target = center_crop_tensor(pd_target, pdfs_target.shape[-2:])

    gt = to_numpy(pdfs_target[0])
    pd_ref = to_numpy(pd_target[0])
    single = to_numpy(single_pred[0])
    joint = to_numpy(joint_pdfs_pred[0])
    aux = to_numpy(aux_pred[0])

    residual_single = np.abs(single - gt)
    residual_joint = np.abs(joint - gt)
    residual_aux = np.abs(aux - gt)

    metrics = {
        "single": compute_metrics(gt, single),
        "joint": compute_metrics(gt, joint),
        "aux": compute_metrics(gt, aux),
    }

    print("Single:", metrics["single"])
    print("Joint:", metrics["joint"])
    print("Aux ep50:", metrics["aux"])

    vmax_recon = robust_vmax([gt, single, joint, aux], percentile=99.5)
    vmax_pd = robust_vmax([pd_ref], percentile=99.5)
    vmax_res = robust_vmax([residual_single, residual_joint, residual_aux], percentile=99.5)

    short_pid = str(args.patient_id)[:12]
    safe_name = f"aux_ep50_qual_R{args.R}_patient{short_pid}_slice{args.slice_idx}"

    out_png = output_dir / f"{safe_name}.png"
    out_pdf = output_dir / f"{safe_name}.pdf"
    out_csv = output_dir / f"{safe_name}_metrics.csv"
    out_npz = output_dir / f"{safe_name}_arrays.npz"

    fig, axes = plt.subplots(2, 4, figsize=(22, 10.5))

    panels = [
        (
            gt,
            "PD-FS Ground Truth",
            "gray",
            0,
            vmax_recon,
        ),
        (
            single,
            "Single VarNet\n"
            f"PSNR {metrics['single']['PSNR']:.2f} | SSIM {metrics['single']['SSIM']:.3f}",
            "gray",
            0,
            vmax_recon,
        ),
        (
            joint,
            "Symmetric Joint\n"
            f"PSNR {metrics['joint']['PSNR']:.2f} | SSIM {metrics['joint']['SSIM']:.3f}",
            "gray",
            0,
            vmax_recon,
        ),
        (
            aux,
            "AuxPDVarNet ep50\n"
            f"PSNR {metrics['aux']['PSNR']:.2f} | SSIM {metrics['aux']['SSIM']:.3f}",
            "gray",
            0,
            vmax_recon,
        ),
        (
            residual_single,
            "|Single - GT|\n"
            f"NMSE {metrics['single']['NMSE']:.4f} | L1 {metrics['single']['L1']:.4f}",
            "magma",
            0,
            vmax_res,
        ),
        (
            residual_joint,
            "|Joint - GT|\n"
            f"NMSE {metrics['joint']['NMSE']:.4f} | L1 {metrics['joint']['L1']:.4f}",
            "magma",
            0,
            vmax_res,
        ),
        (
            residual_aux,
            "|Aux ep50 - GT|\n"
            f"NMSE {metrics['aux']['NMSE']:.4f} | L1 {metrics['aux']['L1']:.4f}",
            "magma",
            0,
            vmax_res,
        ),
        (
            pd_ref,
            "Full PD Auxiliary\n"
            "same patient, same slice",
            "gray",
            0,
            vmax_pd,
        ),
    ]

    for ax, (image, title, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=13, pad=10)
        ax.axis("off")

    fig.suptitle(
        f"Epoch-50 full-PD auxiliary qualitative comparison | R={args.R} | "
        f"patient={short_pid} | slice={args.slice_idx}",
        fontsize=16,
        y=0.985,
    )

    plt.subplots_adjust(
        left=0.02,
        right=0.98,
        bottom=0.03,
        top=0.90,
        wspace=0.04,
        hspace=0.22,
    )

    fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    save_metrics_csv(out_csv, metrics)
    np.savez_compressed(
        out_npz,
        gt=gt,
        single=single,
        joint=joint,
        auxiliary_ep50=aux,
        pd_reference=pd_ref,
        residual_single=residual_single,
        residual_joint=residual_joint,
        residual_aux=residual_aux,
    )

    print("=" * 80)
    print("Saved:")
    print(out_png)
    print(out_pdf)
    print(out_csv)
    print(out_npz)
    print("=" * 80)


if __name__ == "__main__":
    main()
