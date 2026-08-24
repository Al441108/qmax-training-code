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
from src.fft_utils import center_crop
from src.joint_varnet_revised import JointVarNet
from src.auxiliary_varnet import AuxPDVarNet


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", None))
        config = ckpt.get("config", {})
        if state is None:
            state = ckpt
    else:
        state = ckpt
        config = {}

    clean_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        clean_state[k] = v

    return clean_state, config


def find_checkpoint(root, R, keywords, prefer_keywords=None, exclude_keywords=None):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Checkpoint root does not exist: {root}")

    files = []
    for pattern in ["*.pt", "*.pth", "*.ckpt"]:
        files.extend(root.rglob(pattern))

    keywords = [k.lower() for k in keywords]
    prefer_keywords = [k.lower() for k in (prefer_keywords or [])]
    exclude_keywords = [k.lower() for k in (exclude_keywords or [])]

    candidates = []
    for p in files:
        s = str(p).lower()
        if f"r{R}" not in s:
            continue
        if not all(k in s for k in keywords):
            continue
        if any(k in s for k in exclude_keywords):
            continue
        if p.name not in {"model_best.pt", "best.pt"} and "best" not in p.name.lower():
            continue
        candidates.append(p)

    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found under {root} for R={R}, keywords={keywords}"
        )

    def score(p):
        s = str(p).lower()
        pref = 0
        for k in prefer_keywords:
            if k in s:
                pref -= 1
        if p.name == "model_best.pt":
            pref -= 1
        return (pref, len(str(p)), str(p))

    candidates = sorted(candidates, key=score)
    print("Selected checkpoint:", candidates[0])
    return candidates[0]


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


def prepare_one_sample(sample, device):
    pd_kspace = sample["pd_masked_kspace"].unsqueeze(0).to(device)
    pdfs_kspace = sample["pdfs_masked_kspace"].unsqueeze(0).to(device)

    if not torch.is_complex(pd_kspace):
        raise TypeError(f"Expected complex PD k-space, got {pd_kspace.dtype}")
    if not torch.is_complex(pdfs_kspace):
        raise TypeError(f"Expected complex PD-FS k-space, got {pdfs_kspace.dtype}")

    pd_kspace = torch.view_as_real(pd_kspace).float()
    pdfs_kspace = torch.view_as_real(pdfs_kspace).float()

    mask = sample["mask"].unsqueeze(0).to(device).bool()
    mask = mask[:, None, None, :, None]

    pd_target = sample["pd_target_raw"].unsqueeze(0).to(device).float()
    pdfs_target = sample["pdfs_target_raw"].unsqueeze(0).to(device).float()

    if pd_target.ndim == 4 and pd_target.shape[1] == 1:
        pd_target = pd_target.squeeze(1)
    if pdfs_target.ndim == 4 and pdfs_target.shape[1] == 1:
        pdfs_target = pdfs_target.squeeze(1)

    return pd_kspace, pdfs_kspace, mask, pd_target, pdfs_target


def to_numpy(x):
    return x.detach().cpu().float().numpy()


def compute_metrics(gt, pred):
    gt = gt.astype(np.float64)
    pred = pred.astype(np.float64)

    mse = np.mean((pred - gt) ** 2)
    denom = np.sum(gt ** 2) + 1e-12
    nmse = np.sum((pred - gt) ** 2) / denom

    data_range = float(gt.max() - gt.min())
    if data_range <= 1e-12:
        data_range = float(gt.max())
    if data_range <= 1e-12:
        data_range = 1.0

    psnr = 20.0 * np.log10(data_range / np.sqrt(mse + 1e-12))
    ssim = ssim_metric(gt, pred, data_range=data_range)

    scale = np.max(gt) + 1e-8
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


def save_metrics_csv(path, rows):
    fieldnames = ["model", "NMSE", "PSNR", "SSIM", "L1"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description="Plot same-slice comparison: single vs symmetric joint vs auxiliary PD->PD-FS."
    )

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

    if args.single_checkpoint is None:
        args.single_checkpoint = str(
            find_checkpoint(
                root="outputs/varnet_single",
                R=args.R,
                keywords=["pdfs"],
                prefer_keywords=["ep30", "bs8", "full"],
                exclude_keywords=["smoke"],
            )
        )

    if args.joint_checkpoint is None:
        args.joint_checkpoint = str(
            find_checkpoint(
                root="outputs/varnet_joint_revised",
                R=args.R,
                keywords=["joint"],
                prefer_keywords=["ep30", "adaptive", "bs4"],
                exclude_keywords=["smoke"],
            )
        )

    if args.aux_checkpoint is None:
        args.aux_checkpoint = str(
            find_checkpoint(
                root="outputs/varnet_auxiliary",
                R=args.R,
                keywords=["aux", "pdfs"],
                prefer_keywords=["ep30", "bs8", "lr1e4"],
                exclude_keywords=["smoke"],
            )
        )

    print("Single checkpoint:", args.single_checkpoint)
    print("Joint checkpoint:", args.joint_checkpoint)
    print("Aux checkpoint:", args.aux_checkpoint)

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
        available = sorted({int(r["slice_idx"]) for r in dataset.records if str(r["patient_id"]) == str(args.patient_id)})
        raise RuntimeError(
            f"Could not find patient={args.patient_id}, slice={args.slice_idx}. "
            f"Available slices for this patient include: {available[:20]}..."
        )

    sample = dataset[target_index]
    pd_kspace, pdfs_kspace, mask, pd_target, pdfs_target = prepare_one_sample(sample, device)

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

    joint_pd_pred, joint_pdfs_pred = joint_model(
        pd_masked_kspace=pd_kspace,
        pdfs_masked_kspace=pdfs_kspace,
        mask=mask,
    )

    try:
        aux_pred = aux_model(
            pdfs_masked_kspace=pdfs_kspace,
            mask=mask,
            pd_aux_image=pd_target,
        )
    except TypeError:
        aux_pred = aux_model(pdfs_kspace, mask, pd_target)

    single_pred = center_crop(single_pred, pdfs_target.shape[-2], pdfs_target.shape[-1])
    joint_pdfs_pred = center_crop(joint_pdfs_pred, pdfs_target.shape[-2], pdfs_target.shape[-1])
    aux_pred = center_crop(aux_pred, pdfs_target.shape[-2], pdfs_target.shape[-1])

    gt = to_numpy(pdfs_target[0])
    pd_ref = to_numpy(pd_target[0])
    single = to_numpy(single_pred[0])
    joint = to_numpy(joint_pdfs_pred[0])
    aux = to_numpy(aux_pred[0])

    if pd_ref.shape != gt.shape:
        pd_ref_t = center_crop(pd_target, gt.shape[-2], gt.shape[-1])
        pd_ref = to_numpy(pd_ref_t[0])

    residual_single = np.abs(single - gt)
    residual_joint = np.abs(joint - gt)
    residual_aux = np.abs(aux - gt)

    m_single = compute_metrics(gt, single)
    m_joint = compute_metrics(gt, joint)
    m_aux = compute_metrics(gt, aux)

    print("Single:", m_single)
    print("Joint:", m_joint)
    print("Aux:", m_aux)

    vmax_img = robust_vmax([gt, single, joint, aux, pd_ref], percentile=99.5)
    vmax_res = robust_vmax([residual_single, residual_joint, residual_aux], percentile=99.5)

    short_pid = str(args.patient_id)[:12]

    safe_name = f"same_slice_aux_compare_R{args.R}_patient{short_pid}_slice{args.slice_idx}"
    out_png = output_dir / f"{safe_name}.png"
    out_pdf = output_dir / f"{safe_name}.pdf"
    out_csv = output_dir / f"{safe_name}_metrics.csv"
    out_npz = output_dir / f"{safe_name}_arrays.npz"

    fig, axes = plt.subplots(2, 4, figsize=(22, 10.5))

    panels = [
        (gt, "PD-FS Ground Truth", "gray", 0, vmax_img),
        (
            single,
            "Single VarNet\n"
            f"PSNR {m_single['PSNR']:.2f} | SSIM {m_single['SSIM']:.3f}",
            "gray",
            0,
            vmax_img,
        ),
        (
            joint,
            "Symmetric Joint\n"
            f"PSNR {m_joint['PSNR']:.2f} | SSIM {m_joint['SSIM']:.3f}",
            "gray",
            0,
            vmax_img,
        ),
        (
            aux,
            "Auxiliary PD→PD-FS\n"
            f"PSNR {m_aux['PSNR']:.2f} | SSIM {m_aux['SSIM']:.3f}",
            "gray",
            0,
            vmax_img,
        ),
        (
            residual_single,
            "|Single − GT|\n"
            f"NMSE {m_single['NMSE']:.4f} | L1 {m_single['L1']:.4f}",
            "magma",
            0,
            vmax_res,
        ),
        (
            residual_joint,
            "|Joint − GT|\n"
            f"NMSE {m_joint['NMSE']:.4f} | L1 {m_joint['L1']:.4f}",
            "magma",
            0,
            vmax_res,
        ),
        (
            residual_aux,
            "|Auxiliary − GT|\n"
            f"NMSE {m_aux['NMSE']:.4f} | L1 {m_aux['L1']:.4f}",
            "magma",
            0,
            vmax_res,
        ),
        (
            pd_ref,
            "PD Auxiliary Reference\n"
            "same patient, same slice",
            "gray",
            0,
            vmax_img,
        ),
    ]

    for ax, (image, title, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=13, pad=10)
        ax.axis("off")

    fig.suptitle(
        f"PD-FS reconstruction comparison | R={args.R} | "
        f"patient={short_pid} | slice={args.slice_idx}",
        fontsize=16,
        y=0.985,
    )

    plt.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.90, wspace=0.04, hspace=0.22)

    fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    rows = [
        {"model": "single", **m_single},
        {"model": "symmetric_joint", **m_joint},
        {"model": "auxiliary", **m_aux},
    ]
    save_metrics_csv(out_csv, rows)

    np.savez_compressed(
        out_npz,
        gt=gt,
        single=single,
        joint=joint,
        auxiliary=aux,
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
