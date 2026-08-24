"""
QC visualization for PD / PD-FS matched pairs.

Usage:
    python qc_visualize_pairs.py [--n_pairs 30] [--seed 42] [--out_dir <dir>]

Filters:
  - phase-encoding dimension == 368 or 372 (last dim of kspace shape)
  - k-space shapes of PD and PD-FS are identical
  - at most one scan per patient_id

Outputs:
  - PNG per pair  (3-panel: PD | PD-FS | |diff|)
  - qc_summary.csv
"""

import argparse
import ast
import csv
import os
import random

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def parse_shape(s):
    """Parse a string like '(36, 640, 372)' to a tuple of ints."""
    return tuple(ast.literal_eval(s))


def load_center_rss(path):
    """Return the center slice from reconstruction_rss, as float32 array."""
    with h5py.File(path, "r") as f:
        rss = f["reconstruction_rss"][:]          # (n_slices, H, W)
    mid = rss.shape[0] // 2
    return rss[mid].astype(np.float32)


def normalize(img):
    """Scale image to [0, 1]."""
    lo, hi = img.min(), img.max()
    if hi > lo:
        return (img - lo) / (hi - lo)
    return np.zeros_like(img)


# ---------------------------------------------------------------------------
# QC checks
# ---------------------------------------------------------------------------

def check_fat_suppression(pd_img, pdfs_img):
    """PDFS median brightness should be < 0.95 × PD median."""
    pd_med  = np.median(pd_img)
    fs_med  = np.median(pdfs_img)
    ratio   = fs_med / pd_med if pd_med > 0 else 1.0
    ok      = ratio < 0.95
    return ok, ratio


def check_motion_artifact(img):
    """Flag if signal in the outer 5% border is suspiciously high (>2× interior mean)."""
    H, W      = img.shape
    bh        = max(1, int(H * 0.05))
    bw        = max(1, int(W * 0.05))
    border    = np.concatenate([
        img[:bh, :].ravel(),
        img[-bh:, :].ravel(),
        img[bh:-bh, :bw].ravel(),
        img[bh:-bh, -bw:].ravel(),
    ])
    interior  = img[bh:-bh, bw:-bw]
    int_mean  = interior.mean()
    bord_mean = border.mean()
    ratio     = bord_mean / int_mean if int_mean > 0 else 0.0
    ok        = ratio <= 2.0
    return ok, ratio


def check_metal_artifact(img_norm):
    """Flag if >1 % of pixels are saturated (== 1.0 after normalisation)."""
    sat_frac = float((img_norm >= 1.0).mean())
    ok       = sat_frac <= 0.01
    return ok, sat_frac


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_csv", default="/rds/general/user/ah725/home/fastmri_pipeline/metadata/singlecoil_train_pairs.csv")
    parser.add_argument("--n_pairs",   type=int, default=30)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--out_dir",   default="/rds/general/user/ah725/home/fastmri_pipeline/outputs/qc_pairs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)

    # ------------------------------------------------------------------
    # 1. load & filter CSV
    # ------------------------------------------------------------------
    with open(args.pairs_csv, newline="") as fh:
        reader  = csv.DictReader(fh)
        all_rows = list(reader)

    valid = []
    for row in all_rows:
        pd_shape   = parse_shape(row["pd_kspace_shape"])
        pdfs_shape = parse_shape(row["pdfs_kspace_shape"])
        pe_dim     = pd_shape[-1]                       # last dim = phase-encoding
        if pe_dim not in (368, 372):
            continue
        if pd_shape != pdfs_shape:
            continue
        valid.append(row)

    print(f"Valid pairs after filtering: {len(valid)}")

    # ------------------------------------------------------------------
    # 2. sample – at most one scan per patient
    # ------------------------------------------------------------------
    random.shuffle(valid)
    seen_patients = set()
    sampled = []
    for row in valid:
        pid = row["patient_id"]
        if pid in seen_patients:
            continue
        seen_patients.add(pid)
        sampled.append(row)
        if len(sampled) >= args.n_pairs:
            break

    print(f"Sampled {len(sampled)} pairs (up to {args.n_pairs} requested)")

    # ------------------------------------------------------------------
    # 3. generate figures + QC
    # ------------------------------------------------------------------
    summary_rows = []

    for idx, row in enumerate(sampled):
        pd_path   = row["pd_path"]
        pdfs_path = row["pdfs_path"]
        pd_file   = row["pd_file"]
        pdfs_file = row["pdfs_file"]

        pd_raw   = load_center_rss(pd_path)
        pdfs_raw = load_center_rss(pdfs_path)

        pd_norm   = normalize(pd_raw)
        pdfs_norm = normalize(pdfs_raw)
        diff      = np.abs(pd_norm - pdfs_norm)

        # --- QC ---
        fat_ok,  fat_ratio  = check_fat_suppression(pd_raw, pdfs_raw)
        mot_ok,  mot_ratio  = check_motion_artifact(pd_norm)
        met_ok,  met_frac   = check_metal_artifact(pd_norm)

        # also check PD-FS for motion / metal
        mot_ok_fs, mot_ratio_fs = check_motion_artifact(pdfs_norm)
        met_ok_fs, met_frac_fs  = check_metal_artifact(pdfs_norm)

        any_warn = not all([fat_ok, mot_ok, mot_ok_fs, met_ok, met_ok_fs])
        warn_tag = "_WARN" if any_warn else ""

        base_name = f"pair{idx+1:03d}_{pd_file.replace('.h5','')}_{pdfs_file.replace('.h5','')}{warn_tag}.png"
        out_path  = os.path.join(args.out_dir, base_name)

        # --- figure ---
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(pd_norm,   cmap="gray", vmin=0, vmax=1)
        axes[0].set_title(f"PD\n{pd_file}", fontsize=8)
        axes[0].axis("off")

        axes[1].imshow(pdfs_norm, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title(f"PD-FS\n{pdfs_file}", fontsize=8)
        axes[1].axis("off")

        im = axes[2].imshow(diff, cmap="hot", vmin=0, vmax=diff.max() if diff.max() > 0 else 1)
        axes[2].set_title("|PD − PD-FS|", fontsize=8)
        axes[2].axis("off")
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

        warn_lines = []
        if not fat_ok:
            warn_lines.append(f"fat_ratio={fat_ratio:.3f} (>=0.95)")
        if not mot_ok:
            warn_lines.append(f"PD_border_ratio={mot_ratio:.2f}")
        if not mot_ok_fs:
            warn_lines.append(f"PDFS_border_ratio={mot_ratio_fs:.2f}")
        if not met_ok:
            warn_lines.append(f"PD_sat={met_frac:.3f}")
        if not met_ok_fs:
            warn_lines.append(f"PDFS_sat={met_frac_fs:.3f}")

        sup_title = f"Pair {idx+1}  patient={row['patient_id'][:8]}…"
        if warn_lines:
            sup_title += "\nWARN: " + " | ".join(warn_lines)
        fig.suptitle(sup_title, fontsize=8, color="red" if any_warn else "black")

        plt.tight_layout()
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        summary_rows.append({
            "pair_idx":        idx + 1,
            "pd_file":         pd_file,
            "pdfs_file":       pdfs_file,
            "patient_id":      row["patient_id"],
            "kspace_shape":    row["pd_kspace_shape"],
            "fat_ok":          fat_ok,
            "fat_ratio":       f"{fat_ratio:.4f}",
            "motion_pd_ok":    mot_ok,
            "motion_pd_ratio": f"{mot_ratio:.4f}",
            "motion_fs_ok":    mot_ok_fs,
            "motion_fs_ratio": f"{mot_ratio_fs:.4f}",
            "metal_pd_ok":     met_ok,
            "metal_pd_frac":   f"{met_frac:.4f}",
            "metal_fs_ok":     met_ok_fs,
            "metal_fs_frac":   f"{met_frac_fs:.4f}",
            "any_warn":        any_warn,
            "png_file":        base_name,
        })

        status = "WARN" if any_warn else "ok"
        print(f"  [{idx+1:3d}/{len(sampled)}] {status:4s}  {base_name}")

    # ------------------------------------------------------------------
    # 4. write summary CSV
    # ------------------------------------------------------------------
    summary_path = os.path.join(args.out_dir, "qc_summary.csv")
    fieldnames   = list(summary_rows[0].keys())
    with open(summary_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    n_warn = sum(1 for r in summary_rows if r["any_warn"])
    print(f"\nDone. {n_warn}/{len(sampled)} pairs flagged.")
    print(f"PNGs saved to: {args.out_dir}")
    print(f"Summary:       {summary_path}")


if __name__ == "__main__":
    main()
