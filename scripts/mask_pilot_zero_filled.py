#!/usr/bin/env python3

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim


# ----------------------------
# Basic MRI utilities
# ----------------------------

def ifft2c(kspace):
    """
    Centered 2D inverse FFT over the last two dimensions.
    Input shape: [coils, height, width]
    Output shape: [coils, height, width]
    """
    kspace = np.fft.ifftshift(kspace, axes=(-2, -1))
    image = np.fft.ifft2(kspace, axes=(-2, -1), norm="ortho")
    image = np.fft.fftshift(image, axes=(-2, -1))
    return image


def rss_combine(coil_images):
    """
    Root-sum-of-squares coil combination.
    Input shape: [coils, height, width]
    Output shape: [height, width]
    """
    return np.sqrt(np.sum(np.abs(coil_images) ** 2, axis=0))


def center_crop_to_match(a, b):
    """
    Center crop two 2D arrays to their common minimum shape.
    """
    h = min(a.shape[-2], b.shape[-2])
    w = min(a.shape[-1], b.shape[-1])

    def crop(x):
        h0, w0 = x.shape[-2], x.shape[-1]
        top = (h0 - h) // 2
        left = (w0 - w) // 2
        return x[top:top + h, left:left + w]

    return crop(a), crop(b)


def normalise_for_display(x):
    x = np.asarray(x)
    p99 = np.percentile(x, 99)
    if p99 <= 0:
        return x
    return np.clip(x / p99, 0, 1)


def compute_metrics(pred, target):
    """
    Compute NMSE, PSNR and SSIM.
    pred and target should be 2D magnitude images.
    """
    pred, target = center_crop_to_match(pred, target)

    pred = pred.astype(np.float64)
    target = target.astype(np.float64)

    mse = np.mean((pred - target) ** 2)
    nmse = np.sum((pred - target) ** 2) / (np.sum(target ** 2) + 1e-12)

    data_range = float(target.max() - target.min())
    if data_range <= 0:
        psnr = np.nan
        ssim_val = np.nan
    else:
        psnr = 20 * np.log10(float(target.max()) / (np.sqrt(mse) + 1e-12))
        ssim_val = ssim(target, pred, data_range=data_range)

    return {
        "NMSE": nmse,
        "PSNR": psnr,
        "SSIM": ssim_val,
    }


# ----------------------------
# Mask generation
# ----------------------------

def get_center_region(num_lines, center_fraction):
    num_low = int(round(num_lines * center_fraction))
    num_low = max(num_low, 1)
    pad = (num_lines - num_low + 1) // 2
    center = np.zeros(num_lines, dtype=bool)
    center[pad:pad + num_low] = True
    return center, num_low


def make_fastmri_vd_mask(num_lines, acceleration, center_fraction, rng):
    """
    fastMRI-style variable-density random mask:
    fully sampled centre + uniform random outer lines.
    """
    center, num_low = get_center_region(num_lines, center_fraction)
    target_samples = int(round(num_lines / acceleration))
    target_samples = max(target_samples, num_low)

    mask = center.copy()
    outer_indices = np.where(~center)[0]
    remaining = target_samples - num_low

    if remaining > 0:
        chosen = rng.choice(
            outer_indices,
            size=min(remaining, len(outer_indices)),
            replace=False,
        )
        mask[chosen] = True

    return mask


def make_gaussian_vd_mask(num_lines, acceleration, center_fraction, rng, sigma_scale=0.22):
    """
    Gaussian variable-density mask:
    fully sampled centre + Gaussian-probability outer lines.
    sigma_scale controls how wide the Gaussian is relative to num_lines.
    Larger sigma_scale = more high-frequency lines.
    Smaller sigma_scale = more centre-concentrated.
    """
    center, num_low = get_center_region(num_lines, center_fraction)
    target_samples = int(round(num_lines / acceleration))
    target_samples = max(target_samples, num_low)

    mask = center.copy()
    outer_indices = np.where(~center)[0]
    remaining = target_samples - num_low

    if remaining > 0:
        ky = np.arange(num_lines)
        centre = (num_lines - 1) / 2
        sigma = sigma_scale * num_lines

        probs = np.exp(-0.5 * ((ky - centre) / sigma) ** 2)
        probs[center] = 0
        probs = probs[outer_indices]
        probs = probs / probs.sum()

        chosen = rng.choice(
            outer_indices,
            size=min(remaining, len(outer_indices)),
            replace=False,
            p=probs,
        )
        mask[chosen] = True

    return mask


def make_equispaced_mask(num_lines, acceleration, center_fraction):
    """
    Equispaced Cartesian mask:
    fully sampled centre + approximately equally spaced outer lines.
    """
    center, num_low = get_center_region(num_lines, center_fraction)
    target_samples = int(round(num_lines / acceleration))
    target_samples = max(target_samples, num_low)

    mask = center.copy()
    outer_indices = np.where(~center)[0]
    remaining = target_samples - num_low

    if remaining > 0:
        if remaining >= len(outer_indices):
            chosen = outer_indices
        else:
            positions = np.linspace(0, len(outer_indices) - 1, remaining)
            chosen = outer_indices[np.round(positions).astype(int)]
        mask[chosen] = True

    return mask


def make_mask(mask_type, num_lines, acceleration, center_fraction, seed):
    rng = np.random.default_rng(seed)

    if mask_type == "fastmri_vd":
        return make_fastmri_vd_mask(num_lines, acceleration, center_fraction, rng)

    if mask_type == "gaussian_vd":
        return make_gaussian_vd_mask(num_lines, acceleration, center_fraction, rng)

    if mask_type == "equispaced":
        return make_equispaced_mask(num_lines, acceleration, center_fraction)

    raise ValueError(f"Unknown mask_type: {mask_type}")


def apply_1d_mask(kspace_slice, mask_1d, mask_dim):
    """
    Apply 1D mask to one k-space slice.
    kspace_slice shape: [coils, height, width]

    mask_dim:
      -1 means mask along width
      -2 means mask along height
    """
    if mask_dim == -1:
        mask = mask_1d.reshape(1, 1, -1)
    elif mask_dim == -2:
        mask = mask_1d.reshape(1, -1, 1)
    else:
        raise ValueError("mask_dim must be -1 or -2")

    return kspace_slice * mask


# ----------------------------
# Data loading
# ----------------------------

def read_volume_pair(row):
    pd_path = Path(row["pd_new_path"])
    pdfs_path = Path(row["pdfs_new_path"])

    if not pd_path.exists():
        raise FileNotFoundError(f"PD file not found: {pd_path}")
    if not pdfs_path.exists():
        raise FileNotFoundError(f"PDFS file not found: {pdfs_path}")

    return pd_path, pdfs_path


def load_kspace_and_target(h5_path, target_key="reconstruction_rss"):
    with h5py.File(h5_path, "r") as hf:
        kspace = hf["kspace"][()]
        if target_key not in hf:
            raise KeyError(f"{target_key} not found in {h5_path}")
        target = hf[target_key][()]
    return kspace, target


def choose_slice_indices(num_slices):
    """
    Choose 25%, 50%, 75% slice positions.
    """
    idxs = [
        int(round(0.25 * (num_slices - 1))),
        int(round(0.50 * (num_slices - 1))),
        int(round(0.75 * (num_slices - 1))),
    ]
    return sorted(set(idxs))


# ----------------------------
# Plotting
# ----------------------------

def save_figure(target, recon, mask_1d, out_path, title):
    target, recon = center_crop_to_match(target, recon)
    residual = np.abs(recon - target)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    axes[0].imshow(normalise_for_display(target), cmap="gray")
    axes[0].set_title("Target")
    axes[0].axis("off")

    axes[1].imshow(normalise_for_display(recon), cmap="gray")
    axes[1].set_title("Zero-filled")
    axes[1].axis("off")

    axes[2].imshow(normalise_for_display(residual), cmap="magma")
    axes[2].set_title("Residual")
    axes[2].axis("off")

    axes[3].imshow(mask_1d.reshape(1, -1), cmap="gray", aspect="auto")
    axes[3].set_title("1D mask")
    axes[3].set_yticks([])
    axes[3].set_xlabel("k-space line")

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ----------------------------
# Main experiment
# ----------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--metadata_csv",
        default="/rds/general/user/ah725/home/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split.csv",
    )

    parser.add_argument(
        "--out_dir",
        default="/rds/general/user/ah725/home/fastmri_pipeline/outputs/mask_pilot",
    )

    parser.add_argument("--split", default="test")
    parser.add_argument("--num_patients", type=int, default=10)
    parser.add_argument("--prefer_annotated", action="store_true")

    parser.add_argument(
        "--accelerations",
        nargs="+",
        type=int,
        default=[4, 8],
    )

    parser.add_argument(
        "--mask_types",
        nargs="+",
        default=["fastmri_vd", "gaussian_vd", "equispaced"],
    )

    parser.add_argument(
        "--center_fractions",
        nargs="+",
        type=float,
        default=[0.08, 0.04],
        help="Must match accelerations. Example: R=4 -> 0.08, R=8 -> 0.04",
    )

    parser.add_argument(
        "--mask_dim",
        type=int,
        default=-1,
        choices=[-1, -2],
        help="Use -1 for width, -2 for height. If unsure, start with -1.",
    )

    parser.add_argument("--target_key", default="reconstruction_rss")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--save_all_figures", action="store_true")

    args = parser.parse_args()

    if len(args.accelerations) != len(args.center_fractions):
        raise ValueError("accelerations and center_fractions must have the same length.")

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.metadata_csv)

    required_cols = ["split", "patient_id", "pd_new_path", "pdfs_new_path"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in metadata CSV: {col}")

    df = df[df["split"].astype(str) == args.split].copy()

    if args.prefer_annotated and "has_target_pathology" in df.columns:
        df["_annotated_rank"] = df["has_target_pathology"].astype(int)
        df = df.sort_values("_annotated_rank", ascending=False)

    df = df.head(args.num_patients).copy()

    print(f"Using metadata: {args.metadata_csv}")
    print(f"Split: {args.split}")
    print(f"Selected patients/pairs: {len(df)}")
    print(f"Mask dim: {args.mask_dim}")
    print(f"Output: {out_dir}")
    print()

    records = []
    figure_counter = 0

    for row_idx, row in df.iterrows():
        patient_id = str(row["patient_id"])
        pd_path, pdfs_path = read_volume_pair(row)

        for contrast, h5_path in [("PD", pd_path), ("PDFS", pdfs_path)]:
            print(f"Reading {contrast}: {h5_path}")

            kspace_vol, target_vol = load_kspace_and_target(h5_path, args.target_key)

            num_slices = min(kspace_vol.shape[0], target_vol.shape[0])
            slice_indices = choose_slice_indices(num_slices)

            for slice_idx in slice_indices:
                kspace_slice = kspace_vol[slice_idx]  # [coils, height, width]
                target = target_vol[slice_idx]

                if args.mask_dim == -1:
                    num_lines = kspace_slice.shape[-1]
                else:
                    num_lines = kspace_slice.shape[-2]

                for R, cf in zip(args.accelerations, args.center_fractions):
                    for mask_type in args.mask_types:
                        seed = (
                            args.seed
                            + abs(hash((patient_id, contrast, slice_idx, R, mask_type))) % 1_000_000
                        )

                        mask_1d = make_mask(
                            mask_type=mask_type,
                            num_lines=num_lines,
                            acceleration=R,
                            center_fraction=cf,
                            seed=seed,
                        )

                        actual_acc = num_lines / mask_1d.sum()

                        masked_kspace = apply_1d_mask(
                            kspace_slice=kspace_slice,
                            mask_1d=mask_1d,
                            mask_dim=args.mask_dim,
                        )

                        zf = rss_combine(ifft2c(masked_kspace))

                        metrics = compute_metrics(zf, target)

                        records.append(
                            {
                                "patient_id": patient_id,
                                "contrast": contrast,
                                "slice_idx": slice_idx,
                                "R_target": R,
                                "center_fraction": cf,
                                "mask_type": mask_type,
                                "num_lines": num_lines,
                                "num_sampled_lines": int(mask_1d.sum()),
                                "actual_acceleration": actual_acc,
                                "NMSE": metrics["NMSE"],
                                "PSNR": metrics["PSNR"],
                                "SSIM": metrics["SSIM"],
                                "h5_path": str(h5_path),
                            }
                        )

                        should_save = args.save_all_figures

                        # Save a small representative subset by default:
                        # first patient, middle slice, all masks/R/contrasts
                        if not args.save_all_figures:
                            is_first_patient = row_idx == df.index[0]
                            is_middle_slice = slice_idx == slice_indices[len(slice_indices) // 2]
                            should_save = is_first_patient and is_middle_slice

                        if should_save:
                            # Save figures into category folders:
                            # figures/R4/PD/fastmri_vd/
                            # figures/R8/PDFS/gaussian_vd/
                            subdir = fig_dir / f"R{R}" / contrast / mask_type
                            subdir.mkdir(parents=True, exist_ok=True)

                            fname = f"{patient_id}_{contrast}_slice{slice_idx}.png"

                            title = (
                                f"Patient {patient_id} | {contrast} | "
                                f"slice {slice_idx} | R={R} | {mask_type} | "
                                f"actual R={actual_acc:.2f}"
                            )

                            save_figure(
                                target=target,
                                recon=zf,
                                mask_1d=mask_1d,
                                out_path=subdir / fname,
                                title=title,
                            )
                            figure_counter += 1

    metrics_df = pd.DataFrame(records)

    metrics_csv = out_dir / "mask_pilot_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    summary = (
        metrics_df
        .groupby(["mask_type", "R_target", "contrast"])
        .agg(
            NMSE_median=("NMSE", "median"),
            NMSE_IQR_low=("NMSE", lambda x: np.percentile(x, 25)),
            NMSE_IQR_high=("NMSE", lambda x: np.percentile(x, 75)),
            PSNR_median=("PSNR", "median"),
            PSNR_IQR_low=("PSNR", lambda x: np.percentile(x, 25)),
            PSNR_IQR_high=("PSNR", lambda x: np.percentile(x, 75)),
            SSIM_median=("SSIM", "median"),
            SSIM_IQR_low=("SSIM", lambda x: np.percentile(x, 25)),
            SSIM_IQR_high=("SSIM", lambda x: np.percentile(x, 75)),
            actual_acceleration_mean=("actual_acceleration", "mean"),
            actual_acceleration_std=("actual_acceleration", "std"),
            n_images=("NMSE", "count"),
        )
        .reset_index()
    )

    summary_csv = out_dir / "mask_pilot_summary.csv"
    summary.to_csv(summary_csv, index=False)

    print()
    print("Done.")
    print(f"Metrics CSV: {metrics_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Figures saved: {figure_counter}")
    print(f"Figure folder: {fig_dir}")
    print()
    print(summary)


if __name__ == "__main__":
    main()
