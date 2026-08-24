import argparse
import hashlib
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    ssim = None


def stable_seed(patient_id, slice_idx, acceleration):
    seed_string = f"{patient_id}_{slice_idx}_{acceleration}"
    return int(hashlib.md5(seed_string.encode("utf-8")).hexdigest()[:8], 16)


def make_gaussian_vd_mask(num_cols, acceleration, seed=42, center_fraction=None):
    """
    Create a 1D Gaussian variable-density Cartesian mask.
    The same function should be used consistently for zero-filled baseline and later model experiments.
    """
    rng = np.random.default_rng(seed)

    if center_fraction is None:
        if acceleration == 4:
            center_fraction = 0.08
        elif acceleration == 6:
            center_fraction = 0.06
        elif acceleration == 8:
            center_fraction = 0.04
        else:
            raise ValueError(f"Unsupported acceleration: {acceleration}")

    num_low_freqs = int(round(num_cols * center_fraction))
    num_low_freqs = max(1, num_low_freqs)

    mask = np.zeros(num_cols, dtype=np.float32)

    center = num_cols // 2
    half = num_low_freqs // 2
    start = max(0, center - half)
    end = min(num_cols, start + num_low_freqs)

    mask[start:end] = 1.0

    target_samples = int(round(num_cols / acceleration))
    remaining_samples = max(0, target_samples - int(mask.sum()))

    if remaining_samples > 0:
        x = np.arange(num_cols)
        sigma = num_cols / 6.0

        probs = np.exp(-0.5 * ((x - center) / sigma) ** 2)
        probs[mask == 1] = 0.0

        if probs.sum() <= 0:
            raise RuntimeError("Gaussian sampling probabilities sum to zero.")

        probs = probs / probs.sum()

        available = np.where(mask == 0)[0]
        chosen = rng.choice(
            np.arange(num_cols),
            size=min(remaining_samples, len(available)),
            replace=False,
            p=probs,
        )
        mask[chosen] = 1.0

    actual_R = num_cols / float(mask.sum())
    return mask, int(mask.sum()), float(actual_R)


def apply_1d_mask_to_kspace(kspace_slice, mask):
    """
    kspace_slice shape: [coils, height, width]
    mask shape: [width]
    """
    if kspace_slice.ndim != 3:
        raise ValueError(f"Expected kspace slice [coils, height, width], got {kspace_slice.shape}")

    if kspace_slice.shape[-1] != len(mask):
        raise ValueError(
            f"Mask length {len(mask)} does not match kspace width {kspace_slice.shape[-1]}"
        )

    return kspace_slice * mask.reshape(1, 1, -1)


def ifft2c(kspace):
    """
    Centered 2D inverse FFT for multi-coil k-space.
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


def center_crop(img, shape):
    """
    Center crop 2D image to target shape.
    """
    h, w = img.shape
    target_h, target_w = shape

    if target_h > h or target_w > w:
        raise ValueError(f"Cannot crop image shape {img.shape} to larger target shape {shape}")

    top = (h - target_h) // 2
    left = (w - target_w) // 2

    return img[top:top + target_h, left:left + target_w]


def zero_filled_recon(kspace_slice, mask, target_shape):
    masked_kspace = apply_1d_mask_to_kspace(kspace_slice, mask)
    coil_images = ifft2c(masked_kspace)
    rss = rss_combine(coil_images)
    rss_crop = center_crop(rss, target_shape)
    return rss_crop, masked_kspace


def nmse_metric(target, pred):
    target = np.asarray(target, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)
    return float(np.sum((target - pred) ** 2) / (np.sum(target ** 2) + 1e-12))


def psnr_metric(target, pred):
    target = np.asarray(target, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)

    mse = np.mean((target - pred) ** 2)
    if mse <= 0:
        return float("inf")

    data_range = float(target.max() - target.min())
    if data_range <= 0:
        data_range = float(target.max())
    if data_range <= 0:
        data_range = 1.0

    return float(20.0 * np.log10(data_range / np.sqrt(mse)))


def ssim_metric(target, pred):
    if ssim is None:
        raise ImportError(
            "scikit-image is not installed. Install it with: pip install scikit-image"
        )

    target = np.asarray(target, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)

    data_range = float(target.max() - target.min())
    if data_range <= 0:
        data_range = float(target.max())
    if data_range <= 0:
        data_range = 1.0

    return float(ssim(target, pred, data_range=data_range))


def normalise_for_display(img, p_low=1, p_high=99):
    img = np.asarray(img, dtype=np.float32)
    lo, hi = np.percentile(img, [p_low, p_high])
    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo + 1e-8)
    return img


def save_figure(target, recon, mask, patient_id, contrast, slice_idx, acceleration, actual_R, out_path):
    target_n = normalise_for_display(target)
    recon_n = normalise_for_display(recon)
    residual = np.abs(target_n - recon_n)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(target_n, cmap="gray")
    axes[0].set_title("Target")
    axes[0].axis("off")

    axes[1].imshow(recon_n, cmap="gray")
    axes[1].set_title("Zero-filled")
    axes[1].axis("off")

    axes[2].imshow(residual, cmap="magma")
    axes[2].set_title("Residual")
    axes[2].axis("off")

    axes[3].imshow(mask.reshape(1, -1), cmap="gray", aspect="auto")
    axes[3].set_title("1D Gaussian VD mask")
    axes[3].set_xlabel("k-space line")
    axes[3].set_yticks([])

    fig.suptitle(
        f"patient {patient_id} | {contrast} | slice {slice_idx} | "
        f"R={acceleration} | actual R={actual_R:.2f}",
        fontsize=10,
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def process_one_file(
    h5_path,
    patient_id,
    split,
    contrast,
    acceleration,
    out_fig_dir,
    max_figures,
    figure_counter,
    target_key="reconstruction_rss",
):
    rows = []

    h5_path = Path(h5_path)

    with h5py.File(h5_path, "r") as hf:
        if "kspace" not in hf:
            raise KeyError(f"kspace not found in {h5_path}")

        if target_key not in hf:
            raise KeyError(f"{target_key} not found in {h5_path}")

        kspace_all = hf["kspace"]
        target_all = hf[target_key]

        n_slices = kspace_all.shape[0]

        for slice_idx in range(n_slices):
            kspace_slice = kspace_all[slice_idx]
            target = np.asarray(target_all[slice_idx], dtype=np.float32)

            seed = stable_seed(patient_id, slice_idx, acceleration)

            mask, num_samples, actual_R = make_gaussian_vd_mask(
                num_cols=kspace_slice.shape[-1],
                acceleration=acceleration,
                seed=seed,
            )

            recon, _ = zero_filled_recon(
                kspace_slice=kspace_slice,
                mask=mask,
                target_shape=target.shape,
            )

            nmse_val = nmse_metric(target, recon)
            psnr_val = psnr_metric(target, recon)
            ssim_val = ssim_metric(target, recon)

            rows.append(
                {
                    "split": split,
                    "patient_id": patient_id,
                    "contrast": contrast,
                    "slice_idx": slice_idx,
                    "acceleration": acceleration,
                    "mask_type": "gaussian_vd",
                    "num_sampled_lines": num_samples,
                    "actual_R": actual_R,
                    "h5_path": str(h5_path),
                    "NMSE": nmse_val,
                    "PSNR": psnr_val,
                    "SSIM": ssim_val,
                }
            )

            if figure_counter[0] < max_figures:
                out_path = (
                    out_fig_dir
                    / f"R{acceleration}"
                    / contrast
                    / f"patient_{patient_id}_slice_{slice_idx:03d}_{contrast}_R{acceleration}_zf.png"
                )

                save_figure(
                    target=target,
                    recon=recon,
                    mask=mask,
                    patient_id=patient_id,
                    contrast=contrast,
                    slice_idx=slice_idx,
                    acceleration=acceleration,
                    actual_R=actual_R,
                    out_path=out_path,
                )

                figure_counter[0] += 1

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_csv", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--accelerations", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--target_key", type=str, default="reconstruction_rss")
    parser.add_argument("--max_figures_per_acceleration", type=int, default=12)
    args = parser.parse_args()

    metadata_csv = Path(args.metadata_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_fig_dir = out_dir / "figures"

    df = pd.read_csv(metadata_csv)
    df = df[df["split"] == args.split].copy()

    required_cols = ["split", "patient_id", "pd_new_path", "pdfs_new_path"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in metadata CSV: {missing}")

    print("=" * 80)
    print("Full zero-filled baseline with Gaussian VD mask")
    print("=" * 80)
    print(f"metadata_csv: {metadata_csv}")
    print(f"split: {args.split}")
    print(f"patients: {len(df)}")
    print(f"accelerations: {args.accelerations}")
    print(f"out_dir: {out_dir}")
    print("=" * 80)

    all_rows = []

    for acceleration in args.accelerations:
        print(f"\nProcessing acceleration R={acceleration}")
        figure_counter = [0]

        for i, row in df.iterrows():
            patient_id = row["patient_id"]
            pd_path = row["pd_new_path"]
            pdfs_path = row["pdfs_new_path"]

            print(f"[R={acceleration}] patient {patient_id}")

            try:
                pd_rows = process_one_file(
                    h5_path=pd_path,
                    patient_id=patient_id,
                    split=args.split,
                    contrast="PD",
                    acceleration=acceleration,
                    out_fig_dir=out_fig_dir,
                    max_figures=args.max_figures_per_acceleration,
                    figure_counter=figure_counter,
                    target_key=args.target_key,
                )
                all_rows.extend(pd_rows)

                pdfs_rows = process_one_file(
                    h5_path=pdfs_path,
                    patient_id=patient_id,
                    split=args.split,
                    contrast="PDFS",
                    acceleration=acceleration,
                    out_fig_dir=out_fig_dir,
                    max_figures=args.max_figures_per_acceleration,
                    figure_counter=figure_counter,
                    target_key=args.target_key,
                )
                all_rows.extend(pdfs_rows)

            except Exception as e:
                print(f"[FAILED] patient={patient_id}, R={acceleration}: {e}")

    metrics_df = pd.DataFrame(all_rows)

    metrics_csv = out_dir / "zero_filled_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    summary = (
        metrics_df
        .groupby(["mask_type", "acceleration", "contrast"])
        .agg(
            n_slices=("SSIM", "count"),
            NMSE_mean=("NMSE", "mean"),
            NMSE_median=("NMSE", "median"),
            PSNR_mean=("PSNR", "mean"),
            PSNR_median=("PSNR", "median"),
            SSIM_mean=("SSIM", "mean"),
            SSIM_median=("SSIM", "median"),
        )
        .reset_index()
    )

    summary_csv = out_dir / "zero_filled_summary.csv"
    summary.to_csv(summary_csv, index=False)

    print("\n" + "=" * 80)
    print("Done")
    print("=" * 80)
    print(f"Saved metrics: {metrics_csv}")
    print(f"Saved summary: {summary_csv}")
    print(f"Saved figures under: {out_fig_dir}")
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
