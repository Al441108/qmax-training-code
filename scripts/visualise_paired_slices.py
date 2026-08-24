import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def normalise_image(img, p_low=1, p_high=99):
    """Robust normalisation for visualisation."""
    img = np.asarray(img, dtype=np.float32)
    lo, hi = np.percentile(img, [p_low, p_high])
    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo + 1e-8)
    return img


def load_target_slice(h5_path, slice_idx, target_key="reconstruction_rss"):
    with h5py.File(h5_path, "r") as hf:
        if target_key not in hf:
            raise KeyError(f"{target_key} not found in {h5_path}. Keys: {list(hf.keys())}")

        target = hf[target_key]

        n_slices = target.shape[0]
        if slice_idx < 0 or slice_idx >= n_slices:
            raise IndexError(f"slice_idx={slice_idx} out of range for {h5_path}, n_slices={n_slices}")

        img = target[slice_idx]

    return img


def get_num_slices(h5_path, target_key="reconstruction_rss"):
    with h5py.File(h5_path, "r") as hf:
        if target_key in hf:
            return hf[target_key].shape[0]
        if "kspace" in hf:
            return hf["kspace"].shape[0]
        raise KeyError(f"Neither {target_key} nor kspace found in {h5_path}")


def choose_slice_indices(n_slices, fractions=(0.25, 0.50, 0.75)):
    indices = []
    for f in fractions:
        idx = int(round((n_slices - 1) * f))
        idx = max(0, min(n_slices - 1, idx))
        indices.append(idx)
    return sorted(set(indices))


def make_figure(pd_img, pdfs_img, patient_id, split, slice_idx, out_path):
    pd_norm = normalise_image(pd_img)
    pdfs_norm = normalise_image(pdfs_img)

    # Difference is only for rough alignment check.
    # It should not be interpreted as pathology or reconstruction error.
    diff = np.abs(pd_norm - pdfs_norm)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(pd_norm, cmap="gray")
    axes[0].set_title("PD target")
    axes[0].axis("off")

    axes[1].imshow(pdfs_norm, cmap="gray")
    axes[1].set_title("PD-FS target")
    axes[1].axis("off")

    axes[2].imshow(diff, cmap="magma")
    axes[2].set_title("Abs difference\n(alignment check only)")
    axes[2].axis("off")

    fig.suptitle(f"{split} | patient {patient_id} | slice {slice_idx}", fontsize=10)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata_csv",
        type=str,
        required=True,
        help="Path to reorganised_dataset_split.csv",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for paired visualisation figures",
    )
    parser.add_argument(
        "--patients_per_split",
        type=int,
        default=5,
        help="Number of patients to visualise per split",
    )
    parser.add_argument(
        "--target_key",
        type=str,
        default="reconstruction_rss",
    )
    parser.add_argument(
        "--prefer_annotated",
        action="store_true",
        help="Prefer patients with annotation if available",
    )
    args = parser.parse_args()

    metadata_csv = Path(args.metadata_csv)
    out_dir = Path(args.out_dir)

    df = pd.read_csv(metadata_csv)

    required_cols = ["split", "patient_id", "pd_new_path", "pdfs_new_path"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in metadata CSV: {missing}")

    print(f"Loaded metadata: {metadata_csv}")
    print(f"Rows: {len(df)}")
    print(df["split"].value_counts())

    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split].copy()

        if len(split_df) == 0:
            print(f"[WARNING] No rows for split={split}")
            continue

        if args.prefer_annotated and "has_annotation" in split_df.columns:
            split_df = split_df.sort_values("has_annotation", ascending=False)

        split_df = split_df.head(args.patients_per_split)

        print(f"\nProcessing split={split}, selected patients={len(split_df)}")

        for _, row in split_df.iterrows():
            patient_id = row["patient_id"]
            pd_path = Path(row["pd_new_path"])
            pdfs_path = Path(row["pdfs_new_path"])

            if not pd_path.exists():
                print(f"[SKIP] PD path missing: {pd_path}")
                continue
            if not pdfs_path.exists():
                print(f"[SKIP] PDFS path missing: {pdfs_path}")
                continue

            try:
                n_pd = get_num_slices(pd_path, args.target_key)
                n_pdfs = get_num_slices(pdfs_path, args.target_key)
            except Exception as e:
                print(f"[SKIP] Failed to read slices for patient {patient_id}: {e}")
                continue

            n_common = min(n_pd, n_pdfs)
            slice_indices = choose_slice_indices(n_common)

            print(
                f"patient={patient_id} | PD slices={n_pd} | PDFS slices={n_pdfs} | "
                f"using slices={slice_indices}"
            )

            for slice_idx in slice_indices:
                try:
                    pd_img = load_target_slice(pd_path, slice_idx, args.target_key)
                    pdfs_img = load_target_slice(pdfs_path, slice_idx, args.target_key)
                except Exception as e:
                    print(f"[SKIP] Failed patient={patient_id}, slice={slice_idx}: {e}")
                    continue

                out_path = (
                    out_dir
                    / split
                    / f"patient_{patient_id}_slice_{slice_idx:03d}_paired_check.png"
                )

                make_figure(
                    pd_img=pd_img,
                    pdfs_img=pdfs_img,
                    patient_id=patient_id,
                    split=split,
                    slice_idx=slice_idx,
                    out_path=out_path,
                )

    print(f"\nDone. Figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
