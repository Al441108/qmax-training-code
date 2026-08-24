import argparse
import hashlib
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def stable_seed(patient_id, slice_idx, acceleration):
    seed_string = f"{patient_id}_{slice_idx}_{acceleration}"
    return int(hashlib.md5(seed_string.encode("utf-8")).hexdigest()[:8], 16)


def make_gaussian_vd_mask(num_cols, acceleration, seed=42, center_fraction=None):
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


def check_finite(name, arr):
    arr = np.asarray(arr)
    finite = np.isfinite(arr).all()
    print(f"  {name}: finite={finite}, dtype={arr.dtype}, shape={arr.shape}")
    if not finite:
        bad = np.size(arr) - np.isfinite(arr).sum()
        print(f"  WARNING: {name} has {bad} non-finite values")
    return finite


def inspect_pair(row, acceleration, target_key, max_slices_per_patient):
    patient_id = row["patient_id"]
    pd_path = Path(row["pd_new_path"])
    pdfs_path = Path(row["pdfs_new_path"])

    print("\n" + "=" * 100)
    print(f"patient_id: {patient_id}")
    print(f"PD path:    {pd_path}")
    print(f"PDFS path:  {pdfs_path}")

    if not pd_path.exists():
        raise FileNotFoundError(f"PD file not found: {pd_path}")
    if not pdfs_path.exists():
        raise FileNotFoundError(f"PDFS file not found: {pdfs_path}")

    with h5py.File(pd_path, "r") as pd_hf, h5py.File(pdfs_path, "r") as pdfs_hf:
        for key in ["kspace", target_key]:
            if key not in pd_hf:
                raise KeyError(f"{key} missing in PD file: {pd_path}")
            if key not in pdfs_hf:
                raise KeyError(f"{key} missing in PDFS file: {pdfs_path}")

        pd_kspace = pd_hf["kspace"]
        pdfs_kspace = pdfs_hf["kspace"]
        pd_target = pd_hf[target_key]
        pdfs_target = pdfs_hf[target_key]

        print(f"PD kspace shape:      {pd_kspace.shape}")
        print(f"PDFS kspace shape:    {pdfs_kspace.shape}")
        print(f"PD target shape:      {pd_target.shape}")
        print(f"PDFS target shape:    {pdfs_target.shape}")

        same_num_slices = pd_kspace.shape[0] == pdfs_kspace.shape[0]
        same_kspace_hw = pd_kspace.shape[-2:] == pdfs_kspace.shape[-2:]
        same_target_hw = pd_target.shape[-2:] == pdfs_target.shape[-2:]

        print(f"same_num_slices:      {same_num_slices}")
        print(f"same_kspace_hw:       {same_kspace_hw}")
        print(f"same_target_hw:       {same_target_hw}")

        n_slices = min(pd_kspace.shape[0], pdfs_kspace.shape[0], max_slices_per_patient)

        for slice_idx in range(n_slices):
            print(f"\n  slice_idx: {slice_idx}")

            pd_ks = np.asarray(pd_kspace[slice_idx])
            pdfs_ks = np.asarray(pdfs_kspace[slice_idx])
            pd_tgt = np.asarray(pd_target[slice_idx])
            pdfs_tgt = np.asarray(pdfs_target[slice_idx])

            check_finite("PD kspace", pd_ks)
            check_finite("PDFS kspace", pdfs_ks)
            check_finite("PD target", pd_tgt)
            check_finite("PDFS target", pdfs_tgt)

            if pd_ks.shape[-1] != pdfs_ks.shape[-1]:
                print("  WARNING: PD/PDFS kspace width mismatch; same 1D mask length not possible.")
                continue

            seed = stable_seed(patient_id, slice_idx, acceleration)
            mask, num_lines, actual_R = make_gaussian_vd_mask(
                num_cols=pd_ks.shape[-1],
                acceleration=acceleration,
                seed=seed,
            )

            pd_masked = pd_ks * mask.reshape(1, 1, -1)
            pdfs_masked = pdfs_ks * mask.reshape(1, 1, -1)

            print(f"  mask shape: {mask.shape}")
            print(f"  sampled lines: {num_lines}")
            print(f"  actual_R: {actual_R:.4f}")
            check_finite("PD masked kspace", pd_masked)
            check_finite("PDFS masked kspace", pdfs_masked)

            same_mask_check = np.array_equal(mask, mask.copy())
            print(f"  same_mask_for_PD_PDFS: {same_mask_check}")

    return {
        "patient_id": patient_id,
        "pd_path": str(pd_path),
        "pdfs_path": str(pdfs_path),
        "pd_kspace_shape": str(pd_kspace.shape),
        "pdfs_kspace_shape": str(pdfs_kspace.shape),
        "pd_target_shape": str(pd_target.shape),
        "pdfs_target_shape": str(pdfs_target.shape),
        "same_num_slices": same_num_slices,
        "same_kspace_hw": same_kspace_hw,
        "same_target_hw": same_target_hw,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_csv", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--acceleration", type=int, default=4, choices=[4, 6, 8])
    parser.add_argument("--num_patients", type=int, default=5)
    parser.add_argument("--max_slices_per_patient", type=int, default=3)
    parser.add_argument("--target_key", type=str, default="reconstruction_rss")
    parser.add_argument("--out_csv", type=str, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.metadata_csv)
    df = df[df["split"] == args.split].copy()

    required_cols = ["split", "patient_id", "pd_new_path", "pdfs_new_path"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in metadata CSV: {missing}")

    df = df.head(args.num_patients)

    print("=" * 100)
    print("Joint PD/PDFS loader smoke test")
    print("=" * 100)
    print(f"metadata_csv: {args.metadata_csv}")
    print(f"split: {args.split}")
    print(f"num_patients: {len(df)}")
    print(f"acceleration: {args.acceleration}")
    print(f"target_key: {args.target_key}")
    print("=" * 100)

    rows = []
    for _, row in df.iterrows():
        result = inspect_pair(
            row=row,
            acceleration=args.acceleration,
            target_key=args.target_key,
            max_slices_per_patient=args.max_slices_per_patient,
        )
        rows.append(result)

    result_df = pd.DataFrame(rows)

    print("\n" + "=" * 100)
    print("Smoke summary")
    print("=" * 100)
    print(result_df.to_string(index=False))

    print("\nChecks:")
    print("same_num_slices counts:")
    print(result_df["same_num_slices"].value_counts())
    print("\nsame_kspace_hw counts:")
    print(result_df["same_kspace_hw"].value_counts())
    print("\nsame_target_hw counts:")
    print(result_df["same_target_hw"].value_counts())

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(out_csv, index=False)
        print(f"\nSaved smoke summary to: {out_csv}")

    print("\nDone.")


if __name__ == "__main__":
    main()
