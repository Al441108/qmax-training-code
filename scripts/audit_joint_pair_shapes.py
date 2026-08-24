import argparse
from pathlib import Path

import h5py
import pandas as pd


def get_shape(path, target_key):
    path = Path(path)
    if not path.exists():
        return None, None, f"missing file: {path}"

    try:
        with h5py.File(path, "r") as hf:
            if "kspace" not in hf:
                return None, None, f"missing kspace: {path}"
            if target_key not in hf:
                return None, None, f"missing {target_key}: {path}"
            return tuple(hf["kspace"].shape), tuple(hf[target_key].shape), None
    except Exception as e:
        return None, None, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_csv", type=str, required=True)
    parser.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    parser.add_argument("--target_key", type=str, default="reconstruction_rss")
    parser.add_argument("--out_csv", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.metadata_csv)
    df = df[df["split"] == args.split].copy()

    rows = []
    for i, row in df.iterrows():
        patient_id = row["patient_id"]
        pd_path = row["pd_new_path"]
        pdfs_path = row["pdfs_new_path"]

        pd_ks_shape, pd_tgt_shape, pd_err = get_shape(pd_path, args.target_key)
        pdfs_ks_shape, pdfs_tgt_shape, pdfs_err = get_shape(pdfs_path, args.target_key)

        same_num_slices = (
            pd_ks_shape is not None
            and pdfs_ks_shape is not None
            and pd_ks_shape[0] == pdfs_ks_shape[0]
        )
        same_kspace_hw = (
            pd_ks_shape is not None
            and pdfs_ks_shape is not None
            and pd_ks_shape[-2:] == pdfs_ks_shape[-2:]
        )
        same_target_hw = (
            pd_tgt_shape is not None
            and pdfs_tgt_shape is not None
            and pd_tgt_shape[-2:] == pdfs_tgt_shape[-2:]
        )

        rows.append({
            "split": args.split,
            "patient_id": patient_id,
            "pd_path": pd_path,
            "pdfs_path": pdfs_path,
            "pd_kspace_shape": str(pd_ks_shape),
            "pdfs_kspace_shape": str(pdfs_ks_shape),
            "pd_target_shape": str(pd_tgt_shape),
            "pdfs_target_shape": str(pdfs_tgt_shape),
            "pd_error": pd_err,
            "pdfs_error": pdfs_err,
            "same_num_slices": same_num_slices,
            "same_kspace_hw": same_kspace_hw,
            "same_target_hw": same_target_hw,
            "kspace_hw": str(pd_ks_shape[-2:]) if pd_ks_shape is not None else None,
            "target_hw": str(pd_tgt_shape[-2:]) if pd_tgt_shape is not None else None,
            "n_slices": pd_ks_shape[0] if pd_ks_shape is not None else None,
        })

    out = pd.DataFrame(rows)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print("=" * 100)
    print(f"Joint pair shape audit: {args.split}")
    print("=" * 100)
    print(f"patients: {len(out)}")
    print("\nsame_num_slices:")
    print(out["same_num_slices"].value_counts(dropna=False))
    print("\nsame_kspace_hw:")
    print(out["same_kspace_hw"].value_counts(dropna=False))
    print("\nsame_target_hw:")
    print(out["same_target_hw"].value_counts(dropna=False))
    print("\nkspace_hw counts:")
    print(out["kspace_hw"].value_counts(dropna=False).sort_index())
    print("\ntarget_hw counts:")
    print(out["target_hw"].value_counts(dropna=False).sort_index())
    print("\nn_slices summary:")
    print(out["n_slices"].describe())
    print(f"\nSaved: {out_path}")

    bad = out[
        (~out["same_num_slices"])
        | (~out["same_kspace_hw"])
        | (~out["same_target_hw"])
        | out["pd_error"].notna()
        | out["pdfs_error"].notna()
    ]

    if len(bad) > 0:
        print("\nWARNING: problematic pairs found:")
        print(bad[[
            "patient_id",
            "pd_kspace_shape",
            "pdfs_kspace_shape",
            "pd_target_shape",
            "pdfs_target_shape",
            "pd_error",
            "pdfs_error",
            "same_num_slices",
            "same_kspace_hw",
            "same_target_hw",
        ]].to_string(index=False))
    else:
        print("\nAll pairs passed basic shape checks.")


if __name__ == "__main__":
    main()
