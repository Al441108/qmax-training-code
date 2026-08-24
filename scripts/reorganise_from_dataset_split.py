#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path

import pandas as pd


def clean_patient_id(patient_id):
    return str(patient_id).replace("/", "_").replace(" ", "_")


def make_new_name(patient_id, contrast, filename):
    """
    Create readable and unique output filename.
    """
    return f"{clean_patient_id(patient_id)}_{contrast}_{Path(filename).name}"


def find_file(filename, search_roots):
    """
    Search for a file by basename under existing fastMRI folders.
    """
    filename = Path(str(filename)).name
    matches = []

    for root in search_roots:
        root = Path(root)
        if root.exists():
            matches.extend(root.rglob(filename))

    if len(matches) == 1:
        return matches[0]

    if len(matches) == 0:
        raise FileNotFoundError(
            f"Could not find {filename} under:\n"
            + "\n".join(str(r) for r in search_roots)
        )

    raise RuntimeError(
        f"Multiple matches found for {filename}:\n"
        + "\n".join(str(m) for m in matches[:20])
    )


def link_or_copy(src, dst, mode):
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() or dst.is_symlink():
        return "exists"

    if mode == "symlink":
        dst.symlink_to(src)
        return "symlinked"

    if mode == "copy":
        shutil.copy2(src, dst)
        return "copied"

    if mode == "hardlink":
        dst.hardlink_to(src)
        return "hardlinked"

    raise ValueError(f"Unknown mode: {mode}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_split_csv",
        required=True,
        help="CSV containing split, patient_id, pd_file and pdfs_file columns.",
    )

    parser.add_argument(
        "--out_root",
        default="/rds/general/user/ah725/home/fastmri/multicoil_lesion_split",
    )

    parser.add_argument(
        "--mode",
        choices=["symlink", "copy", "hardlink"],
        default="symlink",
    )

    parser.add_argument(
        "--search_roots",
        nargs="+",
        default=[
            "/rds/general/user/ah725/home/fastmri/multicoil_train_paired",
            "/rds/general/user/ah725/home/fastmri/multicoil_val",
        ],
    )

    args = parser.parse_args()

    df = pd.read_csv(args.dataset_split_csv)

    required_cols = [
        "split",
        "patient_id",
        "pd_file",
        "pdfs_file",
        "has_annotation",
        "has_target_pathology",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "metadata").mkdir(parents=True, exist_ok=True)

    records = []

    for _, row in df.iterrows():
        split = str(row["split"]).strip()

        if split not in ["train", "val", "test"]:
            raise ValueError(f"Unexpected split value: {split}")

        patient_id = row["patient_id"]

        pd_src = find_file(row["pd_file"], args.search_roots)
        pdfs_src = find_file(row["pdfs_file"], args.search_roots)

        pd_dst = out_root / split / make_new_name(patient_id, "PD", row["pd_file"])
        pdfs_dst = out_root / split / make_new_name(patient_id, "PDFS", row["pdfs_file"])

        pd_status = link_or_copy(pd_src, pd_dst, args.mode)
        pdfs_status = link_or_copy(pdfs_src, pdfs_dst, args.mode)

        records.append(
            {
                "split": split,
                "patient_id": patient_id,
                "pd_file": row["pd_file"],
                "pdfs_file": row["pdfs_file"],
                "pd_src": str(pd_src),
                "pdfs_src": str(pdfs_src),
                "pd_new_path": str(pd_dst),
                "pdfs_new_path": str(pdfs_dst),
                "has_annotation": row["has_annotation"],
                "has_target_pathology": row["has_target_pathology"],
                "pd_labels": row.get("pd_labels", ""),
                "pdfs_labels": row.get("pdfs_labels", ""),
                "source_csv": row.get("source_csv", ""),
                "pd_status": pd_status,
                "pdfs_status": pdfs_status,
            }
        )

    out_csv = out_root / "metadata" / "reorganised_dataset_split.csv"
    pd.DataFrame(records).to_csv(out_csv, index=False)

    print("Done.")
    print(f"Output root: {out_root}")
    print(f"Reorganised CSV: {out_csv}")
    print()

    print("Pair counts:")
    print(df["split"].value_counts().sort_index())
    print()

    print("Expected h5 file counts:")
    for split, n in df["split"].value_counts().sort_index().items():
        print(f"{split}: {n * 2}")

    print()
    print("Annotation summary:")
    print(
        df.groupby("split")[["has_annotation", "has_target_pathology"]]
        .sum()
        .sort_index()
    )


if __name__ == "__main__":
    main()
