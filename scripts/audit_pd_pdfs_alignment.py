#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def normalise_image(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)

    lo = float(np.percentile(x, 1))
    hi = float(np.percentile(x, 99))

    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)

    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo)
    return x.astype(np.float32)


def gradient_magnitude(x: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(x)
    return np.sqrt(gx ** 2 + gy ** 2)


def corrcoef_safe(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)

    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return float("nan")

    return float(np.corrcoef(a, b)[0, 1])


def center_crop_np(x: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    h, w = x.shape[-2:]
    th, tw = target_shape

    if h < th or w < tw:
        raise RuntimeError(
            f"Cannot crop image from {(h, w)} to {(th, tw)}"
        )

    if h == th and w == tw:
        return x

    top = (h - th) // 2
    left = (w - tw) // 2

    return x[..., top:top + th, left:left + tw]


def match_spatial_shape(
    a: np.ndarray,
    b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    h = min(a.shape[-2], b.shape[-2])
    w = min(a.shape[-1], b.shape[-1])

    return (
        center_crop_np(a, (h, w)),
        center_crop_np(b, (h, w)),
    )


def get_dataset(hf: h5py.File, preferred_keys: List[str]) -> np.ndarray:
    for key in preferred_keys:
        if key in hf:
            return hf[key][()]

    raise KeyError(
        f"None of the expected datasets exist: {preferred_keys}. "
        f"Available keys: {list(hf.keys())}"
    )


def get_path_column(df: pd.DataFrame, candidates: List[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col

    raise KeyError(
        f"Could not find any path column from {candidates}. "
        f"CSV columns: {df.columns.tolist()}"
    )


def best_offset_for_slice(
    pd_volume: np.ndarray,
    pdfs_volume: np.ndarray,
    slice_idx: int,
    offsets: Tuple[int, ...] = (-2, -1, 0, 1, 2),
) -> Tuple[int, Dict[int, float]]:
    pd_img = normalise_image(pd_volume[slice_idx])
    pd_edge = gradient_magnitude(pd_img)

    scores: Dict[int, float] = {}

    for offset in offsets:
        j = slice_idx + offset

        if j < 0 or j >= len(pdfs_volume):
            continue

        pdfs_img = normalise_image(pdfs_volume[j])
        pd_edge_c, pdfs_edge_c = match_spatial_shape(
            pd_edge,
            gradient_magnitude(pdfs_img),
        )

        scores[offset] = corrcoef_safe(
            pd_edge_c,
            pdfs_edge_c,
        )

    valid_scores = {
        k: v for k, v in scores.items()
        if np.isfinite(v)
    }

    if not valid_scores:
        return 0, scores

    best_offset = max(valid_scores, key=valid_scores.get)
    return int(best_offset), scores


def save_alignment_figure(
    patient_id: str,
    position_name: str,
    slice_idx: int,
    pd_img: np.ndarray,
    pdfs_img: np.ndarray,
    output_path: Path,
) -> None:
    pd_img, pdfs_img = match_spatial_shape(pd_img, pdfs_img)

    pd_n = normalise_image(pd_img)
    pdfs_n = normalise_image(pdfs_img)

    pd_edge = gradient_magnitude(pd_n)
    pdfs_edge = gradient_magnitude(pdfs_n)

    overlay = np.zeros(
        (pd_edge.shape[0], pd_edge.shape[1], 3),
        dtype=np.float32,
    )

    pd_edge_scaled = normalise_image(pd_edge)
    pdfs_edge_scaled = normalise_image(pdfs_edge)

    overlay[..., 0] = pd_edge_scaled
    overlay[..., 1] = pdfs_edge_scaled

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(15, 3.2),
    )

    axes[0].imshow(pd_n, cmap="gray")
    axes[0].set_title("PD")

    axes[1].imshow(pdfs_n, cmap="gray")
    axes[1].set_title("PD-FS")

    axes[2].imshow(pd_edge_scaled, cmap="gray")
    axes[2].set_title("PD edge")

    axes[3].imshow(pdfs_edge_scaled, cmap="gray")
    axes[3].set_title("PD-FS edge")

    axes[4].imshow(overlay)
    axes[4].set_title("Edge overlay")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"{patient_id} | {position_name} | slice {slice_idx}",
        fontsize=11,
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit PD/PD-FS paired volume alignment."
    )

    parser.add_argument(
        "--metadata_csv",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
    )

    parser.add_argument(
        "--max_patients",
        type=int,
        default=None,
        help="Optional limit for smoke testing.",
    )

    parser.add_argument(
        "--save_figures",
        action="store_true",
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(args.metadata_csv)

    if "split" in df.columns:
        df = df[
            df["split"].astype(str).str.lower()
            == args.split.lower()
        ].copy()

    if args.max_patients is not None:
        df = df.head(args.max_patients).copy()

    pd_path_col = get_path_column(
        df,
        [
            "pd_new_path",
            "pd_path",
            "PD_path",
            "pd_file",
        ],
    )

    pdfs_path_col = get_path_column(
        df,
        [
            "pdfs_new_path",
            "pdfs_path",
            "PDFS_path",
            "pdfs_file",
        ],
    )

    patient_col = (
        "patient_id"
        if "patient_id" in df.columns
        else df.columns[0]
    )

    audit_rows = []
    offset_rows = []

    for row_idx, row in df.iterrows():
        patient_id = str(row[patient_col])
        pd_path = Path(str(row[pd_path_col]))
        pdfs_path = Path(str(row[pdfs_path_col]))

        audit = {
            "patient_id": patient_id,
            "split": args.split,
            "pd_path": str(pd_path),
            "pdfs_path": str(pdfs_path),
            "pd_exists": pd_path.exists(),
            "pdfs_exists": pdfs_path.exists(),
        }

        if not pd_path.exists() or not pdfs_path.exists():
            audit["status"] = "missing_file"
            audit_rows.append(audit)
            continue

        try:
            with h5py.File(pd_path, "r") as pd_hf, \
                 h5py.File(pdfs_path, "r") as pdfs_hf:

                pd_volume = get_dataset(
                    pd_hf,
                    [
                        "reconstruction_rss",
                        "reconstruction_esc",
                    ],
                )

                pdfs_volume = get_dataset(
                    pdfs_hf,
                    [
                        "reconstruction_rss",
                        "reconstruction_esc",
                    ],
                )

                pd_kspace_shape = (
                    tuple(pd_hf["kspace"].shape)
                    if "kspace" in pd_hf
                    else None
                )

                pdfs_kspace_shape = (
                    tuple(pdfs_hf["kspace"].shape)
                    if "kspace" in pdfs_hf
                    else None
                )

                n_pd = int(pd_volume.shape[0])
                n_pdfs = int(pdfs_volume.shape[0])
                n_common = min(n_pd, n_pdfs)

                same_target_spatial = (
                    tuple(pd_volume.shape[-2:])
                    == tuple(pdfs_volume.shape[-2:])
                )

                same_kspace_spatial = (
                    pd_kspace_shape is not None
                    and pdfs_kspace_shape is not None
                    and tuple(pd_kspace_shape[-2:])
                    == tuple(pdfs_kspace_shape[-2:])
                )

                audit.update({
                    "n_pd": n_pd,
                    "n_pdfs": n_pdfs,
                    "n_common": n_common,
                    "slice_count_diff": n_pd - n_pdfs,
                    "same_num_slices": n_pd == n_pdfs,
                    "pd_target_shape": str(tuple(pd_volume.shape)),
                    "pdfs_target_shape": str(tuple(pdfs_volume.shape)),
                    "pd_kspace_shape": str(pd_kspace_shape),
                    "pdfs_kspace_shape": str(pdfs_kspace_shape),
                    "same_target_spatial": same_target_spatial,
                    "same_kspace_spatial": same_kspace_spatial,
                })

                if n_common < 3:
                    audit["status"] = "too_few_slices"
                    audit_rows.append(audit)
                    continue

                positions = {
                    "q1": max(0, n_common // 4),
                    "mid": max(0, n_common // 2),
                    "q3": min(
                        n_common - 1,
                        (3 * n_common) // 4,
                    ),
                }

                selected_offsets = []
                same_index_corrs = []

                for position_name, slice_idx in positions.items():
                    best_offset, scores = best_offset_for_slice(
                        pd_volume,
                        pdfs_volume,
                        slice_idx,
                    )

                    selected_offsets.append(best_offset)
                    same_index_corrs.append(
                        scores.get(0, float("nan"))
                    )

                    offset_row = {
                        "patient_id": patient_id,
                        "position": position_name,
                        "pd_slice_idx": int(slice_idx),
                        "best_offset": int(best_offset),
                    }

                    for offset in [-2, -1, 0, 1, 2]:
                        offset_row[f"corr_offset_{offset:+d}"] = (
                            scores.get(offset, float("nan"))
                        )

                    offset_rows.append(offset_row)

                    if args.save_figures:
                        pdfs_idx = slice_idx

                        figure_path = (
                            figures_dir
                            / f"{patient_id}_{position_name}_slice{slice_idx}.png"
                        )

                        save_alignment_figure(
                            patient_id=patient_id,
                            position_name=position_name,
                            slice_idx=slice_idx,
                            pd_img=pd_volume[slice_idx],
                            pdfs_img=pdfs_volume[pdfs_idx],
                            output_path=figure_path,
                        )

                finite_same_corrs = [
                    x for x in same_index_corrs
                    if np.isfinite(x)
                ]

                median_same_corr = (
                    float(np.median(finite_same_corrs))
                    if finite_same_corrs
                    else float("nan")
                )

                systematic_offset = (
                    len(selected_offsets) == 3
                    and selected_offsets[0]
                    == selected_offsets[1]
                    == selected_offsets[2]
                    and selected_offsets[0] != 0
                )

                low_similarity = (
                    np.isfinite(median_same_corr)
                    and median_same_corr < 0.15
                )

                audit.update({
                    "offset_q1": selected_offsets[0],
                    "offset_mid": selected_offsets[1],
                    "offset_q3": selected_offsets[2],
                    "median_same_index_edge_corr": median_same_corr,
                    "possible_systematic_offset": systematic_offset,
                    "low_same_index_similarity": low_similarity,
                })

                if not audit["same_num_slices"]:
                    audit["status"] = "slice_count_mismatch"
                elif not audit["same_target_spatial"]:
                    audit["status"] = "target_shape_mismatch"
                elif not audit["same_kspace_spatial"]:
                    audit["status"] = "kspace_shape_mismatch"
                elif systematic_offset:
                    audit["status"] = "possible_systematic_offset"
                elif low_similarity:
                    audit["status"] = "low_anatomical_similarity"
                else:
                    audit["status"] = "no_obvious_problem"

        except Exception as exc:
            audit["status"] = "read_error"
            audit["error"] = repr(exc)

        audit_rows.append(audit)

        print(
            f"[{len(audit_rows)}/{len(df)}] "
            f"{patient_id}: {audit['status']}"
        )

    audit_df = pd.DataFrame(audit_rows)
    offset_df = pd.DataFrame(offset_rows)

    audit_path = (
        args.output_dir
        / "pair_alignment_audit.csv"
    )

    offset_path = (
        args.output_dir
        / "offset_analysis.csv"
    )

    suspicious_path = (
        args.output_dir
        / "suspicious_pairs.csv"
    )

    audit_df.to_csv(
        audit_path,
        index=False,
    )

    offset_df.to_csv(
        offset_path,
        index=False,
    )

    suspicious_df = audit_df[
        audit_df["status"] != "no_obvious_problem"
    ].copy()

    suspicious_df.to_csv(
        suspicious_path,
        index=False,
    )

    summary_lines = [
        f"split: {args.split}",
        f"patients audited: {len(audit_df)}",
        "",
        "status counts:",
        audit_df["status"].value_counts(
            dropna=False
        ).to_string(),
        "",
        f"suspicious patients: {len(suspicious_df)}",
    ]

    summary_path = (
        args.output_dir
        / "alignment_audit_summary.txt"
    )

    summary_path.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print("=" * 80)
    print("Alignment audit complete")
    print("=" * 80)
    print(audit_df["status"].value_counts(dropna=False))
    print(f"Saved: {audit_path}")
    print(f"Saved: {offset_path}")
    print(f"Saved: {suspicious_path}")
    print(f"Saved: {summary_path}")

    if args.save_figures:
        print(f"Figures: {figures_dir}")


if __name__ == "__main__":
    main()