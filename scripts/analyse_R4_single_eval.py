#!/usr/bin/env python3
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def median_iqr(values):
    values = pd.Series(values).dropna().astype(float)
    if len(values) == 0:
        return {
            "median": np.nan,
            "iqr_low": np.nan,
            "iqr_high": np.nan,
        }
    return {
        "median": float(values.median()),
        "iqr_low": float(values.quantile(0.25)),
        "iqr_high": float(values.quantile(0.75)),
    }


def format_median_iqr(values, decimals=4):
    s = median_iqr(values)
    return f"{s['median']:.{decimals}f} [{s['iqr_low']:.{decimals}f}, {s['iqr_high']:.{decimals}f}]"


def safe_ratio(a, b):
    if b == 0 or pd.isna(a) or pd.isna(b):
        return np.nan
    return a / b


def load_data(base_dir):
    base_dir = Path(base_dir)

    paths = {
        "pd_slice": base_dir / "pd_R4_val" / "pd_val_per_slice_metrics.csv",
        "pdfs_slice": base_dir / "pdfs_R4_val" / "pdfs_val_per_slice_metrics.csv",
        "pd_patient": base_dir / "pd_R4_val" / "pd_val_patient_level_metrics.csv",
        "pdfs_patient": base_dir / "pdfs_R4_val" / "pdfs_val_patient_level_metrics.csv",
    }

    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")

    pd_slice = pd.read_csv(paths["pd_slice"])
    pdfs_slice = pd.read_csv(paths["pdfs_slice"])
    pd_patient = pd.read_csv(paths["pd_patient"])
    pdfs_patient = pd.read_csv(paths["pdfs_patient"])

    pd_slice["contrast_label"] = "PD"
    pdfs_slice["contrast_label"] = "PD-FS"
    pd_patient["contrast_label"] = "PD"
    pdfs_patient["contrast_label"] = "PD-FS"

    slice_df = pd.concat([pd_slice, pdfs_slice], ignore_index=True)
    patient_df = pd.concat([pd_patient, pdfs_patient], ignore_index=True)

    return slice_df, patient_df


def make_summary_table(df, level_name):
    metrics = ["NMSE", "PSNR", "SSIM", "L1"]
    rows = []

    for contrast in ["PD", "PD-FS"]:
        for group_name, group_df in [
            ("overall", df[df["contrast_label"] == contrast]),
            ("central", df[(df["contrast_label"] == contrast) & (df["is_edge"] == False)]),
            ("edge", df[(df["contrast_label"] == contrast) & (df["is_edge"] == True)]),
        ]:
            row = {
                "level": level_name,
                "contrast": contrast,
                "group": group_name,
                "n_rows": int(len(group_df)),
                "n_patients": int(group_df["patient_id"].nunique()) if "patient_id" in group_df.columns else np.nan,
            }

            for metric in metrics:
                row[f"{metric}_median_IQR"] = format_median_iqr(group_df[metric], decimals=4)

                s = median_iqr(group_df[metric])
                row[f"{metric}_median"] = s["median"]
                row[f"{metric}_iqr_low"] = s["iqr_low"]
                row[f"{metric}_iqr_high"] = s["iqr_high"]

            rows.append(row)

    return pd.DataFrame(rows)


def make_patient_paired_comparison(patient_df):
    """
    Compare PD vs PD-FS using patient-level central slices.
    This is the most dissertation-friendly R=4 single-contrast comparison.
    """
    central = patient_df[patient_df["is_edge"] == False].copy()

    pd_df = central[central["contrast_label"] == "PD"].copy()
    pdfs_df = central[central["contrast_label"] == "PD-FS"].copy()

    metrics = ["NMSE", "PSNR", "SSIM", "L1"]

    merged = pd_df[["patient_id"] + metrics].merge(
        pdfs_df[["patient_id"] + metrics],
        on="patient_id",
        suffixes=("_PD", "_PDFS"),
    )

    rows = []
    for metric in metrics:
        pd_values = merged[f"{metric}_PD"]
        pdfs_values = merged[f"{metric}_PDFS"]
        diff = pdfs_values - pd_values

        row = {
            "metric": metric,
            "n_patients": int(len(merged)),
            "PD_median_IQR": format_median_iqr(pd_values, decimals=4),
            "PDFS_median_IQR": format_median_iqr(pdfs_values, decimals=4),
            "PDFS_minus_PD_median_IQR": format_median_iqr(diff, decimals=4),
            "PD_median": float(pd_values.median()),
            "PDFS_median": float(pdfs_values.median()),
            "PDFS_minus_PD_median": float(diff.median()),
            "PDFS_over_PD_median_ratio": safe_ratio(float(pdfs_values.median()), float(pd_values.median())),
        }

        # Optional Wilcoxon signed-rank test if scipy is installed.
        try:
            from scipy.stats import wilcoxon

            stat, p = wilcoxon(pdfs_values, pd_values, zero_method="wilcox", alternative="two-sided")
            row["wilcoxon_statistic"] = float(stat)
            row["wilcoxon_p_value"] = float(p)
        except Exception:
            row["wilcoxon_statistic"] = np.nan
            row["wilcoxon_p_value"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows), merged


def save_boxplot(df, metric, out_path, title, patient_level=True):
    """
    Matplotlib-only boxplot. Compatible with older/newer matplotlib versions.
    """
    central_pd = df[(df["contrast_label"] == "PD") & (df["is_edge"] == False)][metric].dropna()
    central_pdfs = df[(df["contrast_label"] == "PD-FS") & (df["is_edge"] == False)][metric].dropna()
    edge_pd = df[(df["contrast_label"] == "PD") & (df["is_edge"] == True)][metric].dropna()
    edge_pdfs = df[(df["contrast_label"] == "PD-FS") & (df["is_edge"] == True)][metric].dropna()

    data = [central_pd, central_pdfs, edge_pd, edge_pdfs]
    labels = ["PD\ncentral", "PD-FS\ncentral", "PD\nedge", "PD-FS\nedge"]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.boxplot(data, showfliers=True)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)

    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.grid(axis="y", alpha=0.3)

    if metric in ["NMSE", "L1"]:
        ax.set_yscale("log")
        ax.set_ylabel(f"{metric} (log scale)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_patient_scatter(merged, metric, out_path):
    pd_col = f"{metric}_PD"
    pdfs_col = f"{metric}_PDFS"

    x = merged[pd_col].astype(float)
    y = merged[pdfs_col].astype(float)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x, y)

    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], linestyle="--")

    ax.set_xlabel(f"PD {metric}")
    ax.set_ylabel(f"PD-FS {metric}")
    ax.set_title(f"Patient-level central: PD vs PD-FS {metric}")
    ax.grid(alpha=0.3)

    if metric in ["NMSE", "L1"]:
        ax.set_xscale("log")
        ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    base_dir = Path("run/eval_single")
    out_dir = base_dir / "R4_analysis"
    fig_dir = out_dir / "figures"

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    slice_df, patient_df = load_data(base_dir)

    slice_df.to_csv(out_dir / "R4_val_slice_level_combined.csv", index=False)
    patient_df.to_csv(out_dir / "R4_val_patient_level_combined.csv", index=False)

    slice_summary = make_summary_table(slice_df, "slice_level")
    patient_summary = make_summary_table(patient_df, "patient_level")

    summary_table = pd.concat([slice_summary, patient_summary], ignore_index=True)
    summary_table.to_csv(out_dir / "R4_val_summary_table.csv", index=False)

    paired_comparison, merged_patient_central = make_patient_paired_comparison(patient_df)
    paired_comparison.to_csv(out_dir / "R4_val_patient_level_paired_comparison.csv", index=False)
    merged_patient_central.to_csv(out_dir / "R4_val_patient_level_central_PD_vs_PDFS_merged.csv", index=False)

    for metric in ["NMSE", "PSNR", "SSIM", "L1"]:
        save_boxplot(
            patient_df,
            metric,
            fig_dir / f"patient_level_{metric}_boxplot.png",
            title=f"R=4 validation patient-level {metric}",
            patient_level=True,
        )

        save_boxplot(
            slice_df,
            metric,
            fig_dir / f"slice_level_{metric}_boxplot.png",
            title=f"R=4 validation slice-level {metric}",
            patient_level=False,
        )

        save_patient_scatter(
            merged_patient_central,
            metric,
            fig_dir / f"patient_level_central_PD_vs_PDFS_{metric}_scatter.png",
        )

    report = {
        "input_base_dir": str(base_dir),
        "output_dir": str(out_dir),
        "n_slice_rows": int(len(slice_df)),
        "n_patient_rows": int(len(patient_df)),
        "n_patients": int(patient_df["patient_id"].nunique()),
        "generated_files": [
            "R4_val_slice_level_combined.csv",
            "R4_val_patient_level_combined.csv",
            "R4_val_summary_table.csv",
            "R4_val_patient_level_paired_comparison.csv",
            "R4_val_patient_level_central_PD_vs_PDFS_merged.csv",
            "figures/patient_level_NMSE_boxplot.png",
            "figures/patient_level_PSNR_boxplot.png",
            "figures/patient_level_SSIM_boxplot.png",
            "figures/patient_level_L1_boxplot.png",
            "figures/slice_level_NMSE_boxplot.png",
            "figures/slice_level_PSNR_boxplot.png",
            "figures/slice_level_SSIM_boxplot.png",
            "figures/slice_level_L1_boxplot.png",
        ],
    }

    with open(out_dir / "R4_analysis_manifest.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\nAnalysis complete.")
    print(f"Output directory: {out_dir}")
    print("\nMain files:")
    print(f"- {out_dir / 'R4_val_summary_table.csv'}")
    print(f"- {out_dir / 'R4_val_patient_level_paired_comparison.csv'}")
    print(f"- {out_dir / 'R4_val_patient_level_central_PD_vs_PDFS_merged.csv'}")
    print(f"- {fig_dir}")


if __name__ == "__main__":
    main()
