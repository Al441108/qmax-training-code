import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_metric(df, metric_col, ylabel, title, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))

    for contrast in ["PD", "PDFS"]:
        sub = df[df["contrast"] == contrast].sort_values("acceleration")

        label = "PD-FS" if contrast == "PDFS" else "PD"

        ax.plot(
            sub["acceleration"],
            sub[metric_col],
            marker="o",
            linewidth=2,
            label=label,
        )

    ax.set_xlabel("Acceleration factor R")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(sorted(df["acceleration"].unique()))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    summary_csv = Path(args.summary_csv)
    out_dir = Path(args.out_dir)

    df = pd.read_csv(summary_csv)

    required_cols = [
        "acceleration",
        "contrast",
        "NMSE_median",
        "SSIM_median",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    print("Loaded summary:")
    print(df.to_string(index=False))

    plot_metric(
        df=df,
        metric_col="SSIM_median",
        ylabel="Median SSIM",
        title="Zero-filled baseline: SSIM vs acceleration",
        out_path=out_dir / "zero_filled_SSIM_median_vs_R.png",
    )

    plot_metric(
        df=df,
        metric_col="NMSE_median",
        ylabel="Median NMSE",
        title="Zero-filled baseline: NMSE vs acceleration",
        out_path=out_dir / "zero_filled_NMSE_median_vs_R.png",
    )

    print(f"\nSaved figures to: {out_dir}")


if __name__ == "__main__":
    main()

