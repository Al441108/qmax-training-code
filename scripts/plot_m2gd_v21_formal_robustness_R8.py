#!/usr/bin/env python3
"""Create publication-ready quantitative figures for formal R=8 robustness.

The statistical unit is the patient. Positive paired improvements always favour
the selected Stage-B actual-q model. This script does not recompute model
predictions or select a checkpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update(
    {
        "font.size": 7,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    }
)


STAGEB = "M2GDv21_StageB_actual_q"
CONDITIONS = [
    "correct",
    "shift8_reflect_+x",
    "same_patient_wrong_slice",
    "wrong_patient_matched_level",
    "missing",
]
DISPLAY = {
    "correct": "Correct",
    "shift8_reflect_+x": "Shift 8",
    "same_patient_wrong_slice": "Wrong slice",
    "wrong_patient_matched_level": "Wrong patient",
    "missing": "Missing",
}
COLORS = {
    "q": "#0F4D92",
    "effective": "#42949E",
    "constant_q": "#7884B4",
    "q1": "#B64342",
    "actual": "#0F4D92",
    "m2u": "#767676",
}


def bootstrap_ci(values: Iterable[float], seed: int, iterations: int = 10000) -> Tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size < 2:
        raise ValueError("At least two finite patient values are required.")
    rng = np.random.default_rng(seed)
    sampled = array[rng.integers(0, array.size, size=(iterations, array.size))]
    means = sampled.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.06, label, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom", ha="left")


def save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def require_files(input_dir: Path) -> dict[str, Path]:
    files = {
        "patient": input_dir / "formal_robustness_patient_level.csv",
        "paired": input_dir / "formal_robustness_paired_patient_delta.csv",
        "bootstrap": input_dir / "formal_robustness_paired_bootstrap_summary.csv",
        "decision": input_dir / "formal_robustness_decision.json",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing formal robustness outputs: " + ", ".join(missing))
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    files = require_files(input_dir)
    decision = json.loads(files["decision"].read_text(encoding="utf-8"))
    if not decision.get("formal_robustness_confirmed", False):
        raise RuntimeError("Formal robustness decision is not confirmed; figures were not generated.")

    patient = pd.read_csv(files["patient"])
    paired = pd.read_csv(files["paired"])
    bootstrap = pd.read_csv(files["bootstrap"])
    actual = patient[patient["model"] == STAGEB].copy()
    if actual["patient_id"].nunique() != 25:
        raise RuntimeError("Expected exactly 25 patients in Stage-B patient-level data.")

    fig = plt.figure(figsize=(7.2, 6.0), facecolor="white")
    grid = fig.add_gridspec(2, 2, height_ratios=[0.92, 1.18], hspace=0.38, wspace=0.38)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, :])

    # a | Reliability mechanism. Patient means and patient-bootstrap 95% CI.
    x = np.arange(len(CONDITIONS), dtype=float)
    mechanism_source = []
    for offset, column, label, color in (
        (-0.07, "q_mean", "Reliability $q$", COLORS["q"]),
        (+0.07, "effective_weight_mean", "Effective weight", COLORS["effective"]),
    ):
        means, lows, highs = [], [], []
        for index, condition in enumerate(CONDITIONS):
            values = actual.loc[actual["condition"] == condition, column].to_numpy(float)
            mean = float(np.mean(values))
            lo, hi = bootstrap_ci(values, args.seed + index + (0 if column == "q_mean" else 100))
            means.append(mean); lows.append(lo); highs.append(hi)
            mechanism_source.append(
                {"panel": "a", "condition": condition, "quantity": column,
                 "mean": mean, "ci95_lower": lo, "ci95_upper": hi, "n_patients": len(values)}
            )
        means = np.asarray(means); lows = np.asarray(lows); highs = np.asarray(highs)
        ax_a.errorbar(x + offset, means, yerr=np.vstack([means - lows, highs - means]),
                      fmt="o-", color=color, lw=1.3, ms=4, capsize=2.2, label=label)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([DISPLAY[item] for item in CONDITIONS], rotation=28, ha="right")
    ax_a.set_ylim(-0.04, 1.02)
    ax_a.set_ylabel("Mean fusion weight")
    ax_a.legend(loc="upper right", fontsize=6.4)
    ax_a.grid(axis="y", color="#E4E4E4", lw=0.6)
    panel_label(ax_a, "a")

    # b | Forest plot: primary L1 comparison for the three hard negatives.
    hard = ["shift8_reflect_+x", "same_patient_wrong_slice", "wrong_patient_matched_level"]
    forest_rows = []
    for condition in hard:
        for reference in ("constant_q", "q1"):
            row = bootstrap[(bootstrap["reference"] == reference) &
                            (bootstrap["condition"] == condition) &
                            (bootstrap["metric"] == "L1")]
            if len(row) != 1:
                raise RuntimeError(f"Missing unique L1 bootstrap row for {reference}/{condition}")
            item = row.iloc[0]
            scale = 100.0 / float(item["reference_mean"])
            forest_rows.append(
                {"condition": condition, "reference": reference,
                 "estimate": float(item["paired_improvement_mean"]) * scale,
                 "low": float(item["paired_mean_ci95_lower"]) * scale,
                 "high": float(item["paired_mean_ci95_upper"]) * scale,
                 "n_patients": int(item["num_patients"])}
            )
    y = np.arange(len(forest_rows))[::-1]
    for yi, row in zip(y, forest_rows):
        color = COLORS[row["reference"]]
        ax_b.plot([row["low"], row["high"]], [yi, yi], color=color, lw=1.5)
        ax_b.plot(row["estimate"], yi, "o", color=color, ms=4)
    ax_b.axvline(0, color="#767676", ls="--", lw=0.8)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels([
        f"{DISPLAY[row['condition']]} vs " + ("constant $q$" if row["reference"] == "constant_q" else "$q=1$")
        for row in forest_rows
    ], fontsize=6.5)
    ax_b.set_xlabel("Relative L1 improvement (%)")
    ax_b.grid(axis="x", color="#E4E4E4", lw=0.6)
    ax_b.text(0.99, 0.02, "Positive favours actual $q$", transform=ax_b.transAxes,
              ha="right", va="bottom", color="#606060", fontsize=6)
    panel_label(ax_b, "b")

    # c | Every patient under each hard negative: actual-q versus q=1.
    q1 = paired[(paired["reference"] == "q1") & paired["condition"].isin(hard)].copy()
    rng = np.random.default_rng(args.seed)
    patient_source = []
    for index, condition in enumerate(hard):
        group = q1[q1["condition"] == condition].sort_values("patient_id")
        values = 1000.0 * group["L1_improvement"].to_numpy(float)
        jitter = rng.uniform(-0.12, 0.12, size=len(values))
        ax_c.scatter(np.full(len(values), index) + jitter, values, s=16,
                     facecolor="#DDE8F4", edgecolor=COLORS["actual"], lw=0.65, alpha=0.95)
        mean = float(values.mean())
        lo, hi = bootstrap_ci(values, args.seed + 500 + index)
        ax_c.errorbar(index, mean, yerr=[[mean - lo], [hi - mean]], fmt="D",
                      color=COLORS["actual"], ms=4.2, capsize=3, lw=1.4, zorder=5)
        for _, row in group.iterrows():
            patient_source.append(
                {"panel": "c", "condition": condition, "patient_id": row["patient_id"],
                 "L1_improvement_actual_over_q1": float(row["L1_improvement"])}
            )
    ax_c.axhline(0, color="#767676", ls="--", lw=0.9)
    ax_c.set_xticks(np.arange(len(hard)))
    ax_c.set_xticklabels([DISPLAY[item] for item in hard])
    ax_c.set_ylabel(r"Patient-paired L1 improvement vs $q=1$ ($\times10^{-3}$)")
    ax_c.grid(axis="y", color="#E4E4E4", lw=0.6)
    ax_c.text(0.99, 0.04, "Dots: patients; diamonds: mean and 95% bootstrap CI",
              transform=ax_c.transAxes, ha="right", va="bottom", fontsize=6.2, color="#606060")
    panel_label(ax_c, "c")

    fig.suptitle("Reliability-adaptive fusion preserves clean performance and suppresses unsafe auxiliary input",
                 fontsize=9, fontweight="bold", y=0.995)
    save_figure(fig, output_dir / "Fig1_formal_robustness_quantitative")
    plt.close(fig)

    source = pd.concat(
        [pd.DataFrame(mechanism_source), pd.DataFrame(forest_rows).assign(panel="b"),
         pd.DataFrame(patient_source)], ignore_index=True, sort=False
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    source.to_csv(output_dir / "Fig1_source_data.csv", index=False)
    (output_dir / "Fig1_legend.txt").write_text(
        "Fig. 1 | Reliability-adaptive fusion under controlled auxiliary-input corruption. "
        "a, Patient-level mean reliability q and effective auxiliary weight across five pre-registered "
        "conditions; error bars show patient-bootstrap 95% confidence intervals (n=25 patients). "
        "b, Relative L1 improvement of Stage-B actual-q over constant-q and q=1 controls; intervals are "
        "paired patient-bootstrap 95% confidence intervals. c, Patient-paired L1 improvement over q=1 "
        "for each hard-negative condition. Each point is one patient averaged over 12 fixed slices; "
        "diamonds show the patient mean and 95% bootstrap confidence interval. Positive values favour "
        "actual-q. One selected training run was evaluated; intervals quantify test-patient variability, "
        "not training-seed variability.\n",
        encoding="utf-8",
    )
    print("Saved quantitative evidence to", output_dir)


if __name__ == "__main__":
    main()
