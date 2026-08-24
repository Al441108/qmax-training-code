#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "epoch",
    "train_pdfs_l1",
    "val_pdfs_volume_l1",
    "val_pdfs_slice_l1",
    "gradient_norm_mean",
    "fusion_diagnostics_json",
]


def parse_run(run_dir: Path):
    log_path = run_dir / "training_log.csv"
    if not log_path.exists():
        return {
            "run_dir": str(run_dir),
            "status": "missing_log",
        }

    df = pd.read_csv(log_path)

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        return {
            "run_dir": str(run_dir),
            "status": f"missing_columns:{missing}",
        }

    config_path = run_dir / "config.json"
    config = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)

    numeric_columns = [
        "train_pdfs_l1",
        "val_pdfs_volume_l1",
        "val_pdfs_slice_l1",
        "gradient_norm_mean",
    ]

    numeric_values = df[numeric_columns].to_numpy(dtype=float)
    has_nonfinite = not np.isfinite(numeric_values).all()

    last3 = df.tail(min(3, len(df)))

    all_ratios = []
    all_alphas = []

    for raw in df["fusion_diagnostics_json"]:
        diagnostics = json.loads(raw)

        for scale_values in diagnostics.values():
            alpha = float(scale_values["alpha"])
            ratio = float(scale_values["aux_to_target_rms"])

            if np.isfinite(alpha):
                all_alphas.append(alpha)

            if np.isfinite(ratio):
                all_ratios.append(ratio)

    max_ratio = max(all_ratios) if all_ratios else float("nan")
    final_mean_ratio = float("nan")
    final_mean_alpha = float("nan")

    if len(df):
        final_diagnostics = json.loads(
            df.iloc[-1]["fusion_diagnostics_json"]
        )

        final_ratios = [
            float(values["aux_to_target_rms"])
            for values in final_diagnostics.values()
        ]
        final_alphas = [
            float(values["alpha"])
            for values in final_diagnostics.values()
        ]

        final_mean_ratio = float(np.mean(final_ratios))
        final_mean_alpha = float(np.mean(final_alphas))

    hard_fail_reasons = []

    if has_nonfinite:
        hard_fail_reasons.append("nonfinite_metric")

    if np.isfinite(max_ratio) and max_ratio >= 1.0:
        hard_fail_reasons.append("aux_ratio_ge_1")

    if len(all_alphas) and not np.isfinite(np.asarray(all_alphas)).all():
        hard_fail_reasons.append("nonfinite_alpha")

    warning_reasons = []

    if np.isfinite(max_ratio) and 0.75 <= max_ratio < 1.0:
        warning_reasons.append("aux_ratio_ge_0.75")

    return {
        "run_dir": str(run_dir),
        "status": "fail" if hard_fail_reasons else "pass",
        "initial_aux_alpha": float(
            config.get("initial_aux_alpha", np.nan)
        ),
        "seed": int(config.get("seed", -1)),
        "completed_epochs": int(df["epoch"].max()),
        "last3_val_volume_l1_mean": float(
            last3["val_pdfs_volume_l1"].mean()
        ),
        "last3_val_volume_l1_std": float(
            last3["val_pdfs_volume_l1"].std(ddof=1)
        ) if len(last3) > 1 else float("nan"),
        "best_val_volume_l1": float(
            df["val_pdfs_volume_l1"].min()
        ),
        "final_val_volume_l1": float(
            df.iloc[-1]["val_pdfs_volume_l1"]
        ),
        "final_train_l1": float(
            df.iloc[-1]["train_pdfs_l1"]
        ),
        "gradient_norm_mean_all_epochs": float(
            df["gradient_norm_mean"].mean()
        ),
        "gradient_norm_max": float(
            df["gradient_norm_mean"].max()
        ),
        "max_aux_to_target_rms": float(max_ratio),
        "final_mean_aux_to_target_rms": final_mean_ratio,
        "final_mean_learned_alpha": final_mean_alpha,
        "hard_fail_reasons": ";".join(hard_fail_reasons),
        "warnings": ";".join(warning_reasons),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    args = parser.parse_args()

    run_dirs = sorted(
        path.parent
        for path in args.root.rglob("training_log.csv")
    )

    rows = [parse_run(run_dir) for run_dir in run_dirs]
    result = pd.DataFrame(rows)

    if len(result) == 0:
        raise RuntimeError(
            f"No training_log.csv found under {args.root}"
        )

    if "last3_val_volume_l1_mean" in result.columns:
        result = result.sort_values(
            [
                "status",
                "last3_val_volume_l1_mean",
                "initial_aux_alpha",
            ],
            na_position="last",
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)

    display_columns = [
        "initial_aux_alpha",
        "seed",
        "completed_epochs",
        "status",
        "last3_val_volume_l1_mean",
        "last3_val_volume_l1_std",
        "final_val_volume_l1",
        "gradient_norm_max",
        "max_aux_to_target_rms",
        "final_mean_aux_to_target_rms",
        "final_mean_learned_alpha",
        "hard_fail_reasons",
        "warnings",
    ]

    print(result[display_columns].to_string(index=False))
    print()
    print("Saved:", args.output_csv)


if __name__ == "__main__":
    main()
