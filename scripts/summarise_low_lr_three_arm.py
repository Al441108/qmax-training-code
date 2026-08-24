#!/usr/bin/env python3
"""Summarise epoch-50 baselines and epoch-51--60 low-LR continuations."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ARMS = {
    "m2u_augmented": Path(
        "outputs/m2_prnf/posthoc_low_lr_50to60/m2u_augmented_seed42"
    ),
    "global_direct": Path(
        "outputs/m2_prnf/posthoc_low_lr_50to60/global_direct_seed42"
    ),
    "hybrid_gain": Path(
        "outputs/m2_prnf/posthoc_low_lr_50to60/hybrid_gain_seed42"
    ),
}


def main() -> None:
    rows = []
    for arm, directory in ARMS.items():
        with (directory / "training_log.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            history = list(csv.DictReader(handle))
        by_epoch = {int(row["epoch"]): row for row in history}
        if 50 not in by_epoch or 60 not in by_epoch:
            raise RuntimeError(f"{arm}: training log is not complete through epoch 60")
        continuation = [by_epoch[epoch] for epoch in range(51, 61)]
        best_continuation = min(
            continuation, key=lambda row: float(row["val_patient_l1"])
        )
        epoch50 = float(by_epoch[50]["val_patient_l1"])
        epoch60 = float(by_epoch[60]["val_patient_l1"])
        branch_summary = json.loads(
            (directory / "final_summary.json").read_text(encoding="utf-8")
        )
        rows.append(
            {
                "arm": arm,
                "epoch50_val_patient_l1": epoch50,
                "epoch60_val_patient_l1": epoch60,
                "epoch60_minus_epoch50_l1": epoch60 - epoch50,
                "best_extension_epoch": int(best_continuation["epoch"]),
                "best_extension_val_patient_l1": float(
                    best_continuation["val_patient_l1"]
                ),
                "best_extension_minus_epoch50_l1": (
                    float(best_continuation["val_patient_l1"]) - epoch50
                ),
                "all_history_best_epoch": int(branch_summary["best_epoch"]),
                "all_history_best_val_patient_l1": float(
                    branch_summary["best_val_patient_l1"]
                ),
            }
        )

    rows.sort(key=lambda row: row["best_extension_val_patient_l1"])
    output = Path("outputs/m2_prnf/posthoc_low_lr_50to60")
    with (output / "three_arm_low_lr_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "protocol_version": "M2-PRNF-R8-posthoc-lowLR-50to60-three-arm-v1",
        "interpretation": (
            "A common low-LR continuation from each arm's epoch-50 model_last; "
            "not an epoch-40 StepLR experiment."
        ),
        "ranking_by_best_extension_l1": [
            row["arm"] for row in rows
        ],
        "results": rows,
    }
    (output / "three_arm_low_lr_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
