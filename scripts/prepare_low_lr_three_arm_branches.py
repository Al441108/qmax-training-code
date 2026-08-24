#!/usr/bin/env python3
"""Create three non-destructive epoch-50 branches for the low-LR study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import torch


ARMS = {
    "m2u_augmented": {
        "source": "outputs/m2_prnf/formal_R8_bs4/m2u_augmented_seed42_ep50",
        "target": "outputs/m2_prnf/posthoc_low_lr_50to60/m2u_augmented_seed42",
        "expected": {
            "variant": "m2u_augmented",
            "epochs": 50,
            "learning_rate": 3e-4,
            "seed": 42,
            "acceleration": 8,
            "pd_aux_acceleration": 2,
            "batch_size": 4,
            "grad_accum_steps": 1,
        },
    },
    "global_direct": {
        "source": (
            "outputs/m2_prnf/formal_R8_bs4/"
            "global_direct_prnf_no_need_seed42_ep50"
        ),
        "target": "outputs/m2_prnf/posthoc_low_lr_50to60/global_direct_seed42",
        "expected": {
            "variant": "prnf_no_need",
            "fusion_design": "global_direct",
            "run_stage": "pilot_extension_15_to_50",
            "epochs": 50,
            "learning_rate": 3e-4,
            "seed": 42,
            "acceleration": 8,
            "pd_aux_acceleration": 2,
            "batch_size": 4,
            "grad_accum_steps": 1,
        },
    },
    "hybrid_gain": {
        "source": (
            "outputs/m2_prnf/formal_R8_bs4/"
            "quality_protected_hybrid_gain_seed42_ep50"
        ),
        "target": "outputs/m2_prnf/posthoc_low_lr_50to60/hybrid_gain_seed42",
        "expected": {
            "variant": "prnf_no_need",
            "fusion_design": "hybrid_direct_residual",
            "run_stage": "quality_gain_extension_15_to_50",
            "epochs": 50,
            "learning_rate": 3e-4,
            "seed": 42,
            "acceleration": 8,
            "pd_aux_acceleration": 2,
            "batch_size": 4,
            "grad_accum_steps": 1,
            "lambda_residual_gain": 0.2,
            "residual_gain_margin_relative": 0.002,
            "residual_gain_ramp_epochs": 5,
        },
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source(name: str, source: Path, expected: dict) -> dict:
    required = [
        "model_last.pt",
        "model_best.pt",
        "config.json",
        "final_summary.json",
        "training_log.csv",
        "train_patient_ids.txt",
        "val_patient_ids.txt",
        "run_corruption_audit.json",
    ]
    missing = [item for item in required if not (source / item).is_file()]
    if missing:
        raise RuntimeError(f"{name}: missing source files: {missing}")

    summary = json.loads((source / "final_summary.json").read_text("utf-8"))
    if int(summary.get("completed_epochs", -1)) != 50:
        raise RuntimeError(f"{name}: final_summary is not an epoch-50 run")

    with (source / "training_log.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 50 or int(rows[-1]["epoch"]) != 50:
        raise RuntimeError(f"{name}: training_log does not end at epoch 50")

    checkpoint = torch.load(
        source / "model_last.pt", map_location="cpu", weights_only=False
    )
    for key in (
        "epoch",
        "best_epoch",
        "best_val",
        "model_state_dict",
        "optimizer_state_dict",
        "config",
        "history",
        "rng_state",
        "sampler_next_epoch",
        "run_corruption_audit",
    ):
        if key not in checkpoint:
            raise RuntimeError(f"{name}: model_last.pt is missing {key}")
    if int(checkpoint["epoch"]) != 50:
        raise RuntimeError(f"{name}: model_last checkpoint is not epoch 50")
    if int(checkpoint["sampler_next_epoch"]) != 50:
        raise RuntimeError(f"{name}: sampler/checkpoint epoch mismatch")
    if len(checkpoint["history"]) != 50:
        raise RuntimeError(f"{name}: checkpoint history does not contain 50 epochs")

    observed = {key: checkpoint["config"].get(key) for key in expected}
    if observed != expected:
        raise RuntimeError(
            f"{name}: source configuration mismatch:\n"
            + json.dumps({"expected": expected, "observed": observed}, indent=2)
        )

    return {
        "source_directory": str(source),
        "source_model_last_sha256": sha256_file(source / "model_last.pt"),
        "source_model_best_sha256": sha256_file(source / "model_best.pt"),
        "source_best_epoch": int(checkpoint["best_epoch"]),
        "source_best_val_patient_l1": float(checkpoint["best_val"]),
        "source_last_epoch": 50,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=".")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()

    provenance = {
        "protocol_version": "M2-PRNF-R8-posthoc-lowLR-50to60-three-arm-v1",
        "branch_point": "epoch-50 model_last.pt for every arm",
        "continuation_epochs": [51, 60],
        "continuation_learning_rate": 3e-5,
        "arms": {},
    }

    for name, specification in ARMS.items():
        source = root / specification["source"]
        target = root / specification["target"]
        report = verify_source(name, source, specification["expected"])

        if target.exists():
            marker = target / "branch_provenance.json"
            if not marker.is_file():
                raise RuntimeError(
                    f"{name}: target exists without provenance marker: {target}"
                )
            installed = json.loads(marker.read_text("utf-8"))
            if installed.get("source_model_last_sha256") != report[
                "source_model_last_sha256"
            ]:
                raise RuntimeError(f"{name}: existing branch has another source")
            print(f"{name}: verified existing branch {target}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, target)
            marker = target / "branch_provenance.json"
            marker.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"{name}: created {target}")

        provenance["arms"][name] = {
            **report,
            "branch_directory": str(target),
        }

    output = root / "outputs/m2_prnf/posthoc_low_lr_50to60"
    output.mkdir(parents=True, exist_ok=True)
    (output / "three_arm_branch_manifest.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
