#!/usr/bin/env python3
"""Fixed-checkpoint PD input intervention: R2-ZF vs full-PD oracle vs No-PD."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


CONDITIONS = ("r2_zf", "full_pd_oracle", "no_pd")
AUX_KEYS = (
    "q_hat",
    "need_mean",
    "gated_aux_to_target_rms",
    "direct_to_target_rms",
    "residual_to_target_rms",
    "raw_auxiliary_to_target_rms",
    "target_auxiliary_cosine",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate R2-ZF/full-PD/no-PD on one frozen Global-direct checkpoint."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def center_crop(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    top = (x.shape[-2] - height) // 2
    left = (x.shape[-1] - width) // 2
    return x[..., top : top + height, left : left + width]


def l1_per_sample(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    scale = target.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    return torch.abs(prediction / scale - target / scale).mean(dim=(-2, -1))


def scalar_per_sample(value: torch.Tensor, batch_size: int) -> list[float]:
    value = value.detach().float().cpu()
    if value.ndim == 0:
        return [float(value.item())] * batch_size
    if value.shape[0] != batch_size:
        raise ValueError(
            f"Auxiliary tensor batch mismatch: {tuple(value.shape)}, batch={batch_size}"
        )
    return value.reshape(batch_size, -1).mean(dim=1).tolist()


def paired_bootstrap_ci(
    values: list[float], replicates: int, seed: int = 20260727
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(replicates, array.size), replace=True).mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


def make_model(config: dict[str, Any], device: torch.device):
    from src.m2_prnf_fusion_pilot_varnet import M2PRNFFusionPilotVarNet

    model = M2PRNFFusionPilotVarNet(
        model_variant=config["variant"],
        fusion_design=config["fusion_design"],
        need_scope=config.get("need_scope", "residual"),
        residual_scale=float(config.get("residual_scale", 0.1)),
        num_cascades=int(config.get("num_cascades", 12)),
        sens_chans=int(config.get("sens_chans", 8)),
        sens_pools=int(config.get("sens_pools", 4)),
        chans=int(config.get("chans", 18)),
        pools=int(config.get("pools", 4)),
        controller_chans=int(config.get("controller_chans", 16)),
        initial_aux_alpha=float(config.get("initial_aux_alpha", 0.1)),
        initial_gate_probability=float(
            config.get("initial_gate_probability", 0.95)
        ),
        initial_need_probability=float(
            config.get("initial_need_probability", 0.95)
        ),
        need_floor=float(config.get("need_floor", 0.25)),
    ).to(device)
    return model


def prepare_common(batch: dict[str, Any], device: torch.device):
    kspace = batch["pdfs_masked_kspace"].to(device, non_blocking=True)
    if not torch.is_complex(kspace):
        raise TypeError(f"Expected complex k-space, got {kspace.dtype}")
    kspace = torch.view_as_real(kspace).float()

    mask = batch["mask"].to(device, non_blocking=True).bool()
    if mask.ndim == 2:
        mask = mask[:, None, None, :, None]
    elif mask.ndim == 1:
        mask = mask[None, None, None, :, None]
    elif mask.ndim != 5:
        raise RuntimeError(f"Unexpected mask shape {tuple(mask.shape)}")

    target = batch["pdfs_target_raw"].to(device, non_blocking=True).float()
    if target.ndim == 4:
        target = target[:, 0]
    return kspace, mask, target


def condition_pd(
    batch: dict[str, Any], condition: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    if condition == "r2_zf":
        pd = batch["pd_aux_image"].to(device, non_blocking=True).float()
        availability = torch.ones(pd.shape[0], device=device)
    elif condition == "full_pd_oracle":
        pd = batch["pd_target_raw"].to(device, non_blocking=True).float().clone()
        flips = batch["pd_flip_lr"]
        if not torch.is_tensor(flips):
            flips = torch.as_tensor(flips)
        for index, flip in enumerate(flips.tolist()):
            if bool(flip):
                pd[index] = torch.flip(pd[index], dims=[-1])
        availability = torch.ones(pd.shape[0], device=device)
    elif condition == "no_pd":
        pd = torch.zeros_like(
            batch["pd_aux_image"].to(device, non_blocking=True).float()
        )
        availability = torch.zeros(pd.shape[0], device=device)
    else:
        raise ValueError(condition)
    if pd.ndim == 4:
        pd = pd[:, 0]
    return pd, availability


def aggregate_patients(slice_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in slice_rows:
        grouped[(row["condition"], row["patient_id"])].append(float(row["l1"]))
    return [
        {
            "condition": condition,
            "patient_id": patient_id,
            "num_slices": len(values),
            "l1": float(np.mean(values)),
        }
        for (condition, patient_id), values in sorted(grouped.items())
    ]


def summarize(
    patient_rows: list[dict[str, Any]], bootstrap_replicates: int
) -> dict[str, Any]:
    by_condition: dict[str, dict[str, float]] = defaultdict(dict)
    for row in patient_rows:
        by_condition[row["condition"]][row["patient_id"]] = float(row["l1"])

    patients = sorted(set.intersection(*(set(by_condition[c]) for c in CONDITIONS)))
    if not patients:
        raise RuntimeError("No patients common to all three conditions")

    condition_means = {
        condition: float(np.mean([by_condition[condition][p] for p in patients]))
        for condition in CONDITIONS
    }
    comparisons = {}
    definitions = {
        "full_improvement_over_r2": ("r2_zf", "full_pd_oracle"),
        "r2_gain_over_no_pd": ("no_pd", "r2_zf"),
        "full_gain_over_no_pd": ("no_pd", "full_pd_oracle"),
    }
    for name, (baseline, candidate) in definitions.items():
        differences = [
            by_condition[baseline][patient] - by_condition[candidate][patient]
            for patient in patients
        ]
        ci_low, ci_high = paired_bootstrap_ci(
            differences, bootstrap_replicates
        )
        comparisons[name] = {
            "definition": f"L1({baseline}) - L1({candidate}); positive favors candidate",
            "mean_difference": float(np.mean(differences)),
            "ci95": [ci_low, ci_high],
            "patients_favoring_candidate": int(np.sum(np.asarray(differences) > 0)),
            "patients_total": len(differences),
            "patient_differences": {
                patient: float(value)
                for patient, value in zip(patients, differences)
            },
        }
    return {
        "num_patients": len(patients),
        "condition_patient_l1": condition_means,
        "comparisons": comparisons,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError(
            "Use --batch-size 1: raw k-space widths vary across patients."
        )
    if args.bootstrap_replicates < 1000:
        raise ValueError("--bootstrap-replicates must be at least 1000")

    project_root = Path(args.project_root).resolve()
    metadata_csv = Path(args.metadata_csv).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    for path in (metadata_csv, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(project_root))

    from src.dataset_paired_multicoil_aux_pd_r2 import (
        PairedMulticoilAuxPDToPDFSDataset,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" not in checkpoint or "config" not in checkpoint:
        raise RuntimeError("Checkpoint must contain model_state_dict and config")
    config = checkpoint["config"]

    if config.get("fusion_design") != "global_direct":
        raise ValueError(
            f"Expected Global-direct, got fusion_design={config.get('fusion_design')}"
        )
    if int(config.get("pd_aux_acceleration", 2)) != 2:
        raise ValueError("Checkpoint was not trained with PD auxiliary R=2")

    patient_ids = config.get(f"{args.split}_patient_ids")
    if args.split == "val":
        patient_ids = config.get("val_patient_ids", patient_ids)
    dataset = PairedMulticoilAuxPDToPDFSDataset(
        metadata_csv=str(metadata_csv),
        split=args.split,
        pdfs_acceleration=int(config.get("acceleration", 8)),
        pd_aux_acceleration=2,
        patient_ids=patient_ids,
        slices_per_patient=None,
        edge_weight=1.0,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = make_model(config, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    slice_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for sample_number, batch in enumerate(loader):
            if args.max_samples is not None and sample_number >= args.max_samples:
                break
            kspace, mask, target = prepare_common(batch, device)
            patient_id = str(batch["patient_id"][0])
            slice_idx = int(batch["slice_idx"][0])
            for condition in CONDITIONS:
                pd, availability = condition_pd(batch, condition, device)
                prediction, aux = model(
                    kspace,
                    mask,
                    pd,
                    availability,
                    return_aux=True,
                )
                prediction = center_crop(
                    prediction, target.shape[-2], target.shape[-1]
                )
                row: dict[str, Any] = {
                    "condition": condition,
                    "patient_id": patient_id,
                    "slice_idx": slice_idx,
                    "l1": float(l1_per_sample(prediction, target)[0].item()),
                }
                for key in AUX_KEYS:
                    if key in aux:
                        row[key] = scalar_per_sample(aux[key], pd.shape[0])[0]
                slice_rows.append(row)
            if (sample_number + 1) % 50 == 0:
                print(f"Evaluated {sample_number + 1}/{len(dataset)} slices", flush=True)

    patient_rows = aggregate_patients(slice_rows)
    summary = summarize(patient_rows, args.bootstrap_replicates)
    audit = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_protocol": config.get("protocol_version"),
        "metadata_csv": str(metadata_csv),
        "metadata_sha256": sha256_file(metadata_csv),
        "split": args.split,
        "device": str(device),
        "num_dataset_slices": len(dataset),
        "max_samples": args.max_samples,
        "conditions": list(CONDITIONS),
        "full_pd_alignment": (
            "pd_target_raw with the dataset's metadata-driven pd_flip_lr applied"
        ),
        "summary": summary,
    }
    write_csv(output_dir / "pd_oracle_slice_metrics.csv", slice_rows)
    write_csv(output_dir / "pd_oracle_patient_metrics.csv", patient_rows)
    (output_dir / "pd_oracle_summary.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
