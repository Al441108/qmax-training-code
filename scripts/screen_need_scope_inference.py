#!/usr/bin/env python3
"""Inference-only screening of residual vs all_auxiliary need scope."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


AUX_KEYS = (
    "q_hat",
    "need_mean",
    "gated_aux_to_target_rms",
    "direct_to_target_rms",
    "residual_to_target_rms",
    "target_auxiliary_cosine",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
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


def tensor_mean(value: torch.Tensor) -> float:
    return float(value.detach().float().mean().cpu().item())


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_patients(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["scope"], row["patient_id"])].append(row)
    result = []
    for (scope, patient_id), group in sorted(grouped.items()):
        output = {
            "scope": scope,
            "patient_id": patient_id,
            "num_slices": len(group),
        }
        numeric = [
            key
            for key, value in group[0].items()
            if key not in {"scope", "patient_id", "slice_idx"}
            and isinstance(value, (int, float))
        ]
        for key in numeric:
            output[key] = float(np.nanmean([float(row[key]) for row in group]))
        result.append(output)
    return result


def bootstrap_ci(values: np.ndarray, replicates: int) -> list[float]:
    rng = np.random.default_rng(20260727)
    samples = rng.choice(
        values, size=(replicates, values.size), replace=True
    ).mean(axis=1)
    return [float(value) for value in np.percentile(samples, [2.5, 97.5])]


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "scripts"))
    from analyze_pd_spatial_helpfulness import build_masks, condition_metrics
    from evaluate_pd_oracle_stage2a import (
        condition_pd,
        make_model,
        prepare_common,
    )
    from src.dataset_paired_multicoil_aux_pd_r2 import (
        PairedMulticoilAuxPDToPDFSDataset,
    )

    metadata_csv = Path(args.metadata_csv).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    for path in (metadata_csv, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    base_config = dict(checkpoint["config"])
    # Compatibility defaults for the original formal prnf_full checkpoint.
    base_config.setdefault("fusion_design", "hybrid_direct_residual")
    base_config.setdefault("need_scope", "residual")
    if base_config.get("variant") != "prnf_full":
        raise ValueError(
            "Inference need-scope screening requires a prnf_full checkpoint; "
            f"got {base_config.get('variant')}"
        )

    models = {}
    for scope in ("residual", "all_auxiliary"):
        config = dict(base_config)
        config["need_scope"] = scope
        model = make_model(config, device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        models[scope] = model

    patient_ids = base_config.get("val_patient_ids") if args.split == "val" else None
    dataset = PairedMulticoilAuxPDToPDFSDataset(
        metadata_csv=str(metadata_csv),
        split=args.split,
        pdfs_acceleration=int(base_config.get("acceleration", 8)),
        pd_aux_acceleration=int(base_config.get("pd_aux_acceleration", 2)),
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

    slice_rows: list[dict[str, Any]] = []
    max_no_pd_scope_difference = 0.0
    with torch.inference_mode():
        for sample_number, batch in enumerate(loader):
            if args.max_samples is not None and sample_number >= args.max_samples:
                break
            kspace, mask, target_tensor = prepare_common(batch, device)
            pd_r2, available = condition_pd(batch, "r2_zf", device)
            pd_zero, unavailable = condition_pd(batch, "no_pd", device)
            predictions = {}
            auxiliaries = {}
            no_pd_predictions = {}
            for scope, model in models.items():
                prediction, aux = model(
                    kspace, mask, pd_r2, available, return_aux=True
                )
                no_pd_prediction, _ = model(
                    kspace, mask, pd_zero, unavailable, return_aux=True
                )
                height, width = target_tensor.shape[-2:]
                top = (prediction.shape[-2] - height) // 2
                left = (prediction.shape[-1] - width) // 2
                predictions[scope] = prediction[
                    ..., top : top + height, left : left + width
                ][0].float().cpu().numpy()
                no_pd_predictions[scope] = no_pd_prediction[
                    ..., top : top + height, left : left + width
                ][0].float().cpu().numpy()
                auxiliaries[scope] = aux

            no_pd_difference = float(
                np.max(
                    np.abs(
                        no_pd_predictions["residual"]
                        - no_pd_predictions["all_auxiliary"]
                    )
                )
            )
            max_no_pd_scope_difference = max(
                max_no_pd_scope_difference, no_pd_difference
            )
            baseline = no_pd_predictions["residual"]
            target = target_tensor[0].float().cpu().numpy()
            masks = build_masks(
                target,
                target - baseline,
                foreground_fraction=0.10,
                high_error_quantile=0.75,
                edge_quantile=0.75,
                active_error_quantile=0.50,
            )
            patient_id = str(batch["patient_id"][0])
            slice_idx = int(batch["slice_idx"][0])
            for scope in ("residual", "all_auxiliary"):
                metrics, _ = condition_metrics(
                    baseline, predictions[scope], target, masks
                )
                row = {
                    "scope": scope,
                    "patient_id": patient_id,
                    "slice_idx": slice_idx,
                    **metrics,
                }
                for key in AUX_KEYS:
                    if key in auxiliaries[scope]:
                        row[key] = tensor_mean(auxiliaries[scope][key])
                slice_rows.append(row)
            if (sample_number + 1) % 50 == 0:
                print(f"Screened {sample_number + 1}/{len(dataset)} slices", flush=True)

    patient_rows = aggregate_patients(slice_rows)
    by_scope = {
        scope: {
            row["patient_id"]: row
            for row in patient_rows
            if row["scope"] == scope
        }
        for scope in ("residual", "all_auxiliary")
    }
    patients = sorted(set(by_scope["residual"]) & set(by_scope["all_auxiliary"]))
    l1_improvement = np.asarray(
        [
            by_scope["residual"][patient]["l1"]
            - by_scope["all_auxiliary"][patient]["l1"]
            for patient in patients
        ],
        dtype=np.float64,
    )
    metric_differences = {}
    for metric in (
        "active_helpful_fraction",
        "active_wrong_direction_fraction",
        "active_harmful_overshoot_fraction",
        "active_abs_delta_error_spearman",
        "mean_helpfulness",
    ):
        values = np.asarray(
            [
                by_scope["all_auxiliary"][patient][metric]
                - by_scope["residual"][patient][metric]
                for patient in patients
            ],
            dtype=np.float64,
        )
        metric_differences[metric] = {
            "definition": "all_auxiliary - residual",
            "mean_difference": float(np.mean(values)),
            "positive_patients": int(np.sum(values > 0)),
            "negative_patients": int(np.sum(values < 0)),
        }

    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_trained_need_scope": base_config.get("need_scope"),
        "warning": (
            "Inference-only screening: the NeedHead was not trained to modulate "
            "the direct path. Negative results cannot rule out trained all_auxiliary."
        ),
        "num_patients": len(patients),
        "num_slices": len(slice_rows) // 2,
        "max_no_pd_scope_output_difference": max_no_pd_scope_difference,
        "all_auxiliary_l1_improvement": {
            "definition": "L1(residual) - L1(all_auxiliary); positive favors all_auxiliary",
            "mean_difference": float(np.mean(l1_improvement)),
            "ci95": bootstrap_ci(l1_improvement, args.bootstrap_replicates),
            "patients_favoring_all_auxiliary": int(np.sum(l1_improvement > 0)),
            "patients_total": len(patients),
        },
        "spatial_metric_differences": metric_differences,
    }
    write_csv(output_dir / "need_scope_screen_slice.csv", slice_rows)
    write_csv(output_dir / "need_scope_screen_patient.csv", patient_rows)
    (output_dir / "need_scope_screen_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
