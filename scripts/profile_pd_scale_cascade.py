#!/usr/bin/env python3
"""Profile recorded PD diagnostics by cascade and scale without training."""

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


PREFERRED_METRICS = (
    "q_hat",
    "need_mean",
    "gated_aux_to_target_rms",
    "direct_to_target_rms",
    "residual_to_target_rms",
    "raw_auxiliary_to_target_rms",
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
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(
    rows: list[dict[str, Any]], group_keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)
    result = []
    excluded = set(group_keys) | {"patient_id", "slice_idx"}
    for key, group in sorted(grouped.items()):
        out = dict(zip(group_keys, key))
        out["num_rows"] = len(group)
        numeric = [
            name
            for name, value in group[0].items()
            if name not in excluded and isinstance(value, (int, float))
        ]
        for name in numeric:
            values = np.asarray([float(item[name]) for item in group], dtype=np.float64)
            out[name] = float(np.nanmean(values))
        result.append(out)
    return result


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "scripts"))

    from evaluate_pd_oracle_stage2a import condition_pd, make_model, prepare_common
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
    config = checkpoint["config"]
    model = make_model(config, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    patient_ids = config.get("val_patient_ids") if args.split == "val" else None
    dataset = PairedMulticoilAuxPDToPDFSDataset(
        metadata_csv=str(metadata_csv),
        split=args.split,
        pdfs_acceleration=int(config.get("acceleration", 8)),
        pd_aux_acceleration=int(config.get("pd_aux_acceleration", 2)),
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
    expected_cascades = int(config.get("num_cascades", 12))
    cell_rows: list[dict[str, Any]] = []
    q_variance_rows: list[dict[str, Any]] = []
    aux_inventory: dict[str, Any] = {}

    with torch.inference_mode():
        for sample_number, batch in enumerate(loader):
            if args.max_samples is not None and sample_number >= args.max_samples:
                break
            kspace, mask, _ = prepare_common(batch, device)
            patient_id = str(batch["patient_id"][0])
            slice_idx = int(batch["slice_idx"][0])
            for condition in ("r2_zf", "full_pd_oracle"):
                pd, availability = condition_pd(batch, condition, device)
                _, aux = model(kspace, mask, pd, availability, return_aux=True)
                if not aux_inventory:
                    aux_inventory = {
                        key: {
                            "type": type(value).__name__,
                            "shape": list(value.shape)
                            if torch.is_tensor(value)
                            else None,
                            "dtype": str(value.dtype)
                            if torch.is_tensor(value)
                            else None,
                        }
                        for key, value in aux.items()
                    }

                matrices: dict[str, np.ndarray] = {}
                for key, value in aux.items():
                    if not torch.is_tensor(value) or value.ndim != 3:
                        continue
                    array = value.detach().float().cpu().numpy()
                    if array.shape[0] != 1 or array.shape[1] != expected_cascades:
                        continue
                    matrices[key] = array[0]
                missing = [
                    key for key in PREFERRED_METRICS if key not in matrices
                ]
                if missing:
                    print(
                        f"Warning: recorded matrices missing for {missing}",
                        flush=True,
                    )
                if not matrices:
                    raise RuntimeError("No [batch,cascade,scale] aux tensors found")

                shapes = {tuple(value.shape) for value in matrices.values()}
                if len(shapes) != 1:
                    raise RuntimeError(f"Inconsistent diagnostic shapes: {shapes}")
                num_cascades, num_scales = next(iter(shapes))
                for cascade in range(num_cascades):
                    for scale in range(num_scales):
                        row = {
                            "condition": condition,
                            "patient_id": patient_id,
                            "slice_idx": slice_idx,
                            "cascade": cascade,
                            "scale": scale,
                        }
                        for key, matrix in matrices.items():
                            row[key] = float(matrix[cascade, scale])
                        cell_rows.append(row)

                if "q_hat" in matrices:
                    q_matrix = matrices["q_hat"]
                    for scale in range(num_scales):
                        q_values = q_matrix[:, scale]
                        q_variance_rows.append(
                            {
                                "condition": condition,
                                "patient_id": patient_id,
                                "slice_idx": slice_idx,
                                "scale": scale,
                                "q_cascade_mean": float(np.mean(q_values)),
                                "q_cascade_std": float(np.std(q_values)),
                                "q_cascade_variance": float(np.var(q_values)),
                                "q_cascade_range": float(
                                    np.max(q_values) - np.min(q_values)
                                ),
                            }
                        )
            if (sample_number + 1) % 50 == 0:
                print(f"Profiled {sample_number + 1}/{len(dataset)} slices", flush=True)

    patient_cells = aggregate(
        cell_rows, ("condition", "patient_id", "cascade", "scale")
    )
    summary_cells = aggregate(
        patient_cells, ("condition", "cascade", "scale")
    )
    patient_q = aggregate(
        q_variance_rows, ("condition", "patient_id", "scale")
    )
    summary_q = aggregate(patient_q, ("condition", "scale"))

    write_csv(output_dir / "scale_cascade_slice.csv", cell_rows)
    write_csv(output_dir / "scale_cascade_patient.csv", patient_cells)
    write_csv(output_dir / "scale_cascade_summary.csv", summary_cells)
    write_csv(output_dir / "q_cascade_variance_patient.csv", patient_q)
    write_csv(output_dir / "q_cascade_variance_summary.csv", summary_q)
    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "fusion_design": config.get("fusion_design"),
        "model_variant": config.get("variant"),
        "num_dataset_slices": len(dataset),
        "num_profiled_slices": len(
            {(row["patient_id"], row["slice_idx"]) for row in cell_rows}
        ),
        "expected_cascades": expected_cascades,
        "conditions": ["r2_zf", "full_pd_oracle"],
        "aux_inventory": aux_inventory,
        "matrix_metrics": sorted(
            {
                key
                for row in cell_rows[:1]
                for key in row
                if key
                not in {"condition", "patient_id", "slice_idx", "cascade", "scale"}
            }
        ),
    }
    (output_dir / "profile_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
