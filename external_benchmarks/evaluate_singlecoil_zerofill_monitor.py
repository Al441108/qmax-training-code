#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from singlecoil_paired_dataset_raw import (
    FSMNetSinglecoilRawGridDataset,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_fsmnet_metric(path: Path):
    spec = importlib.util.spec_from_file_location(
        "fsmnet_official_metric",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import FSMNet metric file: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for name in ("nmse", "psnr", "ssim"):
        if not hasattr(module, name):
            raise RuntimeError(
                f"FSMNet metric file is missing function: {name}"
            )

    return module


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def to_2d_numpy(value, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().float().numpy()
    else:
        value = np.asarray(value, dtype=np.float32)

    while value.ndim > 2 and value.shape[0] == 1:
        value = value[0]

    if value.ndim != 2:
        raise RuntimeError(
            f"{name} must resolve to a 2-D image, got {value.shape}"
        )

    if not np.isfinite(value).all():
        raise RuntimeError(f"{name} contains non-finite values")

    return np.asarray(value, dtype=np.float32)


def official_metrics(metric, target: np.ndarray, prediction: np.ndarray):
    if target.shape != prediction.shape:
        raise RuntimeError(
            f"Metric shape mismatch: {target.shape} vs {prediction.shape}"
        )

    # FSMNet functions expect [slice, height, width].
    if target.ndim == 2:
        target = target[None, ...]
        prediction = prediction[None, ...]

    values = {
        "nmse": float(metric.nmse(target, prediction)),
        "psnr": float(metric.psnr(target, prediction)),
        "ssim": float(metric.ssim(target, prediction)),
    }

    if not all(math.isfinite(value) for value in values.values()):
        raise RuntimeError(f"Non-finite metric result: {values}")

    return values


def verify_optional_slice_index(
    sample: Dict,
    expected_slice_index: int,
) -> None:
    for key in ("slice_index", "slice_idx"):
        if key not in sample:
            continue

        value = sample[key]
        if isinstance(value, torch.Tensor):
            value = value.item()

        if int(value) != expected_slice_index:
            raise RuntimeError(
                f"Dataset/QMax slice mismatch: "
                f"dataset {key}={value}, "
                f"QMax slice_index={expected_slice_index}"
            )
        return


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bench-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--fsmnet-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--qmax-eval-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--mask-rng-seed",
        type=int,
        default=1337,
    )

    args = parser.parse_args()

    manifest_path = (
        args.bench_root / "manifests" / "train_monitor.csv"
    )
    metric_path = args.fsmnet_root / "metric.py"
    qmax_summary_path = args.qmax_eval_dir / "summary.json"
    qmax_slice_path = args.qmax_eval_dir / "slice_metrics.csv"
    qmax_volume_path = args.qmax_eval_dir / "volume_metrics.csv"

    for required in (
        manifest_path,
        metric_path,
        qmax_summary_path,
        qmax_slice_path,
        qmax_volume_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    qmax_summary = json.loads(
        qmax_summary_path.read_text(encoding="utf-8")
    )

    if qmax_summary.get("status") != "complete":
        raise RuntimeError("QMax evaluation is not complete")

    if bool(qmax_summary.get("held_out_accessed")):
        raise RuntimeError(
            "QMax input evaluation unexpectedly accessed held-out data"
        )

    if qmax_summary.get("manifest_name") != "train_monitor.csv":
        raise RuntimeError(
            "QMax evaluation did not use train_monitor.csv"
        )

    if int(qmax_summary.get("evaluated_slices", -1)) != 571:
        raise RuntimeError(
            "Expected the completed 571-slice QMax evaluation"
        )

    manifest_sha256 = sha256_file(manifest_path)

    if manifest_sha256 != qmax_summary.get("manifest_sha256"):
        raise RuntimeError(
            "Manifest SHA256 differs from the QMax evaluation"
        )

    metric_sha256 = sha256_file(metric_path)

    if metric_sha256 != qmax_summary.get("fsmnet_metric_sha256"):
        raise RuntimeError(
            "FSMNet metric.py SHA256 differs from the QMax evaluation"
        )

    qmax_slice_rows = read_csv(qmax_slice_path)
    qmax_volume_rows = read_csv(qmax_volume_path)

    if len(qmax_slice_rows) != 571:
        raise RuntimeError(
            f"Expected 571 QMax slice rows, got {len(qmax_slice_rows)}"
        )

    if len(qmax_volume_rows) != 16:
        raise RuntimeError(
            f"Expected 16 QMax volume rows, got {len(qmax_volume_rows)}"
        )

    required_slice_columns = {
        "volume_id",
        "slice_index",
        "nmse",
        "psnr",
        "ssim",
    }
    if not required_slice_columns.issubset(qmax_slice_rows[0]):
        raise RuntimeError(
            "QMax slice_metrics.csv has unexpected columns: "
            f"{list(qmax_slice_rows[0])}"
        )

    required_volume_columns = {
        "volume_id",
        "nmse",
        "psnr",
        "ssim",
    }
    if not required_volume_columns.issubset(qmax_volume_rows[0]):
        raise RuntimeError(
            "QMax volume_metrics.csv has unexpected columns: "
            f"{list(qmax_volume_rows[0])}"
        )

    print("manifest:", manifest_path)
    print("manifest SHA256:", manifest_sha256)
    print("FSMNet metric SHA256:", metric_sha256)
    print("mask RNG seed:", args.mask_rng_seed)
    print("held out:", False)

    metric = load_fsmnet_metric(metric_path)

    dataset = FSMNetSinglecoilRawGridDataset(
        manifest_path=str(manifest_path),
        fsmnet_root=str(args.fsmnet_root),
        mode="train",
        mask_rng_seed=args.mask_rng_seed,
        deterministic_train_mask=True,
    )

    if len(dataset) != len(qmax_slice_rows):
        raise RuntimeError(
            f"Dataset/QMax length mismatch: "
            f"{len(dataset)} vs {len(qmax_slice_rows)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    slice_rows: List[Dict] = []
    volume_targets = defaultdict(list)
    volume_predictions = defaultdict(list)
    volume_slice_indices = defaultdict(list)

    started = time.perf_counter()

    for index in range(len(dataset)):
        sample = dataset[index]
        qmax_row = qmax_slice_rows[index]

        volume_id = str(qmax_row["volume_id"])
        slice_index = int(qmax_row["slice_index"])

        verify_optional_slice_index(sample, slice_index)

        target = to_2d_numpy(
            sample["target_image"],
            "target_image",
        )
        zero_filled = to_2d_numpy(
            sample["zero_filled_crop"],
            "zero_filled_crop",
        )

        values = official_metrics(
            metric,
            target,
            zero_filled,
        )

        slice_rows.append(
            {
                "model_type": "zero_filled",
                "volume_id": volume_id,
                "slice_index": slice_index,
                **values,
            }
        )

        volume_targets[volume_id].append(target)
        volume_predictions[volume_id].append(zero_filled)
        volume_slice_indices[volume_id].append(slice_index)

        completed = index + 1
        if completed == 1 or completed % 25 == 0 or completed == len(dataset):
            print(
                f"evaluated={completed}/{len(dataset)} "
                f"volume={volume_id} "
                f"slice={slice_index} "
                f"PSNR={values['psnr']:.6f} "
                f"SSIM={values['ssim']:.6f}",
                flush=True,
            )

    evaluation_seconds = time.perf_counter() - started

    volume_rows: List[Dict] = []

    for volume_id in volume_targets:
        order = np.argsort(volume_slice_indices[volume_id])

        target_volume = np.stack(
            [volume_targets[volume_id][i] for i in order],
            axis=0,
        )
        prediction_volume = np.stack(
            [volume_predictions[volume_id][i] for i in order],
            axis=0,
        )

        values = official_metrics(
            metric,
            target_volume,
            prediction_volume,
        )

        volume_rows.append(
            {
                "model_type": "zero_filled",
                "volume_id": volume_id,
                "num_slices": int(target_volume.shape[0]),
                **values,
            }
        )

    volume_rows.sort(key=lambda row: row["volume_id"])

    qmax_by_volume = {
        str(row["volume_id"]): row
        for row in qmax_volume_rows
    }
    zf_by_volume = {
        str(row["volume_id"]): row
        for row in volume_rows
    }

    if set(qmax_by_volume) != set(zf_by_volume):
        raise RuntimeError(
            "QMax and zero-filled volume sets differ: "
            f"QMax-only={sorted(set(qmax_by_volume)-set(zf_by_volume))}, "
            f"ZF-only={sorted(set(zf_by_volume)-set(qmax_by_volume))}"
        )

    paired_rows: List[Dict] = []

    for volume_id in sorted(zf_by_volume):
        qmax = qmax_by_volume[volume_id]
        zf = zf_by_volume[volume_id]

        paired_rows.append(
            {
                "volume_id": volume_id,
                "num_slices": int(zf["num_slices"]),
                "zero_filled_nmse": float(zf["nmse"]),
                "qmax_nmse": float(qmax["nmse"]),
                "nmse_reduction": (
                    float(zf["nmse"]) - float(qmax["nmse"])
                ),
                "zero_filled_psnr": float(zf["psnr"]),
                "qmax_psnr": float(qmax["psnr"]),
                "psnr_gain_db": (
                    float(qmax["psnr"]) - float(zf["psnr"])
                ),
                "zero_filled_ssim": float(zf["ssim"]),
                "qmax_ssim": float(qmax["ssim"]),
                "ssim_gain": (
                    float(qmax["ssim"]) - float(zf["ssim"])
                ),
            }
        )

    slice_nmse_mean = float(
        np.mean([row["nmse"] for row in slice_rows])
    )
    slice_psnr_mean = float(
        np.mean([row["psnr"] for row in slice_rows])
    )
    slice_ssim_mean = float(
        np.mean([row["ssim"] for row in slice_rows])
    )

    volume_nmse_mean = float(
        np.mean([row["nmse"] for row in volume_rows])
    )
    volume_psnr_mean = float(
        np.mean([row["psnr"] for row in volume_rows])
    )
    volume_ssim_mean = float(
        np.mean([row["ssim"] for row in volume_rows])
    )

    summary = {
        "status": "complete",
        "model_type": "zero_filled",
        "dataset_mode": "train",
        "manifest_name": manifest_path.name,
        "manifest_sha256": manifest_sha256,
        "fsmnet_metric_sha256": metric_sha256,
        "mask_rng_seed": args.mask_rng_seed,
        "deterministic_train_mask": True,
        "held_out_accessed": False,
        "available_slices": len(dataset),
        "evaluated_slices": len(slice_rows),
        "evaluated_volumes": len(volume_rows),
        "complete_manifest_evaluation": (
            len(slice_rows) == len(dataset) == 571
        ),
        "evaluation_seconds_including_data_loading": (
            evaluation_seconds
        ),
        "slice_nmse_mean": slice_nmse_mean,
        "slice_psnr_mean": slice_psnr_mean,
        "slice_ssim_mean": slice_ssim_mean,
        "volume_nmse_mean": volume_nmse_mean,
        "volume_psnr_mean": volume_psnr_mean,
        "volume_ssim_mean": volume_ssim_mean,
        "qmax_checkpoint_sha256": qmax_summary[
            "checkpoint_sha256"
        ],
        "qmax_checkpoint_update": qmax_summary[
            "checkpoint_update"
        ],
        "qmax_volume_nmse_mean": float(
            qmax_summary["volume_nmse_mean"]
        ),
        "qmax_volume_psnr_mean": float(
            qmax_summary["volume_psnr_mean"]
        ),
        "qmax_volume_ssim_mean": float(
            qmax_summary["volume_ssim_mean"]
        ),
        "qmax_nmse_reduction_mean": float(
            np.mean([row["nmse_reduction"] for row in paired_rows])
        ),
        "qmax_psnr_gain_db_mean": float(
            np.mean([row["psnr_gain_db"] for row in paired_rows])
        ),
        "qmax_ssim_gain_mean": float(
            np.mean([row["ssim_gain"] for row in paired_rows])
        ),
        "qmax_better_nmse_volumes": int(
            sum(row["nmse_reduction"] > 0 for row in paired_rows)
        ),
        "qmax_better_psnr_volumes": int(
            sum(row["psnr_gain_db"] > 0 for row in paired_rows)
        ),
        "qmax_better_ssim_volumes": int(
            sum(row["ssim_gain"] > 0 for row in paired_rows)
        ),
    }

    if not summary["complete_manifest_evaluation"]:
        raise RuntimeError("Incomplete monitor evaluation")

    write_csv(
        args.output_dir / "slice_metrics.csv",
        slice_rows,
    )
    write_csv(
        args.output_dir / "volume_metrics.csv",
        volume_rows,
    )
    write_csv(
        args.output_dir / "paired_qmax_vs_zerofill.csv",
        paired_rows,
    )

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    print("ZERO-FILLED TRAIN-MONITOR AUDIT COMPLETE")


if __name__ == "__main__":
    main()