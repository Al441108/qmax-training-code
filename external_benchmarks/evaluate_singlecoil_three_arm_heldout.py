#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from singlecoil_paired_dataset_raw import (
    FSMNetSinglecoilRawGridDataset,
    center_crop_real,
)
from src.m2_prnf_qmax_singlecoil import QMaxSinglecoilFull
from src.m2_prnf_qmax_singlecoil_freqaux import (
    QMaxSinglecoilFullFreqAux,
)


EXPECTED = {
    "train_monitor": {"slices": 571, "volumes": 16, "mode": "train"},
    "heldout": {"slices": 1665, "volumes": 45, "mode": "val"},
}
ARMS = ("zero_filled", "qmax_full", "qmax_frequency")
MODEL_ARMS = ("qmax_full", "qmax_frequency")
METRICS = ("nmse", "psnr", "ssim")
MONITOR_REPRODUCTION_TOLERANCES = {
    "qmax_full": {
        "nmse": 1e-4,
        "psnr": 2e-3,
        "ssim": 1e-4,
    },
    "zero_filled": {
        "nmse": 1e-4,
        "psnr": 2e-3,
        "ssim": 1e-4,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def torch_load(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_metric(path: Path):
    spec = importlib.util.spec_from_file_location("fsmnet_metric_locked", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in METRICS:
        if not hasattr(module, name):
            raise RuntimeError(f"FSMNet metric.py lacks {name}")
    return module


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return value
        return value.item()
    if isinstance(value, np.ndarray) and value.size == 1:
        return value.reshape(-1)[0].item()
    return value


def image2d(value: Any, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    else:
        value = np.asarray(value, dtype=np.float32)
    while value.ndim > 2 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2:
        raise RuntimeError(f"{name} did not resolve to 2-D: {value.shape}")
    if not np.isfinite(value).all():
        raise RuntimeError(f"{name} contains non-finite values")
    return np.asarray(value, dtype=np.float32)


def official_metrics(metric, target: np.ndarray, prediction: np.ndarray):
    if target.shape != prediction.shape:
        raise RuntimeError(f"Metric shape mismatch: {target.shape}/{prediction.shape}")
    if target.ndim == 2:
        target = target[None]
        prediction = prediction[None]
    result = {
        "nmse": float(metric.nmse(target, prediction)),
        "psnr": float(metric.psnr(target, prediction)),
        "ssim": float(metric.ssim(target, prediction)),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise RuntimeError(f"Non-finite metric result: {result}")
    return result


def batch_tensor(sample: Mapping[str, Any], key: str, device: torch.device):
    value = sample[key]
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value.unsqueeze(0).to(device)


class IdentityTracker:
    """Uses dataset metadata, with a deterministic slice-reset fallback."""

    def __init__(self, cohort: str):
        self.cohort = cohort
        self.generated_volume = 0
        self.previous_slice = None

    @staticmethod
    def lookup(sample: Mapping[str, Any], names: Sequence[str]):
        for name in names:
            if name in sample:
                return scalar(sample[name])
        return None

    def get(self, sample: Mapping[str, Any], index: int) -> Tuple[str, int]:
        volume = self.lookup(
            sample,
            ("volume_id", "pair_id", "pair_key", "fsmnet_public_pair_id"),
        )
        slice_index = self.lookup(sample, ("slice_index", "slice_idx"))
        if slice_index is None:
            raise RuntimeError(
                f"Dataset sample {index} has no slice_index/slice_idx; "
                f"keys={sorted(sample.keys())}"
            )
        slice_index = int(slice_index)
        if volume is None:
            if self.previous_slice is None or slice_index <= self.previous_slice:
                self.generated_volume += 1
            volume = f"{self.cohort}_volume_{self.generated_volume:04d}"
        self.previous_slice = slice_index
        return str(volume), slice_index


def verify_checkpoint(
    checkpoint: Mapping[str, Any],
    model_name: str,
    checkpoint_path: Path,
) -> None:
    expected = {
        "model_name": model_name,
        "global_update": 100000,
        "seed": 1337,
        "precision": "fp32",
    }

    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise RuntimeError(
                f"{checkpoint_path}: "
                f"{key}={checkpoint.get(key)!r}, "
                f"expected {value!r}"
            )

    if model_name == "QMaxSinglecoilFullFreqAux":
        if checkpoint.get("batch_size") != 4:
            raise RuntimeError(
                "FreqAux checkpoint batch_size != 4"
            )

        if checkpoint.get("frequency_channels") != 64:
            raise RuntimeError(
                "FreqAux checkpoint frequency_channels != 64"
            )
    else:
        # 原QMax正式checkpoint没有保存batch_size字段。
        # 若未来版本提供该字段，则只接受4。
        if checkpoint.get("batch_size") not in (None, 4):
            raise RuntimeError(
                "QMax checkpoint has an unexpected batch_size"
            )

    if "model_state" not in checkpoint:
        raise RuntimeError(
            f"{checkpoint_path}: model_state missing"
        )


def build_models(qmax_checkpoint: Path, freq_checkpoint: Path, device):
    qmax_state = torch_load(qmax_checkpoint)
    freq_state = torch_load(freq_checkpoint)
    verify_checkpoint(qmax_state, "QMaxSinglecoilFull", qmax_checkpoint)
    verify_checkpoint(freq_state, "QMaxSinglecoilFullFreqAux", freq_checkpoint)

    qmax = QMaxSinglecoilFull(
        qmax_variant="qmax_full",
        num_cascades=12,
        chans=18,
        pools=4,
        controller_chans=16,
        initial_aux_alpha=0.1,
        initial_gate_probability=0.95,
    ).to(device)
    frequency = QMaxSinglecoilFullFreqAux(
        frequency_channels=64,
        crop_size=320,
        qmax_variant="qmax_full",
        num_cascades=12,
        chans=18,
        pools=4,
        controller_chans=16,
        initial_aux_alpha=0.1,
        initial_gate_probability=0.95,
    ).to(device)
    qmax.load_state_dict(qmax_state["model_state"], strict=True)
    frequency.load_state_dict(freq_state["model_state"], strict=True)
    qmax.eval()
    frequency.eval()
    return qmax, frequency


@torch.inference_mode()
def evaluate_cohort(
    *, cohort: str, manifest: Path, mode: str, bench_root: Path,
    fsmnet_root: Path, mask_seed: int, qmax, frequency, metric, device
):
    dataset = FSMNetSinglecoilRawGridDataset(
        manifest_path=str(manifest),
        fsmnet_root=str(fsmnet_root),
        mode=mode,
        mask_rng_seed=mask_seed,
        deterministic_train_mask=True,
    )
    expected = EXPECTED[cohort]
    if len(dataset) != expected["slices"]:
        raise RuntimeError(
            f"{cohort}: expected {expected['slices']} slices, got {len(dataset)}"
        )

    tracker = IdentityTracker(cohort)
    slice_rows: List[Dict[str, Any]] = []
    volume_targets: Dict[str, List[np.ndarray]] = defaultdict(list)
    volume_predictions = {
        arm: defaultdict(list) for arm in ARMS
    }
    volume_slices: Dict[str, List[int]] = defaultdict(list)
    inference_seconds = defaultdict(float)

    for index in range(len(dataset)):
        sample = dataset[index]
        volume_id, slice_index = tracker.get(sample, index)
        masked = batch_tensor(sample, "masked_kspace", device)
        mask = batch_tensor(sample, "mask", device)
        pd_image = batch_tensor(sample, "pd_image", device)
        target_tensor = batch_tensor(sample, "target_image", device).squeeze(1)
        zero_tensor = batch_tensor(sample, "zero_filled_crop", device).squeeze(1)
        available = torch.ones(1, device=device)

        torch.cuda.synchronize()
        started = time.perf_counter()
        qmax_raw = qmax(
            pdfs_masked_kspace=masked,
            mask=mask,
            pd_aux_image=pd_image,
            pd_available=available,
        )
        qmax_image = center_crop_real(qmax_raw, 320)
        torch.cuda.synchronize()
        inference_seconds["qmax_full"] += time.perf_counter() - started

        frequency_mean = zero_tensor.mean(dim=(-2, -1), keepdim=True)
        frequency_std = zero_tensor.std(
            dim=(-2, -1), keepdim=True
        ).clamp_min(1e-11)
        torch.cuda.synchronize()
        started = time.perf_counter()
        freq_output = frequency(
            pdfs_masked_kspace=masked,
            mask=mask,
            pd_aux_image=pd_image,
            pd_available=available,
            frequency_mean=frequency_mean,
            frequency_std=frequency_std,
        )
        frequency_image = freq_output["img_fre"]
        torch.cuda.synchronize()
        inference_seconds["qmax_frequency"] += time.perf_counter() - started

        arrays = {
            "zero_filled": image2d(zero_tensor, "zero_filled"),
            "qmax_full": image2d(qmax_image, "qmax_full"),
            "qmax_frequency": image2d(frequency_image, "qmax_frequency"),
        }
        target = image2d(target_tensor, "target")
        volume_targets[volume_id].append(target)
        volume_slices[volume_id].append(slice_index)

        for arm in ARMS:
            values = official_metrics(metric, target, arrays[arm])
            slice_rows.append({
                "cohort": cohort,
                "model_type": arm,
                "volume_id": volume_id,
                "slice_index": slice_index,
                **values,
            })
            volume_predictions[arm][volume_id].append(arrays[arm])

        completed = index + 1
        if completed == 1 or completed % 25 == 0 or completed == len(dataset):
            print(
                f"{cohort}: {completed}/{len(dataset)} "
                f"volume={volume_id} slice={slice_index}",
                flush=True,
            )

    if len(volume_targets) != expected["volumes"]:
        raise RuntimeError(
            f"{cohort}: expected {expected['volumes']} volumes, "
            f"got {len(volume_targets)}"
        )

    volume_rows: List[Dict[str, Any]] = []
    for volume_id in sorted(volume_targets):
        order = np.argsort(volume_slices[volume_id])
        target_volume = np.stack(
            [volume_targets[volume_id][position] for position in order]
        )
        for arm in ARMS:
            prediction_volume = np.stack(
                [volume_predictions[arm][volume_id][position] for position in order]
            )
            volume_rows.append({
                "cohort": cohort,
                "model_type": arm,
                "volume_id": volume_id,
                "num_slices": int(target_volume.shape[0]),
                **official_metrics(metric, target_volume, prediction_volume),
            })

    timing_rows = [
        {
            "cohort": cohort,
            "model_type": arm,
            "total_inference_seconds": inference_seconds[arm],
            "mean_seconds_per_slice": inference_seconds[arm] / len(dataset),
        }
        for arm in MODEL_ARMS
    ]
    return slice_rows, volume_rows, timing_rows


def cohort_summary(volume_rows: Sequence[Mapping[str, Any]]):
    output = []
    for cohort in EXPECTED:
        for arm in ARMS:
            selected = [
                row for row in volume_rows
                if row["cohort"] == cohort and row["model_type"] == arm
            ]
            output.append({
                "cohort": cohort,
                "model_type": arm,
                "num_volumes": len(selected),
                **{
                    f"volume_{metric}_mean": float(
                        np.mean([float(row[metric]) for row in selected])
                    )
                    for metric in METRICS
                },
            })
    return output


def paired_improvements(volume_rows: Sequence[Mapping[str, Any]]):
    lookup = {
        (row["cohort"], row["model_type"], row["volume_id"]): row
        for row in volume_rows
    }
    rows = []
    for cohort in EXPECTED:
        volume_ids = sorted({
            row["volume_id"] for row in volume_rows
            if row["cohort"] == cohort and row["model_type"] == "zero_filled"
        })
        for model in MODEL_ARMS:
            for volume_id in volume_ids:
                zf = lookup[(cohort, "zero_filled", volume_id)]
                prediction = lookup[(cohort, model, volume_id)]
                rows.append({
                    "cohort": cohort,
                    "model_type": model,
                    "volume_id": volume_id,
                    "nmse_reduction": float(zf["nmse"]) - float(prediction["nmse"]),
                    "psnr_gain_db": float(prediction["psnr"]) - float(zf["psnr"]),
                    "ssim_gain": float(prediction["ssim"]) - float(zf["ssim"]),
                })
    return rows


def bootstrap_gap(
    train_values: np.ndarray, test_values: np.ndarray, resamples: int,
    rng: np.random.Generator,
):
    observed = float(test_values.mean() - train_values.mean())
    train_draws = train_values[
        rng.integers(0, len(train_values), (resamples, len(train_values)))
    ].mean(axis=1)
    test_draws = test_values[
        rng.integers(0, len(test_values), (resamples, len(test_values)))
    ].mean(axis=1)
    samples = test_draws - train_draws
    return {
        "train_mean_improvement": float(train_values.mean()),
        "heldout_mean_improvement": float(test_values.mean()),
        "heldout_minus_train_improvement": observed,
        "bootstrap_ci95_low": float(np.quantile(samples, 0.025)),
        "bootstrap_ci95_high": float(np.quantile(samples, 0.975)),
        "significant_degradation": bool(np.quantile(samples, 0.975) < 0.0),
    }


def overfitting_analysis(
    paired_rows: Sequence[Mapping[str, Any]], resamples: int, seed: int
):
    rng = np.random.default_rng(seed)
    report = {}
    metric_columns = {
        "nmse": "nmse_reduction",
        "psnr": "psnr_gain_db",
        "ssim": "ssim_gain",
    }
    for model in MODEL_ARMS:
        metrics = {}
        for metric, column in metric_columns.items():
            train = np.asarray([
                float(row[column]) for row in paired_rows
                if row["model_type"] == model and row["cohort"] == "train_monitor"
            ])
            test = np.asarray([
                float(row[column]) for row in paired_rows
                if row["model_type"] == model and row["cohort"] == "heldout"
            ])
            metrics[metric] = bootstrap_gap(train, test, resamples, rng)
        count = sum(
            int(result["significant_degradation"])
            for result in metrics.values()
        )
        if count == 3:
            evidence = "strong"
        elif count == 2:
            evidence = "moderate"
        elif count == 1:
            evidence = "weak"
        else:
            evidence = "no_clear_evidence"
        report[model] = {
            "method": "baseline_adjusted_train_to_heldout_gain_gap",
            "interpretation": "negative_gap_means_weaker_generalization",
            "bootstrap_resamples": resamples,
            "significantly_degraded_metrics": count,
            "overfitting_evidence": evidence,
            "metrics": metrics,
        }
    return report


def assert_monitor_reproduction(
    summary_rows: Sequence[Mapping[str, Any]], qmax_summary: Mapping[str, Any],
    zf_summary: Mapping[str, Any]
) -> Dict[str, Any]:
    lookup = {
        (row["cohort"], row["model_type"]): row for row in summary_rows
    }
    checks = (
        (lookup[("train_monitor", "qmax_full")], qmax_summary),
        (lookup[("train_monitor", "zero_filled")], zf_summary),
    )
    records = []
    failures = []
    for observed, reference in checks:
        model_type = str(observed["model_type"])
        for metric in METRICS:
            left = float(observed[f"volume_{metric}_mean"])
            right = float(reference[f"volume_{metric}_mean"])
            difference = abs(left - right)
            tolerance = MONITOR_REPRODUCTION_TOLERANCES[model_type][metric]
            record = {
                "model_type": model_type,
                "metric": metric,
                "observed": left,
                "reference": right,
                "absolute_difference": difference,
                "absolute_tolerance": tolerance,
                "passed": difference <= tolerance,
            }
            records.append(record)
            if not record["passed"]:
                failures.append(record)

    audit = {
        "status": "passed" if not failures else "failed",
        "policy": (
            "Metric-specific absolute tolerances for train-monitor "
            "reproduction only: NMSE/SSIM use 1e-4 and PSNR uses 2e-3 dB "
            "for all monitored arms."
        ),
        "checks": records,
    }
    print(json.dumps({"train_monitor_reproduction": audit}, indent=2))
    if failures:
        raise RuntimeError(
            "Train-monitor reproduction failed: "
            + json.dumps(failures, sort_keys=True)
        )
    return audit


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-root", type=Path, required=True)
    parser.add_argument("--fsmnet-root", type=Path, required=True)
    parser.add_argument("--qmax-checkpoint", type=Path, required=True)
    parser.add_argument("--freq-checkpoint", type=Path, required=True)
    parser.add_argument("--freq-verification", type=Path, required=True)
    parser.add_argument("--qmax-monitor-summary", type=Path, required=True)
    parser.add_argument("--zf-monitor-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mask-seed", type=int, default=1337)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=1337)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    device = torch.device("cuda")
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_manifest = args.bench_root / "manifests" / "train_monitor.csv"
    test_manifest = args.bench_root / "manifests" / "test_locked.csv"
    metric_path = args.fsmnet_root / "metric.py"
    for path in (
        train_manifest, test_manifest, metric_path, args.qmax_checkpoint,
        args.freq_checkpoint, args.freq_verification,
        args.qmax_monitor_summary, args.zf_monitor_summary,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    verification = json.loads(args.freq_verification.read_text())
    if verification.get("status") != "passed":
        raise RuntimeError("FreqAux final verification did not pass")
    if verification.get("held_out_accessed") is not False:
        raise RuntimeError("FreqAux verification has invalid held-out status")
    if verification.get("model_last_sha256") != sha256_file(args.freq_checkpoint):
        raise RuntimeError("FreqAux checkpoint SHA does not match verification")

    qmax_summary = json.loads(args.qmax_monitor_summary.read_text())
    zf_summary = json.loads(args.zf_monitor_summary.read_text())
    if qmax_summary.get("held_out_accessed") is not False:
        raise RuntimeError("QMax monitor summary has invalid held-out status")
    if zf_summary.get("held_out_accessed") is not False:
        raise RuntimeError("ZF monitor summary has invalid held-out status")
    if qmax_summary.get("checkpoint_sha256") != sha256_file(args.qmax_checkpoint):
        raise RuntimeError("QMax checkpoint SHA mismatch")
    train_manifest_hash = sha256_file(train_manifest)
    metric_hash = sha256_file(metric_path)
    for summary in (qmax_summary, zf_summary):
        if summary.get("manifest_sha256") != train_manifest_hash:
            raise RuntimeError("Train-monitor manifest SHA mismatch")
        if summary.get("fsmnet_metric_sha256") != metric_hash:
            raise RuntimeError("FSMNet metric.py SHA mismatch")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    access_lock = args.output_dir / "HELDOUT_ACCESS_STARTED.lock"
    complete_lock = args.output_dir / "HELDOUT_EVALUATION_COMPLETE.lock"
    if access_lock.exists() or complete_lock.exists():
        raise RuntimeError(
            "Held-out access lock already exists; refusing a repeated evaluation"
        )

    print("GPU:", torch.cuda.get_device_name(0))
    print("Building frozen models")
    qmax, frequency = build_models(
        args.qmax_checkpoint, args.freq_checkpoint, device
    )
    metric = load_metric(metric_path)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    print("===== TRAIN-MONITOR REPRODUCTION AND FREQAUX PREFLIGHT =====")
    train_slice, train_volume, train_timing = evaluate_cohort(
        cohort="train_monitor",
        manifest=train_manifest,
        mode="train",
        bench_root=args.bench_root,
        fsmnet_root=args.fsmnet_root,
        mask_seed=args.mask_seed,
        qmax=qmax,
        frequency=frequency,
        metric=metric,
        device=device,
    )
    train_summary_rows = cohort_summary(train_volume)
    monitor_reproduction_audit = assert_monitor_reproduction(
        train_summary_rows, qmax_summary, zf_summary
    )
    atomic_json(
        args.output_dir / "train_monitor_reproduction_audit.json",
        monitor_reproduction_audit,
    )
    print("TRAIN-MONITOR REPRODUCTION PASSED")

    access_record = {
        "status": "HELDOUT_ACCESS_STARTED",
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "heldout_manifest": str(test_manifest),
        "heldout_manifest_sha256": sha256_file(test_manifest),
        "fsmnet_metric_sha256": metric_hash,
        "mask_seed": args.mask_seed,
        "qmax_checkpoint_sha256": sha256_file(args.qmax_checkpoint),
        "freq_checkpoint_sha256": sha256_file(args.freq_checkpoint),
        "arms": list(ARMS),
        "train_monitor_reproduction": monitor_reproduction_audit,
    }
    atomic_json(access_lock, access_record)
    print("===== HELD-OUT ACCESS BEGINS =====", flush=True)

    test_slice, test_volume, test_timing = evaluate_cohort(
        cohort="heldout",
        manifest=test_manifest,
        mode="val",
        bench_root=args.bench_root,
        fsmnet_root=args.fsmnet_root,
        mask_seed=args.mask_seed,
        qmax=qmax,
        frequency=frequency,
        metric=metric,
        device=device,
    )

    slice_rows = train_slice + test_slice
    volume_rows = train_volume + test_volume
    timing_rows = train_timing + test_timing
    summary_rows = cohort_summary(volume_rows)
    paired_rows = paired_improvements(volume_rows)
    overfitting = overfitting_analysis(
        paired_rows, args.bootstrap_resamples, args.bootstrap_seed
    )

    write_csv(args.output_dir / "slice_metrics.csv", slice_rows)
    write_csv(args.output_dir / "volume_metrics.csv", volume_rows)
    write_csv(args.output_dir / "cohort_summary.csv", summary_rows)
    write_csv(args.output_dir / "paired_improvements_over_zerofill.csv", paired_rows)
    write_csv(args.output_dir / "inference_timing.csv", timing_rows)
    atomic_json(args.output_dir / "overfitting_analysis.json", overfitting)

    final_summary = {
        "status": "complete",
        "held_out_accessed": True,
        "complete_heldout_evaluation": True,
        "arms": list(ARMS),
        "train_monitor_slices": EXPECTED["train_monitor"]["slices"],
        "train_monitor_volumes": EXPECTED["train_monitor"]["volumes"],
        "heldout_slices": EXPECTED["heldout"]["slices"],
        "heldout_volumes": EXPECTED["heldout"]["volumes"],
        "mask_seed": args.mask_seed,
        "train_manifest_sha256": train_manifest_hash,
        "heldout_manifest_sha256": sha256_file(test_manifest),
        "fsmnet_metric_sha256": metric_hash,
        "qmax_checkpoint_sha256": sha256_file(args.qmax_checkpoint),
        "freq_checkpoint_sha256": sha256_file(args.freq_checkpoint),
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "train_monitor_reproduction": monitor_reproduction_audit,
        "cohort_results": summary_rows,
        "overfitting": overfitting,
    }
    atomic_json(args.output_dir / "summary.json", final_summary)
    atomic_json(
        complete_lock,
        {
            "status": "HELDOUT_EVALUATION_COMPLETE",
            "completed_at_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
            "summary_sha256": sha256_file(args.output_dir / "summary.json"),
        },
    )
    print(json.dumps(final_summary, indent=2, sort_keys=True))
    print("THREE-ARM FORMAL HELD-OUT EVALUATION PASSED")


if __name__ == "__main__":
    main()
