#!/usr/bin/env python3
from __future__ import annotations

"""Locked R=8 comparison of two final fusion models, three M2-U controls,
and a same-manifest zero-filled reconstruction.

Primary clean evaluation:
  * 25 held-out patients / 878 slices from full_clean_manifest.json
  * patient-equal NMSE, PSNR, SSIM and L1
  * patient-level paired bootstrap (10,000 resamples by default)

Robustness evaluation:
  * correct, shift8, same-patient wrong slice, wrong patient and missing PD
  * actual-q for every learned model
  * actual-q versus frozen constant-q and q=1 for both fusion candidates
  * residual-on versus residual-off for the quality-gain fifth arm

The legacy 1,203-slice zero-filled table supplied by the investigator is
written as an external reference only.  It is never entered into paired tests.
"""

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import fastmri
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_m2_prnf import (  # noqa: E402
    IndexedDataset,
    ShapeBucketBatchSampler,
    make_dataset,
    prepare_batch,
    runtime_versions,
    set_seed,
    sha256_file,
)
from scripts.train_m2_prnf_quality_gain import residual_scale_override  # noqa: E402
from src.fft_utils import center_crop  # noqa: E402
from src.m2_prnf_fusion_pilot_varnet import (  # noqa: E402
    M2PRNFFusionPilotVarNet,
)
from src.m2_prnf_varnet import M2PRNFAuxPDVarNet  # noqa: E402

# Reuse the already audited metric, corruption and bootstrap implementations.
from scripts.evaluate_m2_prnf_quality_gain_R8 import (  # noqa: E402
    CONDITION_MANIFEST_PROTOCOL_VERSION,
    MANIFEST_PROTOCOL_VERSION,
    ManifestDataset,
    condition_pd,
    metrics,
)

PROTOCOL_VERSION = "M2-PRNF-R8-v1.6.0-final-five-model-comparison-audited"
CONDITIONS = ("correct", "shift8", "wrong_slice", "wrong_patient", "missing")
ROBUST_CONDITIONS = ("shift8", "wrong_slice", "wrong_patient")
METRICS = ("nmse", "psnr", "ssim", "l1")
DIAGNOSTICS = (
    "q",
    "need",
    "effective_weight",
    "gated_rms",
    "direct_rms",
    "residual_rms",
    "residual_rms_max",
    "residual_direct_ratio",
    "raw_auxiliary_rms",
    "feature_cosine",
    "diagnostics_finite",
)
BASELINE_NAMES = ("m2u_clean", "m2u_augmented", "m2u_augcap_mask")
FUSION_NAMES = ("global_direct", "quality_protected_hybrid_gain")
ACTUAL_MODEL_NAMES = (*BASELINE_NAMES, *FUSION_NAMES)
ZERO_EXACT = "zero_filled_exact_manifest"
GAIN_MODEL = "quality_protected_hybrid_gain"
GAIN_OFF = "quality_protected_hybrid_gain_residual_off"

LEGACY_ZERO_FILLED_ROWS = (
    {
        "mask_type": "gaussian_vd", "acceleration": 4, "contrast": "PD",
        "n_slices": 1203, "NMSE_mean": 0.02004863919766894,
        "NMSE_median": 0.014316015876829624,
        "PSNR_mean": 29.407590973902423,
        "PSNR_median": 29.327974319458008,
        "SSIM_mean": 0.7941131397251973,
        "SSIM_median": 0.8002018401921516,
    },
    {
        "mask_type": "gaussian_vd", "acceleration": 4, "contrast": "PDFS",
        "n_slices": 1203, "NMSE_mean": 0.05009313964487014,
        "NMSE_median": 0.03333534300327301,
        "PSNR_mean": 26.518818050647713,
        "PSNR_median": 27.313507080078125,
        "SSIM_mean": 0.6877886786604834,
        "SSIM_median": 0.7167528183249071,
    },
    {
        "mask_type": "gaussian_vd", "acceleration": 6, "contrast": "PD",
        "n_slices": 1203, "NMSE_mean": 0.0341468013208344,
        "NMSE_median": 0.027183694764971733,
        "PSNR_mean": 26.807945866636306,
        "PSNR_median": 26.642528533935547,
        "SSIM_mean": 0.7203507985475588,
        "SSIM_median": 0.7238421859742832,
    },
    {
        "mask_type": "gaussian_vd", "acceleration": 6, "contrast": "PDFS",
        "n_slices": 1203, "NMSE_mean": 0.07127829081536033,
        "NMSE_median": 0.048491425812244415,
        "PSNR_mean": 24.887298627585444,
        "PSNR_median": 25.70516586303711,
        "SSIM_mean": 0.6153563437158462,
        "SSIM_median": 0.6438370045587283,
    },
    {
        "mask_type": "gaussian_vd", "acceleration": 8, "contrast": "PD",
        "n_slices": 1203, "NMSE_mean": 0.053066051522531514,
        "NMSE_median": 0.04504683241248131,
        "PSNR_mean": 24.622065346338108,
        "PSNR_median": 24.353601455688477,
        "SSIM_mean": 0.6653609205014905,
        "SSIM_median": 0.6697459566032442,
    },
    {
        "mask_type": "gaussian_vd", "acceleration": 8, "contrast": "PDFS",
        "n_slices": 1203, "NMSE_mean": 0.08818617906631997,
        "NMSE_median": 0.06279264390468597,
        "PSNR_mean": 23.80399918813856,
        "PSNR_median": 24.636775970458984,
        "SSIM_mean": 0.5714150951995483,
        "SSIM_median": 0.5991870416168595,
    },
)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_summary(path: Path) -> Mapping[str, Any]:
    summary_path = path.parent / "final_summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"{path}: missing final_summary.json")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _audit_common_checkpoint(
    path: Path,
    checkpoint: Mapping[str, Any],
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    clean_hash: str,
    robust_hash: str,
    metadata_hash: str,
    require_metadata_hash: bool,
) -> int:
    if path.name != "model_best.pt":
        raise RuntimeError(f"Expected model_best.pt, got {path}")
    if int(config.get("acceleration", -1)) != 8:
        raise RuntimeError(f"{path}: acceleration is not R=8")
    if int(config.get("epochs", -1)) != 50:
        raise RuntimeError(f"{path}: configured training budget is not 50 epochs")
    if int(summary.get("completed_epochs", -1)) != 50:
        raise RuntimeError(f"{path}: run did not complete 50 epochs")
    selected = int(checkpoint.get("epoch", -1))
    if not (
        selected
        == int(checkpoint.get("best_epoch", -2))
        == int(summary.get("best_epoch", -3))
    ):
        raise RuntimeError(f"{path}: selected/checkpoint/summary best epoch mismatch")
    if not math.isclose(
        float(checkpoint.get("best_val", float("nan"))),
        float(summary.get("best_val_patient_l1", float("nan"))),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(f"{path}: checkpoint/final-summary best L1 mismatch")
    expected_hashes = {
        "full_clean_manifest_sha256": clean_hash,
        "robustness_manifest_sha256": robust_hash,
    }
    if require_metadata_hash:
        expected_hashes["metadata_sha256"] = metadata_hash
    for key, expected in expected_hashes.items():
        if config.get(key) != expected:
            raise RuntimeError(
                f"{path}: {key} mismatch: {config.get(key)} != {expected}"
            )
    return selected


def load_baseline(
    path: Path,
    expected_variant: str,
    device: torch.device,
    clean_hash: str,
    robust_hash: str,
    metadata_hash: str,
):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    summary = _load_summary(path)
    selected = _audit_common_checkpoint(
        path,
        checkpoint,
        config,
        summary,
        clean_hash,
        robust_hash,
        metadata_hash,
        require_metadata_hash=False,
    )
    if config.get("variant") != expected_variant:
        raise RuntimeError(
            f"{path}: expected {expected_variant}, got {config.get('variant')}"
        )
    if summary.get("variant") != expected_variant:
        raise RuntimeError(f"{path}: final-summary variant mismatch")
    model = M2PRNFAuxPDVarNet(
        variant=expected_variant,
        num_cascades=int(config["num_cascades"]),
        sens_chans=int(config["sens_chans"]),
        sens_pools=int(config["sens_pools"]),
        chans=int(config["chans"]),
        pools=int(config["pools"]),
        controller_chans=int(config["controller_chans"]),
        initial_aux_alpha=float(config["initial_aux_alpha"]),
        initial_gate_probability=float(config["initial_gate_probability"]),
        initial_need_probability=float(config["initial_need_probability"]),
        need_floor=float(config["need_floor"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, config, summary, selected


def load_fusion(
    path: Path,
    expected_name: str,
    device: torch.device,
    clean_hash: str,
    robust_hash: str,
    condition_hash: str,
    metadata_hash: str,
):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    summary = _load_summary(path)
    selected = _audit_common_checkpoint(
        path,
        checkpoint,
        config,
        summary,
        clean_hash,
        robust_hash,
        metadata_hash,
        require_metadata_hash=True,
    )
    if config.get("condition_manifest_sha256") != condition_hash:
        raise RuntimeError(f"{path}: condition-manifest hash mismatch")
    if config.get("variant") != "prnf_no_need":
        raise RuntimeError(f"{path}: expected prnf_no_need")

    expected = {
        "global_direct": {
            "fusion_design": "global_direct",
            "run_stage": "pilot_extension_15_to_50",
            "protocol_version": (
                "M2-PRNF-R8-v1.4.1-global-direct-continuation-15-to-50-audited"
            ),
        },
        "quality_protected_hybrid_gain": {
            "fusion_design": "hybrid_direct_residual",
            "run_stage": "quality_gain_extension_15_to_50",
            "protocol_version": (
                "M2-PRNF-R8-v1.5.0-quality-gain-continuation-15-to-50-audited"
            ),
        },
    }[expected_name]
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(
                f"{path}: {key} mismatch: {config.get(key)!r} != {value!r}"
            )
    if expected_name == GAIN_MODEL:
        gain_protocol = {
            "training_objective": "quality_protected_residual_incremental_gain",
            "lambda_residual_gain": 0.2,
            "residual_gain_margin_relative": 0.002,
        }
        for key, value in gain_protocol.items():
            if config.get(key) != value:
                raise RuntimeError(f"{path}: gain protocol mismatch for {key}")
    if summary.get("fusion_design") != expected["fusion_design"]:
        raise RuntimeError(f"{path}: final-summary fusion-design mismatch")

    model = M2PRNFFusionPilotVarNet(
        model_variant="prnf_no_need",
        fusion_design=str(config["fusion_design"]),
        need_scope=str(config["need_scope"]),
        residual_scale=float(config["residual_scale"]),
        num_cascades=int(config["num_cascades"]),
        sens_chans=int(config["sens_chans"]),
        sens_pools=int(config["sens_pools"]),
        chans=int(config["chans"]),
        pools=int(config["pools"]),
        controller_chans=int(config["controller_chans"]),
        initial_aux_alpha=float(config["initial_aux_alpha"]),
        initial_gate_probability=float(config["initial_gate_probability"]),
        initial_need_probability=float(config["initial_need_probability"]),
        need_floor=float(config["need_floor"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, config, summary, selected


FAIRNESS_KEYS = (
    "metadata_csv",
    "acceleration",
    "pd_aux_acceleration",
    "epochs",
    "learning_rate",
    "batch_size",
    "grad_accum_steps",
    "num_train_patients",
    "num_val_patients",
    "num_cascades",
    "chans",
    "sens_chans",
    "pools",
    "sens_pools",
    "controller_chans",
    "initial_aux_alpha",
    "initial_gate_probability",
    "initial_need_probability",
    "need_floor",
    "seed",
    "train_patient_ids",
    "val_patient_ids",
    "checkpoint_selection_metric",
    "full_clean_manifest_sha256",
    "robustness_manifest_sha256",
    "optimizer",
    "gradient_clip_norm",
)


def fairness_audit(configs: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    reference_name = "m2u_augmented"
    reference = configs[reference_name]
    mismatches = []
    for model_name, config in configs.items():
        for key in FAIRNESS_KEYS:
            if config.get(key) != reference.get(key):
                mismatches.append(
                    {
                        "model": model_name,
                        "key": key,
                        "reference_model": reference_name,
                        "reference": reference.get(key),
                        "observed": config.get(key),
                    }
                )
    augmented = (
        "m2u_augmented",
        "m2u_augcap_mask",
        "global_direct",
        "quality_protected_hybrid_gain",
    )
    mixtures = {
        name: configs[name].get("corrupt_view_mixture") for name in augmented
    }
    mixture_passed = len(
        {json.dumps(value, sort_keys=True) for value in mixtures.values()}
    ) == 1
    clean_mixture_ok = configs["m2u_clean"].get("corrupt_view_mixture") in ({}, None)
    return {
        "reference": reference_name,
        "checked_shared_keys": list(FAIRNESS_KEYS),
        "shared_configuration_passed": not mismatches,
        "shared_configuration_mismatches": mismatches,
        "augmented_mixture_passed": mixture_passed,
        "augmented_mixtures": mixtures,
        "m2u_clean_has_no_corruption_mixture": clean_mixture_ok,
        "metadata_provenance": {
            name: {
                "metadata_csv": config.get("metadata_csv"),
                "metadata_sha256_recorded": config.get("metadata_sha256"),
            }
            for name, config in configs.items()
        },
        "metadata_hash_limitation": (
            "The locked v1.3 M2-U checkpoints predate metadata SHA-256 logging. "
            "Their metadata_csv path is audited, while the two continuation "
            "checkpoints additionally require the current metadata SHA-256."
        ),
        "passed": not mismatches and mixture_passed and clean_mixture_ok,
    }


def _tensor_stat(
    aux: Mapping[str, Any],
    key: str,
    index: int,
    reducer: str = "mean",
    default: float = 0.0,
) -> float:
    value = aux.get(key)
    if not isinstance(value, torch.Tensor):
        return float(default)
    selected = value[index].detach().float()
    if not selected.numel():
        return float(default)
    if reducer == "max":
        return float(selected.max().item())
    if reducer == "valid_mean":
        valid = selected[selected >= 0]
        return float(valid.mean().item()) if valid.numel() else -1.0
    return float(selected.mean().item())


def _diagnostics(aux: Mapping[str, Any], index: int) -> Dict[str, float]:
    tensors = [
        value[index]
        for value in aux.values()
        if isinstance(value, torch.Tensor) and value.shape[0] > index
    ]
    return {
        "q": _tensor_stat(aux, "q", index),
        "need": _tensor_stat(aux, "need_mean", index, default=1.0),
        "effective_weight": _tensor_stat(
            aux, "effective_weight_mean", index, default=1.0
        ),
        "gated_rms": _tensor_stat(aux, "gated_aux_to_target_rms", index),
        "direct_rms": _tensor_stat(aux, "direct_to_target_rms", index),
        "residual_rms": _tensor_stat(aux, "residual_to_target_rms", index),
        "residual_rms_max": _tensor_stat(
            aux, "residual_to_target_rms", index, reducer="max"
        ),
        "residual_direct_ratio": _tensor_stat(
            aux, "residual_to_direct_rms_ratio", index, reducer="valid_mean",
            default=-1.0,
        ),
        "raw_auxiliary_rms": _tensor_stat(
            aux, "raw_auxiliary_to_target_rms", index
        ),
        "feature_cosine": _tensor_stat(aux, "target_auxiliary_cosine", index),
        "diagnostics_finite": float(
            all(bool(torch.isfinite(value).all()) for value in tensors)
        ),
    }


def _forward_timed(
    model,
    kspace,
    mask,
    pd_used,
    available,
    reliability_override,
    need_override,
) -> Tuple[torch.Tensor, Mapping[str, Any], float]:
    if kspace.is_cuda:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        prediction, aux = model(
            kspace,
            mask,
            pd_used,
            available,
            return_aux=True,
            reliability_override=reliability_override,
            need_override=need_override,
        )
        end.record()
        torch.cuda.synchronize()
        seconds = float(start.elapsed_time(end)) / 1000.0
    else:
        started = time.perf_counter()
        prediction, aux = model(
            kspace,
            mask,
            pd_used,
            available,
            return_aux=True,
            reliability_override=reliability_override,
            need_override=need_override,
        )
        seconds = time.perf_counter() - started
    return prediction, aux, seconds


@torch.no_grad()
def evaluate_model(
    model,
    model_name: str,
    loader,
    full_dataset,
    condition_lookup,
    device,
    cohort: str,
    condition: str,
    reliability_override: Optional[float] = None,
    need_override: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    forward_seconds = 0.0
    batches = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for batch in loader:
        kspace, mask, pd, target, indices = prepare_batch(batch, device)
        pd_used, available = condition_pd(
            pd, indices, full_dataset, condition_lookup, condition
        )
        prediction, aux, seconds = _forward_timed(
            model,
            kspace,
            mask,
            pd_used,
            available,
            reliability_override,
            need_override,
        )
        forward_seconds += seconds
        batches += 1
        prediction = center_crop(prediction, target.shape[-2], target.shape[-1])
        for index in range(target.shape[0]):
            rows.append(
                {
                    "model": model_name,
                    "cohort": cohort,
                    "condition": condition,
                    "patient_id": str(batch["patient_id"][index]),
                    "slice_idx": int(batch["slice_idx"][index]),
                    **metrics(prediction[index], target[index]),
                    **_diagnostics(aux, index),
                }
            )
    timing = {
        "model": model_name,
        "cohort": cohort,
        "condition": condition,
        "num_slices": len(rows),
        "num_batches": batches,
        "forward_seconds": forward_seconds,
        "forward_ms_per_slice": 1000.0 * forward_seconds / max(len(rows), 1),
        "peak_gpu_memory_gb": (
            float(torch.cuda.max_memory_allocated(device)) / 1024 ** 3
            if device.type == "cuda"
            else 0.0
        ),
    }
    return rows, timing


@torch.no_grad()
def evaluate_zero_filled(loader, device) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seconds = 0.0
    batches = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for batch in loader:
        kspace, _mask, _pd, target, _indices = prepare_batch(batch, device)
        if kspace.is_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            prediction = fastmri.rss(
                fastmri.complex_abs(fastmri.ifft2c(kspace)), dim=1
            )
            end.record()
            torch.cuda.synchronize()
            seconds += float(start.elapsed_time(end)) / 1000.0
        else:
            started = time.perf_counter()
            prediction = fastmri.rss(
                fastmri.complex_abs(fastmri.ifft2c(kspace)), dim=1
            )
            seconds += time.perf_counter() - started
        batches += 1
        prediction = center_crop(prediction, target.shape[-2], target.shape[-1])
        for index in range(target.shape[0]):
            rows.append(
                {
                    "model": ZERO_EXACT,
                    "cohort": "full_clean",
                    "condition": "correct",
                    "patient_id": str(batch["patient_id"][index]),
                    "slice_idx": int(batch["slice_idx"][index]),
                    **metrics(prediction[index], target[index]),
                    **{key: float("nan") for key in DIAGNOSTICS},
                }
            )
    timing = {
        "model": ZERO_EXACT,
        "cohort": "full_clean",
        "condition": "correct",
        "num_slices": len(rows),
        "num_batches": batches,
        "forward_seconds": seconds,
        "forward_ms_per_slice": 1000.0 * seconds / max(len(rows), 1),
        "peak_gpu_memory_gb": (
            float(torch.cuda.max_memory_allocated(device)) / 1024 ** 3
            if device.type == "cuda"
            else 0.0
        ),
    }
    return rows, timing


def patient_average(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row["model"]),
                str(row["cohort"]),
                str(row["condition"]),
                str(row["patient_id"]),
            )
        ].append(row)
    output = []
    for (model, cohort, condition, patient), values in sorted(groups.items()):
        output.append(
            {
                "model": model,
                "cohort": cohort,
                "condition": condition,
                "patient_id": patient,
                "num_slices": len(values),
                **{
                    key: float(np.nanmean([float(row[key]) for row in values]))
                    for key in (*METRICS, *DIAGNOSTICS)
                },
            }
        )
    return output


def summary_rows(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (str(row["model"]), str(row["cohort"]), str(row["condition"]))
        ].append(row)
    output = []
    for (model, cohort, condition), values in sorted(groups.items()):
        output.append(
            {
                "model": model,
                "cohort": cohort,
                "condition": condition,
                "num_patients": len(values),
                "num_slices": int(sum(int(row["num_slices"]) for row in values)),
                "aggregation": "patient_equal_macro_mean",
                **{
                    key: float(np.nanmean([float(row[key]) for row in values]))
                    for key in (*METRICS, *DIAGNOSTICS)
                },
            }
        )
    return output


def _paired_values(rows, model_a, model_b, cohort, condition, metric):
    values = {
        (str(row["model"]), str(row["patient_id"])): float(row[metric])
        for row in rows
        if row["cohort"] == cohort and row["condition"] == condition
    }
    patients = sorted(
        patient
        for model, patient in values
        if model == model_a and (model_b, patient) in values
    )
    if not patients:
        raise RuntimeError(
            f"No paired patients for {model_a}/{model_b}/{cohort}/{condition}"
        )
    return patients, values


def paired_bootstrap(
    rows,
    model_a,
    model_b,
    cohort,
    condition,
    metric,
    resamples,
    seed,
):
    patients, values = _paired_values(
        rows, model_a, model_b, cohort, condition, metric
    )
    direction = -1.0 if metric in {"nmse", "l1"} else 1.0
    improvement = np.asarray(
        [
            direction
            * (values[(model_a, patient)] - values[(model_b, patient)])
            for patient in patients
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    bootstrap = improvement[
        rng.integers(0, len(improvement), (resamples, len(improvement)))
    ].mean(axis=1)
    return {
        "model_a": model_a,
        "model_b": model_b,
        "cohort": cohort,
        "condition": condition,
        "metric": metric,
        "positive_means_model_a_better": True,
        "mean_improvement": float(improvement.mean()),
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
        "patients_a_better": int((improvement > 0).sum()),
        "patients_equal": int((improvement == 0).sum()),
        "num_patients": len(patients),
    }


def noninferiority_l1(
    rows,
    model,
    control,
    cohort,
    condition,
    margin,
    resamples,
    seed,
):
    patients, values = _paired_values(
        rows, model, control, cohort, condition, "l1"
    )
    degradation = np.asarray(
        [
            (values[(model, patient)] - values[(control, patient)])
            / max(values[(control, patient)], 1e-12)
            for patient in patients
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    bootstrap = degradation[
        rng.integers(0, len(degradation), (resamples, len(degradation)))
    ].mean(axis=1)
    upper = float(np.quantile(bootstrap, 0.95))
    return {
        "model": model,
        "control": control,
        "cohort": cohort,
        "condition": condition,
        "relative_l1_margin": margin,
        "mean_relative_l1_degradation": float(degradation.mean()),
        "one_sided_95_upper": upper,
        "passed": upper <= margin,
        "num_patients": len(patients),
    }


def append_robustness_composite(patient_rows, models):
    existing = list(patient_rows)
    lookup = defaultdict(list)
    for row in existing:
        if (
            row["cohort"] == "robustness"
            and row["condition"] in ROBUST_CONDITIONS
            and row["model"] in models
        ):
            lookup[(row["model"], row["patient_id"])].append(row)
    for (model, patient), values in sorted(lookup.items()):
        if {row["condition"] for row in values} != set(ROBUST_CONDITIONS):
            raise RuntimeError(f"Incomplete robustness composite for {model}/{patient}")
        patient_rows.append(
            {
                "model": model,
                "cohort": "robustness",
                "condition": "robustness_composite",
                "patient_id": patient,
                "num_slices": int(sum(row["num_slices"] for row in values)),
                **{
                    key: float(np.nanmean([float(row[key]) for row in values]))
                    for key in (*METRICS, *DIAGNOSTICS)
                },
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--full_clean_manifest", required=True)
    parser.add_argument("--robustness_manifest", required=True)
    parser.add_argument("--condition_manifest", required=True)
    parser.add_argument("--m2u_clean_checkpoint", required=True)
    parser.add_argument("--m2u_augmented_checkpoint", required=True)
    parser.add_argument("--m2u_augcap_mask_checkpoint", required=True)
    parser.add_argument("--global_direct_checkpoint", required=True)
    parser.add_argument("--quality_gain_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--bootstrap_resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_paths = {
        "full_clean": Path(args.full_clean_manifest).resolve(),
        "robustness": Path(args.robustness_manifest).resolve(),
    }
    manifests = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in manifest_paths.items()
    }
    manifest_hashes = {
        name: sha256_file(path) for name, path in manifest_paths.items()
    }
    for name, manifest in manifests.items():
        if (
            manifest.get("protocol_version") != MANIFEST_PROTOCOL_VERSION
            or manifest.get("cohort") != name
        ):
            raise RuntimeError(f"{name}: manifest protocol/cohort mismatch")
    if int(manifests["full_clean"].get("num_patients", -1)) != 25:
        raise RuntimeError("Full-clean manifest does not contain 25 patients")
    if int(manifests["full_clean"].get("num_slices", -1)) != 878:
        raise RuntimeError("Full-clean manifest does not contain 878 slices")

    condition_path = Path(args.condition_manifest).resolve()
    condition_manifest = json.loads(condition_path.read_text(encoding="utf-8"))
    condition_hash = sha256_file(condition_path)
    if (
        condition_manifest.get("protocol_version")
        != CONDITION_MANIFEST_PROTOCOL_VERSION
        or int(condition_manifest.get("seed", -1)) != args.seed
    ):
        raise RuntimeError("Condition manifest protocol/seed mismatch")
    condition_lookup = {
        int(entry["source_index"]): entry
        for entry in condition_manifest["entries"]
    }
    if len(condition_lookup) != int(condition_manifest["num_entries"]):
        raise RuntimeError("Duplicate source index in condition manifest")

    metadata_path = Path(args.metadata_csv).resolve()
    metadata_hash = sha256_file(metadata_path)
    dataset_args = argparse.Namespace(
        metadata_csv=str(metadata_path),
        acceleration=8,
        pd_aux_acceleration=2,
    )
    full_dataset = IndexedDataset(make_dataset(dataset_args, "val"))
    selected = {
        name: ManifestDataset(full_dataset, manifest)
        for name, manifest in manifests.items()
    }
    loaders = {
        name: DataLoader(
            dataset,
            batch_sampler=ShapeBucketBatchSampler(
                dataset, args.batch_size, False, args.seed + offset
            ),
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        for offset, (name, dataset) in enumerate(selected.items())
    }

    checkpoint_paths = {
        "m2u_clean": Path(args.m2u_clean_checkpoint).resolve(),
        "m2u_augmented": Path(args.m2u_augmented_checkpoint).resolve(),
        "m2u_augcap_mask": Path(args.m2u_augcap_mask_checkpoint).resolve(),
        "global_direct": Path(args.global_direct_checkpoint).resolve(),
        GAIN_MODEL: Path(args.quality_gain_checkpoint).resolve(),
    }
    for path in checkpoint_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    slice_rows: List[Dict[str, Any]] = []
    resource_rows: List[Dict[str, Any]] = []
    configs: Dict[str, Mapping[str, Any]] = {}
    checkpoint_audit: Dict[str, Any] = {}
    constant_q_values: Dict[str, float] = {}

    for name in BASELINE_NAMES:
        model, config, summary, selected_epoch = load_baseline(
            checkpoint_paths[name],
            name,
            device,
            manifest_hashes["full_clean"],
            manifest_hashes["robustness"],
            metadata_hash,
        )
        configs[name] = config
        checkpoint_audit[name] = {
            "path": str(checkpoint_paths[name]),
            "sha256": sha256_file(checkpoint_paths[name]),
            "selected_epoch": selected_epoch,
            "best_val_patient_l1": summary["best_val_patient_l1"],
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "protocol_version": config.get("protocol_version"),
            "code_hashes_recorded_at_training": config.get("code_hashes"),
        }
        rows, timing = evaluate_model(
            model,
            name,
            loaders["full_clean"],
            full_dataset,
            condition_lookup,
            device,
            "full_clean",
            "correct",
        )
        slice_rows.extend(rows)
        timing["parameter_count"] = sum(p.numel() for p in model.parameters())
        resource_rows.append(timing)
        for condition in CONDITIONS:
            rows, _ = evaluate_model(
                model,
                name,
                loaders["robustness"],
                full_dataset,
                condition_lookup,
                device,
                "robustness",
                condition,
            )
            slice_rows.extend(rows)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for name in FUSION_NAMES:
        model, config, summary, selected_epoch = load_fusion(
            checkpoint_paths[name],
            name,
            device,
            manifest_hashes["full_clean"],
            manifest_hashes["robustness"],
            condition_hash,
            metadata_hash,
        )
        configs[name] = config
        checkpoint_audit[name] = {
            "path": str(checkpoint_paths[name]),
            "sha256": sha256_file(checkpoint_paths[name]),
            "selected_epoch": selected_epoch,
            "best_val_patient_l1": summary["best_val_patient_l1"],
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "protocol_version": config.get("protocol_version"),
            "continuation_script_sha256": config.get(
                "continuation_script_sha256"
            ),
            "code_hashes_recorded_at_training": config.get("code_hashes"),
        }
        clean_rows, timing = evaluate_model(
            model,
            name,
            loaders["full_clean"],
            full_dataset,
            condition_lookup,
            device,
            "full_clean",
            "correct",
        )
        slice_rows.extend(clean_rows)
        timing["parameter_count"] = sum(p.numel() for p in model.parameters())
        resource_rows.append(timing)

        patient_q = defaultdict(list)
        for row in clean_rows:
            patient_q[row["patient_id"]].append(float(row["q"]))
        constant_q = float(
            np.mean([np.mean(values) for values in patient_q.values()])
        )
        constant_q_values[name] = constant_q

        for condition in CONDITIONS:
            rows, _ = evaluate_model(
                model,
                name,
                loaders["robustness"],
                full_dataset,
                condition_lookup,
                device,
                "robustness",
                condition,
            )
            slice_rows.extend(rows)
            rows, _ = evaluate_model(
                model,
                f"{name}_constant_q",
                loaders["robustness"],
                full_dataset,
                condition_lookup,
                device,
                "robustness",
                condition,
                reliability_override=constant_q,
            )
            slice_rows.extend(rows)
            rows, _ = evaluate_model(
                model,
                f"{name}_q1",
                loaders["robustness"],
                full_dataset,
                condition_lookup,
                device,
                "robustness",
                condition,
                reliability_override=1.0,
            )
            slice_rows.extend(rows)

        if name == GAIN_MODEL:
            with residual_scale_override(model, 0.0):
                rows, _ = evaluate_model(
                    model,
                    GAIN_OFF,
                    loaders["full_clean"],
                    full_dataset,
                    condition_lookup,
                    device,
                    "full_clean",
                    "correct",
                )
                slice_rows.extend(rows)
                for condition in CONDITIONS:
                    rows, _ = evaluate_model(
                        model,
                        GAIN_OFF,
                        loaders["robustness"],
                        full_dataset,
                        condition_lookup,
                        device,
                        "robustness",
                        condition,
                    )
                    slice_rows.extend(rows)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    zero_rows, zero_timing = evaluate_zero_filled(loaders["full_clean"], device)
    slice_rows.extend(zero_rows)
    zero_timing["parameter_count"] = 0
    resource_rows.append(zero_timing)

    fairness = fairness_audit(configs)
    if not fairness["passed"]:
        raise RuntimeError(
            "Cross-checkpoint fairness audit failed:\n"
            + json.dumps(fairness, indent=2)
        )

    patient_rows = patient_average(slice_rows)
    composite_models = [
        *ACTUAL_MODEL_NAMES,
        *(f"{name}_constant_q" for name in FUSION_NAMES),
        *(f"{name}_q1" for name in FUSION_NAMES),
        GAIN_OFF,
    ]
    append_robustness_composite(patient_rows, composite_models)
    summaries = summary_rows(patient_rows)

    clean_comparisons = []
    clean_pairs = [
        ("global_direct", GAIN_MODEL),
        ("global_direct", "m2u_clean"),
        ("global_direct", "m2u_augmented"),
        ("global_direct", "m2u_augcap_mask"),
        ("global_direct", ZERO_EXACT),
        (GAIN_MODEL, "m2u_clean"),
        (GAIN_MODEL, "m2u_augmented"),
        (GAIN_MODEL, "m2u_augcap_mask"),
        (GAIN_MODEL, ZERO_EXACT),
        ("m2u_augmented", "m2u_clean"),
        ("m2u_augcap_mask", "m2u_augmented"),
    ]
    for model_a, model_b in clean_pairs:
        for metric in METRICS:
            clean_comparisons.append(
                paired_bootstrap(
                    patient_rows,
                    model_a,
                    model_b,
                    "full_clean",
                    "correct",
                    metric,
                    args.bootstrap_resamples,
                    args.seed,
                )
            )

    robustness_comparisons = []
    for candidate in FUSION_NAMES:
        for control in BASELINE_NAMES:
            for condition in (*CONDITIONS, "robustness_composite"):
                for metric in METRICS:
                    robustness_comparisons.append(
                        paired_bootstrap(
                            patient_rows,
                            candidate,
                            control,
                            "robustness",
                            condition,
                            metric,
                            args.bootstrap_resamples,
                            args.seed,
                        )
                    )

    mechanism_comparisons = []
    for candidate in FUSION_NAMES:
        for counterfactual in (
            f"{candidate}_constant_q",
            f"{candidate}_q1",
        ):
            for condition in (*CONDITIONS, "robustness_composite"):
                for metric in METRICS:
                    mechanism_comparisons.append(
                        paired_bootstrap(
                            patient_rows,
                            candidate,
                            counterfactual,
                            "robustness",
                            condition,
                            metric,
                            args.bootstrap_resamples,
                            args.seed,
                        )
                    )
    for cohort, conditions in (
        ("full_clean", ("correct",)),
        ("robustness", (*CONDITIONS, "robustness_composite")),
    ):
        for condition in conditions:
            for metric in METRICS:
                mechanism_comparisons.append(
                    paired_bootstrap(
                        patient_rows,
                        GAIN_MODEL,
                        GAIN_OFF,
                        cohort,
                        condition,
                        metric,
                        args.bootstrap_resamples,
                        args.seed,
                    )
                )

    noninferiority = []
    for candidate in FUSION_NAMES:
        for control in BASELINE_NAMES:
            noninferiority.append(
                noninferiority_l1(
                    patient_rows,
                    candidate,
                    control,
                    "full_clean",
                    "correct",
                    0.005,
                    args.bootstrap_resamples,
                    args.seed,
                )
            )

    summary_lookup = {
        (row["model"], row["cohort"], row["condition"]): row
        for row in summaries
    }
    decision = {
        "protocol_version": PROTOCOL_VERSION,
        "primary_endpoint": "patient-equal full-clean L1",
        "full_clean_cohort": {
            "num_patients": manifests["full_clean"]["num_patients"],
            "num_slices": manifests["full_clean"]["num_slices"],
        },
        "constant_q_values": constant_q_values,
        "clean_patient_macro_metrics": {
            name: {
                metric: summary_lookup[(name, "full_clean", "correct")][metric]
                for metric in METRICS
            }
            for name in (*ACTUAL_MODEL_NAMES, ZERO_EXACT, GAIN_OFF)
        },
        "robustness_patient_macro_l1": {
            name: {
                condition: summary_lookup[(name, "robustness", condition)]["l1"]
                for condition in (*CONDITIONS, "robustness_composite")
            }
            for name in ACTUAL_MODEL_NAMES
        },
        "interpretation_guardrails": [
            (
                "The exact-manifest zero-filled row is directly comparable and "
                "is included in patient-level paired bootstrap."
            ),
            (
                "The supplied 1,203-slice zero-filled table is an external "
                "historical reference only because its cohort differs from the "
                "locked 878-slice full-clean cohort."
            ),
            (
                "Constant q is the patient-equal mean actual q estimated on the "
                "locked full-clean cohort and is used only as a same-checkpoint "
                "mechanistic counterfactual, not for model selection."
            ),
            (
                "Training total losses are not compared because the quality-gain "
                "model contains additional auxiliary objectives."
            ),
        ],
    }

    legacy_rows = [
        {
            **row,
            "L1_mean": "",
            "source": "investigator_supplied_legacy_zero_filled_table",
            "aggregation": "slice_level",
            "paired_with_current_25_patient_cohort": False,
            "comparability": "external_reference_only",
        }
        for row in LEGACY_ZERO_FILLED_ROWS
    ]

    write_csv(output_dir / "final_comparison_per_slice.csv", slice_rows)
    write_csv(output_dir / "final_comparison_patient_level.csv", patient_rows)
    write_csv(output_dir / "final_comparison_summary.csv", summaries)
    write_csv(
        output_dir / "final_comparison_clean_paired_bootstrap.csv",
        clean_comparisons,
    )
    write_csv(
        output_dir / "final_comparison_robustness_paired_bootstrap.csv",
        robustness_comparisons,
    )
    write_csv(
        output_dir / "final_comparison_mechanism_paired_bootstrap.csv",
        mechanism_comparisons,
    )
    write_csv(
        output_dir / "final_comparison_clean_noninferiority.csv",
        noninferiority,
    )
    write_csv(output_dir / "final_comparison_resources.csv", resource_rows)
    write_csv(
        output_dir / "legacy_zero_filled_external_reference.csv", legacy_rows
    )

    audit = {
        "protocol_version": PROTOCOL_VERSION,
        "evaluator": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "device": str(device),
        "runtime_versions": runtime_versions(),
        "manifests": {
            name: {
                "path": str(path),
                "sha256": manifest_hashes[name],
                "num_patients": manifests[name]["num_patients"],
                "num_slices": manifests[name]["num_slices"],
            }
            for name, path in manifest_paths.items()
        },
        "condition_manifest": {
            "path": str(condition_path),
            "sha256": condition_hash,
            "num_entries": condition_manifest["num_entries"],
        },
        "metadata": {
            "path": str(metadata_path),
            "sha256": metadata_hash,
        },
        "checkpoint_audit": checkpoint_audit,
        "cross_checkpoint_fairness": fairness,
        "constant_q_definition": (
            "patient-equal mean actual q on the locked full-clean cohort"
        ),
        "constant_q_values": constant_q_values,
        "bootstrap_resamples": args.bootstrap_resamples,
        "seed": args.seed,
        "exact_zero_filled_definition": (
            "RSS magnitude of inverse-centred-FFT of the identical masked "
            "PD-FS k-space returned by the locked R=8 evaluation dataset"
        ),
    }
    (output_dir / "final_comparison_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    (output_dir / "final_comparison_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
