#!/usr/bin/env python3
from __future__ import annotations

"""Supplemental causal diagnostics for the M2-PRNF quality-gain pilot.

This evaluator never modifies a checkpoint and deliberately uses a new filename.
It has two independently runnable phases:

``causal``
    Global-direct actual-q versus q=1, plus quality-gain residual-on versus
    residual-off on the full-clean and frozen robustness cohorts.

``scales``
    Same-checkpoint residual scale ablations on the full-clean cohort, together
    with image-domain correction direction, corrected-pixel and overshoot
    diagnostics relative to the all-residual-off counterfactual.

The script imports the locked v1.5 evaluator for dataset, checkpoint and metric
handling.  No completed evaluator or training source is overwritten.
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_m2_prnf_quality_gain_R8 import (  # noqa: E402
    CONDITION_MANIFEST_PROTOCOL_VERSION,
    FAIRNESS_KEYS,
    MANIFEST_PROTOCOL_VERSION,
    ManifestDataset,
    append_composite,
    bootstrap_improvement,
    build_model,
    condition_pd,
    evaluate_mode,
    metrics as reconstruction_metrics,
    patient_average,
    summary_rows,
    write_csv,
)
from scripts.train_m2_prnf_fusion import (  # noqa: E402
    IndexedDataset,
    ShapeBucketBatchSampler,
    make_dataset,
    prepare_batch,
    runtime_versions,
    set_seed,
    sha256_file,
)
from src.fft_utils import center_crop  # noqa: E402


PROTOCOL_VERSION = "M2-PRNF-R8-v1.6.0-step2-causal-diagnostics"
CONDITIONS = ("correct", "shift8", "wrong_slice", "wrong_patient", "missing")
ROBUST_CONDITIONS = ("shift8", "wrong_slice", "wrong_patient")
METRICS = ("nmse", "psnr", "ssim", "l1")

GLOBAL_ACTUAL = "global_direct_actual_q"
GLOBAL_Q1 = "global_direct_q1"
GAIN_ON = "quality_gain_residual_on"
GAIN_OFF = "quality_gain_residual_off"

SCALE_MODES: Mapping[str, Tuple[int, ...]] = {
    "all_off": (),
    "h2_only": (0,),
    "h4_only": (1,),
    "h8_only": (2,),
    "h16_only": (3,),
    "h2_h4": (0, 1),
    "h8_h16": (2, 3),
    "all_on": (0, 1, 2, 3),
}
SCALE_LABELS = ("H/2", "H/4", "H/8", "H/16")

CORRECTION_METRICS = (
    "l1_improvement",
    "delta_error_cosine",
    "same_sign_fraction",
    "corrected_pixel_fraction",
    "worsened_pixel_fraction",
    "overshoot_fraction",
    "delta_to_error_rms_ratio",
    "high_error_cosine",
    "high_error_corrected_fraction",
)


def safe_mean(values: Iterable[float]) -> float:
    selected = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(selected)) if selected else float("nan")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def common_config_audit(global_config: Mapping[str, Any], gain_config: Mapping[str, Any]):
    """Compare immutable common training settings while allowing objective fields."""
    mismatches = []
    for key in FAIRNESS_KEYS:
        if global_config.get(key) != gain_config.get(key):
            mismatches.append(
                {
                    "key": key,
                    "global_direct": global_config.get(key),
                    "quality_gain": gain_config.get(key),
                }
            )
    for key in ("corrupt_view_mixture", "need_scope", "residual_scale"):
        if global_config.get(key) != gain_config.get(key):
            mismatches.append(
                {
                    "key": key,
                    "global_direct": global_config.get(key),
                    "quality_gain": gain_config.get(key),
                }
            )
    return {
        "checked_keys": list(FAIRNESS_KEYS)
        + ["corrupt_view_mixture", "need_scope", "residual_scale"],
        "allowed_differences": [
            "fusion_design",
            "run_stage",
            "training_objective",
            "lambda_residual_gain",
            "residual_gain_margin_relative",
            "residual_gain_ramp_epochs",
            "parameter_count",
            "code_hashes",
        ],
        "passed": not mismatches,
        "mismatches": mismatches,
    }


@contextmanager
def residual_level_override(model, active_levels: Sequence[int]):
    """Activate selected H/2--H/16 residual controllers without state drift."""
    complements = [controller.complement for controller in model.controllers]
    if len(complements) != 4 or any(module is None for module in complements):
        raise RuntimeError("Scale diagnostics require four complementary controllers")
    active = {int(level) for level in active_levels}
    if not active.issubset(set(range(4))):
        raise ValueError(f"Invalid active residual levels: {sorted(active)}")
    originals = [module.residual_scale.detach().clone() for module in complements]
    with torch.no_grad():
        for index, (module, original) in enumerate(zip(complements, originals)):
            module.residual_scale.copy_(original if index in active else torch.zeros_like(original))
    try:
        yield
    finally:
        with torch.no_grad():
            for module, original in zip(complements, originals):
                module.residual_scale.copy_(original)


def load_external_m2u_l1(path: Optional[Path], expected_patients: int = 25):
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("patient_rows", [])
    if int(payload.get("num_patients", -1)) != expected_patients or len(rows) != expected_patients:
        raise RuntimeError("M2-U epoch audit must contain exactly 25 patient rows")
    lookup = {str(row["patient_id"]): float(row["l1"]) for row in rows}
    if len(lookup) != expected_patients:
        raise RuntimeError("Duplicate patients in M2-U epoch audit")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "patient_l1": float(payload["patient_l1"]),
        "num_patients": int(payload["num_patients"]),
        "num_slices": int(payload["num_slices"]),
        "lookup": lookup,
    }


def bootstrap_external_l1(
    patient_rows: List[Dict[str, Any]],
    model: str,
    external: Mapping[str, Any],
    resamples: int,
    seed: int,
):
    observed = {
        str(row["patient_id"]): float(row["l1"])
        for row in patient_rows
        if row["model"] == model
        and row["cohort"] == "full_clean"
        and row["condition"] == "correct"
    }
    ids = sorted(set(observed) & set(external["lookup"]))
    if len(ids) != int(external["num_patients"]):
        raise RuntimeError(f"M2-U/{model} patient mismatch: {len(ids)} paired")
    # Positive means the evaluated PRNF mode has lower L1 than M2-U.
    differences = np.asarray(
        [external["lookup"][patient] - observed[patient] for patient in ids],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    boot = differences[rng.integers(0, len(ids), size=(resamples, len(ids)))].mean(1)
    return {
        "model_a": model,
        "model_b": "m2u_augmented_epoch15",
        "cohort": "full_clean",
        "condition": "correct",
        "metric": "l1",
        "positive_means_a_better": True,
        "mean_improvement": float(differences.mean()),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
        "patients_a_better": int((differences > 0).sum()),
        "num_patients": len(ids),
    }


def prepare_inputs(args):
    manifest_paths = {
        "full_clean": Path(args.full_clean_manifest).resolve(),
        "robustness": Path(args.robustness_manifest).resolve(),
    }
    condition_path = Path(args.condition_manifest).resolve()
    metadata_path = Path(args.metadata_csv).resolve()
    manifests = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in manifest_paths.items()
    }
    for name, manifest in manifests.items():
        if (
            manifest.get("protocol_version") != MANIFEST_PROTOCOL_VERSION
            or manifest.get("cohort") != name
        ):
            raise RuntimeError(f"{name}: manifest protocol/cohort mismatch")
    condition_manifest = json.loads(condition_path.read_text(encoding="utf-8"))
    if (
        condition_manifest.get("protocol_version")
        != CONDITION_MANIFEST_PROTOCOL_VERSION
        or int(condition_manifest.get("seed", -1)) != args.seed
    ):
        raise RuntimeError("Condition manifest protocol/seed mismatch")
    condition_lookup = {
        int(entry["source_index"]): entry for entry in condition_manifest["entries"]
    }
    if len(condition_lookup) != int(condition_manifest["num_entries"]):
        raise RuntimeError("Duplicate source index in condition manifest")

    dataset_args = argparse.Namespace(
        metadata_csv=str(metadata_path), acceleration=8, pd_aux_acceleration=2
    )
    source = IndexedDataset(make_dataset(dataset_args, "val"))
    selected = {
        name: ManifestDataset(source, manifest) for name, manifest in manifests.items()
    }
    loaders = {
        name: DataLoader(
            dataset,
            batch_sampler=ShapeBucketBatchSampler(
                dataset, args.batch_size, False, args.seed + offset
            ),
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for offset, (name, dataset) in enumerate(selected.items())
    }
    hashes = {name: sha256_file(path) for name, path in manifest_paths.items()}
    return {
        "manifest_paths": manifest_paths,
        "manifests": manifests,
        "condition_path": condition_path,
        "condition_manifest": condition_manifest,
        "condition_lookup": condition_lookup,
        "metadata_path": metadata_path,
        "source": source,
        "loaders": loaders,
        "hashes": hashes,
        "condition_hash": sha256_file(condition_path),
        "metadata_hash": sha256_file(metadata_path),
    }


def build_locked_models(args, prepared, device, include_global: bool):
    gain_path = Path(args.quality_gain_checkpoint).resolve()
    gain, observed, gain_config, gain_epoch = build_model(
        gain_path,
        "quality_protected_hybrid_gain",
        device,
        prepared["hashes"]["full_clean"],
        prepared["hashes"]["robustness"],
        prepared["condition_hash"],
        prepared["metadata_hash"],
    )
    if observed != "quality_protected_hybrid_gain":
        raise RuntimeError("Quality-gain checkpoint identity mismatch")
    models = {"gain": gain}
    configs = {"gain": gain_config}
    audit = {
        "quality_gain": {
            "path": str(gain_path),
            "sha256": sha256_file(gain_path),
            "selected_epoch": gain_epoch,
            "fusion_design": gain_config["fusion_design"],
        }
    }
    fairness = None
    if include_global:
        global_path = Path(args.global_direct_checkpoint).resolve()
        global_model, observed, global_config, global_epoch = build_model(
            global_path,
            "global_direct",
            device,
            prepared["hashes"]["full_clean"],
            prepared["hashes"]["robustness"],
            prepared["condition_hash"],
            prepared["metadata_hash"],
        )
        if observed != "global_direct":
            raise RuntimeError("Global-direct checkpoint identity mismatch")
        models["global"] = global_model
        configs["global"] = global_config
        audit["global_direct"] = {
            "path": str(global_path),
            "sha256": sha256_file(global_path),
            "selected_epoch": global_epoch,
            "fusion_design": global_config["fusion_design"],
        }
        fairness = common_config_audit(global_config, gain_config)
        if not fairness["passed"]:
            raise RuntimeError(
                "Global/quality-gain fairness audit failed:\n"
                + json.dumps(fairness["mismatches"], indent=2)
            )
    return models, configs, audit, fairness


def causal_phase(args, prepared, device, output_dir, external_m2u):
    models, configs, checkpoint_audit, fairness = build_locked_models(
        args, prepared, device, include_global=True
    )
    global_model, gain_model = models["global"], models["gain"]
    rows: List[Dict[str, Any]] = []

    # Full-clean actual-q versus forced q=1.
    rows += evaluate_mode(
        global_model, GLOBAL_ACTUAL, prepared["loaders"]["full_clean"],
        prepared["source"], prepared["condition_lookup"], device,
        "full_clean", "correct", seed=args.seed,
    )
    rows += evaluate_mode(
        global_model, GLOBAL_Q1, prepared["loaders"]["full_clean"],
        prepared["source"], prepared["condition_lookup"], device,
        "full_clean", "correct", reliability_override=1.0, seed=args.seed,
    )

    # Full-clean fifth-arm residual intervention.
    rows += evaluate_mode(
        gain_model, GAIN_ON, prepared["loaders"]["full_clean"],
        prepared["source"], prepared["condition_lookup"], device,
        "full_clean", "correct", seed=args.seed,
    )
    with residual_level_override(gain_model, ()):
        rows += evaluate_mode(
            gain_model, GAIN_OFF, prepared["loaders"]["full_clean"],
            prepared["source"], prepared["condition_lookup"], device,
            "full_clean", "correct", seed=args.seed,
        )

    # Frozen robustness conditions for all four causal modes.
    for condition in CONDITIONS:
        rows += evaluate_mode(
            global_model, GLOBAL_ACTUAL, prepared["loaders"]["robustness"],
            prepared["source"], prepared["condition_lookup"], device,
            "robustness", condition, seed=args.seed,
        )
        rows += evaluate_mode(
            global_model, GLOBAL_Q1, prepared["loaders"]["robustness"],
            prepared["source"], prepared["condition_lookup"], device,
            "robustness", condition, reliability_override=1.0, seed=args.seed,
        )
        rows += evaluate_mode(
            gain_model, GAIN_ON, prepared["loaders"]["robustness"],
            prepared["source"], prepared["condition_lookup"], device,
            "robustness", condition, seed=args.seed,
        )
        with residual_level_override(gain_model, ()):
            rows += evaluate_mode(
                gain_model, GAIN_OFF, prepared["loaders"]["robustness"],
                prepared["source"], prepared["condition_lookup"], device,
                "robustness", condition, seed=args.seed,
            )

    patient_rows = patient_average(rows)
    model_names = (GLOBAL_ACTUAL, GLOBAL_Q1, GAIN_ON, GAIN_OFF)
    append_composite(patient_rows, model_names)
    summaries = summary_rows(patient_rows)
    comparisons = []
    for first, second, hypothesis in (
        (GLOBAL_ACTUAL, GLOBAL_Q1, "learned_q_effect"),
        (GAIN_ON, GAIN_OFF, "residual_effect"),
    ):
        for metric in METRICS:
            result = bootstrap_improvement(
                patient_rows, first, second, "full_clean", "correct", metric,
                args.bootstrap_resamples, args.seed,
            )
            result["hypothesis"] = hypothesis
            comparisons.append(result)
        for condition in (*CONDITIONS, "robustness_composite"):
            result = bootstrap_improvement(
                patient_rows, first, second, "robustness", condition, "l1",
                args.bootstrap_resamples, args.seed,
            )
            result["hypothesis"] = hypothesis
            comparisons.append(result)

    external_rows = []
    if external_m2u is not None:
        for model in (GLOBAL_ACTUAL, GAIN_ON, GAIN_OFF):
            external_rows.append(
                bootstrap_external_l1(
                    patient_rows, model, external_m2u,
                    args.bootstrap_resamples, args.seed,
                )
            )

    summary_lookup = {
        (row["model"], row["cohort"], row["condition"]): row for row in summaries
    }
    comparison_lookup = {
        (row["model_a"], row["model_b"], row["cohort"], row["condition"], row["metric"]): row
        for row in comparisons
    }
    q_clean = comparison_lookup[(GLOBAL_ACTUAL, GLOBAL_Q1, "full_clean", "correct", "l1")]
    q_robust = comparison_lookup[
        (GLOBAL_ACTUAL, GLOBAL_Q1, "robustness", "robustness_composite", "l1")
    ]
    residual_clean = comparison_lookup[(GAIN_ON, GAIN_OFF, "full_clean", "correct", "l1")]
    residual_robust = comparison_lookup[
        (GAIN_ON, GAIN_OFF, "robustness", "robustness_composite", "l1")
    ]
    decision = {
        "protocol_version": PROTOCOL_VERSION,
        "phase": "causal",
        "learned_q": {
            "clean_l1_test": q_clean,
            "robustness_composite_l1_test": q_robust,
            "interpretation": (
                "Positive means learned actual-q is better than forcing q=1. "
                "Do not raise clean q solely from its mean if robustness favours actual-q."
            ),
        },
        "quality_gain_residual": {
            "clean_l1_test": residual_clean,
            "robustness_composite_l1_test": residual_robust,
            "clean_status": (
                "eligible" if residual_clean["ci95_low"] > 0
                else "borderline" if residual_clean["mean_improvement"] > 0
                else "rejected"
            ),
            "robustness_status": (
                "eligible" if residual_robust["ci95_low"] >= 0
                else "borderline" if residual_robust["ci95_high"] >= 0
                else "rejected"
            ),
        },
        "missing_exact_zero": {
            model: summary_lookup[(model, "robustness", "missing")]["gated_rms"] == 0.0
            for model in model_names
        },
        "m2u_epoch15_l1_comparisons": external_rows,
    }

    write_csv(output_dir / "step2_causal_per_slice.csv", rows)
    write_csv(output_dir / "step2_causal_patient_level.csv", patient_rows)
    write_csv(output_dir / "step2_causal_summary.csv", summaries)
    write_csv(output_dir / "step2_causal_paired_bootstrap.csv", comparisons)
    write_csv(output_dir / "step2_causal_vs_m2u_epoch15.csv", external_rows)
    write_json(output_dir / "step2_causal_decision.json", decision)
    return checkpoint_audit, fairness, decision


def cosine_on_mask(first: torch.Tensor, second: torch.Tensor, mask: torch.Tensor) -> float:
    if int(mask.sum().item()) < 2:
        return float("nan")
    a, b = first[mask].double(), second[mask].double()
    denominator = a.square().sum().sqrt() * b.square().sum().sqrt()
    if float(denominator.item()) <= 1e-12:
        return float("nan")
    return float((a * b).sum().div(denominator).item())


def correction_metrics(on, off, target):
    scale = target.amax().clamp_min(1e-8)
    truth = target / scale
    on_norm, off_norm = on / scale, off / scale
    error = truth - off_norm
    delta = on_norm - off_norm
    old_abs = error.abs()
    new_abs = (truth - on_norm).abs()
    foreground = truth > 0.01
    if int(foreground.sum().item()) == 0:
        foreground = torch.ones_like(truth, dtype=torch.bool)
    active = foreground & (old_abs > 1e-6) & (delta.abs() > 1e-6)
    foreground_errors = old_abs[foreground]
    threshold = torch.quantile(foreground_errors, 0.75)
    high_error = foreground & (old_abs >= threshold) & (old_abs > 1e-6)
    corrected = new_abs < old_abs
    worsened = new_abs > old_abs
    same_sign = (delta * error) > 0
    overshoot = same_sign & (delta.abs() > old_abs)
    delta_rms = delta[foreground].double().square().mean().sqrt()
    error_rms = error[foreground].double().square().mean().sqrt().clamp_min(1e-12)

    def fraction(mask, denominator_mask):
        count = int(denominator_mask.sum().item())
        return float(mask[denominator_mask].float().mean().item()) if count else float("nan")

    return {
        "l1_improvement": float((old_abs.mean() - new_abs.mean()).item()),
        "delta_error_cosine": cosine_on_mask(delta, error, foreground),
        "same_sign_fraction": fraction(same_sign, active),
        "corrected_pixel_fraction": fraction(corrected, foreground),
        "worsened_pixel_fraction": fraction(worsened, foreground),
        "overshoot_fraction": fraction(overshoot, active),
        "delta_to_error_rms_ratio": float((delta_rms / error_rms).item()),
        "high_error_cosine": cosine_on_mask(delta, error, high_error),
        "high_error_corrected_fraction": fraction(corrected, high_error),
        "foreground_pixels": int(foreground.sum().item()),
        "active_pixels": int(active.sum().item()),
        "high_error_pixels": int(high_error.sum().item()),
    }


def standard_row(name, batch, sample, prediction, target, aux):
    residual_values = aux["residual_to_target_rms"][sample]
    ratio_values = aux["residual_to_direct_rms_ratio"][sample]
    valid_ratios = ratio_values[ratio_values >= 0]
    tensors_to_check = (
        residual_values,
        aux["direct_to_target_rms"][sample],
        aux["gated_aux_to_target_rms"][sample],
    )
    return {
        "model": name,
        "cohort": "full_clean",
        "condition": "correct",
        "patient_id": str(batch["patient_id"][sample]),
        "slice_idx": int(batch["slice_idx"][sample]),
        **reconstruction_metrics(prediction[sample], target[sample]),
        "q": float(aux["q"].mean((1, 2))[sample].item()),
        "need": float(aux["need_mean"].mean((1, 2))[sample].item()),
        "effective_weight": float(aux["effective_weight_mean"].mean((1, 2))[sample].item()),
        "gated_rms": float(aux["gated_aux_to_target_rms"].mean((1, 2))[sample].item()),
        "direct_rms": float(aux["direct_to_target_rms"].mean((1, 2))[sample].item()),
        "residual_rms": float(aux["residual_to_target_rms"].mean((1, 2))[sample].item()),
        "residual_rms_max": float(residual_values.max().item()),
        "residual_direct_ratio": float(valid_ratios.mean().item()) if valid_ratios.numel() else -1.0,
        "raw_auxiliary_rms": float(aux["raw_auxiliary_to_target_rms"].mean((1, 2))[sample].item()),
        "feature_cosine": float(aux["target_auxiliary_cosine"].mean((1, 2))[sample].item()),
        "diagnostics_finite": float(
            all(bool(torch.isfinite(value).all()) for value in tensors_to_check)
        ),
    }


@torch.no_grad()
def evaluate_scale_modes(model, loader, device):
    standard_rows: List[Dict[str, Any]] = []
    correction_rows: List[Dict[str, Any]] = []
    for batch in loader:
        kspace, mask, pd, target, _ = prepare_batch(batch, device)
        available = torch.ones(pd.shape[0], device=device)
        predictions, auxiliaries = {}, {}
        for mode, active_levels in SCALE_MODES.items():
            with residual_level_override(model, active_levels):
                prediction, aux = model(
                    kspace, mask, pd, available, return_aux=True
                )
            predictions[mode] = center_crop(
                prediction, target.shape[-2], target.shape[-1]
            )
            auxiliaries[mode] = aux
        off = predictions["all_off"]
        for mode in SCALE_MODES:
            name = f"scale_{mode}"
            prediction, aux = predictions[mode], auxiliaries[mode]
            for sample in range(target.shape[0]):
                standard_rows.append(
                    standard_row(name, batch, sample, prediction, target, aux)
                )
                if mode != "all_off":
                    correction_rows.append(
                        {
                            "mode": name,
                            "reference": "scale_all_off",
                            "patient_id": str(batch["patient_id"][sample]),
                            "slice_idx": int(batch["slice_idx"][sample]),
                            **correction_metrics(
                                prediction[sample], off[sample], target[sample]
                            ),
                        }
                    )
        del predictions, auxiliaries
    return standard_rows, correction_rows


def correction_patient_average(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["mode"], row["reference"], row["patient_id"])].append(row)
    output = []
    for (mode, reference, patient), values in sorted(groups.items()):
        output.append(
            {
                "mode": mode,
                "reference": reference,
                "patient_id": patient,
                "num_slices": len(values),
                **{
                    metric: safe_mean(row[metric] for row in values)
                    for metric in CORRECTION_METRICS
                },
            }
        )
    return output


def bootstrap_scalar(rows, mode, metric, resamples, seed):
    values = np.asarray(
        [float(row[metric]) for row in rows if row["mode"] == mode],
        dtype=np.float64,
    )
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise RuntimeError(f"No finite correction values: {mode}/{metric}")
    rng = np.random.default_rng(seed)
    boot = values[rng.integers(0, values.size, size=(resamples, values.size))].mean(1)
    return {
        "mode": mode,
        "metric": metric,
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
        "patients_positive": int((values > 0).sum()),
        "num_patients": int(values.size),
    }


def scale_phase(args, prepared, device, output_dir, external_m2u):
    models, configs, checkpoint_audit, _ = build_locked_models(
        args, prepared, device, include_global=False
    )
    standard_rows, correction_rows = evaluate_scale_modes(
        models["gain"], prepared["loaders"]["full_clean"], device
    )
    patient_rows = patient_average(standard_rows)
    summaries = summary_rows(patient_rows)
    correction_patients = correction_patient_average(correction_rows)

    comparisons = []
    for mode in SCALE_MODES:
        name = f"scale_{mode}"
        if mode == "all_off":
            continue
        for metric in METRICS:
            comparisons.append(
                bootstrap_improvement(
                    patient_rows, name, "scale_all_off", "full_clean", "correct",
                    metric, args.bootstrap_resamples, args.seed,
                )
            )
    correction_bootstrap = [
        bootstrap_scalar(
            correction_patients, f"scale_{mode}", metric,
            args.bootstrap_resamples, args.seed,
        )
        for mode in SCALE_MODES
        if mode != "all_off"
        for metric in CORRECTION_METRICS
    ]
    external_rows = []
    if external_m2u is not None:
        for mode in SCALE_MODES:
            external_rows.append(
                bootstrap_external_l1(
                    patient_rows, f"scale_{mode}", external_m2u,
                    args.bootstrap_resamples, args.seed,
                )
            )

    summary_lookup = {row["model"]: row for row in summaries}
    l1_tests = {
        row["model_a"]: row for row in comparisons if row["metric"] == "l1"
    }
    ranked = sorted(
        SCALE_MODES,
        key=lambda mode: float(summary_lookup[f"scale_{mode}"]["l1"]),
    )
    mode_results = {}
    for mode in SCALE_MODES:
        name = f"scale_{mode}"
        result = {
            "active_levels": [SCALE_LABELS[index] for index in SCALE_MODES[mode]],
            "clean_metrics": {
                metric: summary_lookup[name][metric] for metric in METRICS
            },
        }
        if mode != "all_off":
            result["l1_vs_all_off"] = l1_tests[name]
            result["l1_status"] = (
                "eligible" if l1_tests[name]["ci95_low"] > 0
                else "borderline" if l1_tests[name]["mean_improvement"] > 0
                else "rejected"
            )
        mode_results[mode] = result
    decision = {
        "protocol_version": PROTOCOL_VERSION,
        "phase": "scales",
        "scale_index_mapping": {
            str(index): label for index, label in enumerate(SCALE_LABELS)
        },
        "ranked_by_clean_l1": ranked,
        "best_clean_l1_mode": ranked[0],
        "mode_results": mode_results,
        "correction_metric_definitions": {
            "normalisation": "Per-slice target maximum.",
            "foreground": "Normalised target > 0.01.",
            "active": "Foreground and |off error| > 1e-6 and |delta| > 1e-6.",
            "high_error": "Top quartile of |target - residual_off| within foreground.",
            "delta": "mode prediction - all-residual-off prediction.",
            "error": "target - all-residual-off prediction.",
            "overshoot": "same-sign correction with |delta| > |error|.",
        },
        "m2u_epoch15_l1_comparisons": external_rows,
    }

    write_csv(output_dir / "step2_scales_per_slice.csv", standard_rows)
    write_csv(output_dir / "step2_scales_patient_level.csv", patient_rows)
    write_csv(output_dir / "step2_scales_summary.csv", summaries)
    write_csv(output_dir / "step2_scales_paired_bootstrap.csv", comparisons)
    write_csv(output_dir / "step2_correction_per_slice.csv", correction_rows)
    write_csv(output_dir / "step2_correction_patient_level.csv", correction_patients)
    write_csv(output_dir / "step2_correction_bootstrap.csv", correction_bootstrap)
    write_csv(output_dir / "step2_scales_vs_m2u_epoch15.csv", external_rows)
    write_json(output_dir / "step2_scales_decision.json", decision)
    return checkpoint_audit, None, decision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("causal", "scales"), required=True)
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--full_clean_manifest", required=True)
    parser.add_argument("--robustness_manifest", required=True)
    parser.add_argument("--condition_manifest", required=True)
    parser.add_argument("--quality_gain_checkpoint", required=True)
    parser.add_argument("--global_direct_checkpoint")
    parser.add_argument("--m2u_augmented_epoch15_audit")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--bootstrap_resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.phase == "causal" and not args.global_direct_checkpoint:
        parser.error("--global_direct_checkpoint is required for phase=causal")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare_inputs(args)
    external_path = (
        Path(args.m2u_augmented_epoch15_audit).resolve()
        if args.m2u_augmented_epoch15_audit else None
    )
    external_m2u = load_external_m2u_l1(external_path)

    if args.phase == "causal":
        checkpoint_audit, fairness, decision = causal_phase(
            args, prepared, device, output_dir, external_m2u
        )
    else:
        checkpoint_audit, fairness, decision = scale_phase(
            args, prepared, device, output_dir, external_m2u
        )

    audit = {
        "protocol_version": PROTOCOL_VERSION,
        "phase": args.phase,
        "evaluator": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
            "mutation_policy": "Read-only checkpoint interventions; no state is saved.",
        },
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "manifests": {
            name: {
                "path": str(prepared["manifest_paths"][name]),
                "sha256": prepared["hashes"][name],
                "num_patients": prepared["manifests"][name]["num_patients"],
                "num_slices": prepared["manifests"][name]["num_slices"],
            }
            for name in prepared["manifest_paths"]
        },
        "condition_manifest": {
            "path": str(prepared["condition_path"]),
            "sha256": prepared["condition_hash"],
            "num_entries": prepared["condition_manifest"]["num_entries"],
        },
        "metadata": {
            "path": str(prepared["metadata_path"]),
            "sha256": prepared["metadata_hash"],
        },
        "checkpoints": checkpoint_audit,
        "common_config_fairness": fairness,
        "external_m2u_epoch15": (
            {key: value for key, value in external_m2u.items() if key != "lookup"}
            if external_m2u is not None else None
        ),
        "runtime_versions": runtime_versions(),
        "bootstrap_resamples": args.bootstrap_resamples,
        "seed": args.seed,
    }
    write_json(output_dir / f"step2_{args.phase}_audit.json", audit)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
