#!/usr/bin/env python3
from __future__ import annotations

"""Imperial-HPC comparison of H/4 RPCRF, global-direct and the fifth arm."""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import torch
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_m2_prnf_fusion import (  # noqa: E402
    IndexedDataset, ShapeBucketBatchSampler,
    locked_code_hashes as fusion_locked_code_hashes, make_dataset,
    prepare_batch, runtime_versions, set_seed, sha256_file,
)
from scripts.train_m2_prnf_quality_gain import (  # noqa: E402
    locked_code_hashes as gain_locked_code_hashes,
    residual_scale_override,
)
from scripts.train_m2_prnf_rpcrf import (  # noqa: E402
    locked_code_hashes as rpcrf_locked_code_hashes,
)
from scripts.preflight_m2_prnf_rpcrf import (  # noqa: E402
    validate_metadata_relocation,
)
from src.fft_utils import center_crop  # noqa: E402
from src.m2_prnf_corruptions import (  # noqa: E402
    load_pd_auxiliary, translate_nonwrapping,
)
from src.m2_prnf_fusion_pilot_varnet import (  # noqa: E402
    FUSION_DESIGNS,
    M2PRNFFusionPilotVarNet,
)
from src.m2_prnf_rpcrf_varnet import M2PRNFRPCRFVarNet  # noqa: E402

PROTOCOL_VERSION = (
    "M2-PRNF-R8-v1.7.1-RPCRF-H4-final-comparison-IC-audited"
)
MANIFEST_PROTOCOL_VERSION = "M2-PRNF-R8-v1.3-bs4-audited"
CONDITION_MANIFEST_PROTOCOL_VERSION = (
    "M2-PRNF-R8-v1.4.1-fusion-pilot-audited"
)
EVALUATOR_HOTFIX_ID = "rpcrf-h4-final-evaluator-ic-v2-bootstrap-ni"
PILOT_ARMS = (
    "global_direct",
    "quality_protected_hybrid_gain",
    "rpcrf_h4_final",
)
GAIN_ARM = "quality_protected_hybrid_gain"
GAIN_RESIDUAL_OFF = "quality_protected_hybrid_gain_residual_off"
RPCRF_ARM = "rpcrf_h4_final"
RPCRF_RESIDUAL_OFF = "rpcrf_h4_final_residual_off"
CONDITIONS = ("correct", "shift8", "wrong_slice", "wrong_patient", "missing")
ROBUST_CONDITIONS = ("shift8", "wrong_slice", "wrong_patient")
METRICS = ("nmse", "psnr", "ssim", "l1")
DIAGNOSTICS = (
    "q", "need", "effective_weight", "gated_rms", "direct_rms",
    "residual_rms", "residual_rms_max", "residual_direct_ratio",
    "raw_auxiliary_rms", "feature_cosine", "background_l1",
    "background_p99", "diagnostics_finite",
)


class ManifestDataset:
    def __init__(self, source: IndexedDataset, manifest: Mapping[str, Any]):
        self.source = source
        wanted = {(str(p["patient_id"]), int(s)) for p in manifest["patients"]
                  for s in p["slice_indices"]}
        self.indices = [i for i, r in enumerate(source.records)
                        if (str(r["patient_id"]), int(r["slice_idx"])) in wanted]
        observed = {(str(source.records[i]["patient_id"]),
                     int(source.records[i]["slice_idx"])) for i in self.indices}
        if observed != wanted:
            raise RuntimeError(f"Manifest entries not found: {sorted(wanted-observed)[:8]}")
        self.records = [source.records[i] for i in self.indices]
        self.patient_rows = source.patient_rows

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        source_index = self.indices[index]
        item = self.source[source_index]
        item["sample_idx"] = int(source_index)
        return item


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_model(
    path: Path, arm: str, device, clean_hash: str, robust_hash: str,
    condition_hash: str, metadata_hash: str,
):
    if path.name != "model_best.pt":
        raise RuntimeError(f"Evaluator requires model_best.pt, got {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    variant = str(config.get("variant"))
    if variant != "prnf_no_need":
        raise RuntimeError(f"{path}: invalid variant {variant}")
    fusion_design = str(config.get("fusion_design"))
    if fusion_design not in FUSION_DESIGNS:
        raise RuntimeError(f"{path}: invalid fusion design {fusion_design}")
    is_gain_candidate = arm == "quality_protected_hybrid_gain"
    is_rpcrf = arm == RPCRF_ARM
    expected_stage = (
        "rpcrf_h4_final_pilot" if is_rpcrf
        else "quality_gain_pilot" if is_gain_candidate else "pilot"
    )
    if config.get("run_stage") != expected_stage:
        raise RuntimeError(
            f"{path}: expected run_stage={expected_stage}, got {config.get('run_stage')}"
        )
    if is_gain_candidate:
        if fusion_design != "hybrid_direct_residual":
            raise RuntimeError(f"{path}: gain candidate is not hybrid")
        if config.get("training_objective") != (
            "quality_protected_residual_incremental_gain"
        ):
            raise RuntimeError(f"{path}: missing quality-gain objective identity")
        expected_gain = {
            "lambda_residual_gain": 0.2,
            "residual_gain_margin_relative": 0.002,
            "residual_gain_ramp_epochs": 5,
        }
        observed_gain = {key: config.get(key) for key in expected_gain}
        if observed_gain != expected_gain:
            raise RuntimeError(
                f"{path}: quality-gain protocol mismatch: {observed_gain}"
            )
    if is_rpcrf:
        if fusion_design != "hybrid_direct_residual":
            raise RuntimeError(f"{path}: RPCRF candidate is not hybrid")
        if config.get("training_objective") != (
            "h4_direction_corrective_residual_only_routing"
        ):
            raise RuntimeError(f"{path}: missing RPCRF objective identity")
        expected_rpcrf = {
            "residual_reliability_power": 1.5,
            "lambda_correction": 0.05,
            "lambda_overshoot": 0.01,
            "overshoot_kappa": 1.25,
            "lambda_residual_gain": 0.10,
            "residual_gain_margin_relative": 0.002,
        }
        observed_rpcrf = {key: config.get(key) for key in expected_rpcrf}
        if observed_rpcrf != expected_rpcrf:
            raise RuntimeError(f"{path}: RPCRF protocol mismatch: {observed_rpcrf}")
        if config.get("residual_levels") != ["H/4"]:
            raise RuntimeError(
                f"{path}: RPCRF must contain only H/4 residual, got "
                f"{config.get('residual_levels')}"
            )
    if int(config.get("acceleration", -1)) != 8 or int(config.get("epochs", -1)) != 15:
        raise RuntimeError(f"{path}: not a pre-registered R=8/15-epoch run")
    summary_path = path.parent / "final_summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"{path}: missing final_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary.get("completed_epochs", -1)) != 15:
        raise RuntimeError(f"{path}: pilot did not complete 15 epochs")
    selected = int(checkpoint.get("epoch", -1))
    if not selected == int(checkpoint.get("best_epoch", -1)) == int(summary.get("best_epoch", -1)):
        raise RuntimeError(f"{path}: selected/checkpoint/summary best epoch mismatch")
    if summary.get("variant") != variant:
        raise RuntimeError(f"{path}: final-summary variant mismatch")
    if summary.get("fusion_design") != fusion_design:
        raise RuntimeError(f"{path}: final-summary fusion-design mismatch")
    if not math.isclose(
        float(checkpoint.get("best_val", float("nan"))),
        float(summary.get("best_val_patient_l1", float("nan"))),
        rel_tol=0.0, abs_tol=1e-12,
    ):
        raise RuntimeError(f"{path}: checkpoint/final-summary best L1 mismatch")
    if config.get("full_clean_manifest_sha256") != clean_hash:
        raise RuntimeError(f"{path}: full-clean manifest hash mismatch")
    if config.get("robustness_manifest_sha256") != robust_hash:
        raise RuntimeError(f"{path}: robustness manifest hash mismatch")
    if config.get("condition_manifest_sha256") != condition_hash:
        raise RuntimeError(f"{path}: condition manifest hash mismatch")
    if config.get("metadata_sha256") != metadata_hash:
        raise RuntimeError(f"{path}: metadata hash mismatch")
    expected_code_hashes = (
        rpcrf_locked_code_hashes() if is_rpcrf
        else gain_locked_code_hashes() if is_gain_candidate
        else fusion_locked_code_hashes()
    )
    if config.get("code_hashes") != expected_code_hashes:
        raise RuntimeError(f"{path}: code hash mismatch")
    current_runtime = runtime_versions()
    checkpoint_runtime = config.get("runtime_versions", {})
    strict_runtime_keys = ("h5py", "scikit_image", "scipy", "fastmri_version")
    runtime_differences = {
        key: {
            "checkpoint": checkpoint_runtime.get(key),
            "current": current_runtime.get(key),
        }
        for key in sorted(set(checkpoint_runtime) | set(current_runtime))
        if checkpoint_runtime.get(key) != current_runtime.get(key)
    }
    strict_runtime_match = all(
        checkpoint_runtime.get(key) == current_runtime.get(key)
        for key in strict_runtime_keys
    )
    if not strict_runtime_match:
        raise RuntimeError(
            f"{path}: strict runtime dependency mismatch: "
            f"{json.dumps(runtime_differences, indent=2)}"
        )
    if is_rpcrf and runtime_differences:
        raise RuntimeError(
            f"{path}: the HPC-trained RPCRF checkpoint must match the current "
            f"runtime exactly: {json.dumps(runtime_differences, indent=2)}"
        )
    model_class = M2PRNFRPCRFVarNet if is_rpcrf else M2PRNFFusionPilotVarNet
    model_kwargs = {}
    if is_rpcrf:
        model_kwargs["residual_reliability_power"] = float(
            config["residual_reliability_power"]
        )
        model_kwargs["residual_levels"] = (2,)
    model = model_class(
        model_variant=variant,
        fusion_design=fusion_design,
        need_scope=str(config["need_scope"]),
        residual_scale=float(config["residual_scale"]),
        num_cascades=int(config["num_cascades"]),
        sens_chans=int(config["sens_chans"]), sens_pools=int(config["sens_pools"]),
        chans=int(config["chans"]), pools=int(config["pools"]),
        controller_chans=int(config["controller_chans"]),
        initial_aux_alpha=float(config["initial_aux_alpha"]),
        initial_gate_probability=float(config["initial_gate_probability"]),
        initial_need_probability=float(config["initial_need_probability"]),
        need_floor=float(config["need_floor"]),
        **model_kwargs,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    observed_name = arm if (is_gain_candidate or is_rpcrf) else fusion_design
    runtime_audit = {
        "mode": (
            "native_hpc_runtime" if is_rpcrf
            else "audited_cross_cluster_checkpoint_inference"
        ),
        "strict_keys_matched": list(strict_runtime_keys),
        "differences": runtime_differences,
        "passed": True,
    }
    return model, observed_name, config, selected, runtime_audit


FAIRNESS_KEYS = (
    "acceleration", "pd_aux_acceleration", "epochs",
    "learning_rate", "num_workers",
    "num_train_patients",
    "num_val_patients", "max_train_batches", "max_val_batches", "num_cascades",
    "chans", "sens_chans", "pools", "sens_pools", "controller_chans",
    "initial_aux_alpha", "initial_gate_probability", "initial_need_probability",
    "need_floor", "lambda_rel", "lambda_rank", "aux_loss_ramp_epochs", "seed",
    "train_patient_ids", "val_patient_ids", "corruption_config",
    "correct_corrupt_loss_weights", "forward_view_proportions",
    "checkpoint_selection_metric", "full_clean_manifest_sha256",
    "robustness_manifest_sha256", "condition_manifest_sha256",
    "metadata_sha256", "optimizer", "gradient_clip_norm",
)


def audit_fairness(configs):
    reference, mismatches = configs["global_direct"], []
    for name, config in configs.items():
        for key in FAIRNESS_KEYS:
            if config.get(key) != reference.get(key):
                mismatches.append({"model": name, "key": key,
                                   "reference": reference.get(key),
                                   "observed": config.get(key)})
    mixtures = {name: configs[name].get("corrupt_view_mixture") for name in configs}
    if len({json.dumps(v, sort_keys=True) for v in mixtures.values()}) != 1:
        mismatches.append({"key": "augmented_corrupt_view_mixture", "observed": mixtures})
    effective_batches = {
        name: int(config.get("batch_size", 0))
        * int(config.get("grad_accum_steps", 0))
        for name, config in configs.items()
    }
    if set(effective_batches.values()) != {4}:
        mismatches.append(
            {"key": "effective_anatomical_batch", "observed": effective_batches}
        )
    return {
        "reference": "global_direct",
        "checked_keys": list(FAIRNESS_KEYS),
        "effective_anatomical_batches": effective_batches,
        "allowed_execution_differences": [
            "metadata_csv path after audited relocation",
            "runtime_versions",
            "batch_size and grad_accum_steps when their product is 4",
        ],
        "training_runtime_limitation": (
            "RPCRF H/4 was trained on Imperial HPC while historical controls "
            "were trained on Isambard. Cross-model pilot comparisons are "
            "contextual; same-checkpoint residual-on/off is the primary causal "
            "mechanism test."
        ),
        "passed": not mismatches,
        "mismatches": mismatches,
    }


def condition_pd(pd, indices, dataset, condition_lookup, condition):
    if condition == "correct":
        return pd, torch.ones(pd.shape[0], device=pd.device)
    if condition == "missing":
        return torch.zeros_like(pd), torch.zeros(pd.shape[0], device=pd.device)
    output = []
    for position, source in enumerate(indices):
        source = int(source)
        if source not in condition_lookup:
            raise RuntimeError(f"Source {source} missing from condition manifest")
        frozen = condition_lookup[source]
        source_record = dataset.records[source]
        if (
            str(source_record["patient_id"]) != frozen["patient_id"]
            or int(source_record["slice_idx"]) != int(frozen["slice_idx"])
        ):
            raise RuntimeError("Frozen source identity drift")
        if condition == "shift8":
            shift = frozen["shift8"]
            image = translate_nonwrapping(
                pd[position], int(shift["dy"]), int(shift["dx"]),
                str(shift["padding_mode"])
            )
        elif condition == "wrong_slice":
            replacement = frozen["wrong_slice"]
            replacement_index = int(replacement["replacement_index"])
            record = dataset.records[replacement_index]
            if (
                str(record["patient_id"]) != replacement["replacement_patient_id"]
                or int(record["slice_idx"]) != int(replacement["replacement_slice_idx"])
            ):
                raise RuntimeError("Frozen wrong-slice replacement identity drift")
            image = load_pd_auxiliary(dataset, replacement_index, pd.device)
        elif condition == "wrong_patient":
            replacement = frozen["wrong_patient"]
            replacement_index = int(replacement["replacement_index"])
            record = dataset.records[replacement_index]
            if (
                str(record["patient_id"]) != replacement["replacement_patient_id"]
                or int(record["slice_idx"]) != int(replacement["replacement_slice_idx"])
            ):
                raise RuntimeError("Frozen wrong-patient replacement identity drift")
            image = load_pd_auxiliary(dataset, replacement_index, pd.device)
        else:
            raise ValueError(condition)
        if image.shape != pd[position].shape:
            raise RuntimeError("Negative auxiliary shape mismatch")
        output.append(image)
    return torch.stack(output), torch.ones(pd.shape[0], device=pd.device)


def metrics(prediction, target):
    scale = float(target.max().clamp_min(1e-8).item())
    pred = prediction.detach().float().cpu().numpy() / scale
    truth = target.detach().float().cpu().numpy() / scale
    difference = pred - truth
    mse = float(np.mean(difference ** 2))
    background = truth <= 0.01
    background_error = np.abs(difference[background])
    return {"nmse": float(np.sum(difference ** 2) / max(np.sum(truth ** 2), 1e-12)),
            "psnr": float(-10.0 * math.log10(max(mse, 1e-12))),
            "ssim": float(structural_similarity(truth, pred, data_range=1.0)),
            "l1": float(np.mean(np.abs(difference))),
            "background_l1": float(np.mean(background_error)) if background_error.size else 0.0,
            "background_p99": float(np.quantile(background_error, 0.99))
            if background_error.size else 0.0}


@torch.no_grad()
def evaluate_mode(model, name, loader, dataset, condition_lookup, device, cohort, condition,
                  reliability_override=None, need_override=None, target_only=False,
                  seed=42):
    rows = []
    for batch in loader:
        kspace, mask, pd, target, indices = prepare_batch(batch, device)
        pd_used, available = condition_pd(
            pd, indices, dataset, condition_lookup, condition
        )
        if target_only:
            available = torch.zeros_like(available)
        prediction, aux = model(
            kspace, mask, pd_used, available, return_aux=True,
            reliability_override=reliability_override, need_override=need_override,
        )
        prediction = center_crop(prediction, target.shape[-2], target.shape[-1])
        for index in range(target.shape[0]):
            residual_values = aux["residual_to_target_rms"][index]
            ratio_values = aux["residual_to_direct_rms_ratio"][index]
            valid_ratios = ratio_values[ratio_values >= 0]
            tensors_to_check = (
                residual_values, aux["direct_to_target_rms"][index],
                aux["gated_aux_to_target_rms"][index],
            )
            rows.append({"model": name, "cohort": cohort, "condition": condition,
                         "patient_id": str(batch["patient_id"][index]),
                         "slice_idx": int(batch["slice_idx"][index]),
                         **metrics(prediction[index], target[index]),
                         "q": float(aux["q"].mean((1, 2))[index].item()),
                         "need": float(aux["need_mean"].mean((1, 2))[index].item()),
                         "effective_weight": float(aux["effective_weight_mean"].mean((1, 2))[index].item()),
                         "gated_rms": float(aux["gated_aux_to_target_rms"].mean((1, 2))[index].item()),
                         "direct_rms": float(aux["direct_to_target_rms"].mean((1, 2))[index].item()),
                         "residual_rms": float(aux["residual_to_target_rms"].mean((1, 2))[index].item()),
                         "residual_rms_max": float(residual_values.max().item()),
                         "residual_direct_ratio": float(valid_ratios.mean().item())
                         if valid_ratios.numel() else -1.0,
                         "raw_auxiliary_rms": float(aux["raw_auxiliary_to_target_rms"].mean((1, 2))[index].item()),
                         "feature_cosine": float(aux["target_auxiliary_cosine"].mean((1, 2))[index].item()),
                         "diagnostics_finite": float(all(
                             bool(torch.isfinite(value).all()) for value in tensors_to_check
                         ))})
    return rows


def patient_average(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["cohort"], row["condition"], row["patient_id"])].append(row)
    output = []
    for (model, cohort, condition, patient), values in sorted(groups.items()):
        output.append({"model": model, "cohort": cohort, "condition": condition,
                       "patient_id": patient, "num_slices": len(values),
                       **{key: float(np.mean([float(row[key]) for row in values]))
                          for key in (*METRICS, *DIAGNOSTICS)}})
    return output


def summary_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["cohort"], row["condition"])].append(row)
    return [{"model": model, "cohort": cohort, "condition": condition,
             "num_patients": len(values),
             **{key: float(np.mean([row[key] for row in values]))
                for key in (*METRICS, *DIAGNOSTICS)}}
            for (model, cohort, condition), values in sorted(groups.items())]


def paired(rows, a, b, cohort, condition, metric):
    values = {(row["model"], row["patient_id"]): float(row[metric]) for row in rows
              if row["cohort"] == cohort and row["condition"] == condition}
    patients = sorted(p for m, p in values if m == a and (b, p) in values)
    if not patients:
        raise RuntimeError(f"No paired patients: {a}/{b}/{cohort}/{condition}")
    return patients, values


def bootstrap_improvement(rows, a, b, cohort, condition, metric, n, seed):
    patients, values = paired(rows, a, b, cohort, condition, metric)
    direction = -1.0 if metric in {"nmse", "l1"} else 1.0
    delta = np.asarray([direction * (values[(a, p)] - values[(b, p)]) for p in patients])
    rng = np.random.default_rng(seed)
    samples = delta[rng.integers(0, len(delta), (n, len(delta)))].mean(1)
    return {"model_a": a, "model_b": b, "cohort": cohort, "condition": condition,
            "metric": metric, "positive_means_a_better": True,
            "mean_improvement": float(delta.mean()),
            "ci95_low": float(np.quantile(samples, 0.025)),
            "ci95_high": float(np.quantile(samples, 0.975)),
            "patients_a_better": int((delta > 0).sum()), "num_patients": len(patients)}


def bootstrap_noninferiority(rows, model, control, cohort, condition, margin, n, seed):
    patients, values = paired(rows, model, control, cohort, condition, "l1")
    degradation = np.asarray([(values[(model, p)] - values[(control, p)])
                              / max(values[(control, p)], 1e-12) for p in patients])
    rng = np.random.default_rng(seed)
    samples = degradation[rng.integers(0, len(degradation), (n, len(degradation)))].mean(1)
    upper = float(np.quantile(samples, 0.95))
    return {"model": model, "control": control, "cohort": cohort,
            "condition": condition, "relative_l1_margin": margin,
            "mean_relative_l1_degradation": float(degradation.mean()),
            "one_sided_95_upper": upper, "passed": upper < margin,
            "num_patients": len(patients)}


def append_composite(rows, models):
    lookup = {(r["model"], r["cohort"], r["condition"], r["patient_id"]): r for r in rows}
    for model in models:
        patients = sorted({r["patient_id"] for r in rows
                           if r["model"] == model and r["cohort"] == "robustness"})
        for patient in patients:
            rows.append({"model": model, "cohort": "robustness",
                         "condition": "robustness_composite", "patient_id": patient,
                         "num_slices": sum(lookup[(model, "robustness", c, patient)]["num_slices"]
                                           for c in ROBUST_CONDITIONS),
                         **{metric: float(np.mean([lookup[(model, "robustness", c, patient)][metric]
                                                   for c in ROBUST_CONDITIONS]))
                            for metric in METRICS},
                         **{key: 0.0 for key in DIAGNOSTICS}})


def spearman(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(spearmanr(x, y).statistic)


def bootstrap_condition_improvement(
    rows, model, better_condition, worse_condition, metric, n, seed
):
    selected = {
        (row["condition"], row["patient_id"]): float(row[metric])
        for row in rows
        if row["model"] == model and row["cohort"] == "robustness"
        and row["condition"] in {better_condition, worse_condition}
    }
    patients = sorted(
        patient for observed_condition, patient in selected
        if observed_condition == better_condition
        and (worse_condition, patient) in selected
    )
    if not patients:
        raise RuntimeError(
            f"No paired patients for {model}: "
            f"{better_condition}/{worse_condition}/{metric}"
        )
    direction = -1.0 if metric in {"nmse", "l1"} else 1.0
    delta = np.asarray([
        direction * (
            selected[(better_condition, patient)]
            - selected[(worse_condition, patient)]
        )
        for patient in patients
    ])
    rng = np.random.default_rng(seed)
    samples = delta[rng.integers(0, len(delta), (n, len(delta)))].mean(1)
    return {
        "model": model,
        "better_condition": better_condition,
        "worse_condition": worse_condition,
        "metric": metric,
        "positive_means_first_condition_better": True,
        "mean_improvement": float(delta.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "patients_improved": int((delta > 0).sum()),
        "num_patients": len(patients),
    }


def bootstrap_condition_noninferiority(
    rows, model, condition, control_condition, margin, n, seed
):
    selected = {
        (row["condition"], row["patient_id"]): float(row["l1"])
        for row in rows
        if row["model"] == model and row["cohort"] == "robustness"
        and row["condition"] in {condition, control_condition}
    }
    patients = sorted(
        patient for observed_condition, patient in selected
        if observed_condition == condition
        and (control_condition, patient) in selected
    )
    if not patients:
        raise RuntimeError(
            f"No paired patients for {model}: "
            f"{condition}/{control_condition}/l1"
        )
    degradation = np.asarray([
        (
            selected[(condition, patient)]
            - selected[(control_condition, patient)]
        ) / max(selected[(control_condition, patient)], 1e-12)
        for patient in patients
    ])
    rng = np.random.default_rng(seed)
    samples = degradation[
        rng.integers(0, len(degradation), (n, len(degradation)))
    ].mean(1)
    upper = float(np.quantile(samples, 0.95))
    return {
        "model": model,
        "condition": condition,
        "control_condition": control_condition,
        "relative_l1_margin": margin,
        "mean_relative_l1_degradation": float(degradation.mean()),
        "one_sided_95_upper": upper,
        "passed": upper <= margin,
        "num_patients": len(patients),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--metadata_identity_csv", required=True)
    parser.add_argument("--metadata_source_root", required=True)
    parser.add_argument("--metadata_runtime_root", required=True)
    parser.add_argument("--full_clean_manifest", required=True)
    parser.add_argument("--robustness_manifest", required=True)
    parser.add_argument("--condition_manifest", required=True)
    parser.add_argument(
        "--m2u_augmented_epoch15_audit", required=True,
        help="Frozen epoch-15 clean audit from the matched M2-U Augmented run.",
    )
    parser.add_argument("--output_dir", required=True)
    for arm in PILOT_ARMS:
        parser.add_argument(f"--{arm}_checkpoint", required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--bootstrap_resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    m2u_epoch15_path = Path(args.m2u_augmented_epoch15_audit).resolve()
    m2u_epoch15 = json.loads(m2u_epoch15_path.read_text(encoding="utf-8"))
    m2u_epoch15_l1 = m2u_epoch15.get(
        "patient_l1", m2u_epoch15.get("val_patient_l1")
    )
    if m2u_epoch15_l1 is None or not math.isfinite(float(m2u_epoch15_l1)):
        raise RuntimeError("M2-U Augmented epoch-15 audit has no finite patient L1")
    m2u_epoch15_l1 = float(m2u_epoch15_l1)

    manifest_paths = {
        "full_clean": Path(args.full_clean_manifest).resolve(),
        "robustness": Path(args.robustness_manifest).resolve(),
    }
    condition_manifest_path = Path(args.condition_manifest).resolve()
    manifests = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in manifest_paths.items()
    }
    hashes = {name: sha256_file(path) for name, path in manifest_paths.items()}
    condition_hash = sha256_file(condition_manifest_path)
    metadata_relocation = validate_metadata_relocation(
        Path(args.metadata_identity_csv),
        Path(args.metadata_csv),
        args.metadata_source_root,
        args.metadata_runtime_root,
    )
    if not metadata_relocation["passed"]:
        raise RuntimeError(
            "Metadata relocation failed:\n"
            + json.dumps(metadata_relocation, indent=2)
        )
    metadata_hash = sha256_file(Path(args.metadata_identity_csv).resolve())
    for name, manifest in manifests.items():
        if (
            manifest.get("protocol_version") != MANIFEST_PROTOCOL_VERSION
            or manifest.get("cohort") != name
        ):
            raise RuntimeError(f"{name}: manifest protocol/cohort mismatch")

    dataset_args = argparse.Namespace(
        metadata_csv=str(Path(args.metadata_csv).resolve()),
        acceleration=8,
        pd_aux_acceleration=2,
    )
    full_dataset = IndexedDataset(make_dataset(dataset_args, "val"))
    condition_manifest = json.loads(
        condition_manifest_path.read_text(encoding="utf-8")
    )
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
    checkpoints = {
        arm: Path(getattr(args, f"{arm}_checkpoint")).resolve()
        for arm in PILOT_ARMS
    }

    slice_rows, checkpoint_audit, configs = [], {}, {}
    for arm, path in checkpoints.items():
        model, observed, config, selected_epoch, runtime_audit = build_model(
            path, arm, device, hashes["full_clean"], hashes["robustness"],
            condition_hash, metadata_hash,
        )
        if observed != arm:
            raise RuntimeError(f"Expected {arm}, checkpoint contains {observed}")
        configs[arm] = config
        checkpoint_audit[arm] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "training_budget_epochs": 15,
            "selected_checkpoint_epoch": selected_epoch,
            "seed": config["seed"],
            "variant": config["variant"],
            "training_objective": config.get(
                "training_objective", "standard_fusion_pilot"
            ),
            "lambda_residual_gain": config.get("lambda_residual_gain", 0.0),
            "residual_gain_margin_relative": config.get(
                "residual_gain_margin_relative", 0.0
            ),
            "runtime_audit": runtime_audit,
            "batch_size": config.get("batch_size"),
            "grad_accum_steps": config.get("grad_accum_steps"),
            "effective_anatomical_batch": (
                int(config.get("batch_size", 0))
                * int(config.get("grad_accum_steps", 0))
            ),
        }
        slice_rows += evaluate_mode(
            model, arm, loaders["full_clean"], full_dataset, condition_lookup,
            device, "full_clean", "correct", seed=args.seed
        )
        # Same-checkpoint causal intervention.  The extra row set answers the
        # specific scientific question targeted by the new objective: did the
        # learned complementary residual improve clean reconstruction beyond
        # this checkpoint's direct path, rather than merely remain active?
        if arm in {GAIN_ARM, RPCRF_ARM}:
            off_name = (
                GAIN_RESIDUAL_OFF if arm == GAIN_ARM else RPCRF_RESIDUAL_OFF
            )
            with residual_scale_override(model, 0.0):
                slice_rows += evaluate_mode(
                    model, off_name, loaders["full_clean"],
                    full_dataset, condition_lookup, device, "full_clean",
                    "correct", seed=args.seed,
                )
        for condition in CONDITIONS:
            slice_rows += evaluate_mode(
                model, arm, loaders["robustness"], full_dataset,
                condition_lookup, device, "robustness", condition,
                seed=args.seed,
            )
        if arm == RPCRF_ARM:
            with residual_scale_override(model, 0.0):
                for condition in CONDITIONS:
                    slice_rows += evaluate_mode(
                        model, RPCRF_RESIDUAL_OFF, loaders["robustness"],
                        full_dataset, condition_lookup, device, "robustness",
                        condition, seed=args.seed,
                    )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fairness = audit_fairness(configs)
    if not fairness["passed"]:
        raise RuntimeError(
            "Cross-checkpoint fairness failed:\n"
            + json.dumps(fairness["mismatches"], indent=2)
        )

    patient_rows = patient_average(slice_rows)
    append_composite(
        patient_rows, (*PILOT_ARMS, RPCRF_RESIDUAL_OFF)
    )
    summaries = summary_rows(patient_rows)
    comparisons = []
    for arm in PILOT_ARMS:
        if arm != "global_direct":
            for metric in METRICS:
                comparisons.append(
                    bootstrap_improvement(
                        patient_rows, arm, "global_direct", "full_clean", "correct",
                        metric, args.bootstrap_resamples, args.seed
                    )
                )
            for condition in (*CONDITIONS, "robustness_composite"):
                comparisons.append(
                    bootstrap_improvement(
                        patient_rows, arm, "global_direct", "robustness", condition,
                        "l1", args.bootstrap_resamples, args.seed
                    )
                )
    for metric in METRICS:
        comparisons.append(
            bootstrap_improvement(
                patient_rows, RPCRF_ARM, GAIN_ARM, "full_clean", "correct",
                metric, args.bootstrap_resamples, args.seed
            )
        )
    mechanism_tests = {}
    for arm, off_name in (
        (GAIN_ARM, GAIN_RESIDUAL_OFF), (RPCRF_ARM, RPCRF_RESIDUAL_OFF)
    ):
        mechanism_tests[arm] = [
            bootstrap_improvement(
                patient_rows, arm, off_name, "full_clean", "correct",
                metric, args.bootstrap_resamples, args.seed
            )
            for metric in METRICS
        ]
        comparisons.extend(mechanism_tests[arm])

    condition_comparisons = []
    condition_noninferiority = []
    for arm in PILOT_ARMS:
        condition_comparisons.append(
            bootstrap_condition_improvement(
                patient_rows, arm, "correct", "missing", "l1",
                args.bootstrap_resamples, args.seed
            )
        )
        for condition in ROBUST_CONDITIONS:
            condition_comparisons.append(
                bootstrap_condition_improvement(
                    patient_rows, arm, condition, "missing", "l1",
                    args.bootstrap_resamples, args.seed
                )
            )

    summary_lookup = {
        (row["model"], row["cohort"], row["condition"]): row
        for row in summaries
    }
    arm_decisions = {}
    for arm in PILOT_ARMS:
        clean_ni = bootstrap_noninferiority(
            patient_rows, arm, "global_direct", "full_clean", "correct", 0.005,
            args.bootstrap_resamples, args.seed
        )
        missing_slices = [
            row for row in slice_rows
            if row["model"] == arm and row["cohort"] == "robustness"
            and row["condition"] == "missing"
        ]
        missing_zero = bool(missing_slices) and max(
            row["gated_rms"] for row in missing_slices
        ) == 0.0
        robustness_correct_l1 = summary_lookup[
            (arm, "robustness", "correct")
        ]["l1"]
        missing_l1 = summary_lookup[(arm, "robustness", "missing")]["l1"]
        corrupt_relative_harm = {
            condition: (
                summary_lookup[(arm, "robustness", condition)]["l1"]
                - missing_l1
            ) / max(missing_l1, 1e-12)
            for condition in ROBUST_CONDITIONS
        }
        clean_benefit_test = bootstrap_condition_improvement(
            patient_rows, arm, "correct", "missing", "l1",
            args.bootstrap_resamples, args.seed
        )
        clean_benefit_status = (
            "eligible" if clean_benefit_test["ci95_low"] > 0.0
            else "borderline" if clean_benefit_test["mean_improvement"] > 0.0
            else "rejected"
        )
        corrupt_safety_tests = {
            condition: bootstrap_condition_noninferiority(
                patient_rows, arm, condition, "missing", 0.005,
                args.bootstrap_resamples, args.seed
            )
            for condition in ROBUST_CONDITIONS
        }
        condition_noninferiority.extend(corrupt_safety_tests.values())
        corrupt_safety_status = {
            condition: (
                "eligible" if test["passed"]
                else "borderline"
                if test["mean_relative_l1_degradation"] <= 0.005
                else "rejected"
            )
            for condition, test in corrupt_safety_tests.items()
        }
        arm_slice_rows = [row for row in slice_rows if row["model"] == arm]
        numerical_stability = {
            "all_diagnostics_finite": all(
                row["diagnostics_finite"] == 1.0 for row in arm_slice_rows
            ),
            "maximum_residual_to_target_rms": max(
                (row["residual_rms_max"] for row in arm_slice_rows), default=0.0
            ),
            "residual_dominance_limit": 1.0,
        }
        numerical_stability["passed"] = (
            numerical_stability["all_diagnostics_finite"]
            and numerical_stability["maximum_residual_to_target_rms"] <= 1.0
        )
        hard_failure = (
            not clean_ni["passed"]
            or not missing_zero
            or not numerical_stability["passed"]
            or clean_benefit_status == "rejected"
            or "rejected" in corrupt_safety_status.values()
        )
        qualification = (
            "rejected" if hard_failure
            else "eligible" if clean_benefit_status == "eligible"
            and all(value == "eligible" for value in corrupt_safety_status.values())
            else "borderline"
        )
        arm_decisions[arm] = {
            "clean_noninferiority_vs_global_direct": clean_ni,
            "missing_exact_zero": missing_zero,
            "clean_l1": summary_lookup[(arm, "full_clean", "correct")]["l1"],
            "robustness_composite_l1": summary_lookup[
                (arm, "robustness", "robustness_composite")
            ]["l1"],
            "clean_auxiliary_benefit_l1": missing_l1 - robustness_correct_l1,
            "clean_auxiliary_benefit_test": clean_benefit_test,
            "clean_auxiliary_benefit_status": clean_benefit_status,
            "corrupt_relative_harm_vs_missing": corrupt_relative_harm,
            "corrupt_safety_tests": corrupt_safety_tests,
            "corrupt_safety_status": corrupt_safety_status,
            "numerical_residual_stability": numerical_stability,
            "qualification": qualification,
            "clean_q": summary_lookup[(arm, "full_clean", "correct")]["q"],
            "wrong_patient_q": summary_lookup[
                (arm, "robustness", "wrong_patient")
            ]["q"],
        }

    for mechanism_arm in (GAIN_ARM, RPCRF_ARM):
        l1_test = next(
            test for test in mechanism_tests[mechanism_arm]
            if test["metric"] == "l1"
        )
        mechanism_status = (
            "eligible" if l1_test["ci95_low"] > 0.0
            else "borderline" if l1_test["mean_improvement"] > 0.0
            else "rejected"
        )
        arm_decisions[mechanism_arm][
            "same_checkpoint_residual_incremental_gain"
        ] = {
            "status": mechanism_status,
            "tests": mechanism_tests[mechanism_arm],
            "interpretation": (
                "Positive values mean residual-on is better than the identical "
                "checkpoint evaluated with residual_scale=0."
            ),
        }
        if mechanism_status == "rejected":
            arm_decisions[mechanism_arm]["qualification"] = "rejected"
        elif (
            mechanism_status == "borderline"
            and arm_decisions[mechanism_arm]["qualification"] == "eligible"
        ):
            arm_decisions[mechanism_arm]["qualification"] = "borderline"

    rpcrf_vs_fifth_l1 = next(
        test for test in comparisons
        if test.get("model_a") == RPCRF_ARM
        and test.get("model_b") == GAIN_ARM
        and test.get("cohort") == "full_clean"
        and test.get("condition") == "correct"
        and test.get("metric") == "l1"
    )
    rpcrf_clean_l1 = arm_decisions[RPCRF_ARM]["clean_l1"]
    rpcrf_robust_l1 = arm_decisions[RPCRF_ARM]["robustness_composite_l1"]
    global_robust_l1 = arm_decisions["global_direct"]["robustness_composite_l1"]
    rpcrf_mechanism_by_metric = {
        test["metric"]: test for test in mechanism_tests[RPCRF_ARM]
    }
    residual_metric_consistency = {
        metric: {
            "mean_improvement": rpcrf_mechanism_by_metric[metric][
                "mean_improvement"
            ],
            "direction_favourable": (
                rpcrf_mechanism_by_metric[metric]["mean_improvement"] > 0.0
            ),
        }
        for metric in METRICS
    }
    residual_metric_consistency["passed"] = all(
        residual_metric_consistency[metric]["direction_favourable"]
        for metric in METRICS
    )
    rpcrf_residual_robustness = bootstrap_improvement(
        patient_rows,
        RPCRF_ARM,
        RPCRF_RESIDUAL_OFF,
        "robustness",
        "robustness_composite",
        "l1",
        args.bootstrap_resamples,
        args.seed,
    )
    robustness_margin = 0.005
    rpcrf_vs_global_robustness_noninferiority = bootstrap_noninferiority(
        patient_rows,
        RPCRF_ARM,
        "global_direct",
        "robustness",
        "robustness_composite",
        robustness_margin,
        args.bootstrap_resamples,
        args.seed,
    )
    rpcrf_residual_robustness_noninferiority = bootstrap_noninferiority(
        patient_rows,
        RPCRF_ARM,
        RPCRF_RESIDUAL_OFF,
        "robustness",
        "robustness_composite",
        robustness_margin,
        args.bootstrap_resamples,
        args.seed,
    )
    rpcrf_off_robust_l1 = summary_lookup[
        (RPCRF_RESIDUAL_OFF, "robustness", "robustness_composite")
    ]["l1"]
    final_promotion_gates = {
        "clean_better_than_fifth_arm": rpcrf_vs_fifth_l1,
        "m2u_augmented_epoch15_contextual_target": {
            "rpcrf_l1": rpcrf_clean_l1,
            "m2u_augmented_epoch15_l1": m2u_epoch15_l1,
            "point_estimate_met": rpcrf_clean_l1 <= m2u_epoch15_l1,
            "included_in_automatic_promotion": False,
            "evidence_scope": (
                "Contextual target only: the M2-U epoch-15 audit was generated "
                "on Isambard and is not a strict cross-runtime significance "
                "test. This point estimate cannot veto promotion."
            ),
        },
        "residual_metric_direction_consistency": residual_metric_consistency,
        "robustness_within_0_5_percent_of_global_direct": {
            "rpcrf_composite_l1": rpcrf_robust_l1,
            "global_direct_composite_l1": global_robust_l1,
            "relative_difference": (
                rpcrf_robust_l1 / max(global_robust_l1, 1e-12) - 1.0
            ),
            "noninferiority": rpcrf_vs_global_robustness_noninferiority,
            "decision_rule": (
                "One-sided patient-bootstrap 95% upper bound of relative "
                "L1 degradation must be below 0.5%."
            ),
            "passed": rpcrf_vs_global_robustness_noninferiority["passed"],
        },
        "same_checkpoint_residual_robustness_safety": {
            "paired_bootstrap": rpcrf_residual_robustness,
            "noninferiority": rpcrf_residual_robustness_noninferiority,
            "residual_on_l1": rpcrf_robust_l1,
            "residual_off_l1": rpcrf_off_robust_l1,
            "relative_difference": (
                rpcrf_robust_l1 / max(rpcrf_off_robust_l1, 1e-12) - 1.0
            ),
            "noninferiority_margin": robustness_margin,
            "decision_rule": (
                "One-sided patient-bootstrap 95% upper bound of relative "
                "L1 degradation must be below 0.5%."
            ),
            "passed": rpcrf_residual_robustness_noninferiority["passed"],
        },
        "missing_pd_exact_zero": {
            "passed": arm_decisions[RPCRF_ARM]["missing_exact_zero"]
        },
        "numerical_residual_stability": arm_decisions[RPCRF_ARM][
            "numerical_residual_stability"
        ],
    }
    final_promotion_gates["passed"] = (
        rpcrf_vs_fifth_l1["ci95_low"] > 0.0
        and final_promotion_gates[
            "robustness_within_0_5_percent_of_global_direct"
        ]["passed"]
        and final_promotion_gates[
            "residual_metric_direction_consistency"
        ]["passed"]
        and final_promotion_gates[
            "same_checkpoint_residual_robustness_safety"
        ]["passed"]
        and final_promotion_gates["missing_pd_exact_zero"]["passed"]
        and final_promotion_gates["numerical_residual_stability"]["passed"]
        and arm_decisions[RPCRF_ARM][
            "same_checkpoint_residual_incremental_gain"
        ]["status"] == "eligible"
    )
    arm_decisions[RPCRF_ARM]["final_promotion_gates"] = final_promotion_gates
    if not final_promotion_gates["passed"]:
        arm_decisions[RPCRF_ARM]["qualification"] = "rejected"

    eligible = [
        arm for arm in PILOT_ARMS
        if arm_decisions[arm]["qualification"] == "eligible"
    ]
    borderline = [
        arm for arm in PILOT_ARMS
        if arm_decisions[arm]["qualification"] == "borderline"
    ]
    rejected = [
        arm for arm in PILOT_ARMS
        if arm_decisions[arm]["qualification"] == "rejected"
    ]
    robust_best = min(
        (arm_decisions[arm]["robustness_composite_l1"] for arm in eligible),
        default=float("inf"),
    )
    robust_band = [
        arm for arm in eligible
        if arm_decisions[arm]["robustness_composite_l1"] <= 1.01 * robust_best
    ]
    recommended = min(
        robust_band, key=lambda arm: arm_decisions[arm]["clean_l1"],
        default="none",
    )
    decision = {
        "protocol_version": PROTOCOL_VERSION,
        "arm_results": arm_decisions,
        "eligible_arms": eligible,
        "borderline_arms": borderline,
        "rejected_arms": rejected,
        "recommended_fusion_design": recommended,
        "recommendation_rule": (
            "Only CI-qualified eligible arms can be automatically recommended. "
            "H/4 RPCRF must significantly beat the fifth arm in clean L1, show "
            "a significant "
            "same-checkpoint residual-on L1 gain, improve NMSE/PSNR/SSIM/L1 "
            "in the favourable direction, preserve missing-PD exact zero and "
            "numerical stability, and pass one-sided patient-bootstrap "
            "non-inferiority tests with a 0.5% margin against both "
            "global-direct robustness and the same checkpoint with its "
            "residual disabled. The imported M2-U Augmented epoch-15 point "
            "estimate is contextual only and cannot veto promotion. "
            "Failure ends architecture iteration. "
            "Among them, retain candidates "
            "within 1% of the best robustness-composite L1 and select the lowest "
            "full-clean patient-macro L1. Confirm scientific judgement from CIs "
            "and representative images before formal training."
        ),
        "formal_variants_after_selection": [
            "prnf_no_rel", "prnf_no_need", "prnf_full"
        ],
        "historical_controls_to_reuse": [
            "m2u_clean", "m2u_augmented", "m2u_augcap_mask", "prnf_full_v1.3"
        ],
    }

    write_csv(output_dir / "fusion_pilot_per_slice.csv", slice_rows)
    write_csv(output_dir / "fusion_pilot_patient_level.csv", patient_rows)
    write_csv(output_dir / "fusion_pilot_summary.csv", summaries)
    write_csv(output_dir / "fusion_pilot_paired_bootstrap.csv", comparisons)
    write_csv(
        output_dir / "fusion_pilot_condition_bootstrap.csv", condition_comparisons
    )
    write_csv(
        output_dir / "fusion_pilot_condition_noninferiority.csv",
        condition_noninferiority,
    )
    write_csv(
        output_dir / "fusion_pilot_final_noninferiority.csv",
        [
            rpcrf_vs_global_robustness_noninferiority,
            rpcrf_residual_robustness_noninferiority,
        ],
    )
    audit = {
        "protocol_version": PROTOCOL_VERSION,
        "evaluator_hotfix": {
            "id": EVALUATOR_HOTFIX_ID,
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
            "scope": (
                "Final three-model locked comparison with same-checkpoint "
                "residual-on/off interventions for the fifth arm and RPCRF; "
                "cross-runtime M2-U epoch-15 is contextual only, while final "
                "robustness safety uses one-sided patient-bootstrap "
                "non-inferiority upper bounds."
            ),
            "completed_pilot_locked_evaluator_sha256": (
                fusion_locked_code_hashes().get(
                    "scripts/evaluate_m2_prnf_fusion_R8.py"
                )
            ),
            "quality_gain_locked_evaluator_sha256": (
                gain_locked_code_hashes().get(
                    "scripts/evaluate_m2_prnf_quality_gain_R8.py"
                )
            ),
            "rpcrf_locked_evaluator_sha256": (
                rpcrf_locked_code_hashes().get(
                    "scripts/evaluate_m2_prnf_rpcrf_R8.py"
                )
            ),
        },
        "manifests": {
            name: {
                "path": str(path),
                "sha256": hashes[name],
                "num_patients": manifests[name]["num_patients"],
                "num_slices": manifests[name]["num_slices"],
            }
            for name, path in manifest_paths.items()
        },
        "condition_manifest": {
            "path": str(condition_manifest_path),
            "sha256": condition_hash,
            "num_entries": condition_manifest["num_entries"],
            "shift8_definition": condition_manifest["shift8_definition"],
        },
        "metadata_sha256": metadata_hash,
        "metadata_relocation": metadata_relocation,
        "m2u_augmented_epoch15_audit": {
            "path": str(m2u_epoch15_path),
            "sha256": sha256_file(m2u_epoch15_path),
            "patient_l1": m2u_epoch15_l1,
            "evidence_scope": (
                "Contextual target imported from Isambard; not formal "
                "cross-runtime paired evidence."
            ),
        },
        "checkpoint_audit": checkpoint_audit,
        "cross_checkpoint_fairness": fairness,
        "code_hashes": {
            "completed_four_arm_pilots": fusion_locked_code_hashes(),
            "quality_gain_candidate": gain_locked_code_hashes(),
            "rpcrf_h4_final_candidate": rpcrf_locked_code_hashes(),
        },
        "runtime_versions": runtime_versions(),
        "bootstrap_resamples": args.bootstrap_resamples,
        "seed": args.seed,
        "robustness_composite_conditions": ROBUST_CONDITIONS,
    }
    (output_dir / "fusion_pilot_evaluation_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    (output_dir / "fusion_pilot_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
