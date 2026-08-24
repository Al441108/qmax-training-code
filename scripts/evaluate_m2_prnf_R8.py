#!/usr/bin/env python3
from __future__ import annotations

"""Frozen v1.2 patient-level evaluator for the M2-PRNF R=8 protocol."""

import argparse
import csv
import json
import math
import random
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

from scripts.train_m2_prnf import (  # noqa: E402
    IndexedDataset, ShapeBucketBatchSampler, locked_code_hashes, make_dataset,
    prepare_batch, runtime_versions, set_seed, sha256_file,
)
from src.fft_utils import center_crop  # noqa: E402
from src.m2_prnf_corruptions import (  # noqa: E402
    HardNegativeSampler, load_pd_auxiliary, translate_nonwrapping,
)
from src.m2_prnf_varnet import M2PRNFAuxPDVarNet, VALID_VARIANTS  # noqa: E402

PROTOCOL_VERSION = "M2-PRNF-R8-v1.3-bs4-audited"
CONDITIONS = ("correct", "shift8", "wrong_slice", "wrong_patient", "missing")
ROBUST_CONDITIONS = ("shift8", "wrong_slice", "wrong_patient")
METRICS = ("nmse", "psnr", "ssim", "l1")


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


def build_model(path: Path, device, clean_hash: str, robust_hash: str):
    if path.name != "model_best.pt":
        raise RuntimeError(f"Evaluator requires model_best.pt, got {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    variant = str(config.get("variant"))
    if variant not in VALID_VARIANTS:
        raise RuntimeError(f"{path}: invalid variant {variant}")
    if int(config.get("acceleration", -1)) != 8 or int(config.get("epochs", -1)) != 50:
        raise RuntimeError(f"{path}: not a formal R=8/50-epoch run")
    summary_path = path.parent / "final_summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"{path}: missing final_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary.get("completed_epochs", -1)) != 50:
        raise RuntimeError(f"{path}: formal run did not complete 50 epochs")
    selected = int(checkpoint.get("epoch", -1))
    if not selected == int(checkpoint.get("best_epoch", -1)) == int(summary.get("best_epoch", -1)):
        raise RuntimeError(f"{path}: selected/checkpoint/summary best epoch mismatch")
    if summary.get("variant") != variant:
        raise RuntimeError(f"{path}: final-summary variant mismatch")
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
    if config.get("code_hashes") != locked_code_hashes():
        raise RuntimeError(f"{path}: code hash mismatch")
    if config.get("runtime_versions") != runtime_versions():
        raise RuntimeError(f"{path}: runtime fingerprint mismatch")
    model = M2PRNFAuxPDVarNet(
        variant=variant, num_cascades=int(config["num_cascades"]),
        sens_chans=int(config["sens_chans"]), sens_pools=int(config["sens_pools"]),
        chans=int(config["chans"]), pools=int(config["pools"]),
        controller_chans=int(config["controller_chans"]),
        initial_aux_alpha=float(config["initial_aux_alpha"]),
        initial_gate_probability=float(config["initial_gate_probability"]),
        initial_need_probability=float(config["initial_need_probability"]),
        need_floor=float(config["need_floor"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, variant, config, selected


FAIRNESS_KEYS = (
    "metadata_csv", "acceleration", "pd_aux_acceleration", "epochs",
    "learning_rate", "batch_size", "grad_accum_steps", "num_workers",
    "num_train_patients",
    "num_val_patients", "max_train_batches", "max_val_batches", "num_cascades",
    "chans", "sens_chans", "pools", "sens_pools", "controller_chans",
    "initial_aux_alpha", "initial_gate_probability", "initial_need_probability",
    "need_floor", "lambda_rel", "lambda_rank", "aux_loss_ramp_epochs", "seed",
    "train_patient_ids", "val_patient_ids", "corruption_config",
    "correct_corrupt_loss_weights", "forward_view_proportions",
    "checkpoint_selection_metric", "full_clean_manifest_sha256",
    "robustness_manifest_sha256", "optimizer", "gradient_clip_norm",
    "code_hashes", "runtime_versions",
)


def audit_fairness(configs):
    reference, mismatches = configs["m2u_clean"], []
    for name, config in configs.items():
        for key in FAIRNESS_KEYS:
            if config.get(key) != reference.get(key):
                mismatches.append({"model": name, "key": key,
                                   "reference": reference.get(key),
                                   "observed": config.get(key)})
    augmented = [name for name in configs if name != "m2u_clean"]
    mixtures = {name: configs[name].get("corrupt_view_mixture") for name in augmented}
    if len({json.dumps(v, sort_keys=True) for v in mixtures.values()}) != 1:
        mismatches.append({"key": "augmented_corrupt_view_mixture", "observed": mixtures})
    if reference.get("corrupt_view_mixture") not in ({}, None):
        mismatches.append({"key": "m2u_clean_corrupt_view_mixture",
                           "observed": reference.get("corrupt_view_mixture")})
    return {"reference": "m2u_clean", "checked_keys": list(FAIRNESS_KEYS),
            "passed": not mismatches, "mismatches": mismatches}


def condition_pd(pd, indices, dataset, sampler, condition, seed):
    if condition == "correct":
        return pd, torch.ones(pd.shape[0], device=pd.device)
    if condition == "missing":
        return torch.zeros_like(pd), torch.zeros(pd.shape[0], device=pd.device)
    output = []
    for position, source in enumerate(indices):
        rng = random.Random(seed + 1_000_003 * int(source))
        if condition == "shift8":
            image = translate_nonwrapping(pd[position], 0, 8, "reflect")
        elif condition == "wrong_slice":
            candidate = sampler.same_patient_wrong_slice(source, rng)
            if candidate is None:
                raise RuntimeError(f"No wrong-slice candidate for {source}")
            image = load_pd_auxiliary(dataset, candidate[0], pd.device)
        elif condition == "wrong_patient":
            candidate = sampler.wrong_patient_matched_level(
                source, tuple(pd[position].shape), rng, top_k=8
            )
            if candidate is None:
                raise RuntimeError(f"No wrong-patient candidate for {source}")
            image = load_pd_auxiliary(dataset, candidate[0], pd.device)
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
    return {"nmse": float(np.sum(difference ** 2) / max(np.sum(truth ** 2), 1e-12)),
            "psnr": float(-10.0 * math.log10(max(mse, 1e-12))),
            "ssim": float(structural_similarity(truth, pred, data_range=1.0)),
            "l1": float(np.mean(np.abs(difference)))}


@torch.no_grad()
def evaluate_mode(model, name, loader, dataset, sampler, device, cohort, condition,
                  reliability_override=None, need_override=None, target_only=False,
                  seed=42):
    rows = []
    for batch in loader:
        kspace, mask, pd, target, indices = prepare_batch(batch, device)
        pd_used, available = condition_pd(pd, indices, dataset, sampler, condition, seed)
        if target_only:
            available = torch.zeros_like(available)
        prediction, aux = model(
            kspace, mask, pd_used, available, return_aux=True,
            reliability_override=reliability_override, need_override=need_override,
        )
        prediction = center_crop(prediction, target.shape[-2], target.shape[-1])
        for index in range(target.shape[0]):
            rows.append({"model": name, "cohort": cohort, "condition": condition,
                         "patient_id": str(batch["patient_id"][index]),
                         "slice_idx": int(batch["slice_idx"][index]),
                         **metrics(prediction[index], target[index]),
                         "q": float(aux["q"].mean((1, 2))[index].item()),
                         "need": float(aux["need_mean"].mean((1, 2))[index].item()),
                         "effective_weight": float(aux["effective_weight_mean"].mean((1, 2))[index].item()),
                         "gated_rms": float(aux["gated_aux_to_target_rms"].mean((1, 2))[index].item())})
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
                          for key in (*METRICS, "q", "need", "effective_weight", "gated_rms")}})
    return output


def summary_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["cohort"], row["condition"])].append(row)
    return [{"model": model, "cohort": cohort, "condition": condition,
             "num_patients": len(values),
             **{key: float(np.mean([row[key] for row in values]))
                for key in (*METRICS, "q", "need", "effective_weight", "gated_rms")}}
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
                         **{key: 0.0 for key in ("q", "need", "effective_weight", "gated_rms")}})


def spearman(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(spearmanr(x, y).statistic)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--full_clean_manifest", required=True)
    parser.add_argument("--robustness_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    for variant in sorted(VALID_VARIANTS):
        parser.add_argument(f"--{variant}_checkpoint", required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--bootstrap_resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_paths = {"full_clean": Path(args.full_clean_manifest).resolve(),
                      "robustness": Path(args.robustness_manifest).resolve()}
    manifests = {name: json.loads(path.read_text(encoding="utf-8"))
                 for name, path in manifest_paths.items()}
    hashes = {name: sha256_file(path) for name, path in manifest_paths.items()}
    for name, manifest in manifests.items():
        if manifest.get("protocol_version") != PROTOCOL_VERSION or manifest.get("cohort") != name:
            raise RuntimeError(f"{name}: manifest protocol/cohort mismatch")

    dataset_args = argparse.Namespace(metadata_csv=str(Path(args.metadata_csv).resolve()),
                                      acceleration=8, pd_aux_acceleration=2)
    full_dataset = IndexedDataset(make_dataset(dataset_args, "val"))
    selected = {name: ManifestDataset(full_dataset, manifest)
                for name, manifest in manifests.items()}
    loaders = {name: DataLoader(dataset,
                batch_sampler=ShapeBucketBatchSampler(dataset, args.batch_size, False,
                                                       args.seed + offset),
                num_workers=args.num_workers, pin_memory=device.type == "cuda")
               for offset, (name, dataset) in enumerate(selected.items())}
    negative_sampler = HardNegativeSampler(full_dataset)
    checkpoints = {v: Path(getattr(args, f"{v}_checkpoint")).resolve()
                   for v in sorted(VALID_VARIANTS)}
    slice_rows, checkpoint_audit, configs, full_model = [], {}, {}, None
    for name, path in checkpoints.items():
        model, observed, config, selected_epoch = build_model(
            path, device, hashes["full_clean"], hashes["robustness"]
        )
        if observed != name:
            raise RuntimeError(f"Expected {name}, checkpoint contains {observed}")
        configs[name] = config
        checkpoint_audit[name] = {"path": str(path), "sha256": sha256_file(path),
                                  "training_budget_epochs": 50,
                                  "selected_checkpoint_epoch": selected_epoch,
                                  "seed": config["seed"]}
        slice_rows += evaluate_mode(model, name, loaders["full_clean"], full_dataset,
                                    negative_sampler, device, "full_clean", "correct",
                                    seed=args.seed)
        for condition in CONDITIONS[1:]:
            slice_rows += evaluate_mode(model, name, loaders["robustness"], full_dataset,
                                        negative_sampler, device, "robustness", condition,
                                        seed=args.seed)
        if name == "prnf_full":
            full_model = model
        else:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    fairness = audit_fairness(configs)
    if not fairness["passed"]:
        raise RuntimeError("Cross-checkpoint fairness failed:\n" +
                           json.dumps(fairness["mismatches"], indent=2))
    if full_model is None:
        raise RuntimeError("Full model missing")

    slice_rows += evaluate_mode(full_model, "prnf_full", loaders["robustness"],
                                full_dataset, negative_sampler, device, "robustness",
                                "correct", seed=args.seed)
    for mode, r_override, n_override, target_only in (
        ("prnf_full_r1", 1.0, None, False),
        ("prnf_full_n1", None, 1.0, False),
        ("prnf_full_r05", 0.5, None, False),
        ("prnf_full_target_only", None, None, True),
    ):
        for condition in (CONDITIONS if not target_only else ("correct",)):
            slice_rows += evaluate_mode(
                full_model, mode, loaders["robustness"], full_dataset,
                negative_sampler, device, "robustness", condition,
                reliability_override=r_override, need_override=n_override,
                target_only=target_only, seed=args.seed,
            )

    patient_rows = patient_average(slice_rows)
    append_composite(patient_rows, ("prnf_full", "m2u_augcap_mask",
                                    "prnf_no_need", "prnf_full_n1"))
    summaries, comparisons = summary_rows(patient_rows), []
    for metric in METRICS:
        comparisons.append(bootstrap_improvement(
            patient_rows, "prnf_full", "m2u_augcap_mask", "full_clean", "correct",
            metric, args.bootstrap_resamples, args.seed))
    for condition in CONDITIONS[1:]:
        for metric in METRICS:
            comparisons.append(bootstrap_improvement(
                patient_rows, "prnf_full", "m2u_augcap_mask", "robustness",
                condition, metric, args.bootstrap_resamples, args.seed))
    for comparator in ("prnf_full_r1", "prnf_full_r05", "prnf_full_n1"):
        for condition in CONDITIONS:
            comparisons.append(bootstrap_improvement(
                patient_rows, "prnf_full", comparator, "robustness", condition,
                "l1", args.bootstrap_resamples, args.seed))
    for comparator in ("m2u_clean", "m2u_augmented", "prnf_no_rel", "prnf_no_need"):
        comparisons.append(bootstrap_improvement(
            patient_rows, "prnf_full", comparator, "full_clean", "correct", "l1",
            args.bootstrap_resamples, args.seed))
        for condition in CONDITIONS[1:]:
            comparisons.append(bootstrap_improvement(
                patient_rows, "prnf_full", comparator, "robustness", condition,
                "l1", args.bootstrap_resamples, args.seed))

    clean_full = bootstrap_noninferiority(
        patient_rows, "prnf_full", "m2u_augcap_mask", "full_clean", "correct",
        0.005, args.bootstrap_resamples, args.seed)
    robust_full = bootstrap_improvement(
        patient_rows, "prnf_full", "m2u_augcap_mask", "robustness",
        "robustness_composite", "l1", args.bootstrap_resamples, args.seed)
    actual = {(r["patient_id"], r["slice_idx"]): r for r in slice_rows
              if r["model"] == "prnf_full" and r["cohort"] == "robustness"
              and r["condition"] == "correct"}
    target_only = {(r["patient_id"], r["slice_idx"]): r for r in slice_rows
                   if r["model"] == "prnf_full_target_only"}
    n1 = {(r["patient_id"], r["slice_idx"]): r for r in slice_rows
          if r["model"] == "prnf_full_n1" and r["condition"] == "correct"}
    common = sorted(set(actual) & set(target_only) & set(n1))
    needs = [actual[k]["need"] for k in common]
    potential = [target_only[k]["l1"] - n1[k]["l1"] for k in common]
    actual_benefit = [target_only[k]["l1"] - actual[k]["l1"] for k in common]
    target_error = [target_only[k]["l1"] for k in common]
    low_cut, high_cut = np.quantile(needs, (0.25, 0.75))
    low = [b for n, b in zip(needs, potential) if n <= low_cut]
    high = [b for n, b in zip(needs, potential) if n >= high_cut]
    need_analysis = {
        "num_slices": len(common),
        "potential_benefit_definition": "L1(target-only)-L1(Full,n=1)",
        "spearman_need_vs_potential_benefit": spearman(needs, potential),
        "spearman_need_vs_target_only_error": spearman(needs, target_error),
        "need_p05": float(np.quantile(needs, 0.05)),
        "need_p95": float(np.quantile(needs, 0.95)),
        "need_p95_minus_p05": float(np.quantile(needs, 0.95)-np.quantile(needs, 0.05)),
        "low_need_quartile_mean_potential_benefit": float(np.mean(low)),
        "high_need_quartile_mean_potential_benefit": float(np.mean(high)),
        "mean_actual_auxiliary_benefit": float(np.mean(actual_benefit)),
    }
    full_vs_no_need_clean = bootstrap_noninferiority(
        patient_rows, "prnf_full", "prnf_no_need", "full_clean", "correct", 0.005,
        args.bootstrap_resamples, args.seed)
    full_vs_no_need_robust = bootstrap_noninferiority(
        patient_rows, "prnf_full", "prnf_no_need", "robustness",
        "robustness_composite", 0.005, args.bootstrap_resamples, args.seed)
    actual_vs_n1 = bootstrap_improvement(
        patient_rows, "prnf_full", "prnf_full_n1", "robustness",
        "robustness_composite", "l1", args.bootstrap_resamples, args.seed)
    potential_correlation = need_analysis["spearman_need_vs_potential_benefit"]
    need_gate = {
        "full_vs_no_need_clean_noninferiority": full_vs_no_need_clean,
        "full_vs_no_need_robustness_noninferiority": full_vs_no_need_robust,
        "actual_vs_n1_robustness": actual_vs_n1,
        "actual_vs_n1_passed": actual_vs_n1["ci95_low"] > 0,
        "need_non_degenerate_passed": need_analysis["need_p95_minus_p05"] >= 0.001,
        "positive_potential_benefit_correlation_passed":
            potential_correlation is not None and potential_correlation > 0,
    }
    need_gate["passed"] = all((full_vs_no_need_clean["passed"],
                               full_vs_no_need_robust["passed"],
                               need_gate["actual_vs_n1_passed"],
                               need_gate["need_non_degenerate_passed"],
                               need_gate["positive_potential_benefit_correlation_passed"]))
    full_missing = [r for r in slice_rows if r["model"] == "prnf_full"
                    and r["cohort"] == "robustness" and r["condition"] == "missing"]
    no_need_missing = [r for r in slice_rows if r["model"] == "prnf_no_need"
                       and r["cohort"] == "robustness" and r["condition"] == "missing"]
    full_missing_zero = max(r["gated_rms"] for r in full_missing) == 0.0
    no_need_missing_zero = max(r["gated_rms"] for r in no_need_missing) == 0.0
    full_core = clean_full["passed"] and robust_full["ci95_low"] > 0 and full_missing_zero
    no_need_clean = bootstrap_noninferiority(
        patient_rows, "prnf_no_need", "m2u_augcap_mask", "full_clean", "correct",
        0.005, args.bootstrap_resamples, args.seed)
    no_need_robust = bootstrap_improvement(
        patient_rows, "prnf_no_need", "m2u_augcap_mask", "robustness",
        "robustness_composite", "l1", args.bootstrap_resamples, args.seed)
    no_need_core = no_need_clean["passed"] and no_need_robust["ci95_low"] > 0 \
        and no_need_missing_zero
    candidate = "prnf_full" if full_core and need_gate["passed"] else \
        "prnf_no_need" if no_need_core else "none"
    decision = {
        "protocol_version": PROTOCOL_VERSION, "clean_noninferiority": clean_full,
        "robustness_composite": robust_full, "missing_exact_zero_passed": full_missing_zero,
        "need_analysis": need_analysis, "need_mechanism_gate": need_gate,
        "full_core_passed": full_core,
        "fallback_no_need": {"clean_noninferiority": no_need_clean,
                             "robustness_composite": no_need_robust,
                             "missing_exact_zero_passed": no_need_missing_zero,
                             "core_passed": no_need_core},
        "recommended_candidate": candidate,
        "multi_seed_confirmation_eligible": candidate != "none",
        "clean_margin_note": "0.5% is pre-registered; justify it with multi-seed control variability before the manuscript claim.",
    }
    comparisons.extend((robust_full, actual_vs_n1, no_need_robust))
    write_csv(output_dir / "m2_prnf_per_slice.csv", slice_rows)
    write_csv(output_dir / "m2_prnf_patient_level.csv", patient_rows)
    write_csv(output_dir / "m2_prnf_summary.csv", summaries)
    write_csv(output_dir / "m2_prnf_paired_bootstrap.csv", comparisons)
    audit = {"manifests": {name: {"path": str(path), "sha256": hashes[name],
                                   "num_patients": manifests[name]["num_patients"],
                                   "num_slices": manifests[name]["num_slices"]}
                               for name, path in manifest_paths.items()},
             "checkpoint_audit": checkpoint_audit,
             "cross_checkpoint_fairness": fairness,
             "code_hashes": locked_code_hashes(), "runtime_versions": runtime_versions(),
             "bootstrap_resamples": args.bootstrap_resamples, "seed": args.seed,
             "robustness_composite_conditions": ROBUST_CONDITIONS}
    (output_dir / "m2_prnf_evaluation_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8")
    (output_dir / "m2_prnf_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
