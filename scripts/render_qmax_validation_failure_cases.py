#!/usr/bin/env python3
from __future__ import annotations

"""Render pre-declared QMax-Full failure cases from locked validation only.

Group A selects the three patients with the highest patient-mean clean L1 for
actual-q QMax-Full, then selects each patient's highest-L1 slice.

Group B ranks the remaining patients by the patient-mean difference
actual-q L1 minus the better of q=1 and constant-q L1, then selects the slice
with the largest corresponding difference.  Group B is therefore a
least-favourable reliability-response analysis, not a calibrated probability
analysis and not a model-selection experiment.

Selection uses the already frozen epoch-60 validation CSV before any image is
rendered.  The held-out test is never accessed.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import render_six_slice_qualitative as qual  # noqa: E402
from scripts.evaluate_m2_prnf_final_comparison_R8 import (  # noqa: E402
    GAIN_MODEL,
    load_baseline,
    load_fusion,
)


PROTOCOL_VERSION = "QMax-final-validation-failure-cases-v1"
FORMAL_QMAX_SHA256 = (
    "1285dd76f7900859d7ca57e68fa4f54509bed540865a7a638397e20d5012b5aa"
)
GROUP_A_MODELS = (
    "zero_filled",
    "m2u_augmented",
    "quality_protected_hybrid_gain",
    "qmax_full",
)
GROUP_B_MODELS = (
    "zero_filled",
    "qmax_full",
    "qmax_q1",
    "qmax_constant_q",
)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _reject_test_path(path: Path) -> None:
    lowered = str(path).lower()
    forbidden = ("heldout", "held_out", "test_manifest", "held-out")
    if any(token in lowered for token in forbidden):
        raise RuntimeError(f"Held-out-test path is forbidden: {path}")


def _clean_mode_lookup(
    rows: Sequence[Mapping[str, str]],
) -> Dict[str, Dict[Tuple[str, int], Dict[str, str]]]:
    required = {
        "mode",
        "cohort",
        "condition",
        "patient_id",
        "slice_idx",
        "l1",
    }
    if not rows:
        raise RuntimeError("Validation slice metrics CSV is empty")
    missing = required.difference(rows[0])
    if missing:
        raise RuntimeError(f"Validation CSV missing columns: {sorted(missing)}")

    output: Dict[str, Dict[Tuple[str, int], Dict[str, str]]] = {
        "full": {},
        "q1": {},
        "constant_q": {},
    }
    for row in rows:
        if row["cohort"] != "full_clean" or row["condition"] != "correct":
            continue
        mode = row["mode"]
        if mode not in output:
            continue
        key = (str(row["patient_id"]), int(row["slice_idx"]))
        if key in output[mode]:
            raise RuntimeError(f"Duplicate clean row for mode={mode}, case={key}")
        float(row["l1"])
        output[mode][key] = dict(row)

    key_sets = {mode: set(table) for mode, table in output.items()}
    if not key_sets["full"]:
        raise RuntimeError("No actual-q full-clean rows found")
    if not (key_sets["full"] == key_sets["q1"] == key_sets["constant_q"]):
        counts = {mode: len(keys) for mode, keys in key_sets.items()}
        raise RuntimeError(f"Clean counterfactual slice sets differ: {counts}")
    return output


def _patient_groups(keys: Iterable[Tuple[str, int]]) -> Dict[str, List[int]]:
    grouped: Dict[str, List[int]] = defaultdict(list)
    for patient, slice_idx in keys:
        grouped[patient].append(slice_idx)
    return {patient: sorted(indices) for patient, indices in grouped.items()}


def _select_cases(
    lookup: Mapping[str, Mapping[Tuple[str, int], Mapping[str, str]]],
) -> Dict[str, List[Dict[str, Any]]]:
    by_patient = _patient_groups(lookup["full"].keys())

    absolute_scores: List[Tuple[float, str]] = []
    q_response_scores: List[Tuple[float, str]] = []
    for patient, slices in by_patient.items():
        actual = np.asarray(
            [float(lookup["full"][(patient, idx)]["l1"]) for idx in slices]
        )
        q1 = np.asarray(
            [float(lookup["q1"][(patient, idx)]["l1"]) for idx in slices]
        )
        constant = np.asarray(
            [float(lookup["constant_q"][(patient, idx)]["l1"]) for idx in slices]
        )
        absolute_scores.append((float(actual.mean()), patient))
        q_response_scores.append((float((actual - np.minimum(q1, constant)).mean()), patient))

    absolute_scores.sort(key=lambda item: (-item[0], item[1]))
    group_a_patients = [patient for _score, patient in absolute_scores[:3]]
    if len(group_a_patients) != 3:
        raise RuntimeError("Fewer than three validation patients")

    q_response_scores.sort(key=lambda item: (-item[0], item[1]))
    group_b_ranked = [
        (score, patient)
        for score, patient in q_response_scores
        if patient not in set(group_a_patients)
    ]
    if len(group_b_ranked) < 3:
        raise RuntimeError("Fewer than three non-overlapping q-response patients")
    group_b_patients = [patient for _score, patient in group_b_ranked[:3]]

    absolute_score_map = dict((patient, score) for score, patient in absolute_scores)
    q_score_map = dict((patient, score) for score, patient in q_response_scores)
    group_a: List[Dict[str, Any]] = []
    group_b: List[Dict[str, Any]] = []

    for rank, patient in enumerate(group_a_patients, start=1):
        slices = by_patient[patient]
        ranked = sorted(
            (
                float(lookup["full"][(patient, idx)]["l1"]),
                idx,
            )
            for idx in slices
        )
        slice_score, slice_idx = ranked[-1]
        group_a.append(
            {
                "patient_id": patient,
                "slice_idx": int(slice_idx),
                "rank": rank,
                "patient_score": absolute_score_map[patient],
                "slice_score": slice_score,
                "selection_rule": (
                    "exploratory validation-only absolute failure: top-three patient "
                    "mean actual-q clean L1, then highest actual-q L1 slice"
                ),
            }
        )

    for rank, patient in enumerate(group_b_patients, start=1):
        slices = by_patient[patient]
        ranked: List[Tuple[float, int, str]] = []
        for idx in slices:
            actual = float(lookup["full"][(patient, idx)]["l1"])
            q1 = float(lookup["q1"][(patient, idx)]["l1"])
            constant = float(lookup["constant_q"][(patient, idx)]["l1"])
            better_mode = "q1" if q1 <= constant else "constant_q"
            ranked.append((actual - min(q1, constant), idx, better_mode))
        slice_score, slice_idx, better_mode = sorted(ranked)[-1]
        group_b.append(
            {
                "patient_id": patient,
                "slice_idx": int(slice_idx),
                "rank": rank,
                "patient_score": q_score_map[patient],
                "slice_score": slice_score,
                "better_counterfactual_on_selected_slice": better_mode,
                "selection_rule": (
                    "exploratory validation-only least-favourable q response: among "
                    "patients not selected in group A, top-three patient mean "
                    "actual-q L1 minus better(q=1, constant-q), then largest-delta slice"
                ),
            }
        )
    return {"group_a": group_a, "group_b": group_b}


@torch.no_grad()
def _predict_qmax(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    device: torch.device,
    reliability_override: float | None,
) -> Dict[str, Any]:
    kspace, mask, pd, target, _indices = qual.prepare_batch(batch, device)
    available = torch.ones(kspace.shape[0], device=device)
    kwargs: Dict[str, Any] = {}
    if reliability_override is not None:
        kwargs["reliability_override"] = float(reliability_override)
    prediction, auxiliary = model(
        kspace, mask, pd, available, return_aux=True, **kwargs
    )
    if reliability_override is None:
        q_value = float(auxiliary["q_hat"].detach().float().mean().item())
    else:
        q_value = float(reliability_override)
    return qual._metric_record(prediction, target, q=q_value)


def _case_stem(group: str, index: int, case: Mapping[str, Any]) -> str:
    return (
        f"{group}{index}_{str(case['patient_id'])[:10]}_"
        f"slice{int(case['slice_idx']):03d}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--full_clean_manifest", required=True)
    parser.add_argument("--robustness_manifest", required=True)
    parser.add_argument("--condition_manifest", required=True)
    parser.add_argument("--validation_slice_metrics", required=True)
    parser.add_argument("--validation_audit", required=True)
    parser.add_argument("--m2u_augmented_checkpoint", required=True)
    parser.add_argument("--quality_gain_checkpoint", required=True)
    parser.add_argument("--qmax_full_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = {
        key: Path(value).resolve()
        for key, value in {
            "metadata_csv": args.metadata_csv,
            "full_clean_manifest": args.full_clean_manifest,
            "robustness_manifest": args.robustness_manifest,
            "condition_manifest": args.condition_manifest,
            "validation_slice_metrics": args.validation_slice_metrics,
            "validation_audit": args.validation_audit,
            "m2u_augmented_checkpoint": args.m2u_augmented_checkpoint,
            "quality_gain_checkpoint": args.quality_gain_checkpoint,
            "qmax_full_checkpoint": args.qmax_full_checkpoint,
        }.items()
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
        _reject_test_path(path)
    for label in ("m2u_augmented_checkpoint", "quality_gain_checkpoint"):
        if paths[label].name != "model_best.pt":
            raise RuntimeError(f"Frozen historical comparison requires model_best.pt: {paths[label]}")

    validation_audit = qual.read_json(paths["validation_audit"])
    if validation_audit.get("status") != "passed":
        raise RuntimeError("Locked validation audit did not pass")
    if validation_audit.get("scope") != "locked validation only; held-out test not accessed":
        raise RuntimeError("Unexpected validation scope")
    qmax_hash = qual.sha256_file(paths["qmax_full_checkpoint"])
    if qmax_hash != FORMAL_QMAX_SHA256:
        raise RuntimeError(f"Unexpected formal QMax checkpoint SHA-256: {qmax_hash}")
    if validation_audit["checkpoint_audit"]["checkpoint_sha256"] != qmax_hash:
        raise RuntimeError("Validation CSV audit is not bound to the requested QMax checkpoint")

    output_dir = Path(args.output_dir).resolve()
    if (output_dir / "qmax_validation_failure_case_audit.json").exists():
        raise RuntimeError(f"Refusing to overwrite completed output: {output_dir}")
    slice_dir = output_dir / "slice_level_figures"
    group_dir = output_dir / "group_contact_sheets"
    slice_dir.mkdir(parents=True, exist_ok=True)
    group_dir.mkdir(parents=True, exist_ok=True)

    qual.set_seed(args.seed)
    qual.configure_matplotlib()
    qual.MODEL_LABELS.update(
        {
            "qmax_q1": "QMax-Full, q=1",
            "qmax_constant_q": "QMax-Full, constant-q",
        }
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clean_lookup = _clean_mode_lookup(_read_csv(paths["validation_slice_metrics"]))
    selections = _select_cases(clean_lookup)
    (output_dir / "selected_failure_cases.json").write_text(
        json.dumps(selections, indent=2), encoding="utf-8"
    )
    all_cases = list(selections["group_a"]) + list(selections["group_b"])

    full_dataset = qual.IndexedDataset(
        qual.make_dataset(
            metadata_csv=str(paths["metadata_csv"]),
            split="val",
            acceleration=8,
            pd_aux_acceleration=2,
        )
    )
    clean_dataset = qual.ManifestDataset(
        full_dataset, qual.read_json(paths["full_clean_manifest"])
    )
    dataset_lookup = qual._record_lookup(clean_dataset)
    for case in all_cases:
        key = (str(case["patient_id"]), int(case["slice_idx"]))
        if key not in dataset_lookup:
            raise RuntimeError(f"Selected case absent from locked manifest: {key}")

    results: Dict[str, Dict[str, Dict[str, Any]]] = {
        qual.case_key(case): {} for case in all_cases
    }
    for case in all_cases:
        batch = qual.get_batch(clean_dataset, dataset_lookup, case)
        reference, zf = qual.predict_reference_and_zf(batch, device)
        results[qual.case_key(case)]["reference"] = reference
        results[qual.case_key(case)]["zero_filled"] = zf

    full_clean_hash = qual.sha256_file(paths["full_clean_manifest"])
    robustness_hash = qual.sha256_file(paths["robustness_manifest"])
    condition_hash = qual.sha256_file(paths["condition_manifest"])
    metadata_hash = qual.sha256_file(paths["metadata_csv"])

    model, _config, _summary, _epoch = load_baseline(
        paths["m2u_augmented_checkpoint"],
        "m2u_augmented",
        device,
        full_clean_hash,
        robustness_hash,
        metadata_hash,
    )
    for case in selections["group_a"]:
        results[qual.case_key(case)]["m2u_augmented"] = qual.predict_aux_model(
            model, qual.get_batch(clean_dataset, dataset_lookup, case), device, q_key=None
        )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model, _config, _summary, _epoch = load_fusion(
        paths["quality_gain_checkpoint"],
        GAIN_MODEL,
        device,
        full_clean_hash,
        robustness_hash,
        condition_hash,
        metadata_hash,
    )
    for case in selections["group_a"]:
        results[qual.case_key(case)]["quality_protected_hybrid_gain"] = qual.predict_aux_model(
            model, qual.get_batch(clean_dataset, dataset_lookup, case), device, q_key="q"
        )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    qmax = qual.load_qmax_full(paths["qmax_full_checkpoint"], device)
    constant_q = float(validation_audit["constant_q"])
    for case in all_cases:
        batch = qual.get_batch(clean_dataset, dataset_lookup, case)
        results[qual.case_key(case)]["qmax_full"] = _predict_qmax(
            qmax, batch, device, reliability_override=None
        )
    for case in selections["group_b"]:
        batch = qual.get_batch(clean_dataset, dataset_lookup, case)
        results[qual.case_key(case)]["qmax_q1"] = _predict_qmax(
            qmax, batch, device, reliability_override=1.0
        )
        results[qual.case_key(case)]["qmax_constant_q"] = _predict_qmax(
            qmax, batch, device, reliability_override=constant_q
        )
    del qmax
    if device.type == "cuda":
        torch.cuda.empty_cache()

    metric_rows: List[Dict[str, Any]] = []
    arrays: Dict[str, np.ndarray] = {}
    for group, cases, model_order in (
        ("A", selections["group_a"], GROUP_A_MODELS),
        ("B", selections["group_b"], GROUP_B_MODELS),
    ):
        for index, case in enumerate(cases, start=1):
            key = qual.case_key(case)
            case_results = results[key]
            stem = _case_stem(group, index, case)
            qual.render_slice_reconstruction(
                case, case_results, model_order, slice_dir / f"{stem}_reconstruction"
            )
            qual.render_slice_error(
                case, case_results, model_order, slice_dir / f"{stem}_absolute_error"
            )
            for model_name in ("reference", *model_order):
                record = case_results[model_name]
                arrays[f"{stem}__{model_name}"] = np.asarray(
                    record["prediction"], dtype=np.float32
                )
                metric_rows.append(
                    {
                        "group": group,
                        "failure_type": (
                            "absolute_clean_error" if group == "A" else "least_favourable_q_response"
                        ),
                        "rank": case["rank"],
                        "patient_id": case["patient_id"],
                        "slice_idx": int(case["slice_idx"]),
                        "patient_selection_score": case["patient_score"],
                        "slice_selection_score": case["slice_score"],
                        "selection_rule": case["selection_rule"],
                        "model": model_name,
                        "l1": record["l1"],
                        "nmse": record["nmse"],
                        "psnr": record["psnr"],
                        "ssim": record["ssim"],
                        "q": record["q"],
                    }
                )

        qual.render_group_reconstruction(
            group,
            cases,
            results,
            model_order,
            group_dir / f"Group_{group}_three_slice_reconstruction_plate",
        )
        qual.render_group_error(
            group,
            cases,
            results,
            model_order,
            group_dir / f"Group_{group}_three_slice_absolute_error_plate",
        )

    qual.write_csv(output_dir / "failure_case_panel_metrics.csv", metric_rows)
    np.savez_compressed(output_dir / "failure_case_source_arrays.npz", **arrays)
    audit = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "passed",
        "scope": "locked validation only; held-out test not accessed",
        "scientific_role": "exploratory failure-case analysis; no model or epoch selection",
        "backend": "Python/matplotlib",
        "device": str(device),
        "selection_performed_before_rendering": True,
        "selection_groups": {
            "A": "highest patient-mean actual-q clean L1; worst slice per patient",
            "B": (
                "excluding group A, highest patient-mean actual-q L1 minus "
                "better(q=1, constant-q); largest-delta slice per patient"
            ),
        },
        "selected_cases": selections,
        "constant_q": constant_q,
        "num_distinct_patients": len({case["patient_id"] for case in all_cases}),
        "num_distinct_slices": len(all_cases),
        "image_integrity": {
            "crop": "same evaluator centre crop for every model",
            "reconstruction_window": "target-derived 99.5th percentile, shared within slice",
            "error_scale": "99.5th percentile across all compared outputs, shared within slice",
            "normalisation": "each slice divided by its target maximum",
            "smoothing": "none",
            "interpolation": "nearest",
            "local_adjustment": "none",
        },
        "input_hashes": {key: qual.sha256_file(path) for key, path in paths.items()},
        "code_hashes": {
            "failure_case_renderer": qual.sha256_file(Path(__file__).resolve()),
            "qualitative_renderer": qual.sha256_file(Path(qual.__file__).resolve()),
        },
        "checkpoint_sha256": {
            "m2u_augmented": qual.sha256_file(paths["m2u_augmented_checkpoint"]),
            "fifth_arm": qual.sha256_file(paths["quality_gain_checkpoint"]),
            "qmax_full": qmax_hash,
        },
        "source_data": {
            "selection": "selected_failure_cases.json",
            "metrics": "failure_case_panel_metrics.csv",
            "arrays": "failure_case_source_arrays.npz",
        },
    }
    (output_dir / "qmax_validation_failure_case_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
