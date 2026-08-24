#!/usr/bin/env python3
from __future__ import annotations

"""Patient-level QMax component and reliability counterfactual evaluation."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.qmax_common import (  # noqa: E402
    IndexedDataset,
    ShapeBucketBatchSampler,
    autocast_context,
    code_hashes,
    make_dataset,
    prepare_batch,
    set_seed,
    sha256_file,
    slice_metrics,
    write_csv,
)
from src.fft_utils import center_crop  # noqa: E402
from src.m2_prnf_corruptions import (  # noqa: E402
    load_pd_auxiliary,
    translate_nonwrapping,
)
from src.m2_prnf_qmax_varnet import (  # noqa: E402
    QMAX_SCALE_NAMES,
    QMaxAuxPDVarNet,
)


CONDITIONS = (
    "correct",
    "shift8",
    "wrong_slice",
    "wrong_patient",
    "missing",
)
MANIFEST_PROTOCOL_VERSION = "M2-PRNF-R8-v1.3-bs4-audited"
CONDITION_MANIFEST_PROTOCOL_VERSION = (
    "M2-PRNF-R8-v1.4.1-fusion-pilot-audited"
)
ROBUST_CONDITIONS = ("shift8", "wrong_slice", "wrong_patient")
METRICS = ("l1", "nmse", "psnr", "ssim")
DIAGNOSTICS = (
    "q",
    "direct_rms",
    "detail_gate",
    "alignment_rms",
    "correction_rms",
    "final_auxiliary_rms",
    "cos_direct_correction",
    "dc_raw_rms",
    "missing_direct_exact_zero",
    "missing_correction_exact_zero",
)


class ManifestDataset:
    def __init__(self, source: IndexedDataset, manifest: Mapping[str, Any]):
        self.source = source
        wanted = {
            (str(patient["patient_id"]), int(slice_index))
            for patient in manifest["patients"]
            for slice_index in patient["slice_indices"]
        }
        self.indices = [
            index
            for index, record in enumerate(source.records)
            if (
                str(record["patient_id"]),
                int(record["slice_idx"]),
            )
            in wanted
        ]
        observed = {
            (
                str(source.records[index]["patient_id"]),
                int(source.records[index]["slice_idx"]),
            )
            for index in self.indices
        }
        if observed != wanted:
            raise RuntimeError(
                f"Manifest entries not found: {sorted(wanted-observed)[:8]}"
            )
        self.records = [source.records[index] for index in self.indices]
        self.patient_rows = source.patient_rows

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        source_index = self.indices[index]
        item = self.source[source_index]
        item["sample_idx"] = int(source_index)
        return item


def condition_pd(
    pd: torch.Tensor,
    indices: Sequence[int],
    dataset: IndexedDataset,
    condition_lookup: Mapping[int, Mapping[str, Any]],
    condition: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if condition == "correct":
        return pd, torch.ones(pd.shape[0], device=pd.device)
    if condition == "missing":
        return torch.zeros_like(pd), torch.zeros(
            pd.shape[0], device=pd.device
        )
    output = []
    for position, source_value in enumerate(indices):
        source = int(source_value)
        if source not in condition_lookup:
            raise RuntimeError(
                f"Source {source} missing from condition manifest"
            )
        frozen = condition_lookup[source]
        source_record = dataset.records[source]
        if (
            str(source_record["patient_id"]) != str(frozen["patient_id"])
            or int(source_record["slice_idx"]) != int(frozen["slice_idx"])
        ):
            raise RuntimeError("Frozen source identity drift")
        if condition == "shift8":
            shift = frozen["shift8"]
            image = translate_nonwrapping(
                pd[position],
                int(shift["dy"]),
                int(shift["dx"]),
                str(shift["padding_mode"]),
            )
        elif condition in {"wrong_slice", "wrong_patient"}:
            replacement = frozen[condition]
            replacement_index = int(replacement["replacement_index"])
            record = dataset.records[replacement_index]
            if (
                str(record["patient_id"])
                != str(replacement["replacement_patient_id"])
                or int(record["slice_idx"])
                != int(replacement["replacement_slice_idx"])
            ):
                raise RuntimeError("Frozen replacement identity drift")
            image = load_pd_auxiliary(
                dataset, replacement_index, pd.device
            )
        else:
            raise ValueError(condition)
        if image.shape != pd[position].shape:
            raise RuntimeError("Conditioned PD shape mismatch")
        output.append(image)
    return torch.stack(output), torch.ones(
        pd.shape[0], device=pd.device
    )


def mode_kwargs(
    mode: str,
    constant_q: Optional[float],
) -> Dict[str, Any]:
    if mode == "full":
        return {}
    if mode == "detail_neutral":
        return {"detail_neutral": True}
    if mode == "alignment_off":
        return {"alignment_off": True}
    if mode == "correction_off":
        return {"correction_off": True}
    if mode == "dc_zero":
        return {"dc_zero": True}
    if mode == "q1":
        return {"reliability_override": 1.0}
    if mode == "constant_q":
        if constant_q is None:
            raise RuntimeError("constant_q has not been estimated")
        return {"reliability_override": float(constant_q)}
    raise ValueError(mode)


@torch.no_grad()
def evaluate_mode(
    *,
    model,
    loader,
    source_dataset,
    condition_lookup,
    device,
    amp,
    cohort,
    condition,
    mode,
    constant_q,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    slice_rows: List[Dict[str, Any]] = []
    scale_sums: Dict[tuple, float] = defaultdict(float)
    scale_counts: Dict[tuple, int] = defaultdict(int)
    forward_kwargs = mode_kwargs(mode, constant_q)
    for batch in loader:
        kspace, mask, pd, target, indices = prepare_batch(batch, device)
        pd_used, available = condition_pd(
            pd,
            indices,
            source_dataset,
            condition_lookup,
            condition,
        )
        with autocast_context(device, amp):
            prediction, auxiliary = model(
                kspace,
                mask,
                pd_used,
                available,
                return_aux=True,
                **forward_kwargs,
            )
        prediction = center_crop(
            prediction.float(), target.shape[-2], target.shape[-1]
        )
        for index in range(target.shape[0]):
            direct = auxiliary["direct_to_target_rms"][index]
            correction = auxiliary["correction_to_target_rms"][index]
            slice_rows.append(
                {
                    "mode": mode,
                    "cohort": cohort,
                    "condition": condition,
                    "patient_id": str(batch["patient_id"][index]),
                    "slice_idx": int(batch["slice_idx"][index]),
                    **slice_metrics(prediction[index], target[index]),
                    "q": float(
                        auxiliary["q_hat"][index].mean().item()
                    ),
                    "direct_rms": float(direct.mean().item()),
                    "detail_gate": float(
                        auxiliary["detail_gate_mean"][index].mean().item()
                    ),
                    "alignment_rms": float(
                        auxiliary["alignment_to_target_rms"][index]
                        .mean()
                        .item()
                    ),
                    "correction_rms": float(correction.mean().item()),
                    "final_auxiliary_rms": float(
                        auxiliary["final_auxiliary_to_target_rms"][index]
                        .mean()
                        .item()
                    ),
                    "cos_direct_correction": float(
                        auxiliary["cos_direct_correction"][index]
                        .mean()
                        .item()
                    ),
                    "dc_raw_rms": float(
                        auxiliary["dc_raw_rms"][index].mean().item()
                    ),
                    "missing_direct_exact_zero": float(
                        bool((direct == 0).all())
                    ),
                    "missing_correction_exact_zero": float(
                        bool((correction == 0).all())
                    ),
                }
            )
        for metric in (
            "q_hat",
            "alpha",
            "direct_to_target_rms",
            "detail_gate_mean",
            "detail_gate_std",
            "detail_gate_min",
            "detail_gate_max",
            "alignment_to_target_rms",
            "correction_to_target_rms",
            "final_auxiliary_to_target_rms",
            "cos_direct_correction",
            "dc_raw_rms",
            "dc_normalized_rms",
        ):
            tensor = auxiliary[metric].detach().float().cpu()
            for cascade in range(tensor.shape[1]):
                for scale in range(tensor.shape[2]):
                    key = (cascade, scale, metric)
                    values = tensor[:, cascade, scale]
                    scale_sums[key] += float(values.sum().item())
                    scale_counts[key] += int(values.numel())
    scale_rows = []
    for cascade in range(model.cascades.__len__()):
        for scale, scale_name in enumerate(QMAX_SCALE_NAMES):
            row = {
                "mode": mode,
                "cohort": cohort,
                "condition": condition,
                "cascade": cascade,
                "scale": scale,
                "scale_name": scale_name,
            }
            for metric in (
                "q_hat",
                "alpha",
                "direct_to_target_rms",
                "detail_gate_mean",
                "detail_gate_std",
                "detail_gate_min",
                "detail_gate_max",
                "alignment_to_target_rms",
                "correction_to_target_rms",
                "final_auxiliary_to_target_rms",
                "cos_direct_correction",
                "dc_raw_rms",
                "dc_normalized_rms",
            ):
                key = (cascade, scale, metric)
                row[metric] = (
                    scale_sums[key] / scale_counts[key]
                    if scale_counts[key]
                    else float("nan")
                )
            scale_rows.append(row)
    return slice_rows, scale_rows


def patient_rows(slice_rows: Sequence[Mapping[str, Any]]):
    groups: Dict[tuple, List[Mapping[str, Any]]] = defaultdict(list)
    for row in slice_rows:
        groups[
            (
                row["mode"],
                row["cohort"],
                row["condition"],
                row["patient_id"],
            )
        ].append(row)
    output = []
    for key, values in sorted(groups.items()):
        mode, cohort, condition, patient_id = key
        output.append(
            {
                "mode": mode,
                "cohort": cohort,
                "condition": condition,
                "patient_id": patient_id,
                "num_slices": len(values),
                **{
                    metric: float(
                        np.mean([float(row[metric]) for row in values])
                    )
                    for metric in (*METRICS, *DIAGNOSTICS)
                },
            }
        )
    return output


def add_robustness_composite(rows: List[Dict[str, Any]]) -> None:
    lookup = {
        (
            row["mode"],
            row["cohort"],
            row["condition"],
            row["patient_id"],
        ): row
        for row in rows
    }
    modes = sorted({row["mode"] for row in rows})
    patients = sorted(
        {
            row["patient_id"]
            for row in rows
            if row["cohort"] == "robustness"
        }
    )
    additions = []
    for mode in modes:
        for patient in patients:
            keys = [
                (mode, "robustness", condition, patient)
                for condition in ROBUST_CONDITIONS
            ]
            if not all(key in lookup for key in keys):
                missing = [
                    condition
                    for condition, key in zip(ROBUST_CONDITIONS, keys)
                    if key not in lookup
                ]
                raise RuntimeError(
                    "Incomplete robustness composite for "
                    f"mode={mode}, patient={patient}; "
                    f"missing_conditions={missing}"
                )
            additions.append(
                {
                    "mode": mode,
                    "cohort": "robustness",
                    "condition": "robustness_composite",
                    "patient_id": patient,
                    "num_slices": int(
                        np.mean([lookup[key]["num_slices"] for key in keys])
                    ),
                    **{
                        metric: float(
                            np.mean([lookup[key][metric] for key in keys])
                        )
                        for metric in (*METRICS, *DIAGNOSTICS)
                    },
                }
            )
    rows.extend(additions)


def summaries(rows: Sequence[Mapping[str, Any]]):
    groups: Dict[tuple, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["mode"], row["cohort"], row["condition"])].append(row)
    output = []
    for (mode, cohort, condition), values in sorted(groups.items()):
        output.append(
            {
                "mode": mode,
                "cohort": cohort,
                "condition": condition,
                "num_patients": len(values),
                **{
                    metric: float(
                        np.mean([float(row[metric]) for row in values])
                    )
                    for metric in (*METRICS, *DIAGNOSTICS)
                },
            }
        )
    return output


def paired_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    candidate: str,
    reference: str,
    cohort: str,
    condition: str,
    metric: str,
    resamples: int,
    seed: int,
) -> Dict[str, Any]:
    def select(mode: str) -> Dict[str, float]:
        selected: Dict[str, float] = {}
        for row in rows:
            if (
                row["mode"] != mode
                or row["cohort"] != cohort
                or row["condition"] != condition
            ):
                continue
            patient = str(row["patient_id"])
            if patient in selected:
                raise RuntimeError(
                    f"Duplicate patient row for {mode} "
                    f"{cohort}/{condition}/{metric}: {patient}"
                )
            selected[patient] = float(row[metric])
        return selected

    candidate_values = select(candidate)
    reference_values = select(reference)
    if not candidate_values or not reference_values:
        raise RuntimeError(
            f"No paired patients for {candidate}/{reference}"
        )
    if set(candidate_values) != set(reference_values):
        candidate_only = sorted(
            set(candidate_values) - set(reference_values)
        )
        reference_only = sorted(
            set(reference_values) - set(candidate_values)
        )
        raise RuntimeError(
            f"Patient sets differ for {candidate}/{reference} "
            f"{cohort}/{condition}/{metric}; "
            f"candidate_only={candidate_only[:8]}, "
            f"reference_only={reference_only[:8]}"
        )
    patients = sorted(candidate_values)
    differences = np.asarray(
        [
            candidate_values[patient] - reference_values[patient]
            for patient in patients
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    draws = rng.integers(
        0, len(differences), size=(int(resamples), len(differences))
    )
    bootstrap = differences[draws].mean(axis=1)
    return {
        "candidate": candidate,
        "reference": reference,
        "cohort": cohort,
        "condition": condition,
        "metric": metric,
        "delta_candidate_minus_reference": float(differences.mean()),
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
        "candidate_better_patients": int((differences < 0).sum())
        if metric in {"l1", "nmse"}
        else int((differences > 0).sum()),
        "num_patients": len(patients),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--full_clean_manifest", required=True)
    parser.add_argument("--robustness_manifest", required=True)
    parser.add_argument("--condition_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--bootstrap_resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Counterfactual evaluation requires CUDA")
    device = torch.device("cuda")
    set_seed(args.seed)
    paths = {}
    for name in (
        "checkpoint",
        "metadata_csv",
        "full_clean_manifest",
        "robustness_manifest",
        "condition_manifest",
    ):
        path = Path(getattr(args, name)).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        paths[name] = path
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(
        paths["checkpoint"], map_location=device, weights_only=False
    )
    config = checkpoint["config"]
    variant = str(config["qmax_variant"])
    model = QMaxAuxPDVarNet(
        qmax_variant=variant, **dict(config["model_kwargs"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    if config["metadata_sha256"] != sha256_file(paths["metadata_csv"]):
        raise RuntimeError("Metadata hash mismatch")
    for key in (
        "full_clean_manifest",
        "robustness_manifest",
        "condition_manifest",
    ):
        expected = config[f"{key}_sha256"]
        observed = sha256_file(paths[key])
        if expected != observed:
            raise RuntimeError(f"{key} hash mismatch")
    if config["code_hashes"] != code_hashes(PROJECT_ROOT):
        raise RuntimeError("Installed QMax code hashes differ from checkpoint")

    source = IndexedDataset(
        make_dataset(
            str(paths["metadata_csv"]),
            "val",
            acceleration=8,
            pd_aux_acceleration=2,
        )
    )
    full_clean_manifest = json.loads(
        paths["full_clean_manifest"].read_text(encoding="utf-8")
    )
    robustness_manifest = json.loads(
        paths["robustness_manifest"].read_text(encoding="utf-8")
    )
    for cohort, manifest in (
        ("full_clean", full_clean_manifest),
        ("robustness", robustness_manifest),
    ):
        if (
            manifest.get("protocol_version") != MANIFEST_PROTOCOL_VERSION
            or manifest.get("cohort") != cohort
        ):
            raise RuntimeError(
                f"{cohort} manifest protocol/cohort mismatch"
            )
    clean_dataset = ManifestDataset(source, full_clean_manifest)
    robust_dataset = ManifestDataset(source, robustness_manifest)
    clean_loader = DataLoader(
        clean_dataset,
        batch_sampler=ShapeBucketBatchSampler(
            clean_dataset, args.batch_size, False, args.seed
        ),
        num_workers=args.num_workers,
        pin_memory=True,
    )
    robust_loader = DataLoader(
        robust_dataset,
        batch_sampler=ShapeBucketBatchSampler(
            robust_dataset, args.batch_size, False, args.seed
        ),
        num_workers=args.num_workers,
        pin_memory=True,
    )
    condition_manifest = json.loads(
        paths["condition_manifest"].read_text(encoding="utf-8")
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

    # Estimate constant q exactly as previously fixed: patient-equal mean
    # actual q on the locked full-clean cohort.
    actual_clean, actual_clean_scale = evaluate_mode(
        model=model,
        loader=clean_loader,
        source_dataset=source,
        condition_lookup=condition_lookup,
        device=device,
        amp=args.amp,
        cohort="full_clean",
        condition="correct",
        mode="full",
        constant_q=None,
    )
    q_by_patient: Dict[str, List[float]] = defaultdict(list)
    for row in actual_clean:
        q_by_patient[str(row["patient_id"])].append(float(row["q"]))
    constant_q = float(
        np.mean(
            [
                float(np.mean(values))
                for values in q_by_patient.values()
            ]
        )
    )

    modes = [
        "full",
        "detail_neutral",
        "alignment_off",
        "correction_off",
        "q1",
        "constant_q",
    ]
    if variant == "qmax_full":
        modes.append("dc_zero")
    all_slice_rows = list(actual_clean)
    all_scale_rows = list(actual_clean_scale)
    for mode in modes:
        if mode != "full":
            rows, scale_rows = evaluate_mode(
                model=model,
                loader=clean_loader,
                source_dataset=source,
                condition_lookup=condition_lookup,
                device=device,
                amp=args.amp,
                cohort="full_clean",
                condition="correct",
                mode=mode,
                constant_q=constant_q,
            )
            all_slice_rows.extend(rows)
            all_scale_rows.extend(scale_rows)
        for condition in CONDITIONS:
            rows, scale_rows = evaluate_mode(
                model=model,
                loader=robust_loader,
                source_dataset=source,
                condition_lookup=condition_lookup,
                device=device,
                amp=args.amp,
                cohort="robustness",
                condition=condition,
                mode=mode,
                constant_q=constant_q,
            )
            all_slice_rows.extend(rows)
            all_scale_rows.extend(scale_rows)

    patient_level = patient_rows(all_slice_rows)
    add_robustness_composite(patient_level)
    summary = summaries(patient_level)
    comparisons = []
    for mode in modes:
        if mode == "full":
            continue
        for cohort, condition in (
            ("full_clean", "correct"),
            ("robustness", "robustness_composite"),
        ):
            for metric in METRICS:
                comparisons.append(
                    paired_bootstrap(
                        patient_level,
                        mode,
                        "full",
                        cohort,
                        condition,
                        metric,
                        args.bootstrap_resamples,
                        args.seed,
                    )
                )

    write_csv(output_dir / "qmax_slice_metrics.csv", all_slice_rows)
    write_csv(output_dir / "qmax_patient_metrics.csv", patient_level)
    write_csv(output_dir / "qmax_summary.csv", summary)
    write_csv(output_dir / "qmax_scale_cascade_diagnostics.csv", all_scale_rows)
    write_csv(output_dir / "qmax_counterfactual_bootstrap.csv", comparisons)
    audit = {
        "protocol_version": "QMax-counterfactual-evaluation-v3",
        "checkpoint": str(paths["checkpoint"]),
        "checkpoint_sha256": sha256_file(paths["checkpoint"]),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "qmax_variant": variant,
        "constant_q_definition": (
            "patient-equal mean actual q on the locked full-clean cohort"
        ),
        "constant_q": constant_q,
        "constant_q_num_patients": len(q_by_patient),
        "modes": modes,
        "dc_on_mode": (
            "full"
            if variant == "qmax_full"
            else None
        ),
        "conditions": list(CONDITIONS),
        "robustness_composite_conditions": list(ROBUST_CONDITIONS),
        "bootstrap_resamples": args.bootstrap_resamples,
        "seed": args.seed,
        "missing_exact_zero": all(
            row["missing_direct_exact_zero"] == 1.0
            and row["missing_correction_exact_zero"] == 1.0
            for row in all_slice_rows
            if row["condition"] == "missing"
        ),
        "input_hashes": {
            key: sha256_file(value)
            for key, value in paths.items()
        },
    }
    (output_dir / "qmax_counterfactual_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2), flush=True)
    if not audit["missing_exact_zero"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
