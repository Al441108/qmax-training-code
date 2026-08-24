#!/usr/bin/env python3
from __future__ import annotations

"""Patient-paired epoch30/40/50/60 trajectory for StageA-Full.

Only locked validation manifests are used.  The formal endpoint remains
epoch60/model_last.pt; earlier checkpoints are trajectory measurements only.
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_qmax_counterfactuals import (  # noqa: E402
    CONDITION_MANIFEST_PROTOCOL_VERSION,
    CONDITIONS,
    MANIFEST_PROTOCOL_VERSION,
    METRICS,
    ROBUST_CONDITIONS,
    ManifestDataset,
    add_robustness_composite,
    evaluate_mode,
    paired_bootstrap,
    patient_rows,
    summaries,
)
from scripts.evaluate_stagea_full_epoch60_validation import (  # noqa: E402
    _assert_finite_metrics,
    _load_json,
    _validate_input_hashes,
)
from scripts.qmax_common import (  # noqa: E402
    IndexedDataset,
    ShapeBucketBatchSampler,
    code_hashes,
    make_dataset,
    set_seed,
    sha256_file,
    write_csv,
)
from src.m2_prnf_qmax_varnet import QMaxAuxPDVarNet  # noqa: E402


PROTOCOL_VERSION = "StageA-Full-LR-trajectory-30-40-50-60-v1"
EPOCHS = (30, 40, 50, 60)


def _checkpoint_path(args: argparse.Namespace, epoch: int) -> Path:
    return Path(getattr(args, f"checkpoint{epoch}")).resolve()


def _validate_checkpoint(
    path: Path,
    epoch: int,
    checkpoint: Mapping[str, Any],
    installed_hashes: Mapping[str, str],
) -> Dict[str, Any]:
    if path.name != "model_last.pt" or path.parent.name != f"epoch{epoch}":
        raise RuntimeError(
            f"Epoch {epoch} must use epoch{epoch}/model_last.pt: {path}"
        )
    required = {
        "epoch",
        "config",
        "model_state_dict",
        "optimizer_state_dict",
        "grad_scaler_state_dict",
        "rng_state",
        "history",
        "code_hashes",
        "run_corruption_audit",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise RuntimeError(f"Epoch {epoch} checkpoint missing keys: {missing}")
    if int(checkpoint["epoch"]) != epoch:
        raise RuntimeError(
            f"Checkpoint/path epoch mismatch: {checkpoint['epoch']} vs {epoch}"
        )
    config = checkpoint["config"]
    if str(config.get("qmax_variant")) != "qmax_full":
        raise RuntimeError(f"Epoch {epoch} is not qmax_full")
    history_epochs = [int(row["epoch"]) for row in checkpoint["history"]]
    if history_epochs != list(range(1, epoch + 1)):
        raise RuntimeError(f"Epoch {epoch} history is not exactly 1..{epoch}")
    if checkpoint.get("code_hashes") != installed_hashes:
        raise RuntimeError(f"Epoch {epoch} top-level code hashes drifted")
    if config.get("code_hashes") != installed_hashes:
        raise RuntimeError(f"Epoch {epoch} config code hashes drifted")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "epoch": epoch,
        "history_exact": True,
        "optimizer_state_present": True,
        "grad_scaler_state_present": True,
        "rng_state_present": True,
        "corruption_state_present": True,
        "init_template_sha256": config.get("init_template_sha256"),
    }


def _relabel(
    rows: Iterable[Dict[str, Any]],
    epoch: int,
    intervention: str,
    checkpoint_hash: str,
) -> None:
    mode = f"epoch{epoch}_{intervention}"
    for row in rows:
        row.update(
            {
                "mode": mode,
                "trajectory_epoch": epoch,
                "intervention": intervention,
                "checkpoint_sha256": checkpoint_hash,
                "formal_selection": bool(epoch == 60),
            }
        )


def _mean_metric(
    rows: list[dict[str, Any]],
    mode: str,
    cohort: str,
    condition: str,
    metric: str,
) -> float:
    values = [
        float(row[metric])
        for row in rows
        if row["mode"] == mode
        and row["cohort"] == cohort
        and row["condition"] == condition
    ]
    if not values:
        raise RuntimeError(f"No rows for {mode}/{cohort}/{condition}/{metric}")
    return float(np.mean(values))


def _add_means_and_relative(
    result: Dict[str, Any], rows: list[dict[str, Any]]
) -> None:
    candidate_mean = _mean_metric(
        rows,
        str(result["candidate"]),
        str(result["cohort"]),
        str(result["condition"]),
        str(result["metric"]),
    )
    reference_mean = _mean_metric(
        rows,
        str(result["reference"]),
        str(result["cohort"]),
        str(result["condition"]),
        str(result["metric"]),
    )
    result["candidate_mean"] = candidate_mean
    result["reference_mean"] = reference_mean
    result["relative_candidate_minus_reference_percent"] = float(
        (candidate_mean / reference_mean - 1.0) * 100.0
    )


def _q_separation(
    rows: list[dict[str, Any]], mode: str, resamples: int, seed: int
) -> Dict[str, Any]:
    clean: Dict[str, float] = {}
    corrupt: Dict[str, float] = {}
    for row in rows:
        if row["mode"] != mode or row["cohort"] != "robustness":
            continue
        patient = str(row["patient_id"])
        if row["condition"] == "correct":
            clean[patient] = float(row["q"])
        elif row["condition"] == "robustness_composite":
            corrupt[patient] = float(row["q"])
    if not clean or set(clean) != set(corrupt):
        raise RuntimeError(f"Incomplete q separation pairing for {mode}")
    patients = sorted(clean)
    differences = np.asarray(
        [clean[p] - corrupt[p] for p in patients], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    draws = rng.integers(
        0, len(differences), size=(int(resamples), len(differences))
    )
    bootstrap = differences[draws].mean(axis=1)
    return {
        "mode": mode,
        "epoch": int(mode.split("_")[0].replace("epoch", "")),
        "definition": "robustness correct q minus corrupt-composite q",
        "delta_q_clean_minus_corrupt": float(differences.mean()),
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
        "clean_q_higher_patients": int((differences > 0).sum()),
        "num_patients": len(patients),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for epoch in EPOCHS:
        parser.add_argument(f"--checkpoint{epoch}", required=True)
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

    if args.seed != 42 or not args.amp:
        raise ValueError("Locked trajectory requires seed=42 and AMP")
    if args.bootstrap_resamples < 1000:
        raise ValueError("At least 1000 bootstrap resamples are required")
    if not torch.cuda.is_available():
        raise RuntimeError("Trajectory evaluation requires CUDA")
    set_seed(args.seed)
    device = torch.device("cuda")

    paths: Dict[str, Path] = {}
    for name in (
        "metadata_csv",
        "full_clean_manifest",
        "robustness_manifest",
        "condition_manifest",
    ):
        path = Path(getattr(args, name)).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        paths[name] = path
    checkpoints = {epoch: _checkpoint_path(args, epoch) for epoch in EPOCHS}
    for path in checkpoints.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "slice": output_dir / "stagea_full_lr_trajectory_slice.csv",
        "patient": output_dir / "stagea_full_lr_trajectory_patient.csv",
        "summary": output_dir / "stagea_full_lr_trajectory_summary.csv",
        "scale": output_dir / "stagea_full_lr_trajectory_scale_cascade.csv",
        "paired": output_dir / "stagea_full_lr_trajectory_paired.csv",
        "q_separation": output_dir / "stagea_full_lr_trajectory_q_separation.csv",
        "audit": output_dir / "stagea_full_lr_trajectory_audit.json",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise RuntimeError("Refusing to overwrite outputs: " + ", ".join(existing))

    clean_manifest = _load_json(paths["full_clean_manifest"])
    robust_manifest = _load_json(paths["robustness_manifest"])
    for cohort, manifest in (
        ("full_clean", clean_manifest),
        ("robustness", robust_manifest),
    ):
        if (
            manifest.get("protocol_version") != MANIFEST_PROTOCOL_VERSION
            or manifest.get("cohort") != cohort
        ):
            raise RuntimeError(f"{cohort} manifest protocol/cohort mismatch")
    condition_manifest = _load_json(paths["condition_manifest"])
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

    source = IndexedDataset(
        make_dataset(
            str(paths["metadata_csv"]),
            "val",
            acceleration=8,
            pd_aux_acceleration=2,
        )
    )
    clean_dataset = ManifestDataset(source, clean_manifest)
    robust_dataset = ManifestDataset(source, robust_manifest)
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

    installed_hashes = code_hashes(PROJECT_ROOT)
    checkpoint_audits: Dict[str, Any] = {}
    actual_slice_rows: list[dict[str, Any]] = []
    correction_slice_rows: list[dict[str, Any]] = []
    all_scale_rows: list[dict[str, Any]] = []

    for epoch in EPOCHS:
        checkpoint = torch.load(
            checkpoints[epoch], map_location="cpu", weights_only=False
        )
        checkpoint_audit = _validate_checkpoint(
            checkpoints[epoch], epoch, checkpoint, installed_hashes
        )
        _validate_input_hashes(checkpoint["config"], paths)
        checkpoint_audits[str(epoch)] = checkpoint_audit
        model = QMaxAuxPDVarNet(
            qmax_variant="qmax_full",
            **dict(checkpoint["config"]["model_kwargs"]),
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        checkpoint_hash = checkpoint_audit["sha256"]

        rows, scale_rows = evaluate_mode(
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
        _relabel(rows, epoch, "full", checkpoint_hash)
        _relabel(scale_rows, epoch, "full", checkpoint_hash)
        actual_slice_rows.extend(rows)
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
                mode="full",
                constant_q=None,
            )
            _relabel(rows, epoch, "full", checkpoint_hash)
            _relabel(scale_rows, epoch, "full", checkpoint_hash)
            actual_slice_rows.extend(rows)
            all_scale_rows.extend(scale_rows)

        rows, scale_rows = evaluate_mode(
            model=model,
            loader=clean_loader,
            source_dataset=source,
            condition_lookup=condition_lookup,
            device=device,
            amp=args.amp,
            cohort="full_clean",
            condition="correct",
            mode="correction_off",
            constant_q=None,
        )
        _relabel(rows, epoch, "correction_off", checkpoint_hash)
        _relabel(scale_rows, epoch, "correction_off", checkpoint_hash)
        correction_slice_rows.extend(rows)
        all_scale_rows.extend(scale_rows)

        del model, checkpoint
        torch.cuda.empty_cache()

    actual_patient = patient_rows(actual_slice_rows)
    add_robustness_composite(actual_patient)
    correction_patient = patient_rows(correction_slice_rows)
    patient_level = actual_patient + correction_patient
    all_slice_rows = actual_slice_rows + correction_slice_rows
    summary_level = summaries(patient_level)
    _assert_finite_metrics(patient_level)

    paired: list[dict[str, Any]] = []
    for epoch in EPOCHS[1:]:
        for cohort, condition in (
            ("full_clean", "correct"),
            ("robustness", "robustness_composite"),
        ):
            for metric in METRICS:
                result = paired_bootstrap(
                    patient_level,
                    candidate=f"epoch{epoch}_full",
                    reference="epoch30_full",
                    cohort=cohort,
                    condition=condition,
                    metric=metric,
                    resamples=args.bootstrap_resamples,
                    seed=args.seed,
                )
                result["comparison_type"] = "trajectory_vs_epoch30"
                _add_means_and_relative(result, patient_level)
                paired.append(result)
    for epoch in EPOCHS:
        for metric in METRICS:
            result = paired_bootstrap(
                patient_level,
                candidate=f"epoch{epoch}_full",
                reference=f"epoch{epoch}_correction_off",
                cohort="full_clean",
                condition="correct",
                metric=metric,
                resamples=args.bootstrap_resamples,
                seed=args.seed,
            )
            result["comparison_type"] = "correction_on_vs_off_clean"
            _add_means_and_relative(result, patient_level)
            paired.append(result)

    q_separation = [
        _q_separation(
            patient_level,
            f"epoch{epoch}_full",
            args.bootstrap_resamples,
            args.seed,
        )
        for epoch in EPOCHS
    ]

    init_hashes = {
        audit.get("init_template_sha256")
        for audit in checkpoint_audits.values()
    }
    if len(init_hashes) != 1:
        raise RuntimeError(f"Init-template hashes differ: {init_hashes}")

    write_csv(outputs["slice"], all_slice_rows)
    write_csv(outputs["patient"], patient_level)
    write_csv(outputs["summary"], summary_level)
    write_csv(outputs["scale"], all_scale_rows)
    write_csv(outputs["paired"], paired)
    write_csv(outputs["q_separation"], q_separation)

    audit = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "passed",
        "scope": "locked validation only; held-out test not accessed",
        "formal_selection_checkpoint": str(checkpoints[60]),
        "earlier_checkpoints_are_supportive_trajectory_only": True,
        "strict_state_dict_load": True,
        "epochs": list(EPOCHS),
        "checkpoint_audits": checkpoint_audits,
        "shared_init_template_sha256": next(iter(init_hashes)),
        "conditions": list(CONDITIONS),
        "robustness_composite_conditions": list(ROBUST_CONDITIONS),
        "bootstrap_unit": "patient",
        "bootstrap_resamples": args.bootstrap_resamples,
        "seed": args.seed,
        "input_hashes": {
            key: sha256_file(value) for key, value in paths.items()
        },
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    outputs["audit"].write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
