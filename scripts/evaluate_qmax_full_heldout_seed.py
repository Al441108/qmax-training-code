#!/usr/bin/env python3
from __future__ import annotations

"""Evaluate one frozen QMax-Full epoch-60 seed once on clean held-out test."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_qmax_counterfactuals import (  # noqa: E402
    ManifestDataset,
    evaluate_mode,
    patient_rows,
    summaries,
)
from scripts.qmax_multiseed_common import (  # noqa: E402
    IndexedDataset,
    ShapeBucketBatchSampler,
    install_amp_diagnostic_quantile_compatibility,
    make_dataset,
    set_seed,
    sha256_file,
    write_csv,
)
from src.m2_prnf_qmax_varnet import QMaxAuxPDVarNet  # noqa: E402


PROTOCOL_VERSION = "QMax-Full-heldout-clean-evaluation-v1"
PANEL_PROTOCOL_VERSION = "QMax-Full-heldout-panel-v1"
ALLOWED_SEEDS = (42, 123, 2026)
ANALYSIS_SEED = 42
METRICS = ("l1", "nmse", "psnr", "ssim")
RUNTIME_SOURCE_KEYS = (
    "src/m2_prnf_varnet.py",
    "src/m2_prnf_corruptions.py",
    "src/m2_prnf_fusion_pilot_varnet.py",
    "src/m2_prnf_qmax_varnet.py",
    "src/qmax_deterministic_corruptions.py",
    "src/dataset_paired_multicoil_aux_pd_r2.py",
    "src/fft_utils.py",
    "src/masks.py",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_runtime_sources(checkpoint: Mapping[str, Any]) -> dict[str, str]:
    recorded = checkpoint.get("code_hashes")
    if not isinstance(recorded, Mapping):
        raise RuntimeError("Checkpoint lacks scientific code hashes")
    observed = {}
    failures = {}
    for relative in RUNTIME_SOURCE_KEYS:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        observed[relative] = digest
        if relative not in recorded or recorded[relative] != digest:
            failures[relative] = {
                "checkpoint": recorded.get(relative), "installed": digest
            }
    if failures:
        raise RuntimeError(
            "Inference-relevant scientific source drift:\n"
            + json.dumps(failures, indent=2)
        )
    return observed


def validate_checkpoint(
    path: Path, checkpoint: Mapping[str, Any], seed: int, panel: Mapping[str, Any]
) -> dict[str, Any]:
    if path.name != "model_last.pt" or path.parent.name != "epoch60":
        raise RuntimeError("Held-out evaluator accepts epoch60/model_last.pt only")
    if int(checkpoint.get("epoch", -1)) != 60:
        raise RuntimeError("Checkpoint epoch is not 60")
    config = checkpoint.get("config", {})
    if int(config.get("seed", -1)) != seed:
        raise RuntimeError("Checkpoint model seed mismatch")
    if config.get("qmax_variant") != "qmax_full":
        raise RuntimeError("Checkpoint is not QMax-Full")
    if config.get("formal_structure_selection_checkpoint") != "epoch60/model_last.pt":
        raise RuntimeError("Formal checkpoint policy mismatch")
    history = checkpoint.get("history", [])
    if [int(row["epoch"]) for row in history] != list(range(1, 61)):
        raise RuntimeError("Checkpoint history is not exactly epochs 1..60")
    digest = sha256_file(path)
    entries = {
        int(row["model_seed"]): row for row in panel.get("checkpoints", [])
    }
    if seed not in entries:
        raise RuntimeError(f"Seed {seed} is absent from frozen panel")
    frozen = entries[seed]
    if str(path) != str(Path(frozen["path"]).resolve()) or digest != frozen["sha256"]:
        raise RuntimeError("Checkpoint path/hash differs from frozen held-out panel")
    source_hashes = validate_runtime_sources(checkpoint)
    return {
        "model_seed": seed,
        "checkpoint": str(path),
        "checkpoint_sha256": digest,
        "checkpoint_epoch": 60,
        "history_epochs_exactly_1_to_60": True,
        "strict_state_dict_load": True,
        "runtime_scientific_source_hashes": source_hashes,
    }


def bootstrap_absolute(
    rows: list[dict[str, Any]], metric: str, resamples: int
) -> dict[str, Any]:
    values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
    if values.size != 34 or not np.isfinite(values).all():
        raise RuntimeError(f"Expected 34 finite patient values for {metric}")
    rng = np.random.default_rng(ANALYSIS_SEED)
    indices = rng.integers(0, values.size, size=(resamples, values.size))
    sampled = values[indices].mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return {
        "metric": metric,
        "patient_equal_mean": float(values.mean()),
        "patient_sample_sd": float(values.std(ddof=1)),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "num_patients": int(values.size),
        "bootstrap_unit": "patient",
        "bootstrap_resamples": resamples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--test_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_seed", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--bootstrap_resamples", type=int, default=10000)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.model_seed not in ALLOWED_SEEDS:
        raise ValueError(f"model_seed must be one of {ALLOWED_SEEDS}")
    if not args.amp or not torch.cuda.is_available():
        raise RuntimeError("Held-out evaluation requires CUDA AMP")
    if args.bootstrap_resamples < 1000:
        raise ValueError("At least 1000 bootstrap resamples are required")

    install_amp_diagnostic_quantile_compatibility()
    set_seed(ANALYSIS_SEED)
    device = torch.device("cuda")
    checkpoint_path = Path(args.checkpoint).resolve()
    metadata = Path(args.metadata_csv).resolve()
    manifest_path = Path(args.test_manifest).resolve()
    for path in (checkpoint_path, metadata, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    panel = load_json(manifest_path)
    if panel.get("protocol_version") != PANEL_PROTOCOL_VERSION or panel.get("status") != "frozen":
        raise RuntimeError("Held-out panel manifest is not frozen protocol v1")
    if int(panel.get("num_patients", -1)) != 34:
        raise RuntimeError("Frozen held-out panel does not contain 34 patients")
    if panel.get("metadata_sha256") != sha256_file(metadata):
        raise RuntimeError("Metadata changed after held-out panel freeze")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"qmax_full_seed{args.model_seed}_epoch60_heldout"
    outputs = {
        "slice": output_dir / f"{prefix}_slice_metrics.csv",
        "patient": output_dir / f"{prefix}_patient_metrics.csv",
        "summary": output_dir / f"{prefix}_summary.csv",
        "scale": output_dir / f"{prefix}_scale_cascade_diagnostics.csv",
        "ci": output_dir / f"{prefix}_patient_bootstrap_ci.csv",
        "audit": output_dir / f"{prefix}_audit.json",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise RuntimeError("Refusing to overwrite held-out outputs: " + ", ".join(existing))

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_audit = validate_checkpoint(
        checkpoint_path, checkpoint, args.model_seed, panel
    )
    model = QMaxAuxPDVarNet(
        qmax_variant="qmax_full", **dict(checkpoint["config"]["model_kwargs"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    source = IndexedDataset(
        make_dataset(str(metadata), "test", acceleration=8, pd_aux_acceleration=2)
    )
    dataset = ManifestDataset(source, panel)
    loader = DataLoader(
        dataset,
        batch_sampler=ShapeBucketBatchSampler(
            dataset, args.batch_size, False, ANALYSIS_SEED
        ),
        num_workers=args.num_workers,
        pin_memory=True,
    )
    slice_level, scale_level = evaluate_mode(
        model=model,
        loader=loader,
        source_dataset=source,
        condition_lookup={},
        device=device,
        amp=True,
        cohort="heldout_test",
        condition="correct",
        mode="full",
        constant_q=None,
    )
    patients = patient_rows(slice_level)
    summary = summaries(patients)
    patient_ids = {str(row["patient_id"]) for row in patients}
    if len(patient_ids) != 34 or len(patients) != 34:
        raise RuntimeError("Held-out evaluation did not produce exactly 34 patient rows")
    for rowset in (slice_level, patients, summary, scale_level):
        for row in rowset:
            row.update(
                {
                    "arm": "stagea_full",
                    "qmax_variant": "qmax_full",
                    "model_seed": args.model_seed,
                    "analysis_seed": ANALYSIS_SEED,
                    "checkpoint_epoch": 60,
                    "checkpoint_sha256": checkpoint_audit["checkpoint_sha256"],
                    "test_manifest_sha256": sha256_file(manifest_path),
                }
            )
    for row in patients:
        for metric in METRICS:
            if not math.isfinite(float(row[metric])):
                raise RuntimeError(f"Non-finite held-out {metric}")
    cis = [bootstrap_absolute(patients, metric, args.bootstrap_resamples) for metric in METRICS]

    write_csv(outputs["slice"], slice_level)
    write_csv(outputs["patient"], patients)
    write_csv(outputs["summary"], summary)
    write_csv(outputs["scale"], scale_level)
    write_csv(outputs["ci"], cis)
    audit = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "passed",
        "scope": "one-time clean held-out test evaluation",
        "model_seed": args.model_seed,
        "analysis_seed": ANALYSIS_SEED,
        "mode": "actual-q / all-components-on only",
        "model_or_epoch_selection_performed": False,
        "checkpoint_audit": checkpoint_audit,
        "metadata_sha256": sha256_file(metadata),
        "test_manifest": str(manifest_path),
        "test_manifest_sha256": sha256_file(manifest_path),
        "num_patients": 34,
        "num_slices": len(slice_level),
        "bootstrap_unit": "patient",
        "bootstrap_resamples": args.bootstrap_resamples,
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    outputs["audit"].write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
