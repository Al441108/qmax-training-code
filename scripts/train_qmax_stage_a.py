#!/usr/bin/env python3
from __future__ import annotations

"""Train QMax-Core or QMax-Full from the audited step-0 template."""

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_qmax_random_init import (  # noqa: E402
    PROTOCOL_VERSION,
    build_from_template,
)
from scripts.qmax_common import (  # noqa: E402
    CorruptionAudit,
    DiagnosticAccumulator,
    IndexedDataset,
    ShapeBucketBatchSampler,
    autocast_context,
    code_hashes,
    l1_per_sample,
    make_dataset,
    make_grad_scaler,
    patient_macro,
    prepare_batch,
    restore_rng_state,
    runtime_versions,
    safe_mean,
    save_checkpoint,
    select_patient_ids,
    set_seed,
    sha256_file,
    slice_metrics,
    validate_resume_config,
    write_csv,
)
from src.fft_utils import center_crop  # noqa: E402
from src.m2_prnf_corruptions import (  # noqa: E402
    CORRUPT_MIXTURE,
    CorruptionConfig,
    HardNegativeSampler,
    paired_discrimination_loss,
)
from src.m2_prnf_qmax_varnet import (  # noqa: E402
    QMAX_SCALE_NAMES,
    QMAX_VARIANTS,
)
from src.qmax_deterministic_corruptions import (  # noqa: E402
    DETERMINISTIC_CORRUPTION_PROTOCOL,
    corrupt_batch_qmax,
    manifest_rows,
)


IMMUTABLE_RESUME_KEYS = (
    "qmax_variant",
    "run_mode",
    "metadata_csv",
    "metadata_sha256",
    "full_clean_manifest",
    "full_clean_manifest_sha256",
    "robustness_manifest",
    "robustness_manifest_sha256",
    "condition_manifest",
    "condition_manifest_sha256",
    "preflight_json",
    "preflight_json_sha256",
    "historical_p0_checkpoint",
    "historical_p0_checkpoint_sha256",
    "init_template",
    "init_template_sha256",
    "acceleration",
    "pd_aux_acceleration",
    "epochs",
    "learning_rate",
    "batch_size",
    "grad_accum_steps",
    "num_workers",
    "num_train_patients",
    "num_val_patients",
    "max_train_batches",
    "max_val_batches",
    "lambda_rel",
    "lambda_rank",
    "lambda_residual_gain",
    "residual_gain_margin_relative",
    "aux_loss_ramp_epochs",
    "residual_gain_ramp_epochs",
    "seed",
    "amp",
    "model_kwargs",
    "train_patient_ids",
    "val_patient_ids",
    "corruption_config",
    "corrupt_view_mixture",
    "deterministic_corruption_protocol",
    "optimizer",
    "gradient_clip_norm",
    "code_hashes",
)


def _mean_diagnostic(
    auxiliary: Mapping[str, torch.Tensor],
    key: str,
    selected: slice,
) -> float:
    return float(auxiliary[key][selected].detach().float().mean().item())


@torch.no_grad()
def evaluate_clean(
    model,
    loader,
    device: torch.device,
    amp: bool,
    max_batches: Optional[int],
) -> Dict[str, Any]:
    model.eval()
    rows: List[Dict[str, Any]] = []
    q_values: List[float] = []
    diagnostic_values: Dict[str, List[float]] = defaultdict(list)
    for batch_index, batch in enumerate(loader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break
        kspace, mask, pd, target, _ = prepare_batch(batch, device)
        available = torch.ones(pd.shape[0], device=device)
        with autocast_context(device, amp):
            correction_off_prediction = model(
                kspace,
                mask,
                pd,
                available,
                correction_off=True,
            )
            prediction, auxiliary = model(
                kspace,
                mask,
                pd,
                available,
                return_aux=True,
            )
        prediction = center_crop(
            prediction.float(), target.shape[-2], target.shape[-1]
        )
        correction_off_prediction = center_crop(
            correction_off_prediction.float(),
            target.shape[-2],
            target.shape[-1],
        )
        correction_off_l1 = l1_per_sample(
            correction_off_prediction, target
        )
        for index in range(target.shape[0]):
            row = {
                "patient_id": str(batch["patient_id"][index]),
                "slice_idx": int(batch["slice_idx"][index]),
                **slice_metrics(prediction[index], target[index]),
                "correction_off_l1": float(
                    correction_off_l1[index].item()
                ),
            }
            rows.append(row)
        q_values.extend(
            auxiliary["q_hat"].mean((1, 2)).float().cpu().tolist()
        )
        for key in (
            "direct_to_target_rms",
            "detail_gate_mean",
            "alignment_to_target_rms",
            "correction_to_target_rms",
            "final_auxiliary_to_target_rms",
            "cos_direct_correction",
            "dc_raw_rms",
        ):
            diagnostic_values[key].extend(
                auxiliary[key].mean((1, 2)).float().cpu().tolist()
            )

    summary = patient_macro(rows)
    correction_by_patient: Dict[str, List[float]] = defaultdict(list)
    actual_by_patient: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        patient = str(row["patient_id"])
        actual_by_patient[patient].append(float(row["l1"]))
        correction_by_patient[patient].append(
            float(row["correction_off_l1"])
        )
    paired_delta = [
        safe_mean(actual_by_patient[patient])
        - safe_mean(correction_by_patient[patient])
        for patient in sorted(actual_by_patient)
    ]
    summary.update(
        {
            "correction_off_patient_l1": safe_mean(
                safe_mean(values)
                for values in correction_by_patient.values()
            ),
            "correction_on_minus_off_patient_l1": safe_mean(paired_delta),
            "patients_correction_on_better": sum(
                value < 0.0 for value in paired_delta
            ),
            "clean_q": safe_mean(q_values),
            **{
                f"clean_{key}": safe_mean(values)
                for key, values in diagnostic_values.items()
            },
        }
    )
    model.train()
    return summary


def _checkpoint_reload_audit(
    *,
    checkpoint_path: Path,
    model,
    optimizer,
    scaler,
    val_loader,
    device: torch.device,
    amp: bool,
) -> Dict[str, Any]:
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    required = {
        "model_state_dict",
        "optimizer_state_dict",
        "grad_scaler_state_dict",
        "rng_state",
        "config",
        "epoch",
        "sampler_next_epoch",
        "history",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise RuntimeError(f"Smoke checkpoint missing: {missing}")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])
    restore_rng_state(checkpoint["rng_state"])
    batch = next(iter(val_loader))
    kspace, mask, pd, target, _ = prepare_batch(batch, device)
    with torch.no_grad(), autocast_context(device, amp):
        prediction = model(
            kspace,
            mask,
            pd,
            torch.ones(pd.shape[0], device=device),
        )
    prediction = center_crop(
        prediction.float(), target.shape[-2], target.shape[-1]
    )
    finite = bool(torch.isfinite(prediction).all())
    return {
        "checkpoint": str(checkpoint_path),
        "strict_model_load": True,
        "optimizer_load": True,
        "grad_scaler_load": True,
        "rng_restore": True,
        "post_restore_batch_finite": finite,
        "post_restore_batch_l1": float(
            l1_per_sample(prediction, target).mean().item()
        ),
        "passed": finite,
    }


def _read_patient_file(path: Path) -> List[str]:
    if not path.is_file():
        raise RuntimeError(f"Resume output is missing {path.name}")
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _validate_resume_artifacts(
    *,
    output_dir: Path,
    checkpoint: Mapping[str, Any],
    train_ids: List[str],
    val_ids: List[str],
) -> None:
    """Reject partial or cross-run resumes before mutating model state."""

    completed_epoch = int(checkpoint["epoch"])
    history = list(checkpoint["history"])
    if len(history) != completed_epoch:
        raise RuntimeError(
            "Resume history length does not equal checkpoint epoch: "
            f"{len(history)} != {completed_epoch}"
        )
    if history and int(history[-1].get("epoch", -1)) != completed_epoch:
        raise RuntimeError("Resume history does not end at checkpoint epoch")
    if _read_patient_file(output_dir / "train_patient_ids.txt") != train_ids:
        raise RuntimeError("Resume train-patient manifest differs")
    if _read_patient_file(output_dir / "val_patient_ids.txt") != val_ids:
        raise RuntimeError("Resume validation-patient manifest differs")

    log_path = output_dir / "training_log.csv"
    if not log_path.is_file():
        raise RuntimeError("Resume output is missing training_log.csv")
    with log_path.open("r", newline="", encoding="utf-8") as handle:
        log_rows = list(csv.DictReader(handle))
    if len(log_rows) != completed_epoch:
        raise RuntimeError(
            "Resume training-log length does not equal checkpoint epoch: "
            f"{len(log_rows)} != {completed_epoch}"
        )
    if log_rows and int(log_rows[-1]["epoch"]) != completed_epoch:
        raise RuntimeError("Resume training log does not end at checkpoint epoch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--init_template", required=True)
    parser.add_argument("--full_clean_manifest", required=True)
    parser.add_argument("--robustness_manifest", required=True)
    parser.add_argument("--condition_manifest", required=True)
    parser.add_argument("--preflight_json", required=True)
    parser.add_argument("--historical_p0_checkpoint", required=True)
    parser.add_argument(
        "--qmax_variant", required=True, choices=sorted(QMAX_VARIANTS)
    )
    parser.add_argument(
        "--run_mode", required=True, choices=("smoke", "formal")
    )
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--acceleration", type=int, default=8)
    parser.add_argument("--pd_aux_acceleration", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_train_patients", type=int, default=None)
    parser.add_argument("--num_val_patients", type=int, default=None)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--lambda_rel", type=float, default=0.05)
    parser.add_argument("--lambda_rank", type=float, default=0.02)
    parser.add_argument("--aux_loss_ramp_epochs", type=int, default=5)
    parser.add_argument("--lambda_residual_gain", type=float, default=0.2)
    parser.add_argument(
        "--residual_gain_margin_relative", type=float, default=0.002
    )
    parser.add_argument(
        "--residual_gain_ramp_epochs", type=int, default=5
    )
    parser.add_argument("--gradient_clip_norm", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    if args.seed != 42:
        raise ValueError("Frozen Stage A requires seed=42")
    if args.acceleration != 8 or args.pd_aux_acceleration != 2:
        raise ValueError("Frozen Stage A requires PD-FS R=8 and PD R=2")
    if args.learning_rate != 3e-4:
        raise ValueError("Epochs 1-30 require learning_rate=3e-4")
    if args.run_mode == "smoke" and args.epochs != 1:
        raise ValueError("Smoke mode must run exactly one epoch")
    if args.run_mode == "formal" and args.epochs != 30:
        raise ValueError("Formal Stage-A training must end at epoch 30")
    if args.batch_size != 4 or args.grad_accum_steps != 1:
        raise ValueError("Frozen Stage A requires batch=4 and accumulation=1")
    frozen_scalars = {
        "lambda_rel": (args.lambda_rel, 0.05),
        "lambda_rank": (args.lambda_rank, 0.02),
        "lambda_residual_gain": (args.lambda_residual_gain, 0.2),
        "residual_gain_margin_relative": (
            args.residual_gain_margin_relative,
            0.002,
        ),
        "gradient_clip_norm": (args.gradient_clip_norm, 10.0),
    }
    drift = {
        name: {"observed": observed, "required": required}
        for name, (observed, required) in frozen_scalars.items()
        if not math.isclose(
            float(observed), float(required), rel_tol=0.0, abs_tol=1e-12
        )
    }
    if drift:
        raise ValueError(f"Frozen Stage-A hyperparameter drift: {drift}")
    if (
        args.aux_loss_ramp_epochs != 5
        or args.residual_gain_ramp_epochs != 5
    ):
        raise ValueError("Frozen Stage A requires both loss ramps to be 5")
    if not args.amp:
        raise ValueError("Frozen Stage A requires AMP with GradScaler")
    if args.run_mode == "formal" and (
        args.num_train_patients is not None
        or args.num_val_patients is not None
        or args.max_train_batches is not None
        or args.max_val_batches is not None
    ):
        raise ValueError("Formal Stage A must use the full train/val cohorts")
    if args.epochs < 1 or args.gradient_clip_norm <= 0:
        raise ValueError("Invalid epoch/gradient settings")

    for name in (
        "metadata_csv",
        "init_template",
        "full_clean_manifest",
        "robustness_manifest",
        "condition_manifest",
        "preflight_json",
        "historical_p0_checkpoint",
    ):
        path = Path(getattr(args, name)).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        setattr(args, name, str(path))

    set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("QMax Stage-A training requires CUDA")
    device = torch.device("cuda")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    full_train = make_dataset(
        args.metadata_csv,
        "train",
        args.acceleration,
        args.pd_aux_acceleration,
    )
    full_val = make_dataset(
        args.metadata_csv,
        "val",
        args.acceleration,
        args.pd_aux_acceleration,
    )
    train_ids = select_patient_ids(full_train, args.num_train_patients)
    val_ids = select_patient_ids(full_val, args.num_val_patients)
    if set(train_ids) & set(val_ids):
        raise RuntimeError("Patient leakage detected")
    train_dataset = IndexedDataset(
        make_dataset(
            args.metadata_csv,
            "train",
            args.acceleration,
            args.pd_aux_acceleration,
            train_ids,
        )
    )
    val_dataset = IndexedDataset(
        make_dataset(
            args.metadata_csv,
            "val",
            args.acceleration,
            args.pd_aux_acceleration,
            val_ids,
        )
    )
    train_sampler = ShapeBucketBatchSampler(
        train_dataset, args.batch_size, True, args.seed
    )
    val_sampler = ShapeBucketBatchSampler(
        val_dataset, args.batch_size, False, args.seed
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    negative_sampler = HardNegativeSampler(train_dataset)
    corruption_config = CorruptionConfig()

    template_path = Path(args.init_template)
    template = torch.load(
        template_path, map_location="cpu", weights_only=False
    )
    if template.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("Initialisation template protocol mismatch")
    if int(template.get("seed", -1)) != args.seed:
        raise RuntimeError("Initialisation template seed mismatch")
    model = build_from_template(template, args.qmax_variant).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    amp_enabled = bool(args.amp)
    scaler = make_grad_scaler(amp_enabled)
    hashes = code_hashes(PROJECT_ROOT)
    preflight_path = Path(args.preflight_json)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "passed":
        raise RuntimeError("Bound QMax preflight did not pass")
    if preflight.get("code_hashes") != hashes:
        raise RuntimeError(
            "Installed code differs from the passed QMax preflight"
        )
    expected_preflight_inputs = {
        "metadata": sha256_file(Path(args.metadata_csv)),
        "init_template": sha256_file(template_path),
        "full_clean_manifest": sha256_file(
            Path(args.full_clean_manifest)
        ),
        "robustness_manifest": sha256_file(
            Path(args.robustness_manifest)
        ),
        "condition_manifest": sha256_file(
            Path(args.condition_manifest)
        ),
        "historical_p0_checkpoint": sha256_file(
            Path(args.historical_p0_checkpoint)
        ),
    }
    if preflight.get("input_hashes") != expected_preflight_inputs:
        raise RuntimeError(
            "Training inputs differ from the passed QMax preflight"
        )
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    q_compatibility_parameter_count = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if ".reliability.spatial_out." in name
        or ".reliability.channel_out." in name
    )

    config = vars(args).copy()
    config["resume"] = None
    config.update(
        {
            "protocol_version": "QMax-StageA-R8-R2-train-v2",
            "random_initialisation": True,
            "initialisation_protocol": PROTOCOL_VERSION,
            "init_template_sha256": sha256_file(template_path),
            "model_kwargs": dict(template["model_kwargs"]),
            "parameter_count": int(parameter_count),
            "q_compatibility_only_parameter_count": int(
                q_compatibility_parameter_count
            ),
            "active_parameter_count_excluding_q_compatibility_outputs": int(
                parameter_count - q_compatibility_parameter_count
            ),
            "train_patient_ids": train_ids,
            "val_patient_ids": val_ids,
            "metadata_sha256": sha256_file(Path(args.metadata_csv)),
            "full_clean_manifest_sha256": sha256_file(
                Path(args.full_clean_manifest)
            ),
            "robustness_manifest_sha256": sha256_file(
                Path(args.robustness_manifest)
            ),
            "condition_manifest_sha256": sha256_file(
                Path(args.condition_manifest)
            ),
            "preflight_json_sha256": sha256_file(preflight_path),
            "historical_p0_checkpoint_sha256": (
                expected_preflight_inputs["historical_p0_checkpoint"]
            ),
            "optimizer": {
                "name": "Adam",
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.0,
            },
            "correct_corrupt_loss_weights": {
                "correct": 0.7,
                "second_view": 0.3,
            },
            "corrupt_view_mixture": dict(CORRUPT_MIXTURE),
            "corruption_config": asdict(corruption_config),
            "deterministic_corruption_protocol": (
                DETERMINISTIC_CORRUPTION_PROTOCOL
            ),
            "reliability_definition": (
                "existing detached PairReliabilityHead(target,U0)"
            ),
            "gain_reference": (
                "same checkpoint, clean PD, actual q, correction_off"
            ),
            "gain_reference_stop_gradient": True,
            "checkpoint_selection_metric": (
                "patient-level clean validation L1"
            ),
            "formal_structure_selection_checkpoint": (
                "epoch30/model_last.pt"
            ),
            "code_hashes": hashes,
            "runtime_versions": runtime_versions(),
        }
    )

    start_epoch = 1
    best_epoch = 0
    best_val = float("inf")
    history: List[Dict[str, Any]] = []
    run_corruption_audit = CorruptionAudit()
    if args.resume:
        checkpoint_path = Path(args.resume).resolve()
        if (
            checkpoint_path.name != "model_last.pt"
            or checkpoint_path.parent != output_dir
        ):
            raise RuntimeError(
                "Resume must use model_last.pt from this exact output_dir"
            )
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        required = {
            "model_state_dict",
            "optimizer_state_dict",
            "grad_scaler_state_dict",
            "config",
            "rng_state",
            "code_hashes",
            "epoch",
            "sampler_next_epoch",
            "history",
            "run_corruption_audit",
            "best_epoch",
            "best_val",
        }
        missing = sorted(required - set(checkpoint))
        if missing:
            raise RuntimeError(f"Resume checkpoint missing keys: {missing}")
        validate_resume_config(
            checkpoint["config"], config, IMMUTABLE_RESUME_KEYS
        )
        if checkpoint["code_hashes"] != hashes:
            raise RuntimeError("Resume code hashes differ")
        installed_config_path = output_dir / "config.json"
        if not installed_config_path.is_file():
            raise RuntimeError("Resume output has no config.json")
        installed_config = json.loads(
            installed_config_path.read_text(encoding="utf-8")
        )
        validate_resume_config(
            installed_config, config, IMMUTABLE_RESUME_KEYS
        )
        _validate_resume_artifacts(
            output_dir=output_dir,
            checkpoint=checkpoint,
            train_ids=train_ids,
            val_ids=val_ids,
        )
        model.load_state_dict(
            checkpoint["model_state_dict"], strict=True
        )
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])
        restore_rng_state(checkpoint["rng_state"])
        run_corruption_audit.load_state_dict(
            checkpoint["run_corruption_audit"]
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        best_epoch = int(checkpoint["best_epoch"])
        best_val = float(checkpoint["best_val"])
        history = list(checkpoint["history"])
        if int(checkpoint["sampler_next_epoch"]) != int(
            checkpoint["epoch"]
        ):
            raise RuntimeError("Resume sampler epoch mismatch")
        if start_epoch > args.epochs:
            raise RuntimeError("Requested run is already complete")
    else:
        config_path = output_dir / "config.json"
        if config_path.exists():
            raise RuntimeError(
                f"Fresh run output already contains config: {config_path}"
            )
        config_path.write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )
        (output_dir / "train_patient_ids.txt").write_text(
            "\n".join(train_ids), encoding="utf-8"
        )
        (output_dir / "val_patient_ids.txt").write_text(
            "\n".join(val_ids), encoding="utf-8"
        )

    log_fields = (
        "epoch",
        "train_total_loss",
        "train_recon_loss",
        "train_clean_l1",
        "train_corrupt_l1",
        "train_reliability_bce",
        "train_rank_loss",
        "train_correction_gain_loss",
        "train_correction_off_l1",
        "train_correction_on_minus_off_l1",
        "train_gain_violation_rate",
        "train_q_clean",
        "train_q_corrupt",
        "train_direct_rms_clean",
        "train_direct_rms_corrupt",
        "train_correction_rms_clean",
        "train_correction_rms_corrupt",
        "train_final_aux_rms_clean",
        "train_final_aux_rms_corrupt",
        "train_gradient_norm",
        "val_patient_l1",
        "val_patient_nmse",
        "val_patient_psnr",
        "val_patient_ssim",
        "val_correction_off_patient_l1",
        "val_correction_on_minus_off_patient_l1",
        "val_patients_correction_on_better",
        "val_clean_q",
        "epoch_seconds",
        "peak_gpu_memory_gb",
    )
    log_path = output_dir / "training_log.csv"
    if start_epoch == 1:
        with log_path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=log_fields).writeheader()

    print(
        json.dumps(
            {
                "qmax_variant": args.qmax_variant,
                "run_mode": args.run_mode,
                "device": str(device),
                "parameter_count": parameter_count,
                "amp": amp_enabled,
                "train_patients": len(train_ids),
                "val_patients": len(val_ids),
                "train_slices": len(train_dataset),
                "val_slices": len(val_dataset),
            },
            indent=2,
        ),
        flush=True,
    )

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        train_sampler.epoch = epoch - 1
        torch.cuda.reset_peak_memory_stats()
        stats: Dict[str, List[float]] = defaultdict(list)
        diagnostics = DiagnosticAccumulator()
        epoch_corruption_audit = CorruptionAudit()
        first_manifest: List[Dict[str, Any]] = []
        optimizer.zero_grad(set_to_none=True)
        accumulation = 0

        for batch_index, batch in enumerate(train_loader, start=1):
            if (
                args.max_train_batches is not None
                and batch_index > args.max_train_batches
            ):
                break
            kspace, mask, pd, target, indices = prepare_batch(
                batch, device
            )
            corrupt = corrupt_batch_qmax(
                pd_aux=pd,
                sample_indices=indices,
                dataset=train_dataset,
                negative_sampler=negative_sampler,
                epoch=epoch,
                global_seed=args.seed,
                config=corruption_config,
                view_index=1,
                occurrence_indices=[0] * len(indices),
                stream_id="qmax_train_corrupt",
            )
            epoch_corruption_audit.add(corrupt.records)
            run_corruption_audit.add(corrupt.records)
            if batch_index <= 4:
                first_manifest.extend(manifest_rows(corrupt.records))
            base = pd.shape[0]
            clean_available = torch.ones(base, device=device)

            with torch.no_grad(), autocast_context(device, amp_enabled):
                correction_off_prediction = model(
                    kspace,
                    mask,
                    pd,
                    clean_available,
                    correction_off=True,
                )
            correction_off_prediction = center_crop(
                correction_off_prediction.float(),
                target.shape[-2],
                target.shape[-1],
            )
            correction_off_l1 = l1_per_sample(
                correction_off_prediction, target
            ).detach()
            del correction_off_prediction

            paired_kspace = torch.cat([kspace, kspace], dim=0)
            paired_mask = torch.cat([mask, mask], dim=0)
            paired_pd = torch.cat([pd, corrupt.image], dim=0)
            paired_available = torch.cat(
                [torch.ones_like(corrupt.availability), corrupt.availability],
                dim=0,
            )
            with autocast_context(device, amp_enabled):
                prediction, auxiliary = model(
                    paired_kspace,
                    paired_mask,
                    paired_pd,
                    paired_available,
                    return_aux=True,
                )
                prediction = center_crop(
                    prediction.float(),
                    target.shape[-2],
                    target.shape[-1],
                )
                clean_l1_per_sample = l1_per_sample(
                    prediction[:base], target
                )
                clean_l1 = clean_l1_per_sample.mean()
                corrupt_l1 = l1_per_sample(
                    prediction[base:], target
                ).mean()
                reconstruction_loss = (
                    0.7 * clean_l1 + 0.3 * corrupt_l1
                )

                required_l1 = (
                    1.0 - args.residual_gain_margin_relative
                ) * correction_off_l1
                correction_gain_per_sample = F.relu(
                    clean_l1_per_sample - required_l1
                )
                correction_gain_loss = (
                    correction_gain_per_sample.mean()
                )
                gain_violation = (
                    clean_l1_per_sample > required_l1
                ).float().mean()

                logits_clean = auxiliary["q_logits"][:base]
                logits_corrupt = auxiliary["q_logits"][base:]
                clean_targets = torch.ones_like(logits_clean)
                corrupt_targets = (
                    corrupt.reliability_target[:, None, :]
                    .expand_as(logits_corrupt)
                )
                clean_bce = F.binary_cross_entropy_with_logits(
                    logits_clean, clean_targets
                )
                reliability_mask = torch.tensor(
                    [
                        record.get("condition") != "missing"
                        for record in corrupt.records
                    ],
                    device=device,
                    dtype=torch.bool,
                )
                if bool(reliability_mask.any()):
                    corrupt_bce = F.binary_cross_entropy_with_logits(
                        logits_corrupt[reliability_mask],
                        corrupt_targets[reliability_mask],
                    )
                    reliability_bce = 0.5 * (
                        clean_bce + corrupt_bce
                    )
                else:
                    reliability_bce = clean_bce
                rank_loss, _ = paired_discrimination_loss(
                    auxiliary["q_hat"][:base],
                    auxiliary["q_hat"][base:],
                    corrupt.reliability_target,
                    corrupt.records,
                )
                auxiliary_ramp = min(
                    1.0,
                    epoch / max(1, args.aux_loss_ramp_epochs),
                )
                gain_ramp = min(
                    1.0,
                    epoch / max(1, args.residual_gain_ramp_epochs),
                )
                total_loss = reconstruction_loss + auxiliary_ramp * (
                    args.lambda_rel * reliability_bce
                    + args.lambda_rank * rank_loss
                ) + gain_ramp * (
                    args.lambda_residual_gain * correction_gain_loss
                )

            if not bool(torch.isfinite(total_loss)):
                raise RuntimeError(
                    f"Non-finite loss at epoch {epoch}, batch {batch_index}"
                )
            for key in DiagnosticAccumulator.METRICS:
                if not bool(torch.isfinite(auxiliary[key]).all()):
                    raise RuntimeError(
                        f"Non-finite {key} at epoch {epoch}, "
                        f"batch {batch_index}"
                    )

            scaler.scale(
                total_loss / args.grad_accum_steps
            ).backward()
            accumulation += 1
            final_available_batch = (
                batch_index == len(train_loader)
                or (
                    args.max_train_batches is not None
                    and batch_index == args.max_train_batches
                )
            )
            should_step = (
                accumulation == args.grad_accum_steps
                or final_available_batch
            )
            if should_step:
                scaler.unscale_(optimizer)
                if accumulation < args.grad_accum_steps:
                    correction_factor = (
                        args.grad_accum_steps / accumulation
                    )
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction_factor)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.gradient_clip_norm
                )
                if not bool(torch.isfinite(gradient_norm)):
                    raise RuntimeError("Non-finite gradient norm")
                stats["gradient"].append(float(gradient_norm.item()))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                accumulation = 0

            stats["total"].append(float(total_loss.detach().item()))
            stats["recon"].append(
                float(reconstruction_loss.detach().item())
            )
            stats["clean"].append(float(clean_l1.detach().item()))
            stats["corrupt"].append(float(corrupt_l1.detach().item()))
            stats["bce"].append(
                float(reliability_bce.detach().item())
            )
            stats["rank"].append(float(rank_loss.detach().item()))
            stats["gain"].append(
                float(correction_gain_loss.detach().item())
            )
            stats["correction_off"].append(
                float(correction_off_l1.mean().item())
            )
            stats["on_minus_off"].append(
                float(
                    (
                        clean_l1_per_sample.detach()
                        - correction_off_l1
                    )
                    .mean()
                    .item()
                )
            )
            stats["gain_violation"].append(
                float(gain_violation.detach().item())
            )
            stats["q_clean"].append(
                _mean_diagnostic(auxiliary, "q_hat", slice(0, base))
            )
            stats["q_corrupt"].append(
                _mean_diagnostic(auxiliary, "q_hat", slice(base, None))
            )
            for condition, selected in (
                ("clean", slice(0, base)),
                ("corrupt", slice(base, None)),
            ):
                diagnostics.add(auxiliary, selected, condition)
                for label, key in (
                    ("direct", "direct_to_target_rms"),
                    ("correction", "correction_to_target_rms"),
                    ("final", "final_auxiliary_to_target_rms"),
                ):
                    stats[f"{label}_{condition}"].append(
                        _mean_diagnostic(auxiliary, key, selected)
                    )

            if batch_index == 1 or batch_index % 25 == 0:
                print(
                    f"epoch={epoch:02d} "
                    f"batch={batch_index:04d}/{len(train_loader)} "
                    f"clean={clean_l1.item():.6f} "
                    f"corrupt={corrupt_l1.item():.6f} "
                    f"correction_off={correction_off_l1.mean().item():.6f} "
                    f"q={stats['q_clean'][-1]:.3f}/"
                    f"{stats['q_corrupt'][-1]:.3f}",
                    flush=True,
                )

        manifest_path = output_dir / (
            f"epoch_{epoch:02d}_first_batches_corruption_manifest.json"
        )
        manifest_path.write_text(
            json.dumps(
                {
                    "protocol": DETERMINISTIC_CORRUPTION_PROTOCOL,
                    "epoch": epoch,
                    "qmax_variant": args.qmax_variant,
                    "rows": first_manifest,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        validation = evaluate_clean(
            model,
            val_loader,
            device,
            amp_enabled,
            args.max_val_batches,
        )
        epoch_seconds = time.time() - epoch_start
        row = {
            "epoch": epoch,
            "train_total_loss": safe_mean(stats["total"]),
            "train_recon_loss": safe_mean(stats["recon"]),
            "train_clean_l1": safe_mean(stats["clean"]),
            "train_corrupt_l1": safe_mean(stats["corrupt"]),
            "train_reliability_bce": safe_mean(stats["bce"]),
            "train_rank_loss": safe_mean(stats["rank"]),
            "train_correction_gain_loss": safe_mean(stats["gain"]),
            "train_correction_off_l1": safe_mean(
                stats["correction_off"]
            ),
            "train_correction_on_minus_off_l1": safe_mean(
                stats["on_minus_off"]
            ),
            "train_gain_violation_rate": safe_mean(
                stats["gain_violation"]
            ),
            "train_q_clean": safe_mean(stats["q_clean"]),
            "train_q_corrupt": safe_mean(stats["q_corrupt"]),
            "train_direct_rms_clean": safe_mean(
                stats["direct_clean"]
            ),
            "train_direct_rms_corrupt": safe_mean(
                stats["direct_corrupt"]
            ),
            "train_correction_rms_clean": safe_mean(
                stats["correction_clean"]
            ),
            "train_correction_rms_corrupt": safe_mean(
                stats["correction_corrupt"]
            ),
            "train_final_aux_rms_clean": safe_mean(
                stats["final_clean"]
            ),
            "train_final_aux_rms_corrupt": safe_mean(
                stats["final_corrupt"]
            ),
            "train_gradient_norm": safe_mean(stats["gradient"]),
            "val_patient_l1": validation["patient_l1"],
            "val_patient_nmse": validation["patient_nmse"],
            "val_patient_psnr": validation["patient_psnr"],
            "val_patient_ssim": validation["patient_ssim"],
            "val_correction_off_patient_l1": validation[
                "correction_off_patient_l1"
            ],
            "val_correction_on_minus_off_patient_l1": validation[
                "correction_on_minus_off_patient_l1"
            ],
            "val_patients_correction_on_better": validation[
                "patients_correction_on_better"
            ],
            "val_clean_q": validation["clean_q"],
            "epoch_seconds": epoch_seconds,
            "peak_gpu_memory_gb": (
                torch.cuda.max_memory_allocated() / 1024**3
            ),
        }
        history.append(row)
        with log_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=log_fields).writerow(row)
        write_csv(
            output_dir / f"epoch_{epoch:02d}_diagnostics.csv",
            diagnostics.rows(epoch, QMAX_SCALE_NAMES),
        )
        (output_dir / f"epoch_{epoch:02d}_validation.json").write_text(
            json.dumps(validation, indent=2), encoding="utf-8"
        )
        (
            output_dir / f"epoch_{epoch:02d}_corruption_audit.json"
        ).write_text(
            json.dumps(epoch_corruption_audit.summary(), indent=2),
            encoding="utf-8",
        )
        (
            output_dir / "run_corruption_audit.json"
        ).write_text(
            json.dumps(run_corruption_audit.summary(), indent=2),
            encoding="utf-8",
        )

        if validation["patient_l1"] < best_val:
            best_val = float(validation["patient_l1"])
            best_epoch = int(epoch)
            save_checkpoint(
                path=output_dir / "model_best_within_30.pt",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                best_epoch=best_epoch,
                best_val=best_val,
                config=config,
                history=history,
                corruption_audit=run_corruption_audit.state_dict(),
            )
        save_checkpoint(
            path=output_dir / "model_last.pt",
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            best_epoch=best_epoch,
            best_val=best_val,
            config=config,
            history=history,
            corruption_audit=run_corruption_audit.state_dict(),
        )
        if epoch == 30:
            save_checkpoint(
                path=output_dir / "epoch30" / "model_last.pt",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                best_epoch=best_epoch,
                best_val=best_val,
                config=config,
                history=history,
                corruption_audit=run_corruption_audit.state_dict(),
            )
        print(json.dumps(row, indent=2), flush=True)

    if args.run_mode == "smoke":
        reload_audit = _checkpoint_reload_audit(
            checkpoint_path=output_dir / "model_last.pt",
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            val_loader=val_loader,
            device=device,
            amp=amp_enabled,
        )
        (output_dir / "smoke_resume_audit.json").write_text(
            json.dumps(reload_audit, indent=2), encoding="utf-8"
        )
        if not reload_audit["passed"]:
            raise RuntimeError("Smoke checkpoint reload audit failed")

    final_summary = {
        "protocol_version": config["protocol_version"],
        "qmax_variant": args.qmax_variant,
        "run_mode": args.run_mode,
        "completed_epochs": args.epochs,
        "best_within_30_epoch": best_epoch,
        "best_within_30_val_patient_l1": best_val,
        "formal_selection_checkpoint": (
            str(output_dir / "epoch30" / "model_last.pt")
            if args.run_mode == "formal"
            else None
        ),
        "parameter_count": parameter_count,
        "q_compatibility_only_parameter_count": (
            q_compatibility_parameter_count
        ),
        "init_template_sha256": config["init_template_sha256"],
    }
    (output_dir / "final_summary.json").write_text(
        json.dumps(final_summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(final_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
