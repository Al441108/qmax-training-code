#!/usr/bin/env python3
from __future__ import annotations

"""Strict, external acceptance audit for a QMax Stage-A one-epoch smoke run.

This script is intentionally not imported by training or preflight. It verifies
that a smoke checkpoint is bound to the passed preflight and that the new QMax
branches actually moved away from their step-0 initialisation.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import torch


EXPECTED_SCALES = 4
EXPECTED_CASCADES = 12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be a mapping")
    return value


def tensor_max_abs(value: torch.Tensor) -> float:
    if not torch.is_tensor(value):
        raise TypeError(f"Expected tensor, got {type(value)!r}")
    if value.numel() == 0:
        return 0.0
    return float(value.detach().float().abs().max().item())


def tensor_max_abs_delta(
    observed: torch.Tensor,
    initial: torch.Tensor,
) -> float:
    if observed.shape != initial.shape:
        raise RuntimeError(
            f"Tensor shape mismatch: {tuple(observed.shape)} "
            f"!= {tuple(initial.shape)}"
        )
    return tensor_max_abs(observed.detach().cpu() - initial.detach().cpu())


def module_delta(
    observed: Mapping[str, torch.Tensor],
    initial: Mapping[str, torch.Tensor],
    prefix: str,
) -> float:
    keys = (f"{prefix}.weight", f"{prefix}.bias")
    deltas = []
    for key in keys:
        if key not in observed or key not in initial:
            raise RuntimeError(f"Required parameter missing: {key}")
        deltas.append(tensor_max_abs_delta(observed[key], initial[key]))
    return max(deltas, default=0.0)


def audit_scale_modules(
    observed: Mapping[str, torch.Tensor],
    initial: Mapping[str, torch.Tensor],
    module_suffix: str,
) -> Dict[str, float]:
    return {
        f"scale_{scale}": module_delta(
            observed,
            initial,
            f"controllers.{scale}.{module_suffix}",
        )
        for scale in range(EXPECTED_SCALES)
    }


def audit_cascade_outputs(
    observed: Mapping[str, torch.Tensor],
    initial: Mapping[str, torch.Tensor],
) -> Dict[str, float]:
    return {
        f"cascade_{cascade}": module_delta(
            observed,
            initial,
            f"cascades.{cascade}.regulariser.out_conv",
        )
        for cascade in range(EXPECTED_CASCADES)
    }


def audit_dc_columns(
    observed: Mapping[str, torch.Tensor],
    initial: Mapping[str, torch.Tensor],
) -> Dict[str, Dict[str, float]]:
    output: Dict[str, Dict[str, float]] = {}
    for scale in range(EXPECTED_SCALES):
        for head in ("detail_head", "correction_head"):
            key = f"controllers.{scale}.{head}.in_proj.weight"
            if key not in observed or key not in initial:
                raise RuntimeError(f"Required DC input parameter missing: {key}")
            current = observed[key].detach().cpu()
            core = initial[key].detach().cpu()
            if (
                current.ndim != 4
                or core.ndim != 4
                or current.shape[0] != core.shape[0]
                or current.shape[1] != core.shape[1] + 1
                or current.shape[2:] != core.shape[2:]
            ):
                raise RuntimeError(
                    f"Unexpected Core/Full DC extension for {key}: "
                    f"{tuple(core.shape)} -> {tuple(current.shape)}"
                )
            output[key] = {
                "dc_column_max_abs": tensor_max_abs(current[:, -1:]),
                "old_columns_max_abs_delta": tensor_max_abs_delta(
                    current[:, : core.shape[1]],
                    core,
                ),
            }
    return output


def all_finite_tensors(values: Iterable[Any]) -> bool:
    for value in values:
        if torch.is_tensor(value):
            if not bool(torch.isfinite(value.detach().float()).all()):
                return False
    return True


def finite_history(history: Any) -> bool:
    if not isinstance(history, list) or len(history) != 1:
        return False
    row = history[0]
    if not isinstance(row, Mapping) or int(row.get("epoch", -1)) != 1:
        return False
    for value in row.values():
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            return False
    return True


def optional_finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--init_template", required=True)
    parser.add_argument("--preflight_json", required=True)
    parser.add_argument("--smoke_resume_audit", required=True)
    parser.add_argument(
        "--expected_variant",
        required=True,
        choices=("qmax_core", "qmax_full"),
    )
    parser.add_argument("--output_json", required=True)
    parser.add_argument(
        "--minimum_parameter_delta",
        type=float,
        default=1e-12,
    )
    parser.add_argument("--maximum_epoch_seconds", type=float, default=None)
    parser.add_argument("--maximum_peak_gpu_memory_gb", type=float, default=None)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    template_path = Path(args.init_template).resolve()
    preflight_path = Path(args.preflight_json).resolve()
    resume_audit_path = Path(args.smoke_resume_audit).resolve()
    output_path = Path(args.output_json).resolve()
    for path in (
        checkpoint_path,
        template_path,
        preflight_path,
        resume_audit_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    checkpoint = require_mapping(
        torch.load(checkpoint_path, map_location="cpu", weights_only=False),
        "checkpoint",
    )
    template = require_mapping(
        torch.load(template_path, map_location="cpu", weights_only=False),
        "initialisation template",
    )
    preflight = require_mapping(
        json.loads(preflight_path.read_text(encoding="utf-8")),
        "preflight JSON",
    )
    resume_audit = require_mapping(
        json.loads(resume_audit_path.read_text(encoding="utf-8")),
        "smoke resume audit",
    )

    required_checkpoint_keys = {
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "grad_scaler_state_dict",
        "config",
        "history",
        "rng_state",
        "code_hashes",
        "sampler_next_epoch",
        "run_corruption_audit",
    }
    missing_checkpoint_keys = sorted(
        required_checkpoint_keys - set(checkpoint)
    )
    if missing_checkpoint_keys:
        raise RuntimeError(
            f"Checkpoint missing keys: {missing_checkpoint_keys}"
        )
    config = require_mapping(checkpoint.get("config", {}), "checkpoint config")
    observed_state = require_mapping(
        checkpoint.get("model_state_dict", {}),
        "checkpoint model_state_dict",
    )
    initial_state = require_mapping(
        template.get("p1_state_dict", {}),
        "template p1_state_dict",
    )

    detail_deltas = audit_scale_modules(
        observed_state, initial_state, "detail_head.out"
    )
    alignment_deltas = audit_scale_modules(
        observed_state, initial_state, "alignment_head.out"
    )
    correction_deltas = audit_scale_modules(
        observed_state, initial_state, "correction_head.out"
    )
    q_deltas = audit_scale_modules(
        observed_state, initial_state, "reliability.global_out"
    )
    cascade_deltas = audit_cascade_outputs(observed_state, initial_state)
    dc_columns = (
        audit_dc_columns(observed_state, initial_state)
        if args.expected_variant == "qmax_full"
        else {}
    )

    threshold = float(args.minimum_parameter_delta)
    history = checkpoint.get("history")
    history_row = history[0] if isinstance(history, list) and history else {}
    epoch_seconds = optional_finite_float(
        history_row.get("epoch_seconds")
    )
    peak_memory = optional_finite_float(
        history_row.get("peak_gpu_memory_gb")
    )
    checkpoint_code_hashes = checkpoint.get("code_hashes")
    preflight_code_hashes = preflight.get("code_hashes")
    preflight_inputs = preflight.get("input_hashes", {})
    config_input_hashes = {
        "metadata": config.get("metadata_sha256"),
        "init_template": config.get("init_template_sha256"),
        "full_clean_manifest": config.get(
            "full_clean_manifest_sha256"
        ),
        "robustness_manifest": config.get(
            "robustness_manifest_sha256"
        ),
        "condition_manifest": config.get(
            "condition_manifest_sha256"
        ),
        "historical_p0_checkpoint": config.get(
            "historical_p0_checkpoint_sha256"
        ),
    }
    resume_checkpoint_value = resume_audit.get("checkpoint")
    try:
        resume_checkpoint_path = (
            Path(str(resume_checkpoint_value)).resolve()
            if resume_checkpoint_value is not None
            else None
        )
    except (OSError, RuntimeError, ValueError):
        resume_checkpoint_path = None
    optimizer_state = checkpoint.get("optimizer_state_dict", {})
    scaler_state = checkpoint.get("grad_scaler_state_dict", {})
    rng_state = checkpoint.get("rng_state", {})

    checks = {
        "checkpoint_has_required_keys": not missing_checkpoint_keys,
        "checkpoint_epoch_is_one": safe_int(checkpoint.get("epoch")) == 1,
        "sampler_epoch_is_one": (
            safe_int(checkpoint.get("sampler_next_epoch")) == 1
        ),
        "history_is_one_finite_epoch": finite_history(history),
        "variant_matches": (
            config.get("qmax_variant") == args.expected_variant
        ),
        "run_mode_is_smoke": config.get("run_mode") == "smoke",
        "configured_epochs_is_one": safe_int(config.get("epochs")) == 1,
        "preflight_passed": preflight.get("status") == "passed",
        "checkpoint_code_hashes_match_preflight": (
            checkpoint_code_hashes == preflight_code_hashes
        ),
        "checkpoint_code_hashes_match_config": (
            checkpoint_code_hashes == config.get("code_hashes")
        ),
        "preflight_inputs_match_checkpoint_config": (
            preflight_inputs == config_input_hashes
        ),
        "init_template_hash_matches_config": (
            sha256_file(template_path)
            == config.get("init_template_sha256")
        ),
        "preflight_hash_matches_config": (
            sha256_file(preflight_path)
            == config.get("preflight_json_sha256")
        ),
        "resume_audit_passed": resume_audit.get("passed") is True,
        "resume_audit_binds_exact_checkpoint": (
            resume_checkpoint_path == checkpoint_path
        ),
        "resume_audit_strict_model_load": (
            resume_audit.get("strict_model_load") is True
        ),
        "resume_audit_optimizer_load": (
            resume_audit.get("optimizer_load") is True
        ),
        "resume_audit_grad_scaler_load": (
            resume_audit.get("grad_scaler_load") is True
        ),
        "resume_audit_rng_restore": (
            resume_audit.get("rng_restore") is True
        ),
        "resume_audit_post_restore_finite": (
            resume_audit.get("post_restore_batch_finite") is True
        ),
        "optimizer_state_is_nonempty": (
            isinstance(optimizer_state, Mapping)
            and bool(optimizer_state.get("state"))
        ),
        "grad_scaler_state_is_nonempty": (
            isinstance(scaler_state, Mapping) and bool(scaler_state)
        ),
        "rng_state_has_all_streams": (
            isinstance(rng_state, Mapping)
            and set(rng_state)
            == {"python", "numpy", "torch_cpu", "torch_cuda"}
        ),
        "all_model_tensors_finite": all_finite_tensors(
            observed_state.values()
        ),
        "all_detail_outputs_changed": all(
            value > threshold for value in detail_deltas.values()
        ),
        "all_alignment_outputs_changed": all(
            value > threshold for value in alignment_deltas.values()
        ),
        "all_correction_outputs_changed": all(
            value > threshold for value in correction_deltas.values()
        ),
        "all_q_global_outputs_changed": all(
            value > threshold for value in q_deltas.values()
        ),
        "all_cascade_outputs_changed": all(
            value > threshold for value in cascade_deltas.values()
        ),
        "all_dc_columns_changed": (
            args.expected_variant != "qmax_full"
            or all(
                values["dc_column_max_abs"] > threshold
                for values in dc_columns.values()
            )
        ),
        "epoch_time_within_limit": (
            args.maximum_epoch_seconds is None
            or (
                epoch_seconds is not None
                and epoch_seconds <= args.maximum_epoch_seconds
            )
        ),
        "memory_within_limit": (
            args.maximum_peak_gpu_memory_gb is None
            or (
                peak_memory is not None
                and peak_memory <= args.maximum_peak_gpu_memory_gb
            )
        ),
    }
    status = "passed" if all(checks.values()) else "failed"
    result = {
        "status": status,
        "audit_version": "QMax-smoke-external-audit-v1",
        "expected_variant": args.expected_variant,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "init_template": str(template_path),
        "init_template_sha256": sha256_file(template_path),
        "preflight_json": str(preflight_path),
        "preflight_json_sha256": sha256_file(preflight_path),
        "smoke_resume_audit": str(resume_audit_path),
        "smoke_resume_audit_sha256": sha256_file(resume_audit_path),
        "minimum_parameter_delta": threshold,
        "missing_checkpoint_keys": missing_checkpoint_keys,
        "checks": checks,
        "parameter_max_abs_delta": {
            "detail_output": detail_deltas,
            "alignment_output": alignment_deltas,
            "correction_output": correction_deltas,
            "q_global_output": q_deltas,
            "cascade_output": cascade_deltas,
        },
        "dc_input_columns": dc_columns,
        "epoch_seconds": epoch_seconds,
        "peak_gpu_memory_gb": peak_memory,
        "limits": {
            "maximum_epoch_seconds": args.maximum_epoch_seconds,
            "maximum_peak_gpu_memory_gb": (
                args.maximum_peak_gpu_memory_gb
            ),
        },
    }
    atomic_write_json(output_path, result)
    print(json.dumps(result, indent=2, allow_nan=False))
    if status != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
