#!/usr/bin/env python3
from __future__ import annotations

"""Strict real-batch GPU preflight for QMax replication seeds."""

import argparse
import itertools
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_qmax_multiseed_init import (  # noqa: E402
    ALLOWED_REPLICATION_SEEDS,
    PROTOCOL_VERSION,
    build_from_template,
    initialisation_source_hashes,
)
from scripts.qmax_multiseed_common import (  # noqa: E402
    IndexedDataset,
    ShapeBucketBatchSampler,
    autocast_context,
    code_hashes,
    install_amp_diagnostic_quantile_compatibility,
    l1_per_sample,
    make_dataset,
    make_grad_scaler,
    prepare_batch,
    runtime_versions,
    set_seed,
    sha256_file,
)
from src.fft_utils import center_crop, ifft2c, rss_combine  # noqa: E402
from src.m2_prnf_corruptions import (  # noqa: E402
    CorruptionConfig,
    HardNegativeSampler,
    paired_discrimination_loss,
)
from src.m2_prnf_fusion_pilot_varnet import (  # noqa: E402
    M2PRNFFusionPilotVarNet,
)
from src.m2_prnf_qmax_varnet import (  # noqa: E402
    QMaxAuxPDVarNet,
    qmax_dc_input_columns,
    qmax_shared_state,
)
from src.qmax_deterministic_corruptions import (  # noqa: E402
    DETERMINISTIC_CORRUPTION_PROTOCOL,
    corrupt_batch_qmax,
    manifest_rows,
)


def max_state_difference(
    first: Mapping[str, torch.Tensor],
    second: Mapping[str, torch.Tensor],
) -> float:
    if set(first) != set(second):
        raise RuntimeError("State keys differ")
    return max(
        (
            float((first[key] - second[key]).abs().max().item())
            for key in first
        ),
        default=0.0,
    )


def output_difference(
    first: torch.Tensor,
    second: torch.Tensor,
) -> Dict[str, float]:
    difference = (first - second).detach().abs()
    return {
        "maximum_absolute_difference": float(difference.max().item()),
        "mean_absolute_difference": float(difference.mean().item()),
    }


def _p0_from_p1(template: Mapping[str, Any]):
    kwargs = dict(template["model_kwargs"])
    # P0 must be constructed from the same random stream as the replication
    # template.  A hard-coded seed=42 preserves the trivially zero initial
    # output but changes the adapter/direct-path diagnostics for seeds
    # 123/2026, producing a false P0/P1 direct mismatch.
    set_seed(int(template["seed"]))
    p0 = M2PRNFFusionPilotVarNet(
        model_variant="prnf_no_need",
        fusion_design="hybrid_direct_residual",
        need_scope="residual",
        residual_scale=0.1,
        initial_need_probability=0.95,
        need_floor=0.25,
        **kwargs,
    )
    return p0


def _initial_controller_audit(
    core: QMaxAuxPDVarNet,
    full: QMaxAuxPDVarNet,
) -> Dict[str, Any]:
    set_seed(42)
    target = torch.randn(2, 36, 24, 24)
    auxiliary_u0 = torch.randn_like(target)
    dc = torch.randn(2, 1, 96, 96).abs()
    dc = dc / dc.square().mean((1, 2, 3), keepdim=True).sqrt().clamp_min(
        1e-8
    )
    raw_rms = torch.tensor([0.5, 1.5])
    availability = torch.ones(2)
    alpha = torch.tensor(0.1)

    core_controller = core.controllers[0]
    full_controller = full.controllers[0]
    with torch.no_grad():
        core_output, core_aux = core_controller(
            target,
            auxiliary_u0,
            torch.zeros_like(dc),
            raw_rms,
            availability,
            alpha,
        )
        full_output, full_aux = full_controller(
            target,
            auxiliary_u0,
            dc,
            raw_rms,
            availability,
            alpha,
        )
    expected_direct = (
        core_aux["q_hat"][:, None, None, None] * alpha * auxiliary_u0
    )
    expected_output = target + expected_direct
    return {
        "detail_gate_core_max_abs_from_one": float(
            (core_aux["detail_gate_mean"] - 1.0).abs().max().item()
        ),
        "detail_gate_full_max_abs_from_one": float(
            (full_aux["detail_gate_mean"] - 1.0).abs().max().item()
        ),
        "alignment_core_max_rms": float(
            core_aux["alignment_to_target_rms"].max().item()
        ),
        "alignment_full_max_rms": float(
            full_aux["alignment_to_target_rms"].max().item()
        ),
        "correction_core_max_rms": float(
            core_aux["correction_to_target_rms"].max().item()
        ),
        "correction_full_max_rms": float(
            full_aux["correction_to_target_rms"].max().item()
        ),
        "u_core_max_abs_from_u0": float(
            core_aux["selected_minus_u0_max_abs"].max().item()
        ),
        "u_full_max_abs_from_u0": float(
            full_aux["selected_minus_u0_max_abs"].max().item()
        ),
        "core_direct_formula_max_abs": float(
            (core_output - expected_output).abs().max().item()
        ),
        "p1_p2_initial_controller_output_max_abs": float(
            (core_output - full_output).abs().max().item()
        ),
        "p1_p2_initial_q_max_abs": float(
            (core_aux["q_hat"] - full_aux["q_hat"]).abs().max().item()
        ),
    }


def _gradient_value(parameter: torch.Tensor) -> float:
    if parameter.grad is None:
        return 0.0
    return float(parameter.grad.detach().float().norm().item())


def _unlock_gradient_snapshot(model: QMaxAuxPDVarNet) -> Dict[str, Any]:
    """Capture gradients at every zero-initialised QMax barrier."""

    detail = {
        f"scale_{index}": _gradient_value(
            controller.detail_head.out.weight
        )
        for index, controller in enumerate(model.controllers)
    }
    alignment = {
        f"scale_{index}": _gradient_value(
            controller.alignment_head.out.weight
        )
        for index, controller in enumerate(model.controllers)
    }
    correction = {
        f"scale_{index}": _gradient_value(
            controller.correction_head.out.weight
        )
        for index, controller in enumerate(model.controllers)
    }
    q_global = {
        f"scale_{index}": _gradient_value(
            controller.reliability.global_out.weight
        )
        for index, controller in enumerate(model.controllers)
    }
    cascade_output = {
        f"cascade_{index}": _gradient_value(
            cascade.regulariser.out_conv.weight
        )
        for index, cascade in enumerate(model.cascades)
    }
    dc_columns = {}
    for name, module in model.named_modules():
        if name.endswith("detail_head.in_proj") or name.endswith(
            "correction_head.in_proj"
        ):
            value = (
                module.weight.grad[:, -1:]
                if module.weight.grad is not None
                else None
            )
            dc_columns[name] = (
                float(value.detach().float().norm().item())
                if value is not None
                else 0.0
            )
    return {
        "cascade_output_projection": cascade_output,
        "detail_output_projection": detail,
        "alignment_output_projection": alignment,
        "correction_output_projection": correction,
        "q_global_output": q_global,
        "dc_input_columns": dc_columns,
    }


def _unlock_parameter_snapshot(
    model: QMaxAuxPDVarNet,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Clone only the small tensors needed to prove optimiser updates."""

    snapshot: Dict[str, Dict[str, torch.Tensor]] = {
        "cascade_output_projection": {
            f"cascade_{index}": (
                cascade.regulariser.out_conv.weight.detach().clone()
            )
            for index, cascade in enumerate(model.cascades)
        },
        "detail_output_projection": {
            f"scale_{index}": (
                controller.detail_head.out.weight.detach().clone()
            )
            for index, controller in enumerate(model.controllers)
        },
        "alignment_output_projection": {
            f"scale_{index}": (
                controller.alignment_head.out.weight.detach().clone()
            )
            for index, controller in enumerate(model.controllers)
        },
        "correction_output_projection": {
            f"scale_{index}": (
                controller.correction_head.out.weight.detach().clone()
            )
            for index, controller in enumerate(model.controllers)
        },
        "dc_input_columns": {},
    }
    for name, module in model.named_modules():
        if name.endswith("detail_head.in_proj") or name.endswith(
            "correction_head.in_proj"
        ):
            snapshot["dc_input_columns"][name] = (
                module.weight[:, -1:].detach().clone()
            )
    return snapshot


def _unlock_parameter_delta(
    before: Mapping[str, Mapping[str, torch.Tensor]],
    after: Mapping[str, Mapping[str, torch.Tensor]],
) -> Dict[str, Dict[str, float]]:
    if set(before) != set(after):
        raise RuntimeError("Unlock parameter groups changed during preflight")
    result: Dict[str, Dict[str, float]] = {}
    for group in before:
        if set(before[group]) != set(after[group]):
            raise RuntimeError(
                f"Unlock parameter keys changed for group {group}"
            )
        result[group] = {
            name: float(
                (after[group][name] - before[group][name])
                .detach()
                .float()
                .abs()
                .max()
                .item()
            )
            for name in before[group]
        }
    return result


def _all_nested_finite(values: Mapping[str, Mapping[str, float]]) -> bool:
    return all(
        math.isfinite(float(value))
        for group in values.values()
        for value in group.values()
    )


def _execution_trace(model) -> tuple[list[str], list[Any]]:
    events: list[str] = []
    handles = []
    for scale, controller in enumerate(model.controllers):
        handles.append(
            controller.register_forward_hook(
                lambda _module, _inputs, _output, scale=scale: events.append(
                    f"aux_scale_{scale}"
                )
            )
        )
    for cascade, block in enumerate(model.cascades):
        handles.append(
            block.register_forward_hook(
                lambda _module, _inputs, _output, cascade=cascade: events.append(
                    f"cascade_done_{cascade}"
                )
            )
        )
    return events, handles


def _historical_p0_audit(checkpoint_path: Path) -> Dict[str, Any]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    required = {"model_state_dict", "config", "code_hashes", "epoch"}
    missing = sorted(required - set(checkpoint))
    if missing:
        raise RuntimeError(
            f"Historical P0 checkpoint missing keys: {missing}"
        )
    config = dict(checkpoint["config"])
    checkpoint_hashes = dict(checkpoint["code_hashes"])
    if config.get("code_hashes") != checkpoint_hashes:
        raise RuntimeError(
            "Historical P0 config/top-level code hashes disagree"
        )
    observed_hashes = {}
    hash_mismatches = {}
    for relative, expected in checkpoint_hashes.items():
        path = PROJECT_ROOT / relative
        if not path.is_file():
            hash_mismatches[relative] = {
                "expected": expected,
                "observed": "missing",
            }
            continue
        observed = sha256_file(path)
        observed_hashes[relative] = observed
        if observed != expected:
            hash_mismatches[relative] = {
                "expected": expected,
                "observed": observed,
            }

    frozen_config = {
        "variant": "prnf_no_need",
        "fusion_design": "hybrid_direct_residual",
        "need_scope": "residual",
        "acceleration": 8,
        "pd_aux_acceleration": 2,
        "seed": 42,
        "num_cascades": 12,
        "chans": 18,
        "sens_chans": 8,
        "pools": 4,
        "sens_pools": 4,
        "controller_chans": 16,
        "initial_aux_alpha": 0.1,
        "initial_gate_probability": 0.95,
        "lambda_rel": 0.05,
        "lambda_rank": 0.02,
        "lambda_residual_gain": 0.2,
        "residual_gain_margin_relative": 0.002,
    }
    config_mismatches = {
        key: {"expected": expected, "observed": config.get(key)}
        for key, expected in frozen_config.items()
        if config.get(key) != expected
    }
    model = M2PRNFFusionPilotVarNet(
        model_variant=str(config["variant"]),
        fusion_design=str(config["fusion_design"]),
        need_scope=str(config["need_scope"]),
        residual_scale=float(config.get("residual_scale", 0.1)),
        num_cascades=int(config["num_cascades"]),
        sens_chans=int(config["sens_chans"]),
        sens_pools=int(config["sens_pools"]),
        chans=int(config["chans"]),
        pools=int(config["pools"]),
        controller_chans=int(config["controller_chans"]),
        initial_aux_alpha=float(config["initial_aux_alpha"]),
        initial_gate_probability=float(
            config["initial_gate_probability"]
        ),
        initial_need_probability=float(
            config.get("initial_need_probability", 0.95)
        ),
        need_floor=float(config.get("need_floor", 0.25)),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    stored_parameter_count = int(config.get("parameter_count", -1))
    parameter_count_matches = parameter_count == stored_parameter_count
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "strict_model_load": True,
        "checkpoint_code_hashes": checkpoint_hashes,
        "observed_current_hashes": observed_hashes,
        "code_hash_mismatches": hash_mismatches,
        "frozen_config": frozen_config,
        "config_mismatches": config_mismatches,
        "parameter_count": parameter_count,
        "stored_parameter_count": stored_parameter_count,
        "parameter_count_matches": parameter_count_matches,
        "passed": (
            not hash_mismatches
            and not config_mismatches
            and parameter_count_matches
        ),
        "interpretation": (
            "Historical performance reference only; its batch-RNG "
            "corruption sequence is not paired to P1/P2."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--init_template", required=True)
    parser.add_argument("--full_clean_manifest", required=True)
    parser.add_argument("--robustness_manifest", required=True)
    parser.add_argument("--condition_manifest", required=True)
    parser.add_argument("--historical_p0_checkpoint", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--corruption_manifest_json", required=True)
    parser.add_argument("--formal_batch_size", type=int, default=4)
    parser.add_argument(
        "--formal_max_gpu_memory_gb", type=float, default=110.0
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    install_amp_diagnostic_quantile_compatibility()

    if args.seed not in ALLOWED_REPLICATION_SEEDS:
        raise ValueError(
            "Replication preflight seed must be one of "
            f"{ALLOWED_REPLICATION_SEEDS}, got {args.seed}"
        )
    if args.formal_batch_size != 4:
        raise ValueError("Frozen replication preflight requires batch=4")
    if not args.amp:
        raise ValueError("Frozen preflight requires mixed precision")
    for name in (
        "metadata_csv",
        "init_template",
        "full_clean_manifest",
        "robustness_manifest",
        "condition_manifest",
        "historical_p0_checkpoint",
    ):
        path = Path(getattr(args, name)).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        setattr(args, name, str(path))
    if not torch.cuda.is_available():
        raise RuntimeError("GPU preflight requires CUDA")
    device = torch.device("cuda")
    set_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()

    template_path = Path(args.init_template)
    historical_p0_audit = _historical_p0_audit(
        Path(args.historical_p0_checkpoint)
    )
    template = torch.load(
        template_path, map_location="cpu", weights_only=False
    )
    if template.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("Initialisation protocol mismatch")
    if int(template.get("seed", -1)) != int(args.seed):
        raise RuntimeError(
            "Initialisation template seed does not match requested "
            f"preflight seed: template={template.get('seed')}, "
            f"requested={args.seed}"
        )
    if template.get("source_hashes") != initialisation_source_hashes(
        PROJECT_ROOT
    ):
        raise RuntimeError(
            "Initialisation template was generated by different source code; "
            "archive the stale template and regenerate it"
        )
    template_audit = dict(template["initialisation_audit"])
    if not template_audit.get("passed"):
        raise RuntimeError("Initialisation template audit did not pass")

    core_cpu = build_from_template(template, "qmax_core")
    full_cpu = build_from_template(template, "qmax_full")
    shared_state_difference = max_state_difference(
        qmax_shared_state(core_cpu), qmax_shared_state(full_cpu)
    )
    dc_columns = qmax_dc_input_columns(full_cpu)
    dc_column_max_abs = max(
        (
            float(value.abs().max().item())
            for value in dc_columns.values()
        ),
        default=float("nan"),
    )
    controller_audit = _initial_controller_audit(core_cpu, full_cpu)

    dataset = IndexedDataset(
        make_dataset(args.metadata_csv, "train", 8, 2)
    )
    sampler = ShapeBucketBatchSampler(
        dataset, args.formal_batch_size, False, args.seed
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    manifest_batches = list(itertools.islice(iter(loader), 4))
    if not manifest_batches:
        raise RuntimeError("Training loader is empty")
    batch = manifest_batches[0]
    kspace, mask, pd, target, indices = prepare_batch(batch, device)
    if pd.shape[0] != args.formal_batch_size:
        raise RuntimeError(
            f"First real batch is {pd.shape[0]}, expected 4"
        )
    pd_masked_kspace = batch["pd_masked_kspace"]
    pd_reconstructed = rss_combine(
        ifft2c(pd_masked_kspace), coil_dim=1
    )
    pd_reconstructed = center_crop(
        pd_reconstructed, pd.shape[-2], pd.shape[-1]
    )
    flip_flags = [
        bool(value) for value in batch["pd_flip_lr"]
    ]
    for index, should_flip in enumerate(flip_flags):
        if should_flip:
            pd_reconstructed[index] = torch.flip(
                pd_reconstructed[index], dims=(-1,)
            )
    pd_reconstruction_max_abs = float(
        (
            pd_reconstructed.float()
            - pd.detach().cpu().float()
        )
        .abs()
        .max()
        .item()
    )
    pd_mask_lines = (
        batch["pd_aux_mask"]
        .reshape(pd.shape[0], -1)
        .float()
        .sum(dim=1)
        .tolist()
    )
    reported_pd_lines = [
        int(value) for value in batch["pd_aux_num_sampled_lines"]
    ]
    reported_pd_actual_r = [
        float(value) for value in batch["pd_aux_actual_R"]
    ]
    pd_provenance = {
        "construction": (
            "RSS magnitude of IFFT of R2 masked multicoil PD k-space, "
            "then dataset crop and metadata-driven left-right flip"
        ),
        "configured_acceleration": [
            int(value) for value in batch["pd_aux_acceleration"]
        ],
        "mask_sampled_lines": pd_mask_lines,
        "reported_sampled_lines": reported_pd_lines,
        "reported_actual_R": reported_pd_actual_r,
        "reconstruction_max_abs_difference": pd_reconstruction_max_abs,
        "sampled_line_counts_match": all(
            int(round(observed)) == reported
            for observed, reported in zip(
                pd_mask_lines, reported_pd_lines
            )
        ),
    }
    negative_sampler = HardNegativeSampler(dataset)
    corruption_config = CorruptionConfig()
    corruption_manifests = {"p0": [], "p1": [], "p2": []}
    corrupt_batches = {}
    for manifest_batch_index, manifest_batch in enumerate(
        manifest_batches, start=1
    ):
        manifest_pd = (
            manifest_batch["pd_aux_image"]
            .to(device, non_blocking=True)
            .float()
        )
        if manifest_pd.ndim == 4:
            manifest_pd = manifest_pd[:, 0]
        manifest_indices = [
            int(value) for value in manifest_batch["sample_idx"]
        ]
        for arm in ("p0", "p1", "p2"):
            corrupted = corrupt_batch_qmax(
                manifest_pd,
                manifest_indices,
                dataset,
                negative_sampler,
                epoch=1,
                global_seed=args.seed,
                config=corruption_config,
                view_index=1,
                occurrence_indices=[0] * len(manifest_indices),
                stream_id="qmax_train_corrupt",
            )
            rows = manifest_rows(corrupted.records)
            for row in rows:
                row["batch_index"] = manifest_batch_index
            corruption_manifests[arm].extend(rows)
            if manifest_batch_index == 1:
                corrupt_batches[arm] = corrupted
    corruption_identical = (
        corruption_manifests["p0"]
        == corruption_manifests["p1"]
        == corruption_manifests["p2"]
    )
    corruption_manifest_path = Path(
        args.corruption_manifest_json
    ).resolve()
    corruption_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    corruption_manifest_path.write_text(
        json.dumps(
            {
                "protocol": DETERMINISTIC_CORRUPTION_PROTOCOL,
                "corruption_config": asdict(corruption_config),
                "identical_across_p0_p1_p2": corruption_identical,
                "p0": corruption_manifests["p0"],
                "p1": corruption_manifests["p1"],
                "p2": corruption_manifests["p2"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Real-data step-0 functional equality is evaluated sequentially to avoid
    # holding three 12-cascade models on the GPU at once.
    real_outputs: Dict[str, torch.Tensor] = {}
    real_q: Dict[str, torch.Tensor] = {}
    real_direct: Dict[str, torch.Tensor] = {}
    execution_traces: Dict[str, Dict[str, Any]] = {}
    final_sampled_residuals: Dict[str, float] = {}
    for arm in ("p1", "p2", "p0"):
        if arm == "p1":
            candidate = build_from_template(template, "qmax_core")
        elif arm == "p2":
            candidate = build_from_template(template, "qmax_full")
        else:
            candidate = _p0_from_p1(template)
        candidate = candidate.to(device).eval()
        events, handles = _execution_trace(candidate)
        try:
            # Run the cross-arm step-0 equality/runtime-hook audit in FP32.
            # Historical P0 computes diagnostic quantiles, and PyTorch does
            # not implement quantile for FP16.  Using FP32 for all three arms
            # also preserves an exact, dtype-matched P0/P1/P2 comparison.
            # The separate four-step trainability audit below still uses the
            # frozen AMP/GradScaler training path.
            with torch.no_grad():
                prediction, auxiliary = candidate(
                    kspace[:1],
                    mask[:1],
                    pd[:1],
                    torch.ones(1, device=device),
                    return_aux=True,
                )
        finally:
            for handle in handles:
                handle.remove()
        real_outputs[arm] = prediction.detach().float().cpu()
        real_q[arm] = auxiliary["q_hat"].detach().float().cpu()
        real_direct[arm] = (
            auxiliary["direct_to_target_rms"].detach().float().cpu()
        )
        final_event = f"cascade_done_{len(candidate.cascades) - 1}"
        last_cascade_position = (
            max(
                index
                for index, event in enumerate(events)
                if event == final_event
            )
            if final_event in events
            else -1
        )
        auxiliary_after_final_cascade = any(
            event.startswith("aux_")
            for event in events[last_cascade_position + 1 :]
        )
        execution_traces[arm] = {
            "events": events,
            "num_auxiliary_calls": sum(
                event.startswith("aux_") for event in events
            ),
            "num_completed_cascades": sum(
                event.startswith("cascade_done_") for event in events
            ),
            "final_event": events[-1] if events else None,
            "auxiliary_after_final_cascade": (
                auxiliary_after_final_cascade
            ),
        }
        if arm in {"p1", "p2"}:
            final_sampled_residuals[arm] = float(
                auxiliary["final_sampled_kspace_residual_max_abs"]
                .max()
                .item()
            )
        del candidate, prediction, auxiliary
        torch.cuda.empty_cache()

    p1_p2_output_difference = output_difference(
        real_outputs["p1"], real_outputs["p2"]
    )
    p1_p2_q_difference = output_difference(
        real_q["p1"], real_q["p2"]
    )
    p1_p2_direct_difference = output_difference(
        real_direct["p1"], real_direct["p2"]
    )
    p0_p1_q_difference = output_difference(
        real_q["p0"], real_q["p1"]
    )
    p0_p1_direct_difference = output_difference(
        real_direct["p0"], real_direct["p1"]
    )
    p0_p1_output_difference = output_difference(
        real_outputs["p0"], real_outputs["p1"]
    )

    # Missing and DC-bypass safety checks use a fresh Full model.
    full = build_from_template(template, "qmax_full").to(device).eval()
    unavailable = torch.zeros(1, device=device)
    with torch.no_grad(), autocast_context(device, args.amp):
        missing_a, missing_aux = full(
            kspace[:1],
            mask[:1],
            pd[:1],
            unavailable,
            return_aux=True,
        )
        missing_b = full(
            kspace[:1],
            mask[:1],
            torch.flip(pd[:1], dims=(-2, -1)),
            unavailable,
        )
        dc_on_missing = full(
            kspace[:1],
            mask[:1],
            pd[:1],
            unavailable,
            dc_zero=False,
        )
        dc_zero_missing = full(
            kspace[:1],
            mask[:1],
            pd[:1],
            unavailable,
            dc_zero=True,
        )
    missing_content_difference = output_difference(
        missing_a.float(), missing_b.float()
    )
    dc_bypass_difference = output_difference(
        dc_on_missing.float(), dc_zero_missing.float()
    )
    missing_direct_max = float(
        missing_aux["direct_to_target_rms"].abs().max().item()
    )
    missing_correction_max = float(
        missing_aux["correction_to_target_rms"].abs().max().item()
    )
    del (
        full,
        missing_a,
        missing_b,
        missing_aux,
        dc_on_missing,
        dc_zero_missing,
    )
    torch.cuda.empty_cache()

    # The frozen step-0 function contains three staged barriers:
    #   step 1 opens each VarNet regulariser output projection;
    #   step 2 can then open the QMax head output projections;
    #   later steps can then propagate into the QMax head input projections.
    # QMax-Full's extra DC columns are one stage deeper and their input is
    # itself state-dependent: the sampled residual is initially zero. A fixed
    # four-step deadline is therefore not a valid hard gate for those columns.
    # Preflight instead proves that real nonzero DC evidence reaches every DC
    # column and that every corresponding parameter requires gradients. The
    # complete one-epoch smoke audit is the hard gate for an actual DC-column
    # optimiser update.
    full = build_from_template(template, "qmax_full").to(device).train()
    q_compatibility_parameter_count = sum(
        parameter.numel()
        for name, parameter in full.named_parameters()
        if ".reliability.spatial_out." in name
        or ".reliability.channel_out." in name
    )
    optimizer = torch.optim.Adam(full.parameters(), lr=3e-4)
    scaler = make_grad_scaler(args.amp)
    corrupt = corrupt_batches["p2"]
    base = pd.shape[0]
    paired_kspace = torch.cat([kspace, kspace], dim=0)
    paired_mask = torch.cat([mask, mask], dim=0)
    paired_pd = torch.cat([pd, corrupt.image], dim=0)
    paired_available = torch.cat(
        [torch.ones_like(corrupt.availability), corrupt.availability],
        dim=0,
    )
    clean_available = torch.ones(base, device=device)
    auxiliary_ramp = 1.0 / 5.0
    gain_ramp = 1.0 / 5.0
    step_losses = []
    unlock_steps = []
    initial_unlock_parameters = _unlock_parameter_snapshot(full)
    final_auxiliary = None
    dc_input_activity: Dict[str, float] = {}
    dc_column_requires_grad: Dict[str, bool] = {}
    dc_activity_handles = []
    for name, module in full.named_modules():
        if name.endswith("detail_head.in_proj") or name.endswith(
            "correction_head.in_proj"
        ):
            dc_input_activity[name] = 0.0
            dc_column_requires_grad[name] = bool(
                module.weight.requires_grad
            )

            def capture_dc_input(
                _module,
                inputs,
                *,
                module_name=name,
            ):
                if len(inputs) != 1 or inputs[0].ndim != 4:
                    raise RuntimeError(
                        f"Unexpected DC input for {module_name}"
                    )
                activity = float(
                    inputs[0][:, -1:]
                    .detach()
                    .float()
                    .abs()
                    .max()
                    .item()
                )
                dc_input_activity[module_name] = max(
                    dc_input_activity[module_name],
                    activity,
                )

            dc_activity_handles.append(
                module.register_forward_pre_hook(capture_dc_input)
            )
    for step in (1, 2, 3, 4):
        before_step = _unlock_parameter_snapshot(full)
        scaler_scale_before = float(scaler.get_scale())
        optimizer.zero_grad(set_to_none=True)

        # This is intentionally identical to the detached clean
        # correction-off reference used by formal epoch-1 training.
        with torch.no_grad(), autocast_context(device, args.amp):
            correction_off_prediction = full(
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

        with autocast_context(device, args.amp):
            prediction, auxiliary = full(
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
            reconstruction = 0.7 * clean_l1 + 0.3 * corrupt_l1

            required_l1 = 0.998 * correction_off_l1
            correction_gain_loss = F.relu(
                clean_l1_per_sample - required_l1
            ).mean()

            logits_clean = auxiliary["q_logits"][:base]
            logits_corrupt = auxiliary["q_logits"][base:]
            clean_bce = F.binary_cross_entropy_with_logits(
                logits_clean,
                torch.ones_like(logits_clean),
            )
            corrupt_targets = (
                corrupt.reliability_target[:, None, :]
                .expand_as(logits_corrupt)
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
            loss = reconstruction + auxiliary_ramp * (
                0.05 * reliability_bce + 0.02 * rank_loss
            ) + gain_ramp * (
                0.2 * correction_gain_loss
            )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Non-finite preflight loss at step {step}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            full.parameters(), 10.0
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise RuntimeError("Non-finite preflight gradient")
        step_losses.append(float(loss.detach().item()))
        gradients = _unlock_gradient_snapshot(full)
        if not _all_nested_finite(gradients):
            raise RuntimeError(
                f"Non-finite unlock gradient at step {step}"
            )
        scaler.step(optimizer)
        scaler.update()
        after_step = _unlock_parameter_snapshot(full)
        step_parameter_delta = _unlock_parameter_delta(
            before_step, after_step
        )
        unlock_steps.append(
            {
                "step": step,
                "loss": float(loss.detach().item()),
                "reconstruction_loss": float(
                    reconstruction.detach().item()
                ),
                "clean_l1": float(clean_l1.detach().item()),
                "corrupt_l1": float(corrupt_l1.detach().item()),
                "reliability_bce": float(
                    reliability_bce.detach().item()
                ),
                "rank_loss": float(rank_loss.detach().item()),
                "correction_gain_loss": float(
                    correction_gain_loss.detach().item()
                ),
                "correction_off_l1_mean": float(
                    correction_off_l1.mean().item()
                ),
                "auxiliary_ramp": auxiliary_ramp,
                "gain_ramp": gain_ramp,
                "gradient_norm_before_clip": float(
                    gradient_norm.detach().float().item()
                ),
                "grad_scaler_scale_before": scaler_scale_before,
                "grad_scaler_scale_after": float(scaler.get_scale()),
                "gradients": gradients,
                "parameter_max_abs_delta": step_parameter_delta,
                "optimizer_changed_cascade_output": any(
                    value > 0.0
                    for value in step_parameter_delta[
                        "cascade_output_projection"
                    ].values()
                ),
            }
        )
        final_auxiliary = auxiliary
    for handle in dc_activity_handles:
        handle.remove()

    final_gradients = unlock_steps[-1]["gradients"]
    detail_gradients = final_gradients["detail_output_projection"]
    alignment_gradients = final_gradients[
        "alignment_output_projection"
    ]
    correction_gradients = final_gradients[
        "correction_output_projection"
    ]
    q_gradients = final_gradients["q_global_output"]
    dc_column_gradients = final_gradients["dc_input_columns"]
    cascade_output_gradients = final_gradients[
        "cascade_output_projection"
    ]
    target_backbone_gradient = cascade_output_gradients["cascade_0"]
    final_unlock_parameters = _unlock_parameter_snapshot(full)
    cumulative_parameter_delta = _unlock_parameter_delta(
        initial_unlock_parameters, final_unlock_parameters
    )
    all_gradients_finite = all(
        _all_nested_finite(step["gradients"])
        for step in unlock_steps
    )
    all_outputs_finite = bool(
        torch.isfinite(final_auxiliary["q_hat"]).all()
        and torch.isfinite(
            final_auxiliary["final_auxiliary_to_target_rms"]
        ).all()
        and torch.isfinite(
            final_auxiliary["final_sampled_kspace_residual_max_abs"]
        ).all()
    )
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3

    checks = {
        "template_audit_passed": bool(template_audit["passed"]),
        "historical_p0_source_and_config_bound": bool(
            historical_p0_audit["passed"]
        ),
        "pd_aux_is_reconstructed_r2_zero_filled_rss": (
            pd_reconstruction_max_abs < 1e-5
            and pd_provenance["sampled_line_counts_match"]
            and all(
                value == 2
                for value in pd_provenance["configured_acceleration"]
            )
        ),
        "p1_p2_shared_max_diff_zero": shared_state_difference == 0.0,
        "p2_dc_columns_zero": dc_column_max_abs == 0.0,
        "initial_detail_gate_one": (
            controller_audit["detail_gate_core_max_abs_from_one"] == 0.0
            and controller_audit[
                "detail_gate_full_max_abs_from_one"
            ]
            == 0.0
        ),
        "initial_alignment_zero": (
            controller_audit["alignment_core_max_rms"] == 0.0
            and controller_audit["alignment_full_max_rms"] == 0.0
        ),
        "initial_correction_zero": (
            controller_audit["correction_core_max_rms"] == 0.0
            and controller_audit["correction_full_max_rms"] == 0.0
        ),
        "initial_u_equals_u0": (
            controller_audit["u_core_max_abs_from_u0"] < 1e-6
            and controller_audit["u_full_max_abs_from_u0"] < 1e-6
        ),
        "initial_direct_formula_exact": (
            controller_audit["core_direct_formula_max_abs"] < 1e-6
        ),
        "p1_p2_output_equivalent": (
            p1_p2_output_difference["maximum_absolute_difference"] < 1e-5
        ),
        "p1_p2_q_equal": (
            p1_p2_q_difference["maximum_absolute_difference"] == 0.0
        ),
        "p1_p2_direct_equal": (
            p1_p2_direct_difference["maximum_absolute_difference"] < 1e-6
        ),
        "p0_p1_q_equal": (
            p0_p1_q_difference["maximum_absolute_difference"] == 0.0
        ),
        "p0_p1_direct_equal": (
            p0_p1_direct_difference["maximum_absolute_difference"] < 1e-6
        ),
        "p0_p1_initial_output_equivalent": (
            p0_p1_output_difference["maximum_absolute_difference"] < 1e-5
        ),
        "corruption_identical": corruption_identical,
        "missing_direct_zero": missing_direct_max == 0.0,
        "missing_correction_zero": missing_correction_max == 0.0,
        "missing_content_invariant": (
            missing_content_difference["maximum_absolute_difference"] == 0.0
        ),
        "dc_cannot_bypass_m": (
            dc_bypass_difference["maximum_absolute_difference"] == 0.0
        ),
        "detail_projection_gradients_nonzero": all(
            value > 0.0 for value in detail_gradients.values()
        ),
        "alignment_projection_gradients_nonzero": all(
            value > 0.0 for value in alignment_gradients.values()
        ),
        "correction_projection_gradients_nonzero": all(
            value > 0.0 for value in correction_gradients.values()
        ),
        "q_head_gradients_nonzero": all(
            value > 0.0 for value in q_gradients.values()
        ),
        "target_backbone_gradient_nonzero": all(
            value > 0.0
            for value in cascade_output_gradients.values()
        ),
        "dc_evidence_reaches_all_input_columns": (
            len(dc_input_activity) == 8
            and all(value > 0.0 for value in dc_input_activity.values())
        ),
        "dc_evidence_columns_require_grad": (
            len(dc_column_requires_grad) == 8
            and all(dc_column_requires_grad.values())
        ),
        "cascade_output_parameters_updated": all(
            value > 0.0
            for value in cumulative_parameter_delta[
                "cascade_output_projection"
            ].values()
        ),
        "detail_projection_parameters_updated": all(
            value > 0.0
            for value in cumulative_parameter_delta[
                "detail_output_projection"
            ].values()
        ),
        "alignment_projection_parameters_updated": all(
            value > 0.0
            for value in cumulative_parameter_delta[
                "alignment_output_projection"
            ].values()
        ),
        "correction_projection_parameters_updated": all(
            value > 0.0
            for value in cumulative_parameter_delta[
                "correction_output_projection"
            ].values()
        ),
        "forward_backward_finite": (
            all_gradients_finite and all_outputs_finite
        ),
        "memory_within_limit": peak_memory
        <= args.formal_max_gpu_memory_gb,
        "no_auxiliary_module_after_final_cascade_update": all(
            trace["num_auxiliary_calls"] == 48
            and trace["num_completed_cascades"] == 12
            and trace["final_event"] == "cascade_done_11"
            and not trace["auxiliary_after_final_cascade"]
            for arm, trace in execution_traces.items()
            if arm in {"p1", "p2"}
        ),
        "soft_dc_output_contract_finite": all(
            math.isfinite(value)
            for value in final_sampled_residuals.values()
        ),
    }
    result = {
        "status": "passed" if all(checks.values()) else "failed",
        "amp_diagnostic_quantile_compatibility": (
            "detached FP16/BF16 need percentiles cast to FP32 only"
        ),
        "protocol_version": (
            "QMax-StageA-multiseed-preflight-v7-staged-unlock"
        ),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "amp": bool(args.amp),
        "checks": checks,
        "template_audit": template_audit,
        "historical_p0_audit": historical_p0_audit,
        "pd_aux_provenance": pd_provenance,
        "core_full_shared_state_max_diff": shared_state_difference,
        "p2_dc_input_column_max_abs": dc_column_max_abs,
        "initial_controller_audit": controller_audit,
        "real_initial_function_audit": {
            "p1_p2_output": p1_p2_output_difference,
            "p1_p2_q": p1_p2_q_difference,
            "p1_p2_direct": p1_p2_direct_difference,
            "p0_p1_q": p0_p1_q_difference,
            "p0_p1_direct": p0_p1_direct_difference,
            "p0_p1_output": p0_p1_output_difference,
        },
        "execution_trace_audit": execution_traces,
        "final_output_dc_contract": {
            "type": "learned soft-DC VarNet; no final hard projection",
            "strict_sampled_kspace_consistency_claimed": False,
            "initial_final_sampled_residual_max_abs": (
                final_sampled_residuals
            ),
            "post_optimisation_step_full_sampled_residual_max_abs": float(
                final_auxiliary["final_sampled_kspace_residual_max_abs"]
                .max()
                .item()
            ),
            "post_final_cascade_auxiliary_module": False,
        },
        "safety": {
            "missing_direct_max": missing_direct_max,
            "missing_correction_max": missing_correction_max,
            "missing_content_difference": missing_content_difference,
            "dc_bypass_difference": dc_bypass_difference,
        },
        "gradient_audit": {
            "design": (
                "four real AMP/GradScaler/Adam steps using the exact "
                "formal epoch-1 normalized reconstruction, reliability, "
                "ranking, and correction-gain losses"
            ),
            "step_losses": step_losses,
            "steps": unlock_steps,
            "detail_output_projection": detail_gradients,
            "alignment_output_projection": alignment_gradients,
            "correction_output_projection": correction_gradients,
            "q_global_output": q_gradients,
            "cascade_output_projection": cascade_output_gradients,
            "dc_input_columns": dc_column_gradients,
            "dc_input_activity_max_abs": dc_input_activity,
            "dc_column_requires_grad": dc_column_requires_grad,
            "dc_column_update_within_four_steps": {
                key: value > 0.0
                for key, value in cumulative_parameter_delta[
                    "dc_input_columns"
                ].items()
            },
            "dc_column_update_within_four_steps_is_hard_gate": False,
            "dc_column_update_hard_gate": (
                "external one-epoch qmax_full smoke audit"
            ),
            "target_backbone": target_backbone_gradient,
            "cumulative_parameter_max_abs_delta": (
                cumulative_parameter_delta
            ),
            "all_finite": all_gradients_finite,
        },
        "resource_audit": {
            "anatomical_batch": args.formal_batch_size,
            "forward_batch": int(paired_pd.shape[0]),
            "peak_gpu_memory_gb": peak_memory,
            "maximum_gpu_memory_gb": args.formal_max_gpu_memory_gb,
            "visible_gpu_total_gb": (
                torch.cuda.get_device_properties(0).total_memory / 1024**3
            ),
            "q_compatibility_only_parameter_count": (
                q_compatibility_parameter_count
            ),
            "q_compatibility_outputs_used_for_fusion": False,
        },
        "corruption_manifest": {
            "path": str(corruption_manifest_path),
            "sha256": sha256_file(corruption_manifest_path),
            "identical_across_p0_p1_p2": corruption_identical,
        },
        "input_hashes": {
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
            "historical_p0_checkpoint": historical_p0_audit[
                "checkpoint_sha256"
            ],
        },
        "code_hashes": code_hashes(PROJECT_ROOT),
        "runtime_versions": runtime_versions(),
    }
    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
