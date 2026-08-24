#!/usr/bin/env python3
from __future__ import annotations

"""Strict real-data/GPU preflight for QMax-CompactSwin Stage B."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_qmax_stage_b_init import (  # noqa: E402
    PROTOCOL_VERSION,
    build_stage_b_from_template,
    maximum_difference,
)
from scripts.qmax_stage_b_versioning import (  # noqa: E402
    AUDIT_VERSION,
    RUNTIME_VERSION,
    STRUCTURE_VERSION,
    manifest_digest,
    sha256_file,
    stage_b_audit_hashes,
    stage_b_runtime_hashes,
    stage_b_structure_hashes,
)
from scripts.qmax_common import (  # noqa: E402
    IndexedDataset,
    ShapeBucketBatchSampler,
    autocast_context,
    l1_per_sample,
    make_dataset,
    prepare_batch,
    runtime_versions,
    set_seed,
)
from src.fft_utils import center_crop  # noqa: E402
from src.m2_prnf_qmax_compactswin_varnet import (  # noqa: E402
    COMPACTSWIN_LAYER_SCALE_INIT,
    COMPACTSWIN_WINDOW_SIZE,
    CompactSwinStage,
)
from src.m2_prnf_qmax_varnet import (  # noqa: E402
    QMAX_VARIANTS,
    qmax_dc_input_columns,
    qmax_shared_state,
)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _all_finite_gradients(model: torch.nn.Module) -> bool:
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    return bool(gradients) and all(
        bool(torch.isfinite(gradient).all()) for gradient in gradients
    )


def _gradient_max(model: torch.nn.Module, pattern: str) -> float:
    values = [
        float(parameter.grad.detach().abs().max().item())
        for name, parameter in model.named_parameters()
        if pattern in name and parameter.grad is not None
    ]
    return max(values, default=0.0)


def _temporarily_activate_out_convs(model: torch.nn.Module):
    saved = []
    with torch.no_grad():
        for cascade in model.cascades:
            module = cascade.regulariser.out_conv
            saved.append(
                (
                    module,
                    module.weight.detach().clone(),
                    module.bias.detach().clone(),
                )
            )
            module.weight.normal_(mean=0.0, std=1e-3)
            module.bias.zero_()
    return saved


def _restore_out_convs(saved) -> None:
    with torch.no_grad():
        for module, weight, bias in saved:
            module.weight.copy_(weight)
            module.bias.copy_(bias)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--stage_b_init_template", required=True)
    parser.add_argument(
        "--qmax_variant", required=True, choices=sorted(QMAX_VARIANTS)
    )
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_gpu_memory_gb", type=float, default=110.0)
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    if args.seed != 42 or args.batch_size != 4:
        raise ValueError("Frozen Stage B requires seed=42 and batch=4")
    if not args.amp:
        raise ValueError("Frozen Stage B requires AMP")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-B preflight requires CUDA")
    device = torch.device("cuda")
    set_seed(args.seed)

    paths = {}
    for name in (
        "metadata_csv",
        "stage_b_init_template",
    ):
        path = Path(getattr(args, name)).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        paths[name] = path

    stage_b_template = torch.load(
        paths["stage_b_init_template"],
        map_location="cpu",
        weights_only=False,
    )
    if stage_b_template.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("Stage-B template protocol mismatch")
    if set(stage_b_template.get("variants", [])) != set(QMAX_VARIANTS):
        raise RuntimeError("Stage-B template does not contain Core and Full")
    structure_hashes = stage_b_structure_hashes(PROJECT_ROOT)
    runtime_hashes = stage_b_runtime_hashes(PROJECT_ROOT)
    audit_hashes = stage_b_audit_hashes(PROJECT_ROOT)
    if stage_b_template.get("structure_hashes") != structure_hashes:
        raise RuntimeError(
            "Stage-B structural code differs from the random-init template"
        )
    if stage_b_template.get("structure_digest") != manifest_digest(
        structure_hashes
    ):
        raise RuntimeError("Stage-B template structure digest differs")

    qmax_variant = str(args.qmax_variant)
    core = build_stage_b_from_template(
        stage_b_template, "qmax_core"
    ).cpu()
    full = build_stage_b_from_template(
        stage_b_template, "qmax_full"
    ).cpu()
    shared_keys = sorted(qmax_shared_state(core))
    shared_difference = maximum_difference(
        core.state_dict(), full.state_dict(), shared_keys
    )
    q_keys = [
        key for key in shared_keys if ".reliability." in key
    ]
    q_difference = max(
        (
            float(
                (
                    core.state_dict()[key] - full.state_dict()[key]
                )
                .abs()
                .max()
                .item()
            )
            for key in q_keys
        ),
        default=0.0,
    )
    full_dc_columns = qmax_dc_input_columns(full)
    full_dc_columns_max_abs = max(
        (
            float(value.abs().max().item())
            for value in full_dc_columns.values()
        ),
        default=float("nan"),
    )
    # Isolated attention audit makes trainability observable despite the
    # reconstruction output projections being zero at model step 0.
    isolated = CompactSwinStage(
        72,
        144,
        heads=6,
        blocks=2,
        window_size=8,
        activation_checkpointing=True,
        capture_padding_audit=True,
    ).to(device)
    isolated.train()
    isolated_input = torch.randn(
        1, 72, 39, 41, device=device, requires_grad=True
    )
    # This isolated test checks structural gradient connectivity.
    # Run it in FP32 to avoid AMP underflow through the 1e-3 LayerScale.
    isolated_output = isolated(isolated_input.float())
    isolated_loss = isolated_output.float().square().mean()
    isolated_loss.backward()
    isolated_audit = {
        "input_hw": [39, 41],
        "output_hw": list(isolated_output.shape[-2:]),
        "padding": dict(isolated.last_padding_audit),
        "qkv_gradient_max": _gradient_max(isolated, "attention.qkv"),
        "ffn_gradient_max": _gradient_max(isolated, "ffn_project"),
        "layer_scale_gradient_max": _gradient_max(
            isolated, "layer_scale_"
        ),
        "input_gradient_finite": bool(
            torch.isfinite(isolated_input.grad).all()
        ),
    }
    del isolated, isolated_input, isolated_output, isolated_loss
    torch.cuda.empty_cache()

    dataset = IndexedDataset(make_dataset(args.metadata_csv, "train", 8, 2))
    sampler = ShapeBucketBatchSampler(
        dataset, args.batch_size, False, args.seed
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    batch = next(iter(loader))
    kspace, mask, pd, target, _ = prepare_batch(batch, device)
    if pd.shape[0] != args.batch_size:
        raise RuntimeError(
            f"First real batch has {pd.shape[0]} samples, expected 4"
        )

    step0_available = torch.ones(1, device=device)
    core = core.to(device).eval()
    with torch.no_grad(), autocast_context(device, args.amp):
        core_output, core_auxiliary = core(
            kspace[:1],
            mask[:1],
            pd[:1],
            step0_available,
            return_aux=True,
        )
    core_output = core_output.detach().float().cpu()
    core_q = core_auxiliary["q"].detach().float().cpu()
    core_direct = (
        core_auxiliary["direct_to_target_rms"].detach().float().cpu()
    )
    core = core.cpu()
    torch.cuda.empty_cache()

    full = full.to(device).eval()
    with torch.no_grad(), autocast_context(device, args.amp):
        full_output, full_auxiliary = full(
            kspace[:1],
            mask[:1],
            pd[:1],
            step0_available,
            return_aux=True,
        )
    full_output = full_output.detach().float().cpu()
    full_q = full_auxiliary["q"].detach().float().cpu()
    full_direct = (
        full_auxiliary["direct_to_target_rms"].detach().float().cpu()
    )
    full = full.cpu()
    initial_function_audit = {
        "output_max_abs_difference": float(
            (core_output - full_output).abs().max().item()
        ),
        "q_max_abs_difference": float(
            (core_q - full_q).abs().max().item()
        ),
        "direct_rms_max_abs_difference": float(
            (core_direct - full_direct).abs().max().item()
        ),
    }
    del core, full, core_auxiliary, full_auxiliary
    torch.cuda.empty_cache()

    b1 = build_stage_b_from_template(
        stage_b_template, qmax_variant
    ).to(device)
    b1.train()
    b1.set_padding_audit(True)
    torch.cuda.reset_peak_memory_stats()
    paired_kspace = torch.cat([kspace, kspace], dim=0)
    paired_mask = torch.cat([mask, mask], dim=0)
    paired_pd = torch.cat([pd, torch.roll(pd, 1, dims=-1)], dim=0)
    paired_available = torch.ones(2 * pd.shape[0], device=device)
    with autocast_context(device, args.amp):
        prediction, auxiliary = b1(
            paired_kspace,
            paired_mask,
            paired_pd,
            paired_available,
            return_aux=True,
        )
        prediction = center_crop(
            prediction.float(), target.shape[-2], target.shape[-1]
        )
        reconstruction = l1_per_sample(
            prediction[: pd.shape[0]], target
        ).mean()
        q_logits = auxiliary["q_logits"]
        labels = torch.cat(
            [
                torch.ones_like(q_logits[: pd.shape[0]]),
                torch.zeros_like(q_logits[pd.shape[0] :]),
            ],
            dim=0,
        )
        loss = reconstruction + 0.05 * F.binary_cross_entropy_with_logits(
            q_logits, labels
        )
    loss.backward()
    peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
    full_gradient_audit = {
        "loss": float(loss.detach().item()),
        "all_present_gradients_finite": _all_finite_gradients(b1),
        "q_head_gradient_max": _gradient_max(b1, ".reliability."),
        "target_out_gradient_max": _gradient_max(b1, ".out_conv."),
        "prediction_finite": bool(torch.isfinite(prediction).all()),
        "auxiliary_finite": all(
            bool(torch.isfinite(value).all())
            for value in auxiliary.values()
            if isinstance(value, torch.Tensor)
        ),
    }
    b1.zero_grad(set_to_none=True)

    b1.eval()
    with torch.no_grad(), autocast_context(device, args.amp):
        normal = b1(kspace, mask, pd, torch.ones_like(pd[:, 0, 0]))
        q1 = b1(
            kspace[:1],
            mask[:1],
            pd[:1],
            torch.ones(1, device=device),
            reliability_override=1.0,
        )
        constant = b1(
            kspace[:1],
            mask[:1],
            pd[:1],
            torch.ones(1, device=device),
            reliability_override=0.5,
        )
    normal = center_crop(
        normal.float(), target.shape[-2], target.shape[-1]
    )
    initial_architecture = b1.architecture_audit()

    # Activate only temporary output projections so missing invariance is not
    # a vacuous consequence of the canonical zero-output initialisation.
    saved_out = _temporarily_activate_out_convs(b1)
    with torch.no_grad(), autocast_context(device, args.amp):
        missing_zero = b1(
            kspace[:1],
            mask[:1],
            torch.zeros_like(pd[:1]),
            torch.zeros(1, device=device),
        )
        missing_noise = b1(
            kspace[:1],
            mask[:1],
            torch.randn_like(pd[:1]),
            torch.zeros(1, device=device),
        )
    _restore_out_convs(saved_out)
    missing_difference = float(
        (missing_zero - missing_noise).abs().max().item()
    )

    layer_scale_expected = float(
        torch.tensor(COMPACTSWIN_LAYER_SCALE_INIT).item()
    )
    checks = {
        "paired_core_full_shared_parameter_max_diff_zero": (
            shared_difference == 0.0
        ),
        "paired_core_full_q_parameter_max_diff_zero": q_difference == 0.0,
        "q_parameter_keys_present": len(q_keys) > 0,
        "full_dc_input_columns_zero": full_dc_columns_max_abs == 0.0,
        "full_dc_input_column_count_exact": len(full_dc_columns) == 8,
        "core_full_initial_output_equivalent": (
            initial_function_audit["output_max_abs_difference"] < 1e-5
        ),
        "core_full_initial_q_identical": (
            initial_function_audit["q_max_abs_difference"] == 0.0
        ),
        "core_full_initial_direct_equivalent": (
            initial_function_audit["direct_rms_max_abs_difference"] < 1e-6
        ),
        "layer_scale_exact": (
            initial_architecture["layer_scale_min"]
            == layer_scale_expected
            and initial_architecture["layer_scale_max"]
            == layer_scale_expected
        ),
        "window_size_exact": (
            initial_architecture["window_size"]
            == COMPACTSWIN_WINDOW_SIZE
        ),
        "activation_checkpointing_enabled": bool(
            initial_architecture["activation_checkpointing"]
        ),
        "isolated_stage_shape_preserved": (
            isolated_audit["output_hw"] == isolated_audit["input_hw"]
        ),
        "isolated_stage_padded_to_window": (
            all(
                value % COMPACTSWIN_WINDOW_SIZE == 0
                for value in isolated_audit["padding"]["padded_hw"]
            )
        ),
        "isolated_attention_gradient_nonzero": (
            isolated_audit["qkv_gradient_max"] > 0.0
        ),
        "isolated_ffn_gradient_nonzero": (
            isolated_audit["ffn_gradient_max"] > 0.0
        ),
        "isolated_layer_scale_gradient_nonzero": (
            isolated_audit["layer_scale_gradient_max"] > 0.0
        ),
        "full_forward_backward_finite": (
            full_gradient_audit["all_present_gradients_finite"]
            and full_gradient_audit["prediction_finite"]
            and full_gradient_audit["auxiliary_finite"]
        ),
        "q_head_gradient_nonzero": (
            full_gradient_audit["q_head_gradient_max"] > 0.0
        ),
        "target_output_gradient_nonzero": (
            full_gradient_audit["target_out_gradient_max"] > 0.0
        ),
        "missing_pd_content_invariant": missing_difference == 0.0,
        "q_override_interfaces_callable": (
            bool(torch.isfinite(q1).all())
            and bool(torch.isfinite(constant).all())
        ),
        "peak_gpu_memory_within_limit": (
            peak_memory_gb < args.max_gpu_memory_gb
        ),
        "normal_output_finite": bool(torch.isfinite(normal).all()),
        "normal_output_shape_matches_target": (
            tuple(normal.shape[-2:]) == tuple(target.shape[-2:])
        ),
    }
    report = {
        "protocol_version": "QMax-StageB-independent-preflight-v1",
        "structure_version": STRUCTURE_VERSION,
        "runtime_version": RUNTIME_VERSION,
        "audit_version": AUDIT_VERSION,
        "status": "passed" if all(checks.values()) else "failed",
        "qmax_variant": qmax_variant,
        "checks": checks,
        "shared_parameter_max_diff": shared_difference,
        "shared_q_max_diff": q_difference,
        "full_dc_columns_max_abs": full_dc_columns_max_abs,
        "num_full_dc_columns": len(full_dc_columns),
        "initial_function_audit": initial_function_audit,
        "num_shared_keys": len(shared_keys),
        "num_shared_q_keys": len(q_keys),
        "isolated_compactswin_stage_audit": isolated_audit,
        "architecture_audit": initial_architecture,
        "full_gradient_audit": full_gradient_audit,
        "missing_output_max_abs_difference": missing_difference,
        "peak_gpu_memory_gb": peak_memory_gb,
        "max_gpu_memory_gb": args.max_gpu_memory_gb,
        "input_hashes": {
            name: sha256_file(path) for name, path in paths.items()
        },
        "structure_hashes": structure_hashes,
        "structure_digest": manifest_digest(structure_hashes),
        "runtime_tool_hashes": runtime_hashes,
        "runtime_tool_digest": manifest_digest(runtime_hashes),
        "audit_tool_hashes": audit_hashes,
        "audit_tool_digest": manifest_digest(audit_hashes),
        "runtime_versions": runtime_versions(),
    }
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if report["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
