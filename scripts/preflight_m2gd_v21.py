#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.auxiliary_corruptions_v21 import (
    CorruptionConfig,
    border_only,
    border_reliability_target,
    paired_discrimination_loss,
    ranking_margin_by_scale,
    scale_targets_from_base,
    shift_reliability_target,
    translate_nonwrapping,
    wrong_slice_reliability_target,
)
from src.dataset_paired_multicoil_aux_pd_r2 import (
    PairedMulticoilAuxPDToPDFSDataset,
)
from src.fft_utils import center_crop
from src.m2gd_v21_auxiliary_varnet import (
    M2GDv21AuxPDVarNet,
    load_m2gd_v2_for_v21,
)
from src.m2gd_v2_auxiliary_varnet import M2GDv2AuxPDVarNet


def load_checkpoint(path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def state_dict_from_checkpoint(checkpoint):
    for key in ("model_state_dict", "model", "state_dict", "net", "network"):
        value = checkpoint.get(key) if isinstance(checkpoint, dict) else None
        if isinstance(value, dict):
            return value
    raise RuntimeError("No model state dict in checkpoint.")


def prepare_batch(batch, device):
    kspace = batch["pdfs_masked_kspace"].to(device)
    kspace = torch.view_as_real(kspace).float()
    mask = batch["mask"].to(device).bool()[:, None, None, :, None]
    pd_aux = batch["pd_aux_image"].to(device).float()
    target = batch["pdfs_target_raw"].to(device).float()
    if pd_aux.ndim == 4:
        pd_aux = pd_aux[:, 0]
    if target.ndim == 4:
        target = target[:, 0]
    return kspace, mask, pd_aux, target


def corruption_unit_tests(device) -> dict:
    config = CorruptionConfig()
    config.validate()
    image = torch.arange(64 * 64, dtype=torch.float32, device=device).reshape(64, 64)

    shifted = {}
    for mode in ("reflect", "replicate", "zero"):
        shifted[mode] = translate_nonwrapping(image, 4, -4, mode)
        assert shifted[mode].shape == image.shape
    assert shift_reliability_target(4, config) == 0.35
    # The label is a function of severity only, never padding mode.
    target_by_padding = {
        mode: shift_reliability_target(4, config)
        for mode in shifted
    }
    assert len(set(target_by_padding.values())) == 1

    border_results = {}
    for mode in ("reflect", "replicate", "zero"):
        result = border_only(image, 8, mode)
        border_results[mode] = result
        assert torch.equal(result[8:-8, 8:-8], image[8:-8, 8:-8])
    assert border_reliability_target(8, config) == 0.85

    # Verify zero padding does not wrap content to the opposite side.
    impulse = torch.zeros(32, 32, device=device)
    impulse[16, 30] = 1.0
    zero_shifted = translate_nonwrapping(impulse, 0, 4, "zero")
    if float(zero_shifted[:, :4].abs().sum().item()) != 0.0:
        raise RuntimeError("Zero-padded translation wrapped content circularly.")

    target_matrix = {
        "clean": scale_targets_from_base(1.0, config, False),
        **{
            f"shift{magnitude}": scale_targets_from_base(
                shift_reliability_target(magnitude, config), config, True
            )
            for magnitude in (2, 4, 8)
        },
        "border8": scale_targets_from_base(
            border_reliability_target(8, config), config, True
        ),
        **{
            f"wrong_slice_delta_{delta:.3f}": scale_targets_from_base(
                wrong_slice_reliability_target(delta), config, True
            )
            for delta in (0.05, 0.10, 0.20)
        },
        "wrong_patient": scale_targets_from_base(
            config.reliability_wrong_patient, config, False
        ),
        "missing": scale_targets_from_base(
            config.reliability_missing, config, False
        ),
    }
    target_tensors = {
        key: torch.tensor(value, device=device, dtype=torch.float32)
        for key, value in target_matrix.items()
    }
    margin_matrix = {
        "shift2": ranking_margin_by_scale(
            target_tensors["shift2"],
            {"condition": "shift", "magnitude_linf": 2},
        ),
        "shift4": ranking_margin_by_scale(
            target_tensors["shift4"],
            {"condition": "shift", "magnitude_linf": 4},
        ),
        "shift8": ranking_margin_by_scale(
            target_tensors["shift8"],
            {"condition": "shift", "magnitude_linf": 8},
        ),
        "wrong_slice_delta_0.050": ranking_margin_by_scale(
            target_tensors["wrong_slice_delta_0.050"],
            {"condition": "wrong_slice", "delta_z_norm": 0.05},
        ),
        "wrong_slice_delta_0.100": ranking_margin_by_scale(
            target_tensors["wrong_slice_delta_0.100"],
            {"condition": "wrong_slice", "delta_z_norm": 0.10},
        ),
        "wrong_slice_delta_0.200": ranking_margin_by_scale(
            target_tensors["wrong_slice_delta_0.200"],
            {"condition": "wrong_slice", "delta_z_norm": 0.20},
        ),
    }
    shift_means = [float(margin_matrix[key].mean()) for key in ("shift2", "shift4", "shift8")]
    wrong_slice_means = [
        float(margin_matrix[key].mean())
        for key in (
            "wrong_slice_delta_0.050",
            "wrong_slice_delta_0.100",
            "wrong_slice_delta_0.200",
        )
    ]
    if not shift_means[0] < shift_means[1] < shift_means[2]:
        raise RuntimeError(f"Shift margins are not severity-aware: {shift_means}")
    if not wrong_slice_means[0] < wrong_slice_means[1] < wrong_slice_means[2]:
        raise RuntimeError(
            f"Wrong-slice margins are not distance-aware: {wrong_slice_means}"
        )

    border_target = target_tensors["border8"][None, :]
    q_clean = torch.ones(1, 2, 4, device=device)
    q_border_aligned = border_target[:, None, :].expand_as(q_clean)
    aligned_loss, _ = paired_discrimination_loss(
        q_clean,
        q_border_aligned,
        border_target,
        [{"condition": "border"}],
    )
    conflicting_loss, _ = paired_discrimination_loss(
        q_clean,
        q_clean,
        border_target,
        [{"condition": "border"}],
    )
    if float(aligned_loss.item()) != 0.0 or float(conflicting_loss.item()) <= 0.0:
        raise RuntimeError(
            "Border target-aligned loss unit test failed: "
            f"aligned={aligned_loss.item()}, equal_q={conflicting_loss.item()}"
        )

    return {
        "padding_target_independence": True,
        "border_central_content_unchanged": True,
        "no_circular_wrap": True,
        "shift4_target": target_by_padding,
        "border8_target": border_reliability_target(8, config),
        "per_scale_target_matrix": {
            key: [float(value) for value in values]
            for key, values in target_matrix.items()
        },
        "per_scale_ranking_margin_matrix": {
            key: [float(value) for value in values.detach().cpu()]
            for key, values in margin_matrix.items()
        },
        "border_target_aligned_loss": float(aligned_loss.item()),
        "border_equal_q_conflict_detection_loss": float(
            conflicting_loss.item()
        ),
    }


def make_v2(config, device):
    return M2GDv2AuxPDVarNet(
        num_cascades=int(config.get("num_cascades", 12)),
        sens_chans=int(config.get("sens_chans", 8)),
        sens_pools=int(config.get("sens_pools", 4)),
        chans=int(config.get("chans", 18)),
        pools=int(config.get("pools", 4)),
        mask_center=True,
        initial_aux_alpha=float(config.get("initial_aux_alpha", 0.1)),
        initial_gate_probability=float(config.get("initial_gate_probability", 0.99)),
    ).to(device)


def make_v21(config, device, initial_gate_probability):
    return M2GDv21AuxPDVarNet(
        num_cascades=int(config.get("num_cascades", 12)),
        sens_chans=int(config.get("sens_chans", 8)),
        sens_pools=int(config.get("sens_pools", 4)),
        chans=int(config.get("chans", 18)),
        pools=int(config.get("pools", 4)),
        mask_center=True,
        initial_aux_alpha=float(config.get("initial_aux_alpha", 0.1)),
        initial_gate_probability=initial_gate_probability,
    ).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--m2gd_v2_checkpoint", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--initial_gate_probability", type=float, default=0.7436)
    parser.add_argument("--max_initial_relative_difference", type=float, default=0.02)
    parser.add_argument("--acceleration", type=int, default=8)
    parser.add_argument("--pd_aux_acceleration", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Preflight must run on a GPU compute node.")
    print("GPU:", torch.cuda.get_device_name(0))
    unit_results = corruption_unit_tests(device)

    dataset = PairedMulticoilAuxPDToPDFSDataset(
        metadata_csv=args.metadata_csv,
        split="val",
        pdfs_acceleration=args.acceleration,
        pd_aux_acceleration=args.pd_aux_acceleration,
        slices_per_patient=None,
        edge_weight=1.0,
    )
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))
    kspace, mask, pd_aux, target = prepare_batch(batch, device)

    checkpoint_path = Path(args.m2gd_v2_checkpoint)
    checkpoint = load_checkpoint(checkpoint_path, "cpu")
    if int(checkpoint.get("epoch", -1)) != 5:
        raise RuntimeError(
            "V2.1 Stage A must branch from the audited V2 Epoch-5 checkpoint."
        )
    config = checkpoint.get("config", {})
    expected_source_config = {
        "acceleration": args.acceleration,
        "pd_aux_acceleration": args.pd_aux_acceleration,
        "curriculum": "smoke5",
        "num_cascades": 12,
        "pools": 4,
        "chans": 18,
    }
    provenance_mismatches = {
        key: {"expected": expected, "observed": config.get(key)}
        for key, expected in expected_source_config.items()
        if config.get(key) != expected
    }
    if provenance_mismatches:
        raise RuntimeError(
            "V2 source checkpoint provenance mismatch: "
            f"{provenance_mismatches}"
        )
    source_state = state_dict_from_checkpoint(checkpoint)
    if source_state and all(key.startswith("module.") for key in source_state):
        source_state = {key[len("module."):]: value for key, value in source_state.items()}

    # Run the audited V2 reference and release it before allocating V2.1.
    v2 = make_v2(config, device)
    v2.load_state_dict(source_state, strict=True)
    v2.eval()
    with torch.no_grad():
        reference = v2(
            pdfs_masked_kspace=kspace,
            mask=mask,
            pd_aux_image=pd_aux,
            pd_available=torch.ones(1, device=device),
        )
        reference = center_crop(reference, target.shape[-2], target.shape[-1]).cpu()
    del v2
    torch.cuda.empty_cache()

    model = make_v21(config, device, args.initial_gate_probability)
    transfer_report = load_m2gd_v2_for_v21(
        model, checkpoint_path, map_location=device
    )
    model.eval()
    availability_one = torch.ones(1, device=device)
    with torch.no_grad():
        v21_prediction, clean_aux = model(
            pdfs_masked_kspace=kspace,
            mask=mask,
            pd_aux_image=pd_aux,
            pd_available=availability_one,
            return_aux=True,
        )
        v21_prediction = center_crop(
            v21_prediction, target.shape[-2], target.shape[-1]
        ).cpu()

    expected_q_shape = (int(config.get("num_cascades", 12)), 4)
    if tuple(clean_aux["q_hat"].shape[-2:]) != expected_q_shape:
        raise RuntimeError(
            "V2.1 emitted an unexpected per-cascade/per-scale q_hat shape: "
            f"{tuple(clean_aux['q_hat'].shape)}."
        )
    if tuple(clean_aux["q_logits"].shape) != tuple(clean_aux["q_hat"].shape):
        raise RuntimeError("q_logits and q_hat shapes do not match.")

    mean_abs_difference = float(
        torch.mean(torch.abs(v21_prediction - reference)).item()
    )
    reference_scale = float(torch.mean(torch.abs(reference)).clamp_min(1e-8).item())
    relative_difference = mean_abs_difference / reference_scale
    if relative_difference > args.max_initial_relative_difference:
        raise RuntimeError(
            "M2-GD v2.1 initialization is too far from the audited V2 model: "
            f"relative difference={relative_difference:.6f}, "
            f"limit={args.max_initial_relative_difference:.6f}."
        )

    # Missing-PD invariance: auxiliary content must have no effect when m=0.
    availability_zero = torch.zeros(1, device=device)
    variants = {
        "clean_content": pd_aux,
        "all_zero": torch.zeros_like(pd_aux),
        "spatially_flipped": torch.flip(pd_aux, dims=(-2, -1)),
        "random": torch.randn_like(pd_aux),
    }
    missing_outputs = {}
    missing_aux = {}
    with torch.no_grad():
        for name, value in variants.items():
            prediction, aux = model(
                pdfs_masked_kspace=kspace,
                mask=mask,
                pd_aux_image=value,
                pd_available=availability_zero,
                return_aux=True,
            )
            missing_outputs[name] = prediction.cpu()
            missing_aux[name] = {
                "max_abs_q": float(aux["q"].abs().max().item()),
                "max_gated_rms": float(
                    aux["gated_aux_to_target_rms"].abs().max().item()
                ),
            }
    reference_missing = missing_outputs["clean_content"]
    missing_max_difference = max(
        float((value - reference_missing).abs().max().item())
        for value in missing_outputs.values()
    )
    if missing_max_difference != 0.0:
        raise RuntimeError(
            f"Missing-PD full-model invariance failed: max diff={missing_max_difference}"
        )
    if any(
        stats["max_abs_q"] != 0.0 or stats["max_gated_rms"] != 0.0
        for stats in missing_aux.values()
    ):
        raise RuntimeError("Missing-PD q or gated RMS was not exactly zero.")

    # Two-step complete Stage-A backward test. The second step is necessary
    # because the final linear weight starts at zero, so context convolutions
    # may legitimately receive zero gradient on the first backward pass.
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "reliability_head" in name
    model.train()
    shifted = torch.stack(
        [translate_nonwrapping(image, 8, 0, "reflect") for image in pd_aux], dim=0
    )
    shifted_alternate = torch.stack(
        [translate_nonwrapping(image, 8, 0, "zero") for image in pd_aux], dim=0
    )
    tripled_kspace = torch.cat([kspace, kspace, kspace], dim=0)
    tripled_mask = torch.cat([mask, mask, mask], dim=0)
    tripled_pd = torch.cat([pd_aux, shifted, shifted_alternate], dim=0)
    tripled_availability = torch.ones(3, device=device)
    config_targets = CorruptionConfig()
    shift8_targets = torch.tensor(
        scale_targets_from_base(
            shift_reliability_target(8, config_targets),
            config_targets,
            True,
        ),
        device=device,
        dtype=pd_aux.dtype,
    )[None, :]
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-2,
    )
    step_reports = []
    context_gradient_norm = 0.0
    recon_loss = torch.tensor(float("nan"), device=device)
    for optimizer_step in range(1, 3):
        optimizer.zero_grad(set_to_none=True)
        prediction, aux = model(
            pdfs_masked_kspace=tripled_kspace,
            mask=tripled_mask,
            pd_aux_image=tripled_pd,
            pd_available=tripled_availability,
            return_aux=True,
            detach_q_for_fusion=True,
        )
        prediction = center_crop(
            prediction, target.shape[-2], target.shape[-1]
        )
        recon_scale = target.amax(
            dim=(-2, -1), keepdim=True
        ).clamp_min(1e-8)
        recon_loss = torch.abs(
            prediction[:1] / recon_scale - target / recon_scale
        ).mean().detach()
        clean_logits = aux["q_logits"][:1]
        corrupt_logits = aux["q_logits"][1:2]
        clean_q = aux["q_hat"][:1]
        corrupt_q = aux["q_hat"][1:2]
        alternate_q = aux["q_hat"][2:3]
        bce_loss = 0.5 * (
            F.binary_cross_entropy_with_logits(
                clean_logits, torch.ones_like(clean_logits)
            )
            + F.binary_cross_entropy_with_logits(
                corrupt_logits,
                shift8_targets[:, None, :].expand_as(corrupt_logits),
            )
        )
        rank_loss, rank_diagnostics = paired_discrimination_loss(
            clean_q,
            corrupt_q,
            shift8_targets,
            [{"condition": "shift", "magnitude_linf": 8}],
        )
        padding_loss = F.mse_loss(corrupt_q, alternate_q)
        composite = bce_loss + rank_loss + 0.25 * padding_loss
        total_loss = 0.05 * composite
        total_loss.backward()

        unexpected_gradients = [
            name
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
            and "reliability_head" not in name
            and float(parameter.grad.abs().max().item()) != 0.0
        ]
        if unexpected_gradients:
            raise RuntimeError(
                "Stage-A backward reached frozen parameters: "
                f"{unexpected_gradients[:10]}"
            )
        head_gradient_sq = sum(
            float(parameter.grad.square().sum().item())
            for name, parameter in model.named_parameters()
            if "reliability_head" in name and parameter.grad is not None
        )
        head_gradient_norm = math.sqrt(head_gradient_sq)
        context_gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if "reliability_head.context.0.weight" in name
            and parameter.grad is not None
        ]
        context_gradient_norm = math.sqrt(
            sum(float(gradient.square().sum().item()) for gradient in context_gradients)
        )
        step_reports.append(
            {
                "step": optimizer_step,
                "bce_loss": float(bce_loss.item()),
                "rank_loss": float(rank_loss.item()),
                "padding_consistency_loss": float(padding_loss.item()),
                "total_loss": float(total_loss.item()),
                "head_gradient_norm": head_gradient_norm,
                "context_conv_gradient_norm": context_gradient_norm,
                "rank_diagnostics": rank_diagnostics,
            }
        )
        if not math.isfinite(head_gradient_norm) or head_gradient_norm <= 0:
            raise RuntimeError(
                f"Invalid reliability-head gradient at step {optimizer_step}: "
                f"{head_gradient_norm}."
            )
        optimizer.step()

    if not math.isfinite(context_gradient_norm) or context_gradient_norm <= 0:
        raise RuntimeError(
            "Context convolution did not receive a non-zero gradient after "
            "two optimizer steps."
        )
    new_parameter_count = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if "reliability_head" in name
    )

    result = {
        "status": "passed",
        "gpu": torch.cuda.get_device_name(0),
        "corruption_unit_tests": unit_results,
        "m2gd_v2_transfer_report": transfer_report,
        "initial_clean_comparison": {
            "mean_absolute_output_difference": mean_abs_difference,
            "relative_output_l1_difference": relative_difference,
            "maximum_allowed": args.max_initial_relative_difference,
            "clean_q_hat_mean": float(clean_aux["q_hat"].mean().item()),
            "clean_gated_rms_mean": float(
                clean_aux["gated_aux_to_target_rms"].mean().item()
            ),
        },
        "missing_invariance": {
            "max_output_difference": missing_max_difference,
            "variants": missing_aux,
        },
        "backward_test": {
            "reconstruction_loss": float(recon_loss.item()),
            "two_optimizer_steps": step_reports,
            "final_context_conv_gradient_norm": context_gradient_norm,
            "only_reliability_head_trainable": True,
            "new_reliability_parameter_count": int(new_parameter_count),
        },
        "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 1024 ** 3,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
