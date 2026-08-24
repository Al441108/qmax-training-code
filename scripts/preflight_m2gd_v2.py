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

from src.auxiliary_corruptions_v2 import (
    CorruptionConfig,
    border_only,
    border_reliability_target,
    shift_reliability_target,
    translate_nonwrapping,
)
from src.dataset_paired_multicoil_aux_pd_r2 import (
    PairedMulticoilAuxPDToPDFSDataset,
)
from src.fft_utils import center_crop
from src.m2gd_v2_auxiliary_varnet import (
    M2GDv2AuxPDVarNet,
    load_m2u_backbone,
)
from src.m2u_auxiliary_varnet_optimized import M2UAuxPDVarNet


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

    return {
        "padding_target_independence": True,
        "border_central_content_unchanged": True,
        "no_circular_wrap": True,
        "shift4_target": target_by_padding,
        "border8_target": border_reliability_target(8, config),
    }


def make_m2u(config, device):
    return M2UAuxPDVarNet(
        num_cascades=int(config.get("num_cascades", 12)),
        sens_chans=int(config.get("sens_chans", 8)),
        sens_pools=int(config.get("sens_pools", 4)),
        chans=int(config.get("chans", 18)),
        pools=int(config.get("pools", 4)),
        mask_center=True,
        initial_aux_alpha=float(config.get("initial_aux_alpha", 0.1)),
    ).to(device)


def make_v2(config, device, initial_gate_probability):
    return M2GDv2AuxPDVarNet(
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
    parser.add_argument("--m2u_checkpoint", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--initial_gate_probability", type=float, default=0.99)
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

    checkpoint_path = Path(args.m2u_checkpoint)
    checkpoint = load_checkpoint(checkpoint_path, "cpu")
    config = checkpoint.get("config", {})
    source_state = state_dict_from_checkpoint(checkpoint)
    if source_state and all(key.startswith("module.") for key in source_state):
        source_state = {key[len("module."):]: value for key, value in source_state.items()}

    # Run the reference and release it before allocating v2 to avoid GH200 OOM.
    m2u = make_m2u(config, device)
    m2u.load_state_dict(source_state, strict=True)
    m2u.eval()
    with torch.no_grad():
        reference = m2u(
            pdfs_masked_kspace=kspace,
            mask=mask,
            pd_aux_image=pd_aux,
        )
        reference = center_crop(reference, target.shape[-2], target.shape[-1]).cpu()
    del m2u
    torch.cuda.empty_cache()

    model = make_v2(config, device, args.initial_gate_probability)
    transfer_report = load_m2u_backbone(model, checkpoint_path, map_location=device)
    model.eval()
    availability_one = torch.ones(1, device=device)
    with torch.no_grad():
        v2_prediction, clean_aux = model(
            pdfs_masked_kspace=kspace,
            mask=mask,
            pd_aux_image=pd_aux,
            pd_available=availability_one,
            return_aux=True,
        )
        v2_prediction = center_crop(
            v2_prediction, target.shape[-2], target.shape[-1]
        ).cpu()

    mean_abs_difference = float(torch.mean(torch.abs(v2_prediction - reference)).item())
    reference_scale = float(torch.mean(torch.abs(reference)).clamp_min(1e-8).item())
    relative_difference = mean_abs_difference / reference_scale
    if relative_difference > args.max_initial_relative_difference:
        raise RuntimeError(
            "M2-GD v2 identity initialization is too far from M2-U: "
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

    # Single-batch forward/backward gradient test for the new modules.
    model.train()
    shifted = torch.stack(
        [translate_nonwrapping(image, 8, 0, "reflect") for image in pd_aux], dim=0
    )
    prediction, aux = model(
        pdfs_masked_kspace=kspace,
        mask=mask,
        pd_aux_image=shifted,
        pd_available=availability_one,
        return_aux=True,
    )
    prediction = center_crop(prediction, target.shape[-2], target.shape[-1])
    scale = target.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    recon_loss = torch.abs(prediction / scale - target / scale).mean()
    target_q = torch.full_like(aux["q_hat"], 0.05)
    reliability_loss = F.binary_cross_entropy(aux["q_hat"], target_q)
    loss = recon_loss + 0.05 * reliability_loss
    model.zero_grad(set_to_none=True)
    loss.backward()
    new_gradient_sq = 0.0
    for name, parameter in model.named_parameters():
        if any(fragment in name for fragment in (
            "reliability_head", "channel_gate", "spatial_gate"
        )) and parameter.grad is not None:
            new_gradient_sq += float(parameter.grad.square().sum().item())
    new_gradient_norm = math.sqrt(new_gradient_sq)
    if not math.isfinite(new_gradient_norm) or new_gradient_norm <= 0:
        raise RuntimeError(
            f"New reliability/gate modules received invalid gradient {new_gradient_norm}."
        )

    result = {
        "status": "passed",
        "gpu": torch.cuda.get_device_name(0),
        "corruption_unit_tests": unit_results,
        "m2u_transfer_report": transfer_report,
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
            "reliability_loss": float(reliability_loss.item()),
            "new_module_gradient_norm": new_gradient_norm,
        },
        "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 1024 ** 3,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
