#!/usr/bin/env python3
"""Train M2-GD for reliability-aware PD-assisted PD-FS reconstruction.

The script preserves the controlled M2-U training setup while adding:
  * synthetic auxiliary corruption curriculum;
  * graded, protocol-defined supervision of q_hat (never availability-masked q);
  * optional one-sided auxiliary contribution budget;
  * clean patient-level validation checkpoint selection;
  * clean/shift2/shift4/shift8/missing gate, contribution and L1 diagnostics;
  * full-model missing-PD invariance safety check.

Wrong-patient PD is deliberately excluded from training and remains an unseen
stress test for the separate robustness evaluation pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_paired_multicoil_aux_pd_r2 import (
    PairedMulticoilAuxPDToPDFSDataset,
)
from src.fft_utils import center_crop
from src.m2gd_auxiliary_varnet import M2GDAuxPDVarNet


CONDITION_NAMES = ("clean", "missing", "shift", "blur", "noise")
CONDITION_TO_ID = {name: index for index, name in enumerate(CONDITION_NAMES)}
DIAGNOSTIC_CONDITIONS = ("clean", "shift2", "shift4", "shift8", "missing")


def capture_rng_state() -> Dict[str, object]:
    """Capture all RNG state needed for an exact interrupted-run continuation."""
    state: Dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _to_cpu_byte_tensor(value: object) -> torch.Tensor:
    """Convert a serialized RNG state to the CPU ByteTensor PyTorch expects."""
    if not torch.is_tensor(value):
        value = torch.as_tensor(value, dtype=torch.uint8)

    return (
        value.detach()
        .to(device="cpu", dtype=torch.uint8)
        .contiguous()
    )


def restore_rng_state(state: Dict[str, object]) -> None:
    """Restore Python, NumPy, CPU PyTorch and CUDA RNG states."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])

    # torch.set_rng_state requires a CPU torch.uint8 tensor.
    torch.set_rng_state(
        _to_cpu_byte_tensor(state["torch"])
    )

    if torch.cuda.is_available() and "cuda" in state:
        cuda_states = state["cuda"]

        if torch.is_tensor(cuda_states):
            cuda_states = [cuda_states]

        cuda_states = [
            _to_cpu_byte_tensor(cuda_state)
            for cuda_state in cuda_states
        ]

        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError(
                "CUDA RNG-state count does not match visible GPU count: "
                f"checkpoint={len(cuda_states)}, "
                f"current={torch.cuda.device_count()}"
            )

        torch.cuda.set_rng_state_all(cuda_states)


class ShapeBucketBatchSampler:
    def __init__(self, dataset, batch_size, shuffle=False, seed=42):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

        buckets = defaultdict(list)
        for idx, record in enumerate(dataset.records):
            with h5py.File(record["pdfs_path"], "r") as hf:
                shape_key = tuple(hf["kspace"].shape[1:])
            buckets[shape_key].append(idx)

        self.buckets = dict(buckets)
        self._num_batches = sum(
            (len(indices) + self.batch_size - 1) // self.batch_size
            for indices in self.buckets.values()
        )

        print(
            f"ShapeBucketBatchSampler: {len(self.buckets)} shape buckets, "
            f"{self._num_batches} batches, batch_size={self.batch_size}, "
            f"shuffle={self.shuffle}"
        )

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        all_batches = []

        for indices in self.buckets.values():
            indices = list(indices)
            if self.shuffle:
                rng.shuffle(indices)

            for start in range(0, len(indices), self.batch_size):
                all_batches.append(indices[start:start + self.batch_size])

        if self.shuffle:
            rng.shuffle(all_batches)

        self.epoch += 1
        yield from all_batches

    def __len__(self):
        return self._num_batches


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_aux_batch(batch, device: torch.device):
    required = [
        "pdfs_masked_kspace",
        "pd_aux_image",
        "pdfs_target_raw",
        "mask",
    ]
    for key in required:
        if key not in batch:
            raise KeyError(f"Missing batch key: {key}")

    pdfs_kspace = batch["pdfs_masked_kspace"].to(
        device,
        non_blocking=True,
    )
    if not torch.is_complex(pdfs_kspace):
        raise TypeError(
            f"Expected complex PDFS k-space, got {pdfs_kspace.dtype}"
        )
    pdfs_kspace = torch.view_as_real(pdfs_kspace).float()

    mask = batch["mask"].to(device, non_blocking=True).bool()
    mask = mask[:, None, None, :, None]

    pd_aux = batch["pd_aux_image"].to(
        device,
        non_blocking=True,
    ).float()
    pdfs_target = batch["pdfs_target_raw"].to(
        device,
        non_blocking=True,
    ).float()

    if pd_aux.ndim == 4 and pd_aux.shape[1] == 1:
        pd_aux = pd_aux.squeeze(1)
    if pdfs_target.ndim == 4 and pdfs_target.shape[1] == 1:
        pdfs_target = pdfs_target.squeeze(1)

    if pd_aux.ndim != 3:
        raise RuntimeError(
            f"Expected PD auxiliary [B,H,W], got {tuple(pd_aux.shape)}"
        )
    if pdfs_target.ndim != 3:
        raise RuntimeError(
            f"Expected PDFS target [B,H,W], got {tuple(pdfs_target.shape)}"
        )

    return pdfs_kspace, mask, pd_aux, pdfs_target


def l1_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise RuntimeError(
            f"Prediction/target mismatch: "
            f"{tuple(prediction.shape)} vs {tuple(target.shape)}"
        )

    scale = target.amax(
        dim=(-2, -1),
        keepdim=True,
    ).clamp_min(1e-8)

    return torch.abs(
        prediction / scale - target / scale
    ).mean(dim=(-2, -1))


def safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else float("nan")


def validate_binary_indicator(indicator: torch.Tensor, name: str) -> None:
    """Fail loudly if a hard indicator silently becomes a soft gate."""
    values = torch.unique(indicator.detach())
    valid = torch.logical_or(values == 0, values == 1).all()
    if not bool(valid.item()):
        raise RuntimeError(
            f"{name} must contain only hard 0/1 values, got "
            f"{values.cpu().tolist()}"
        )


def zero_fill_shift(
    image: torch.Tensor,
    shift_y: int,
    shift_x: int,
) -> torch.Tensor:
    """Translate [H,W] without circular wrap-around."""
    if image.ndim != 2:
        raise RuntimeError(f"Expected [H,W], got {tuple(image.shape)}")

    height, width = image.shape
    output = torch.zeros_like(image)

    if abs(shift_y) >= height or abs(shift_x) >= width:
        return output

    src_y0 = max(0, -shift_y)
    src_y1 = min(height, height - shift_y)
    dst_y0 = max(0, shift_y)
    dst_y1 = min(height, height + shift_y)

    src_x0 = max(0, -shift_x)
    src_x1 = min(width, width - shift_x)
    dst_x0 = max(0, shift_x)
    dst_x1 = min(width, width + shift_x)

    output[dst_y0:dst_y1, dst_x0:dst_x1] = image[
        src_y0:src_y1,
        src_x0:src_x1,
    ]
    return output


def gaussian_kernel_2d(
    kernel_size: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if kernel_size % 2 == 0 or kernel_size < 3:
        raise ValueError("Gaussian kernel size must be odd and at least 3")
    if sigma <= 0:
        raise ValueError("Gaussian sigma must be positive")

    coordinates = torch.arange(
        kernel_size,
        device=device,
        dtype=dtype,
    ) - (kernel_size - 1) / 2
    kernel_1d = torch.exp(-0.5 * (coordinates / sigma).square())
    kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1e-12)
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d / kernel_2d.sum().clamp_min(1e-12)


def gaussian_blur_single(
    image: torch.Tensor,
    sigma: float,
    kernel_size: int,
) -> torch.Tensor:
    kernel = gaussian_kernel_2d(
        kernel_size=kernel_size,
        sigma=sigma,
        device=image.device,
        dtype=image.dtype,
    ).view(1, 1, kernel_size, kernel_size)
    padding = kernel_size // 2
    padded = F.pad(
        image[None, None],
        (padding, padding, padding, padding),
        mode="reflect",
    )
    return F.conv2d(padded, kernel)[0, 0]


def curriculum_factor(
    epoch: int,
    warmup_epochs: int,
    ramp_epochs: int,
) -> float:
    """0 during clean warm-up, then linearly ramp to the full mixture."""
    if epoch <= warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return 1.0
    return min(1.0, (epoch - warmup_epochs) / float(ramp_epochs))


def effective_condition_probabilities(
    args,
    epoch: int,
) -> Dict[str, float]:
    factor = curriculum_factor(
        epoch=epoch,
        warmup_epochs=args.curriculum_warmup_epochs,
        ramp_epochs=args.curriculum_ramp_epochs,
    )

    probabilities = {
        "missing": factor * args.prob_missing,
        "shift": factor * args.prob_shift,
        "blur": factor * args.prob_blur,
        "noise": factor * args.prob_noise,
    }
    # During the ramp, interpolate from an all-clean warm-up to the explicitly
    # configured base mixture. `prob_clean` is therefore a real parameter,
    # rather than a redundant consistency-only CLI option.
    probabilities["clean"] = 1.0 - factor * (1.0 - args.prob_clean)

    if probabilities["clean"] < -1e-8:
        raise RuntimeError(
            f"Invalid effective corruption probabilities: {probabilities}"
        )
    probabilities["clean"] = max(0.0, probabilities["clean"])
    return probabilities


def sample_condition(
    probabilities: Dict[str, float],
    rng: random.Random,
) -> str:
    draw = rng.random()
    cumulative = 0.0
    for condition in CONDITION_NAMES:
        cumulative += probabilities[condition]
        if draw <= cumulative:
            return condition
    return "clean"


def interpolate_reliability(
    value: float,
    lower: float,
    upper: float,
    reliability_at_lower: float,
    reliability_at_upper: float,
) -> float:
    """Linearly map corruption severity to a pre-registered trust target."""
    if upper <= lower:
        return float(reliability_at_upper)
    fraction = min(1.0, max(0.0, (value - lower) / (upper - lower)))
    return float(
        reliability_at_lower
        + fraction * (reliability_at_upper - reliability_at_lower)
    )


def shift_reliability_target(magnitude: int, args) -> float:
    """Use a monotonic protocol target for mild, moderate and severe shifts."""
    if magnitude <= 2:
        return float(args.reliability_shift_2)
    if magnitude <= 4:
        return interpolate_reliability(
            magnitude, 2, 4, args.reliability_shift_2, args.reliability_shift_4
        )
    return interpolate_reliability(
        magnitude, 4, 8, args.reliability_shift_4, args.reliability_shift_8
    )


def corrupt_pd_batch(
    pd_aux: torch.Tensor,
    epoch: int,
    batch_index: int,
    args,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create per-sample corruption, hard availability and q_hat labels.

    q_hat targets are protocol-defined *global trust* levels, not final fusion
    weights. Local channel/spatial gates still determine where present PD is
    used. This preserves limited assistance for mild misregistration while
    requiring abstention for severe or absent PD.

    Availability labels:
      missing -> 0
      every present-but-degraded condition -> 1
    """
    probabilities = effective_condition_probabilities(args, epoch)
    rng = random.Random(
        args.seed + epoch * 1_000_003 + batch_index * 10_007
    )

    corrupted = pd_aux.clone()
    batch_size = pd_aux.shape[0]
    availability = torch.ones(
        batch_size,
        device=pd_aux.device,
        dtype=pd_aux.dtype,
    )
    reliability_target = torch.full_like(
        availability,
        fill_value=float(args.reliability_clean),
    )
    condition_ids = torch.empty(
        batch_size,
        device=pd_aux.device,
        dtype=torch.long,
    )

    for sample_index in range(batch_size):
        condition = sample_condition(probabilities, rng)
        condition_ids[sample_index] = CONDITION_TO_ID[condition]

        if condition == "clean":
            continue

        if condition == "missing":
            availability[sample_index] = 0.0
            reliability_target[sample_index] = args.reliability_missing
            corrupted[sample_index].zero_()

        elif condition == "shift":
            magnitude = rng.choice(args.shift_magnitudes)
            reliability_target[sample_index] = shift_reliability_target(
                magnitude, args
            )
            shift_y = rng.choice((-magnitude, 0, magnitude))
            shift_x = rng.choice((-magnitude, 0, magnitude))
            if shift_y == 0 and shift_x == 0:
                shift_x = magnitude if rng.random() < 0.5 else -magnitude
            corrupted[sample_index] = zero_fill_shift(
                corrupted[sample_index],
                shift_y=shift_y,
                shift_x=shift_x,
            )

        elif condition == "blur":
            sigma = rng.uniform(args.blur_sigma_min, args.blur_sigma_max)
            reliability_target[sample_index] = interpolate_reliability(
                sigma,
                args.blur_sigma_min,
                args.blur_sigma_max,
                args.reliability_blur_light,
                args.reliability_blur_severe,
            )
            corrupted[sample_index] = gaussian_blur_single(
                corrupted[sample_index],
                sigma=sigma,
                kernel_size=args.blur_kernel_size,
            )

        elif condition == "noise":
            ratio = rng.uniform(args.noise_std_min, args.noise_std_max)
            reliability_target[sample_index] = interpolate_reliability(
                ratio,
                args.noise_std_min,
                args.noise_std_max,
                args.reliability_noise_light,
                args.reliability_noise_severe,
            )
            sample_std = corrupted[sample_index].std().clamp_min(1e-8)
            noise_generator = torch.Generator(device=pd_aux.device)
            noise_generator.manual_seed(
                args.seed
                + epoch * 1_000_003
                + batch_index * 10_007
                + sample_index * 101
            )
            noise = torch.randn(
                corrupted[sample_index].shape,
                device=pd_aux.device,
                dtype=pd_aux.dtype,
                generator=noise_generator,
            )
            corrupted[sample_index] = (
                corrupted[sample_index] + ratio * sample_std * noise
            ).clamp_min(0.0)

        else:
            raise RuntimeError(f"Unhandled corruption condition: {condition}")

    validate_binary_indicator(availability, "pd_available")
    return corrupted, availability, reliability_target, condition_ids


def fixed_condition_pd(
    pd_aux: torch.Tensor,
    condition: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = pd_aux.shape[0]
    availability = torch.ones(
        batch_size,
        device=pd_aux.device,
        dtype=pd_aux.dtype,
    )

    if condition == "clean":
        output = pd_aux
    elif condition.startswith("shift"):
        shift_pixels = int(condition.removeprefix("shift"))
        output = torch.stack(
            [
                zero_fill_shift(
                    image,
                    shift_y=0,
                    shift_x=shift_pixels,
                )
                for image in pd_aux
            ],
            dim=0,
        )
    elif condition == "missing":
        output = torch.zeros_like(pd_aux)
        availability.zero_()
    else:
        raise ValueError(f"Unsupported diagnostic condition: {condition}")

    validate_binary_indicator(availability, "pd_available")
    return output, availability


def parameter_gradient_norm(model: torch.nn.Module, name_fragment: str) -> float:
    total = 0.0
    found = False
    for name, parameter in model.named_parameters():
        if name_fragment not in name or parameter.grad is None:
            continue
        found = True
        total += float(parameter.grad.detach().square().sum().item())
    return math.sqrt(total) if found else float("nan")


def add_condition_statistics(
    storage: Dict[str, Dict[str, List[float]]],
    condition_ids: torch.Tensor,
    aux: Dict[str, torch.Tensor],
) -> None:
    q_hat_per_sample = aux["q_hat"].detach().mean(dim=1)
    q_per_sample = aux["q"].detach().mean(dim=1)
    rms_per_sample = aux["gated_aux_to_target_rms"].detach().mean(dim=(1, 2))

    for sample_index in range(condition_ids.shape[0]):
        condition = CONDITION_NAMES[int(condition_ids[sample_index].item())]
        storage[condition]["q_hat"].append(
            float(q_hat_per_sample[sample_index].item())
        )
        storage[condition]["q"].append(
            float(q_per_sample[sample_index].item())
        )
        storage[condition]["rms"].append(
            float(rms_per_sample[sample_index].item())
        )
        storage[condition]["count"].append(1.0)


def empty_condition_statistics() -> Dict[str, Dict[str, List[float]]]:
    return {
        condition: {
            "q_hat": [],
            "q": [],
            "rms": [],
            "count": [],
        }
        for condition in CONDITION_NAMES
    }


@torch.no_grad()
def evaluate_clean(model, loader, device, max_val_batches=None):
    """Clean-PD validation; checkpoint selection uses patient-level mean L1."""
    model.eval()

    overall: List[float] = []
    edge: List[float] = []
    central: List[float] = []
    clean_q_hat: List[float] = []
    clean_rms: List[float] = []
    rows = []
    patient_values: Dict[str, List[float]] = defaultdict(list)

    for batch_index, batch in enumerate(loader, start=1):
        if max_val_batches is not None and batch_index > max_val_batches:
            break

        pdfs_kspace, mask, pd_aux, target = prepare_aux_batch(batch, device)
        availability = torch.ones(
            pd_aux.shape[0],
            device=device,
            dtype=pd_aux.dtype,
        )
        validate_binary_indicator(availability, "pd_available")

        prediction, aux = model(
            pdfs_masked_kspace=pdfs_kspace,
            mask=mask,
            pd_aux_image=pd_aux,
            pd_available=availability,
            return_aux=True,
        )
        prediction = center_crop(
            prediction,
            crop_h=target.shape[-2],
            crop_w=target.shape[-1],
        )

        if not torch.isfinite(prediction).all():
            raise RuntimeError("Non-finite PD-FS prediction during validation")

        losses = l1_per_sample(prediction, target)
        if not torch.isfinite(losses).all():
            raise RuntimeError("Non-finite PD-FS validation loss")

        clean_q_hat.extend(aux["q_hat"].detach().mean(dim=1).cpu().tolist())
        clean_rms.extend(
            aux["gated_aux_to_target_rms"]
            .detach()
            .mean(dim=(1, 2))
            .cpu()
            .tolist()
        )

        for sample_idx in range(target.shape[0]):
            value = float(losses[sample_idx].item())

            is_edge_value = batch["is_edge"][sample_idx]
            is_edge = bool(
                is_edge_value.item()
                if torch.is_tensor(is_edge_value)
                else is_edge_value
            )

            slice_value = batch["slice_idx"][sample_idx]
            slice_idx = int(
                slice_value.item()
                if torch.is_tensor(slice_value)
                else slice_value
            )
            patient_id = str(batch["patient_id"][sample_idx])

            overall.append(value)
            patient_values[patient_id].append(value)
            (edge if is_edge else central).append(value)
            rows.append(
                {
                    "patient_id": patient_id,
                    "slice_idx": slice_idx,
                    "is_edge": is_edge,
                    "pdfs_l1": value,
                }
            )

    patient_means = [safe_mean(values) for values in patient_values.values()]
    results = {
        "pdfs_patient_l1": safe_mean(patient_means),
        "pdfs_slice_l1": safe_mean(overall),
        "pdfs_edge_l1": safe_mean(edge),
        "pdfs_central_l1": safe_mean(central),
        "clean_q_hat_mean": safe_mean(clean_q_hat),
        "clean_gated_rms_mean": safe_mean(clean_rms),
        "num_patients": len(patient_means),
        "num_slices": len(overall),
        "num_edge_slices": len(edge),
        "num_central_slices": len(central),
    }

    model.train()
    return results, rows


@torch.no_grad()
def evaluate_gate_conditions(
    model,
    loader,
    device,
    max_batches: int,
) -> Dict[str, float]:
    """Audit trust, actual injection and reconstruction across fixed PD states."""
    if max_batches <= 0:
        results = {
            f"diag_{metric}_{condition}": float("nan")
            for condition in DIAGNOSTIC_CONDITIONS
            for metric in ("q_hat", "rms", "l1")
        }
        results["diag_q_monotonic"] = float("nan")
        results["diag_rms_monotonic"] = float("nan")
        return results

    model.eval()
    q_values = {condition: [] for condition in DIAGNOSTIC_CONDITIONS}
    rms_values = {condition: [] for condition in DIAGNOSTIC_CONDITIONS}
    l1_values = {condition: [] for condition in DIAGNOSTIC_CONDITIONS}

    for batch_index, batch in enumerate(loader, start=1):
        if batch_index > max_batches:
            break

        pdfs_kspace, mask, pd_aux, target = prepare_aux_batch(batch, device)

        for condition in DIAGNOSTIC_CONDITIONS:
            condition_pd, availability = fixed_condition_pd(
                pd_aux,
                condition=condition,
            )
            prediction, aux = model(
                pdfs_masked_kspace=pdfs_kspace,
                mask=mask,
                pd_aux_image=condition_pd,
                pd_available=availability,
                return_aux=True,
            )
            prediction = center_crop(
                prediction,
                crop_h=target.shape[-2],
                crop_w=target.shape[-1],
            )
            q_values[condition].extend(
                aux["q_hat"].detach().mean(dim=1).cpu().tolist()
            )
            rms_values[condition].extend(
                aux["gated_aux_to_target_rms"]
                .detach()
                .mean(dim=(1, 2))
                .cpu()
                .tolist()
            )
            l1_values[condition].extend(
                l1_per_sample(prediction, target).detach().cpu().tolist()
            )

    results = {}
    for condition in DIAGNOSTIC_CONDITIONS:
        results[f"diag_q_hat_{condition}"] = safe_mean(q_values[condition])
        results[f"diag_rms_{condition}"] = safe_mean(rms_values[condition])
        results[f"diag_l1_{condition}"] = safe_mean(l1_values[condition])

    q_sequence = [results[f"diag_q_hat_{condition}"] for condition in DIAGNOSTIC_CONDITIONS]
    rms_sequence = [results[f"diag_rms_{condition}"] for condition in DIAGNOSTIC_CONDITIONS]
    results["diag_q_monotonic"] = float(
        q_sequence[0] > q_sequence[1] > q_sequence[2] > q_sequence[3]> q_sequence[4]
    )
    results["diag_rms_monotonic"] = float(
        rms_sequence[0] > rms_sequence[1] > rms_sequence[2] > rms_sequence[3]> q_sequence[4]
    )

    model.train()
    return results


@torch.no_grad()
def run_missing_pd_invariance_check(
    model,
    loader,
    device,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> Dict[str, float]:
    """With m=0, reconstruction must not depend on the supplied PD image."""
    model.eval()
    batch = next(iter(loader))
    pdfs_kspace, mask, pd_aux, _ = prepare_aux_batch(batch, device)

    availability = torch.zeros(
        pd_aux.shape[0],
        device=device,
        dtype=pd_aux.dtype,
    )
    validate_binary_indicator(availability, "pd_available")

    # Deterministic, deliberately incompatible input. Do not consume the
    # training RNG state merely to run this structural safety test.
    alternative_pd = -pd_aux + pd_aux.mean(dim=(-2, -1), keepdim=True)
    prediction_a, aux_a = model(
        pdfs_masked_kspace=pdfs_kspace,
        mask=mask,
        pd_aux_image=pd_aux,
        pd_available=availability,
        return_aux=True,
    )
    prediction_b, aux_b = model(
        pdfs_masked_kspace=pdfs_kspace,
        mask=mask,
        pd_aux_image=alternative_pd,
        pd_available=availability,
        return_aux=True,
    )

    max_difference = float(
        torch.max(torch.abs(prediction_a - prediction_b)).item()
    )
    max_aux_rms = float(
        torch.max(
            torch.stack(
                [
                    aux_a["gated_aux_to_target_rms"].abs().max(),
                    aux_b["gated_aux_to_target_rms"].abs().max(),
                ]
            )
        ).item()
    )

    if not torch.allclose(
        prediction_a,
        prediction_b,
        atol=atol,
        rtol=rtol,
    ):
        raise RuntimeError(
            "Missing-PD invariance FAILED: reconstruction changed when "
            f"m=0 and PD input changed; max_abs_diff={max_difference:.3e}"
        )
    if max_aux_rms > atol:
        raise RuntimeError(
            "Missing-PD hard gate FAILED: non-zero gated auxiliary RMS "
            f"with m=0; max={max_aux_rms:.3e}"
        )

    model.train()
    return {
        "missing_invariance_max_abs_diff": max_difference,
        "missing_invariance_max_aux_rms": max_aux_rms,
    }


def make_model(args, device):
    return M2GDAuxPDVarNet(
        num_cascades=args.num_cascades,
        sens_chans=args.sens_chans,
        sens_pools=args.sens_pools,
        chans=args.chans,
        pools=args.pools,
        mask_center=True,
        initial_q=args.initial_q,
        initial_local_gate=args.initial_local_gate,
        contribution_budgets=args.contribution_budgets,
    ).to(device)


def select_patient_ids(dataset, limit: Optional[int]):
    patient_ids = list(
        dict.fromkeys(
            str(row["patient_id"])
            for row in dataset.patient_rows
        )
    )

    if limit is None:
        return patient_ids
    if limit < 1:
        raise ValueError("Patient limit must be at least 1 or omitted")

    selected = patient_ids[:limit]
    if len(selected) != limit:
        raise RuntimeError(
            f"Requested {limit} patients, found {len(selected)}"
        )
    return selected


TRAINING_LOG_COLUMNS = [
    "epoch",
    "curriculum_factor",
    "train_total_loss",
    "train_recon_l1",
    "train_reliability_loss",
    "train_weighted_reliability_loss",
    "train_budget_loss",
    "train_weighted_budget_loss",
    "val_pdfs_patient_l1",
    "val_pdfs_slice_l1",
    "val_pdfs_edge_l1",
    "val_pdfs_central_l1",
    "val_clean_q_hat_mean",
    "val_clean_gated_rms_mean",
    "gradient_norm_mean",
    "q_predictor_gradient_norm_mean",
    "channel_gate_gradient_norm_mean",
    "spatial_gate_gradient_norm_mean",
    "epoch_seconds",
    "peak_gpu_memory_gb",
    "learning_rate",
]

for condition_name in CONDITION_NAMES:
    TRAINING_LOG_COLUMNS.extend(
        [
            f"train_fraction_{condition_name}",
            f"train_q_hat_{condition_name}",
            f"train_q_{condition_name}",
            f"train_gated_rms_{condition_name}",
        ]
    )

for condition_name in DIAGNOSTIC_CONDITIONS:
    TRAINING_LOG_COLUMNS.extend(
        [
            f"diag_q_hat_{condition_name}",
            f"diag_rms_{condition_name}",
            f"diag_l1_{condition_name}",
        ]
    )

TRAINING_LOG_COLUMNS.extend(["diag_q_monotonic", "diag_rms_monotonic"])


def initialise_training_log(path):
    with open(path, "w", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=TRAINING_LOG_COLUMNS).writeheader()


def append_training_log(path, row):
    with open(path, "a", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=TRAINING_LOG_COLUMNS).writerow(row)


def save_slice_metrics(path, rows):
    columns = ["patient_id", "slice_idx", "is_edge", "pdfs_l1"]
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_patient_ids(path, patient_ids):
    with open(path, "w", encoding="utf-8") as file:
        for patient_id in sorted(patient_ids):
            file.write(f"{patient_id}\n")


def verify_resume_config(
    checkpoint_config: dict,
    current_config: dict,
) -> None:
    fields = [
        "metadata_csv",
        "acceleration",
        "pd_aux_acceleration",
        "num_cascades",
        "chans",
        "sens_chans",
        "pools",
        "sens_pools",
        "batch_size",
        "seed",
        "train_patient_ids",
        "val_patient_ids",
        "fusion_type",
        "fusion_scales",
        "target_contrast",
        "initial_q",
        "initial_local_gate",
        "learning_rate",
        "lambda_rel",
        "lambda_budget",
        "contribution_budgets",
        "prob_clean",
        "prob_missing",
        "prob_shift",
        "prob_blur",
        "prob_noise",
        "curriculum_warmup_epochs",
        "curriculum_ramp_epochs",
        "shift_magnitudes",
        "blur_kernel_size",
        "blur_sigma_min",
        "blur_sigma_max",
        "noise_std_min",
        "noise_std_max",
        "reliability_clean",
        "reliability_missing",
        "reliability_shift_2",
        "reliability_shift_4",
        "reliability_shift_8",
        "reliability_blur_light",
        "reliability_blur_severe",
        "reliability_noise_light",
        "reliability_noise_severe",
    ]

    mismatches = []
    for field in fields:
        previous = checkpoint_config.get(field)
        current = current_config.get(field)
        if previous != current:
            mismatches.append(
                f"{field}: checkpoint={previous!r}, current={current!r}"
            )

    if mismatches:
        raise RuntimeError(
            "Resume configuration does not match checkpoint:\n"
            + "\n".join(mismatches)
        )


def validate_arguments(args) -> None:
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.learning_rate <= 0:
        raise ValueError("--learning_rate must be positive")
    if args.lambda_rel <= 0:
        raise ValueError(
            "--lambda_rel must be positive: q_hat must be trained explicitly"
        )
    if args.lambda_budget < 0:
        raise ValueError("--lambda_budget cannot be negative")
    for name in ("initial_q", "initial_local_gate"):
        value = getattr(args, name)
        if not 0.0 < value < 1.0:
            raise ValueError(f"--{name} must lie strictly in (0, 1), got {value}")

    reliability_values = {
        "clean": args.reliability_clean,
        "shift_2": args.reliability_shift_2,
        "shift_4": args.reliability_shift_4,
        "shift_8": args.reliability_shift_8,
        "missing": args.reliability_missing,
        "blur_light": args.reliability_blur_light,
        "blur_severe": args.reliability_blur_severe,
        "noise_light": args.reliability_noise_light,
        "noise_severe": args.reliability_noise_severe,
    }
    if any(value < 0.0 or value > 1.0 for value in reliability_values.values()):
        raise ValueError(
            f"All reliability targets must lie in [0, 1]: {reliability_values}"
        )
    if not (
        args.reliability_clean
        >= args.reliability_shift_2
        >= args.reliability_shift_4
        >= args.reliability_shift_8
        >= args.reliability_missing
    ):
        raise ValueError("Shift reliability targets must decrease with shift severity")
    if args.reliability_blur_light < args.reliability_blur_severe:
        raise ValueError("Blur reliability must decrease with blur severity")
    if args.reliability_noise_light < args.reliability_noise_severe:
        raise ValueError("Noise reliability must decrease with noise severity")
    if not math.isclose(args.reliability_missing, 0.0, abs_tol=1e-8):
        raise ValueError("Missing-PD reliability target must be 0.0")

    if args.contribution_budgets is not None:
        if len(args.contribution_budgets) != args.pools:
            raise ValueError(
                f"--contribution_budgets requires {args.pools} values, "
                f"got {len(args.contribution_budgets)}"
            )
        if any(value <= 0 for value in args.contribution_budgets):
            raise ValueError("Contribution budgets must be positive")
    elif args.lambda_budget > 0:
        raise ValueError(
            "--lambda_budget > 0 requires calibrated "
            "--contribution_budgets"
        )

    probabilities = {
        "clean": args.prob_clean,
        "missing": args.prob_missing,
        "shift": args.prob_shift,
        "blur": args.prob_blur,
        "noise": args.prob_noise,
    }
    if any(value < 0 or value > 1 for value in probabilities.values()):
        raise ValueError(f"All condition probabilities must be in [0,1]: {probabilities}")
    if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-8):
        raise ValueError(
            "Condition probabilities must sum to 1.0, got "
            f"{sum(probabilities.values()):.8f}: {probabilities}"
        )
    if args.prob_clean <= 0:
        raise ValueError("Clean PD probability must be positive")
    if args.blur_kernel_size % 2 == 0 or args.blur_kernel_size < 3:
        raise ValueError("--blur_kernel_size must be odd and at least 3")
    if args.blur_sigma_min <= 0 or args.blur_sigma_max < args.blur_sigma_min:
        raise ValueError("Invalid blur sigma range")
    if args.noise_std_min < 0 or args.noise_std_max < args.noise_std_min:
        raise ValueError("Invalid noise standard-deviation range")
    if not args.shift_magnitudes or any(value <= 0 for value in args.shift_magnitudes):
        raise ValueError("--shift_magnitudes must contain positive integers")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "M2-GD reliability-aware global-local disagreement-gated "
            "auxiliary PD to PD-FS VarNet training."
        )
    )
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument(
        "--acceleration", type=int, choices=[4, 6, 8], required=True
    )
    parser.add_argument(
        "--pd_aux_acceleration",
        type=int,
        choices=[2, 4, 6, 8],
        default=2,
    )
    parser.add_argument("--num_train_patients", type=int, default=None)
    parser.add_argument("--num_val_patients", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_cascades", type=int, default=12)
    parser.add_argument("--chans", type=int, default=18)
    parser.add_argument("--sens_chans", type=int, default=8)
    parser.add_argument("--pools", type=int, default=4)
    parser.add_argument("--sens_pools", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--gate_diagnostic_batches", type=int, default=1)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume", default=None)

    parser.add_argument("--initial_q", type=float, default=0.8)
    parser.add_argument("--initial_local_gate", type=float, default=0.35)
    parser.add_argument("--lambda_rel", type=float, default=0.05)
    parser.add_argument("--lambda_budget", type=float, default=0.0)
    parser.add_argument(
        "--contribution_budgets",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Calibrated one-sided H/2...H/16 contribution budgets. "
            "Required when lambda_budget > 0."
        ),
    )

    # Protocol-defined global-trust targets. These are deliberately distinct
    # from the learned local gates and are not claimed to be optimal alpha.
    parser.add_argument("--reliability_clean", type=float, default=1.0)
    parser.add_argument("--reliability_missing", type=float, default=0.0)
    parser.add_argument("--reliability_shift_2", type=float, default=0.65)
    parser.add_argument("--reliability_shift_4", type=float, default=0.35)
    parser.add_argument("--reliability_shift_8", type=float, default=0.05)
    parser.add_argument("--reliability_blur_light", type=float, default=0.40)
    parser.add_argument("--reliability_blur_severe", type=float, default=0.10)
    parser.add_argument("--reliability_noise_light", type=float, default=0.40)
    parser.add_argument("--reliability_noise_severe", type=float, default=0.10)

    parser.add_argument("--prob_clean", type=float, default=0.65)
    parser.add_argument("--prob_missing", type=float, default=0.10)
    parser.add_argument("--prob_shift", type=float, default=0.10)
    parser.add_argument("--prob_blur", type=float, default=0.075)
    parser.add_argument("--prob_noise", type=float, default=0.075)
    parser.add_argument("--curriculum_warmup_epochs", type=int, default=2)
    parser.add_argument("--curriculum_ramp_epochs", type=int, default=3)
    parser.add_argument(
        "--shift_magnitudes", type=int, nargs="+", default=[2, 4, 8]
    )
    parser.add_argument("--blur_kernel_size", type=int, default=7)
    parser.add_argument("--blur_sigma_min", type=float, default=1.5)
    parser.add_argument("--blur_sigma_max", type=float, default=3.0)
    parser.add_argument("--noise_std_min", type=float, default=0.15)
    parser.add_argument("--noise_std_max", type=float, default=0.30)
    parser.add_argument(
        "--skip_missing_invariance_check",
        action="store_true",
        help="Skip the two-forward full-model m=0 invariance test.",
    )

    args = parser.parse_args()
    validate_arguments(args)

    metadata_path = Path(args.metadata_csv).resolve()
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata CSV does not exist: {metadata_path}")
    args.metadata_csv = str(metadata_path)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    resume_path = Path(args.resume).resolve() if args.resume else None
    if resume_path is not None and not resume_path.exists():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("M2-GD reliability-aware auxiliary PD VarNet training")
    print("=" * 80)
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Metadata:", args.metadata_csv)
    print("PD-FS target acceleration:", args.acceleration)
    print("PD auxiliary acceleration:", args.pd_aux_acceleration)
    print("Epochs:", args.epochs)
    print("Loss: recon +", args.lambda_rel, "* reliability +", args.lambda_budget, "* budget")
    print("Base curriculum probabilities:", {
        name: getattr(args, f"prob_{name}") for name in CONDITION_NAMES
    })
    print("Wrong-patient PD training: DISABLED (unseen stress test)")
    print("Output directory:", output_dir)
    print("Resume checkpoint:", resume_path)
    print("=" * 80)

    common_dataset_args = {
        "metadata_csv": args.metadata_csv,
        "pdfs_acceleration": args.acceleration,
        "pd_aux_acceleration": args.pd_aux_acceleration,
        "slices_per_patient": None,
        "edge_weight": 1.0,
    }

    full_train = PairedMulticoilAuxPDToPDFSDataset(
        split="train", **common_dataset_args
    )
    full_val = PairedMulticoilAuxPDToPDFSDataset(
        split="val", **common_dataset_args
    )

    train_patient_ids = select_patient_ids(full_train, args.num_train_patients)
    val_patient_ids = select_patient_ids(full_val, args.num_val_patients)

    overlap = set(train_patient_ids) & set(val_patient_ids)
    if overlap:
        raise RuntimeError(f"Patient leakage detected: {sorted(overlap)}")

    train_dataset = PairedMulticoilAuxPDToPDFSDataset(
        split="train",
        patient_ids=train_patient_ids,
        **common_dataset_args,
    )
    val_dataset = PairedMulticoilAuxPDToPDFSDataset(
        split="val",
        patient_ids=val_patient_ids,
        **common_dataset_args,
    )

    train_sampler = ShapeBucketBatchSampler(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    val_sampler = ShapeBucketBatchSampler(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print("Train patients:", len(train_patient_ids))
    print("Validation patients:", len(val_patient_ids))
    print("Train slices:", len(train_dataset))
    print("Validation slices:", len(val_dataset))
    print("Train batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("Patient leakage check: PASSED")

    save_patient_ids(output_dir / "train_patient_ids.txt", train_patient_ids)
    save_patient_ids(output_dir / "val_patient_ids.txt", val_patient_ids)

    model = make_model(args, device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print("Model parameters:", parameter_count)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    config = vars(args).copy()
    config.update(
        {
            "metadata_csv": args.metadata_csv,
            "output_dir": str(output_dir),
            "resume": str(resume_path) if resume_path is not None else None,
            "train_patient_ids": train_patient_ids,
            "val_patient_ids": val_patient_ids,
            "train_patients": len(train_patient_ids),
            "val_patients": len(val_patient_ids),
            "train_slices": len(train_dataset),
            "val_slices": len(val_dataset),
            "parameter_count": parameter_count,
            "loss": (
                "target-scaled PD-FS L1 + lambda_rel * BCE(q_hat, "
                "reliability_target) + lambda_budget * one-sided "
                "feature contribution budget"
            ),
            "reliability_supervision_target": "q_hat (not availability-masked q)",
            "reliability_labels": {
                "clean": args.reliability_clean,
                "missing": args.reliability_missing,
                "shift_2": args.reliability_shift_2,
                "shift_4": args.reliability_shift_4,
                "shift_8": args.reliability_shift_8,
                "blur": [args.reliability_blur_light, args.reliability_blur_severe],
                "noise": [args.reliability_noise_light, args.reliability_noise_severe],
            },
            "wrong_patient_training": False,
            "availability_semantics": "hard binary 0/1 only; missing=0, otherwise=1",
            "mask": (
                "Gaussian variable-density 1D Cartesian, "
                "PD-FS target stream only"
            ),
            "model": (
                "M2-GD: availability-constrained global reliability plus "
                "factorised channel-spatial disagreement gating"
            ),
            "target_contrast": "PD-FS",
            "auxiliary_contrast": "PD",
            "auxiliary_full_PD_access": False,
            "auxiliary_source": "zero-filled RSS from undersampled PD k-space",
            "fusion_type": "global_local_disagreement_gated",
            "fusion_scales": [
                "H/2",
                "H/4",
                "H/8",
                "H/16",
            ][:args.pools],
            "full_resolution_pd_fusion": False,
            "pd_encoder_shared_across_cascades": True,
            "pd_auxiliary_flip_correction": True,
            "data_consistency": "PD-FS only; applied after complex image model term",
            "checkpoint_selection_metric": "patient-level mean clean validation PDFS L1",
        }
    )

    consistency_config = config.copy()
    consistency_config["resume"] = None

    start_epoch = 1
    best_val = float("inf")
    best_epoch = 0
    epoch_times: List[float] = []
    history: List[dict] = []

    if resume_path is not None:
        checkpoint = torch.load(
            resume_path,
            map_location=device,
            weights_only=False,
        )
        for key in [
            "epoch",
            "model_state_dict",
            "optimizer_state_dict",
            "config",
            "rng_state",
            "train_sampler_epoch",
            "val_sampler_epoch",
        ]:
            if key not in checkpoint:
                raise KeyError(f"Resume checkpoint missing key: {key}")

        verify_resume_config(checkpoint["config"], consistency_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = int(checkpoint["epoch"]) + 1
        best_val = float(checkpoint.get("best_val", float("inf")))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        epoch_times = list(checkpoint.get("epoch_times", []))
        history = list(checkpoint.get("history", []))
        train_sampler.epoch = int(checkpoint["train_sampler_epoch"])
        val_sampler.epoch = int(checkpoint["val_sampler_epoch"])
        restore_rng_state(checkpoint["rng_state"])

        print("Resume loaded:", resume_path)
        print("Starting epoch:", start_epoch)

        if start_epoch > args.epochs:
            raise RuntimeError(
                "Checkpoint has already completed the requested epochs."
            )

    with open(output_dir / "config.json", "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    training_log_path = output_dir / "training_log.csv"
    if resume_path is None:
        initialise_training_log(training_log_path)
    elif not training_log_path.exists():
        # A resumed run may deliberately use a new output directory. Preserve
        # the complete history and retain a valid CSV header in that case.
        initialise_training_log(training_log_path)
        for historical_row in history:
            append_training_log(training_log_path, historical_row)

    if not args.skip_missing_invariance_check:
        safety = run_missing_pd_invariance_check(
            model=model,
            loader=val_loader,
            device=device,
        )
        print(
            "Full-model missing-PD invariance: PASSED | "
            f"max reconstruction diff={safety['missing_invariance_max_abs_diff']:.3e} | "
            f"max auxiliary RMS={safety['missing_invariance_max_aux_rms']:.3e}"
        )
        with open(
            output_dir / "initial_safety_checks.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(safety, file, indent=2)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        model.train()

        total_losses: List[float] = []
        recon_losses: List[float] = []
        reliability_losses: List[float] = []
        weighted_reliability_losses: List[float] = []
        budget_losses: List[float] = []
        weighted_budget_losses: List[float] = []
        gradient_norms: List[float] = []
        q_gradient_norms: List[float] = []
        channel_gradient_norms: List[float] = []
        spatial_gradient_norms: List[float] = []
        condition_statistics = empty_condition_statistics()

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        effective_probs = effective_condition_probabilities(args, epoch)
        print(
            f"Epoch {epoch} curriculum factor="
            f"{curriculum_factor(epoch, args.curriculum_warmup_epochs, args.curriculum_ramp_epochs):.3f} "
            f"effective probabilities={effective_probs}"
        )

        for batch_index, batch in enumerate(train_loader, start=1):
            if (
                args.max_train_batches is not None
                and batch_index > args.max_train_batches
            ):
                break

            pdfs_kspace, mask, clean_pd, target = prepare_aux_batch(batch, device)
            pd_aux, pd_available, reliability_target, condition_ids = (
                corrupt_pd_batch(
                    clean_pd,
                    epoch=epoch,
                    batch_index=batch_index,
                    args=args,
                )
            )
            validate_binary_indicator(pd_available, "pd_available")

            prediction, aux = model(
                pdfs_masked_kspace=pdfs_kspace,
                mask=mask,
                pd_aux_image=pd_aux,
                pd_available=pd_available,
                return_aux=True,
            )
            prediction = center_crop(
                prediction,
                crop_h=target.shape[-2],
                crop_w=target.shape[-1],
            )

            if not torch.isfinite(prediction).all():
                raise RuntimeError(
                    f"Non-finite prediction at epoch {epoch}, batch {batch_index}"
                )

            loss_recon = l1_per_sample(prediction, target).mean()

            q_hat = aux["q_hat"]
            reliability_targets = reliability_target[:, None].expand_as(q_hat)
            # Critical: supervise q_hat, never q = availability * q_hat.
            loss_reliability = F.binary_cross_entropy(
                q_hat,
                reliability_targets,
            )
            loss_budget = aux["budget_loss"]
            weighted_loss_reliability = args.lambda_rel * loss_reliability
            weighted_loss_budget = args.lambda_budget * loss_budget

            loss_total = (
                loss_recon
                + weighted_loss_reliability
                + weighted_loss_budget
            )

            for name, value in (
                ("prediction", prediction),
                ("reconstruction loss", loss_recon),
                ("reliability loss", loss_reliability),
                ("weighted reliability loss", weighted_loss_reliability),
                ("budget loss", loss_budget),
                ("weighted budget loss", weighted_loss_budget),
                ("total loss", loss_total),
                ("q_hat", q_hat),
            ):
                if not torch.isfinite(value).all():
                    raise RuntimeError(
                        f"Non-finite {name} at epoch {epoch}, batch {batch_index}"
                    )

            optimizer.zero_grad(set_to_none=True)
            loss_total.backward()

            q_grad = parameter_gradient_norm(model, "global_reliability")
            channel_grad = parameter_gradient_norm(model, "channel_gate")
            spatial_grad = parameter_gradient_norm(model, "spatial_gate")

            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=10.0,
            )
            if not torch.isfinite(gradient_norm):
                raise RuntimeError(
                    f"Non-finite gradient at epoch {epoch}, batch {batch_index}"
                )
            if not math.isfinite(q_grad) or q_grad <= 0:
                raise RuntimeError(
                    "Global reliability predictor received no finite positive "
                    f"gradient at epoch {epoch}, batch {batch_index}: {q_grad}"
                )

            optimizer.step()

            total_losses.append(float(loss_total.item()))
            recon_losses.append(float(loss_recon.item()))
            reliability_losses.append(float(loss_reliability.item()))
            weighted_reliability_losses.append(float(weighted_loss_reliability.item()))
            budget_losses.append(float(loss_budget.item()))
            weighted_budget_losses.append(float(weighted_loss_budget.item()))
            gradient_norms.append(float(gradient_norm.item()))
            q_gradient_norms.append(q_grad)
            channel_gradient_norms.append(channel_grad)
            spatial_gradient_norms.append(spatial_grad)
            add_condition_statistics(condition_statistics, condition_ids, aux)

            if batch_index == 1 or batch_index % 10 == 0:
                print(
                    f"Epoch {epoch:02d}/{args.epochs} | "
                    f"Batch {batch_index:04d}/{len(train_loader)} | "
                    f"total={loss_total.item():.6f} | "
                    f"recon={loss_recon.item():.6f} | "
                    f"rel={loss_reliability.item():.6f} | "
                    f"w_rel={weighted_loss_reliability.item():.6f} | "
                    f"budget={loss_budget.item():.6f} | "
                    f"w_budget={weighted_loss_budget.item():.6f} | "
                    f"q_grad={q_grad:.3e}",
                    flush=True,
                )

        val_results, slice_rows = evaluate_clean(
            model,
            val_loader,
            device,
            args.max_val_batches,
        )
        diagnostic_results = evaluate_gate_conditions(
            model=model,
            loader=val_loader,
            device=device,
            max_batches=args.gate_diagnostic_batches,
        )

        epoch_seconds = time.time() - epoch_start
        epoch_times.append(epoch_seconds)
        peak_gpu_memory = (
            torch.cuda.max_memory_allocated() / 1024**3
            if device.type == "cuda"
            else 0.0
        )

        total_condition_count = sum(
            len(condition_statistics[name]["count"])
            for name in CONDITION_NAMES
        )

        epoch_row = {
            "epoch": epoch,
            "curriculum_factor": curriculum_factor(
                epoch,
                args.curriculum_warmup_epochs,
                args.curriculum_ramp_epochs,
            ),
            "train_total_loss": safe_mean(total_losses),
            "train_recon_l1": safe_mean(recon_losses),
            "train_reliability_loss": safe_mean(reliability_losses),
            "train_weighted_reliability_loss": safe_mean(
                weighted_reliability_losses
            ),
            "train_budget_loss": safe_mean(budget_losses),
            "train_weighted_budget_loss": safe_mean(weighted_budget_losses),
            "val_pdfs_patient_l1": val_results["pdfs_patient_l1"],
            "val_pdfs_slice_l1": val_results["pdfs_slice_l1"],
            "val_pdfs_edge_l1": val_results["pdfs_edge_l1"],
            "val_pdfs_central_l1": val_results["pdfs_central_l1"],
            "val_clean_q_hat_mean": val_results["clean_q_hat_mean"],
            "val_clean_gated_rms_mean": val_results["clean_gated_rms_mean"],
            "gradient_norm_mean": safe_mean(gradient_norms),
            "q_predictor_gradient_norm_mean": safe_mean(q_gradient_norms),
            "channel_gate_gradient_norm_mean": safe_mean(channel_gradient_norms),
            "spatial_gate_gradient_norm_mean": safe_mean(spatial_gradient_norms),
            "epoch_seconds": epoch_seconds,
            "peak_gpu_memory_gb": peak_gpu_memory,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **diagnostic_results,
        }

        for condition_name in CONDITION_NAMES:
            stats = condition_statistics[condition_name]
            count = len(stats["count"])
            epoch_row[f"train_fraction_{condition_name}"] = (
                count / total_condition_count if total_condition_count else float("nan")
            )
            epoch_row[f"train_q_hat_{condition_name}"] = safe_mean(stats["q_hat"])
            epoch_row[f"train_q_{condition_name}"] = safe_mean(stats["q"])
            epoch_row[f"train_gated_rms_{condition_name}"] = safe_mean(stats["rms"])

        history.append(epoch_row)
        append_training_log(training_log_path, epoch_row)

        print(
            f"Epoch {epoch:02d}/{args.epochs} completed | "
            f"train_recon={epoch_row['train_recon_l1']:.6f} | "
            f"train_rel={epoch_row['train_reliability_loss']:.6f} | "
            f"val_patient={epoch_row['val_pdfs_patient_l1']:.6f} | "
            f"diag q clean/2/4/8/missing="
            f"{epoch_row['diag_q_hat_clean']:.3f}/"
            f"{epoch_row['diag_q_hat_shift2']:.3f}/"
            f"{epoch_row['diag_q_hat_shift4']:.3f}/"
            f"{epoch_row['diag_q_hat_shift8']:.3f}/"
            f"{epoch_row['diag_q_hat_missing']:.3f} | "
            f"diag L1 clean/2/4/8/missing="
            f"{epoch_row['diag_l1_clean']:.4f}/"
            f"{epoch_row['diag_l1_shift2']:.4f}/"
            f"{epoch_row['diag_l1_shift4']:.4f}/"
            f"{epoch_row['diag_l1_shift8']:.4f}/"
            f"{epoch_row['diag_l1_missing']:.4f} | "
            f"time={epoch_seconds:.1f}s | peak_gpu={peak_gpu_memory:.2f}GB",
            flush=True,
        )

        improved = val_results["pdfs_patient_l1"] < best_val
        if improved:
            best_val = float(val_results["pdfs_patient_l1"])
            best_epoch = epoch

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_results": val_results,
            "diagnostic_results": diagnostic_results,
            "best_val": best_val,
            "best_epoch": best_epoch,
            "epoch_times": epoch_times,
            "history": history,
            "config": consistency_config,
            "rng_state": capture_rng_state(),
            "train_sampler_epoch": train_sampler.epoch,
            "val_sampler_epoch": val_sampler.epoch,
        }

        torch.save(checkpoint, output_dir / "model_last.pt")

        if improved:
            torch.save(checkpoint, output_dir / "model_best.pt")
            save_slice_metrics(
                output_dir / "best_val_per_slice_metrics.csv",
                slice_rows,
            )
            print(
                f"New best checkpoint at epoch {epoch}: "
                f"patient-level clean val L1={best_val:.6f}",
                flush=True,
            )

    summary = {
        "best_epoch": best_epoch,
        "best_val_pdfs_patient_l1": best_val,
        "completed_epochs": args.epochs,
        "mean_epoch_seconds": safe_mean(epoch_times),
        "parameter_count": parameter_count,
        "checkpoint_selection_metric": "patient-level mean clean validation PDFS L1",
    }

    if history:
        diagnostic_keys = [
            f"diag_{metric}_{condition}"
            for condition in DIAGNOSTIC_CONDITIONS
            for metric in ("q_hat", "rms", "l1")
        ] + ["diag_q_monotonic", "diag_rms_monotonic"]
        summary["final_gate_diagnostics"] = {
            key: history[-1][key] for key in diagnostic_keys
        }

    with open(
        output_dir / "training_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    print("=" * 80)
    print("M2-GD training finished")
    print("Best epoch:", best_epoch)
    print("Best patient-level clean validation PDFS L1:", best_val)
    print("Output:", output_dir)
    print("=" * 80)


if __name__ == "__main__":
    main()
