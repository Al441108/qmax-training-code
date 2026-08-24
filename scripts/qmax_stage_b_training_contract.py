#!/usr/bin/env python3
from __future__ import annotations

"""Frozen Stage-B loss, optimiser and learning-rate semantics."""

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

import torch
import torch.nn.functional as F

from scripts.qmax_common import l1_per_sample
from src.m2_prnf_corruptions import paired_discrimination_loss


CONTRACT_VERSION = "QMax-StageB-independent-training-contract-v1"
INITIAL_LEARNING_RATE = 3e-4
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 1
LAMBDA_RELIABILITY = 0.05
LAMBDA_RANK = 0.02
LAMBDA_CORRECTION_GAIN = 0.2
CORRECTION_GAIN_MARGIN_RELATIVE = 0.002
AUXILIARY_RAMP_EPOCHS = 5
CORRECTION_GAIN_RAMP_EPOCHS = 5
GRADIENT_CLIP_NORM = 10.0


def learning_rate_for_epoch(epoch: int) -> float:
    if int(epoch) <= 40:
        return 3e-4
    if int(epoch) <= 50:
        return 1e-4
    return 3e-5


def build_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        model.parameters(),
        lr=INITIAL_LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )


def optimizer_spec() -> Dict[str, Any]:
    return {
        "name": "Adam",
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.0,
    }


@dataclass(frozen=True)
class StageBLosses:
    total: torch.Tensor
    reconstruction: torch.Tensor
    clean_l1: torch.Tensor
    corrupt_l1: torch.Tensor
    clean_l1_per_sample: torch.Tensor
    reliability_bce: torch.Tensor
    rank: torch.Tensor
    correction_gain: torch.Tensor
    gain_violation: torch.Tensor


def compute_losses(
    *,
    prediction: torch.Tensor,
    target: torch.Tensor,
    auxiliary: Mapping[str, torch.Tensor],
    base_batch_size: int,
    correction_off_l1: torch.Tensor,
    corruption_records: Sequence[Mapping[str, Any]],
    reliability_target: torch.Tensor,
    epoch: int,
) -> StageBLosses:
    """Compute the exact frozen objective for one paired clean/corrupt batch."""

    base = int(base_batch_size)
    clean_l1_per_sample = l1_per_sample(prediction[:base], target)
    clean_l1 = clean_l1_per_sample.mean()
    corrupt_l1 = l1_per_sample(prediction[base:], target).mean()
    reconstruction = 0.7 * clean_l1 + 0.3 * corrupt_l1

    required_l1 = (
        1.0 - CORRECTION_GAIN_MARGIN_RELATIVE
    ) * correction_off_l1
    correction_gain = F.relu(
        clean_l1_per_sample - required_l1
    ).mean()
    gain_violation = (
        clean_l1_per_sample > required_l1
    ).float().mean()

    logits_clean = auxiliary["q_logits"][:base]
    logits_corrupt = auxiliary["q_logits"][base:]
    clean_bce = F.binary_cross_entropy_with_logits(
        logits_clean, torch.ones_like(logits_clean)
    )
    corrupt_targets = reliability_target[:, None, :].expand_as(
        logits_corrupt
    )
    reliability_mask = torch.tensor(
        [
            record.get("condition") != "missing"
            for record in corruption_records
        ],
        device=prediction.device,
        dtype=torch.bool,
    )
    if bool(reliability_mask.any()):
        corrupt_bce = F.binary_cross_entropy_with_logits(
            logits_corrupt[reliability_mask],
            corrupt_targets[reliability_mask],
        )
        reliability_bce = 0.5 * (clean_bce + corrupt_bce)
    else:
        reliability_bce = clean_bce

    rank, _ = paired_discrimination_loss(
        auxiliary["q_hat"][:base],
        auxiliary["q_hat"][base:],
        reliability_target,
        corruption_records,
    )
    auxiliary_ramp = min(
        1.0, int(epoch) / max(1, AUXILIARY_RAMP_EPOCHS)
    )
    gain_ramp = min(
        1.0,
        int(epoch) / max(1, CORRECTION_GAIN_RAMP_EPOCHS),
    )
    total = reconstruction + auxiliary_ramp * (
        LAMBDA_RELIABILITY * reliability_bce
        + LAMBDA_RANK * rank
    ) + gain_ramp * (
        LAMBDA_CORRECTION_GAIN * correction_gain
    )
    return StageBLosses(
        total=total,
        reconstruction=reconstruction,
        clean_l1=clean_l1,
        corrupt_l1=corrupt_l1,
        clean_l1_per_sample=clean_l1_per_sample,
        reliability_bce=reliability_bce,
        rank=rank,
        correction_gain=correction_gain,
        gain_violation=gain_violation,
    )
