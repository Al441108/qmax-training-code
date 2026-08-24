from __future__ import annotations

"""H/4-only Reliability-Protected Corrective Residual Fusion (RPCRF) model.

Only new filenames are used so installing this overlay cannot change the code
hashes of completed v1.3 checkpoints.  The reconstruction backbone, PD encoder,
per-cascade adapters, alpha parameters and cascade ordering are inherited
unchanged from :mod:`src.m2_prnf_varnet`.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from src.m2_prnf_varnet import (
    M2PRNFAuxPDVarNet,
    PairReliabilityHead,
    TargetNeedHead,
)


FUSION_DESIGNS = {
    "legacy_local_direct",
    "global_direct",
    "residual_only",
    "hybrid_direct_residual",
}

PRNF_VARIANTS = {"prnf_no_rel", "prnf_no_need", "prnf_full"}


def _logit(probability: float) -> float:
    probability = min(1.0 - 1e-6, max(1e-6, float(probability)))
    return math.log(probability / (1.0 - probability))


def _group_count(chans: int) -> int:
    for groups in (8, 4, 2, 1):
        if chans % groups == 0:
            return groups
    return 1


class ComplementaryResidualHead(nn.Module):
    """Predict a bounded local residual and an explicit local transfer gate.

    The final residual projection is zero-initialised while the residual scale
    is fixed to a non-zero registered value. This gives a neutral correction
    at initialisation without the
    zero-times-zero gradient dead zone.
    """

    def __init__(
        self, chans: int, hidden_chans: int = 16, residual_scale: float = 0.1
    ):
        super().__init__()
        if not 0.0 < float(residual_scale) <= 1.0:
            raise ValueError("residual_scale must lie in (0,1]")
        self.register_buffer(
            "residual_scale", torch.tensor(float(residual_scale)), persistent=True
        )
        hidden_chans = int(hidden_chans)
        groups = _group_count(hidden_chans)
        self.context = nn.Sequential(
            nn.Conv2d(4 * chans, hidden_chans, 1, bias=False),
            nn.GroupNorm(groups, hidden_chans),
            nn.GELU(),
            nn.Conv2d(
                hidden_chans, hidden_chans, 3, padding=1,
                groups=hidden_chans, bias=False
            ),
            nn.Conv2d(hidden_chans, hidden_chans, 1, bias=False),
            nn.GroupNorm(groups, hidden_chans),
            nn.GELU(),
        )
        self.residual_out = nn.Conv2d(hidden_chans, chans, 1, bias=False)
        self.spatial_out = nn.Conv2d(hidden_chans, 1, 1)
        self.channel_out = nn.Linear(hidden_chans, chans)
        nn.init.zeros_(self.residual_out.weight)
        gate_bias = _logit(0.5)
        nn.init.zeros_(self.spatial_out.weight)
        nn.init.constant_(self.spatial_out.bias, gate_bias)
        nn.init.zeros_(self.channel_out.weight)
        nn.init.constant_(self.channel_out.bias, gate_bias)

    def forward(
        self, target: torch.Tensor, auxiliary: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if target.shape != auxiliary.shape:
            raise RuntimeError(
                f"Complementary features must align, got {target.shape}/{auxiliary.shape}"
            )
        evidence = torch.cat(
            [target, auxiliary, torch.abs(target - auxiliary), target * auxiliary],
            dim=1,
        )
        context = self.context(evidence)
        # tanh makes the predicted correction explicitly bounded.  The fixed,
        # non-zero scale preserves a gradient for the zero-initialised output.
        residual = torch.tanh(self.residual_out(context))
        spatial = torch.sigmoid(self.spatial_out(context))
        pooled = context.mean(dim=(-2, -1))
        channel = torch.sigmoid(self.channel_out(pooled))[:, :, None, None]
        return residual, spatial, channel


class FusionPilotScaleController(nn.Module):
    """Scale controller with an optional H/4 corrective branch."""

    def __init__(
        self,
        target_chans: int,
        hidden_chans: int,
        initial_gate_probability: float,
        initial_need_probability: float,
        model_variant: str,
        fusion_design: str,
        need_floor: float,
        need_scope: str,
        residual_scale: float,
        residual_reliability_power: float = 1.5,
        residual_enabled: bool = False,
        scale_index: int = -1,
    ):
        super().__init__()
        if model_variant not in PRNF_VARIANTS:
            raise ValueError(model_variant)
        if fusion_design not in FUSION_DESIGNS:
            raise ValueError(fusion_design)
        if need_scope not in {"residual", "all_auxiliary"}:
            raise ValueError(need_scope)
        if not 0.0 <= float(need_floor) < 1.0:
            raise ValueError("need_floor must lie in [0,1)")
        self.model_variant = model_variant
        self.fusion_design = fusion_design
        self.need_floor = float(need_floor)
        self.need_scope = need_scope
        if float(residual_reliability_power) < 1.0:
            raise ValueError("residual_reliability_power must be >= 1")
        self.residual_reliability_power = float(residual_reliability_power)
        self.residual_enabled = bool(residual_enabled)
        self.scale_index = int(scale_index)

        base_seed = int(torch.initial_seed())
        self.reliability = None
        if model_variant in {"prnf_no_need", "prnf_full"}:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(base_seed + 10_000 + int(target_chans))
                self.reliability = PairReliabilityHead(
                    target_chans, hidden_chans, initial_gate_probability
                )

        self.need = None
        if model_variant in {"prnf_no_rel", "prnf_full"}:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(base_seed + 20_000 + int(target_chans))
                self.need = TargetNeedHead(
                    target_chans, hidden_chans, initial_need_probability
                )

        self.complement = None
        if (
            self.residual_enabled
            and fusion_design in {"residual_only", "hybrid_direct_residual"}
        ):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(base_seed + 40_000 + int(target_chans))
                self.complement = ComplementaryResidualHead(
                    target_chans, hidden_chans, residual_scale
                )

    @staticmethod
    def _relative_rms(term: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        numerator = term.detach().square().mean((1, 2, 3)).sqrt()
        denominator = target.detach().square().mean((1, 2, 3)).sqrt().clamp_min(1e-8)
        return numerator / denominator

    @staticmethod
    def _feature_cosine(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        first_flat = first.detach().flatten(1)
        second_flat = second.detach().flatten(1)
        numerator = (first_flat * second_flat).sum(dim=1)
        denominator = (
            first_flat.square().sum(dim=1).sqrt()
            * second_flat.square().sum(dim=1).sqrt()
        ).clamp_min(1e-8)
        return numerator / denominator

    def forward(
        self,
        target: torch.Tensor,
        auxiliary: torch.Tensor,
        dc_evidence: torch.Tensor,
        availability: torch.Tensor,
        variant: str,
        alpha: torch.Tensor,
        reliability_override: Optional[float] = None,
        need_override: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if variant != self.model_variant:
            raise RuntimeError(f"Variant drift: {variant} != {self.model_variant}")
        batch = target.shape[0]
        ones_scalar = target.new_ones(batch)
        ones_spatial = target.new_ones(batch, 1, *target.shape[-2:])
        ones_channel = target.new_ones(batch, target.shape[1], 1, 1)
        zeros = torch.zeros_like(target)

        if self.reliability is not None:
            q_logit, q_hat, legacy_spatial, legacy_channel = self.reliability(
                target, auxiliary
            )
        else:
            q_logit = target.new_zeros(batch)
            q_hat = ones_scalar
            legacy_spatial, legacy_channel = ones_spatial, ones_channel
        if reliability_override is not None:
            q_hat = target.new_full((batch,), float(reliability_override))
            legacy_spatial, legacy_channel = ones_spatial, ones_channel

        if self.need is not None:
            need = self.need(target.detach(), dc_evidence.detach())
        else:
            need = ones_spatial
        if need_override is not None:
            need = target.new_full(ones_spatial.shape, float(need_override))
        need_factor = self.need_floor + (1.0 - self.need_floor) * need

        residual, local_spatial, local_channel = zeros, ones_spatial, ones_channel
        if self.complement is not None:
            # The corrective head is instantiated only at H/4. Non-reentrant
            # checkpointing keeps the L40S microbatch-2 memory envelope bounded.
            if self.training and torch.is_grad_enabled():
                residual, local_spatial, local_channel = checkpoint(
                    self.complement, target, auxiliary, use_reentrant=False
                )
            else:
                residual, local_spatial, local_channel = self.complement(
                    target, auxiliary
                )

        m = availability.to(target.dtype).view(-1, 1, 1, 1)
        q = q_hat[:, None, None, None]
        # Direct anatomical guidance keeps the calibrated reliability q.  The
        # higher-risk corrective branch is protected more strictly without
        # changing clean availability semantics or introducing a second head.
        q_residual = q.pow(self.residual_reliability_power)
        direct_raw = alpha * auxiliary
        residual_scale = (
            self.complement.residual_scale if self.complement is not None
            else target.new_tensor(0.0)
        )
        residual_raw = residual_scale * local_spatial * local_channel * residual

        if self.fusion_design == "legacy_local_direct":
            direct_term = m * q * legacy_spatial * legacy_channel * need_factor * direct_raw
            residual_term = zeros
        elif self.fusion_design == "global_direct":
            direct_need = need_factor if self.need_scope == "all_auxiliary" else 1.0
            direct_term = m * q * direct_need * direct_raw
            residual_term = zeros
        elif self.fusion_design == "residual_only":
            direct_term = zeros
            residual_term = m * q_residual * need_factor * residual_raw
        elif self.fusion_design == "hybrid_direct_residual":
            direct_need = need_factor if self.need_scope == "all_auxiliary" else 1.0
            direct_term = m * q * direct_need * direct_raw
            residual_term = m * q_residual * need_factor * residual_raw
        else:  # pragma: no cover
            raise RuntimeError(self.fusion_design)

        total_term = direct_term + residual_term
        fused = target + total_term
        direct_rms = self._relative_rms(direct_term, target)
        residual_rms = self._relative_rms(residual_term, target)
        total_rms = self._relative_rms(total_term, target)
        residual_direct_ratio = torch.where(
            direct_rms > 1e-8,
            residual_rms / direct_rms.clamp_min(1e-8),
            direct_rms.new_full(direct_rms.shape, -1.0),
        )
        diagnostics = {
            "q_logits": q_logit,
            "q_hat": q_hat,
            "q": availability.to(target.dtype) * q_hat,
            "reliability_spatial_mean": legacy_spatial.detach().mean((1, 2, 3)),
            "reliability_channel_mean": legacy_channel.detach().mean((1, 2, 3)),
            "need_mean": need.detach().mean((1, 2, 3)),
            "need_p05": need.detach().flatten(1).quantile(0.05, dim=1),
            "need_p95": need.detach().flatten(1).quantile(0.95, dim=1),
            "need_factor_mean": need_factor.detach().mean((1, 2, 3)),
            "effective_weight_mean": (m * q).detach().mean((1, 2, 3)),
            "residual_effective_weight_mean": (
                m * q_residual
            ).detach().mean((1, 2, 3)),
            "residual_reliability_power": target.new_full(
                (batch,), self.residual_reliability_power
            ),
            "gated_aux_to_target_rms": total_rms,
            "direct_to_target_rms": direct_rms,
            "residual_to_target_rms": residual_rms,
            "total_aux_to_target_rms": total_rms,
            "local_spatial_gate_mean": local_spatial.detach().mean((1, 2, 3)),
            "local_channel_gate_mean": local_channel.detach().mean((1, 2, 3)),
            "raw_residual_to_target_rms": self._relative_rms(residual, target),
            "residual_to_direct_rms_ratio": residual_direct_ratio,
            "residual_scale": residual_scale.detach().expand(batch),
            "residual_enabled": target.new_full(
                (batch,), float(self.residual_enabled)
            ),
            "scale_index": target.new_full((batch,), float(self.scale_index)),
            "raw_auxiliary_to_target_rms": self._relative_rms(auxiliary, target),
            "target_feature_rms": (
                target.detach().square().mean((1, 2, 3)).sqrt()
            ),
            "auxiliary_feature_rms": (
                auxiliary.detach().square().mean((1, 2, 3)).sqrt()
            ),
            "target_auxiliary_cosine": self._feature_cosine(target, auxiliary),
            "alpha": alpha.detach().expand(batch),
        }
        return fused, diagnostics


class M2PRNFRPCRFVarNet(M2PRNFAuxPDVarNet):
    """Global-direct PRNF plus an H/4-only protected corrective residual."""

    def __init__(
        self,
        model_variant: str = "prnf_no_need",
        fusion_design: str = "global_direct",
        need_scope: str = "residual",
        residual_scale: float = 0.1,
        residual_reliability_power: float = 1.5,
        residual_levels: Tuple[int, ...] = (2,),
        **kwargs,
    ):
        if model_variant not in PRNF_VARIANTS:
            raise ValueError(f"Unknown PRNF variant: {model_variant}")
        if fusion_design not in FUSION_DESIGNS:
            raise ValueError(f"Unknown fusion design: {fusion_design}")
        controller_chans = int(kwargs.get("controller_chans", 16))
        initial_gate_probability = float(kwargs.get("initial_gate_probability", 0.95))
        initial_need_probability = float(kwargs.get("initial_need_probability", 0.95))
        need_floor = float(kwargs.get("need_floor", 0.25))
        chans = int(kwargs.get("chans", 18))
        pools = int(kwargs.get("pools", 4))
        residual_levels = tuple(int(level) for level in residual_levels)
        if residual_levels != (2,):
            raise ValueError(
                "The frozen final candidate requires residual_levels=(2,) "
                "(H/4 only)"
            )
        if pools < 2:
            raise ValueError("H/4 residual requires pools >= 2")
        super().__init__(variant=model_variant, **kwargs)
        self.fusion_design = fusion_design
        self.need_scope = need_scope
        self.residual_reliability_power = float(residual_reliability_power)
        self.residual_levels = residual_levels

        # Replacement happens after all reconstruction parameters have been
        # built, so their seed-matched initial states remain unchanged.
        self.controllers = nn.ModuleList(
            [
                FusionPilotScaleController(
                    chans * (2 ** level),
                    controller_chans,
                    initial_gate_probability,
                    initial_need_probability,
                    model_variant,
                    fusion_design,
                    need_floor,
                    need_scope,
                    residual_scale,
                    residual_reliability_power,
                    residual_enabled=level in residual_levels,
                    scale_index=level,
                )
                for level in range(1, pools + 1)
            ]
        )


def shared_reconstruction_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Return state used by the strict M2-U compatibility preflight."""
    return {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith("controllers.")
    }
