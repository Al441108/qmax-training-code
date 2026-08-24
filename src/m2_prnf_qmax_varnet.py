from __future__ import annotations

"""Frozen Stage-A QMax models for R=2 auxiliary PD reconstruction.

This module deliberately imports the audited fifth-arm building blocks instead
of modifying them.  Existing checkpoints therefore retain their original code
hashes while QMax-Core and QMax-Full live under new class names.

QMax-Core:
    fixed binomial stable/detail split + learned detail selector + learned
    alignment + pair-aware direct injection + signed post-direct correction.

QMax-Full:
    QMax-Core plus detached, RMS-normalised sampled PD-FS DC residual evidence
    supplied only to the detail selector and correction head.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import fastmri
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastmri.models.varnet import SensitivityModel
from torch.utils.checkpoint import checkpoint

from src.m2_prnf_varnet import (
    ConvBlock,
    PDFeatureEncoder,
    PairReliabilityHead,
    _center_crop_or_pad,
    _normalise_per_sample,
    _pad_to_multiple,
    _unpad,
)


QMAX_VARIANTS = {"qmax_core", "qmax_full"}
QMAX_SCALE_NAMES = ("H/2", "H/4", "H/8", "H/16")
QMAX_CORRECTION_BETA = 0.1


def _group_count(chans: int) -> int:
    for groups in (8, 4, 2, 1):
        if chans % groups == 0:
            return groups
    return 1


def _relative_rms(term: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    numerator = term.detach().square().mean((1, 2, 3)).sqrt()
    denominator = target.detach().square().mean((1, 2, 3)).sqrt().clamp_min(1e-8)
    return numerator / denominator


def _feature_cosine(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first_flat = first.detach().flatten(1)
    second_flat = second.detach().flatten(1)
    numerator = (first_flat * second_flat).sum(dim=1)
    denominator = (
        first_flat.square().sum(dim=1).sqrt()
        * second_flat.square().sum(dim=1).sqrt()
    ).clamp_min(1e-8)
    return numerator / denominator


class FixedBinomialBlur(nn.Module):
    """Depthwise 3x3 binomial Gaussian with reflect padding."""

    def __init__(self, chans: int):
        super().__init__()
        kernel = torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]
        ) / 16.0
        weight = kernel.view(1, 1, 3, 3).repeat(int(chans), 1, 1, 1)
        self.register_buffer("weight", weight, persistent=True)
        self.chans = int(chans)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1] != self.chans:
            raise RuntimeError(
                f"Expected [B,{self.chans},H,W], got {tuple(value.shape)}"
            )
        return F.conv2d(
            F.pad(value, (1, 1, 1, 1), mode="reflect"),
            self.weight,
            groups=self.chans,
        )


class ZeroOutputEvidenceHead(nn.Module):
    """Local evidence head with a strictly zero initial output projection."""

    def __init__(self, in_chans: int, out_chans: int, hidden_chans: int):
        super().__init__()
        hidden_chans = int(hidden_chans)
        groups = _group_count(hidden_chans)
        # There is intentionally no normalisation before in_proj.  In
        # QMax-Full the appended DC column is zero initialised and therefore
        # cannot change pre-convolution statistics.
        self.in_proj = nn.Conv2d(in_chans, hidden_chans, 1, bias=False)
        self.context = nn.Sequential(
            nn.GroupNorm(groups, hidden_chans),
            nn.GELU(),
            nn.Conv2d(
                hidden_chans,
                hidden_chans,
                3,
                padding=1,
                groups=hidden_chans,
                bias=False,
            ),
            nn.Conv2d(hidden_chans, hidden_chans, 1, bias=False),
            nn.GroupNorm(groups, hidden_chans),
            nn.GELU(),
        )
        self.out = nn.Conv2d(hidden_chans, out_chans, 1, bias=True)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        return self.out(self.context(self.in_proj(evidence)))


class QMaxScaleController(nn.Module):
    """Scale-specific controller shared across all cascades."""

    def __init__(
        self,
        target_chans: int,
        hidden_chans: int,
        initial_gate_probability: float,
        qmax_variant: str,
    ):
        super().__init__()
        if qmax_variant not in QMAX_VARIANTS:
            raise ValueError(qmax_variant)
        self.target_chans = int(target_chans)
        self.qmax_variant = str(qmax_variant)
        self.uses_dc_evidence = self.qmax_variant == "qmax_full"

        base_seed = int(torch.initial_seed())
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(base_seed + 10_000 + self.target_chans)
            self.reliability = PairReliabilityHead(
                self.target_chans,
                int(hidden_chans),
                float(initial_gate_probability),
            )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(base_seed + 50_000 + self.target_chans)
            self.detail_head = ZeroOutputEvidenceHead(
                5 * self.target_chans + int(self.uses_dc_evidence),
                self.target_chans,
                int(hidden_chans),
            )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(base_seed + 60_000 + self.target_chans)
            self.alignment_head = ZeroOutputEvidenceHead(
                5 * self.target_chans,
                self.target_chans,
                int(hidden_chans),
            )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(base_seed + 70_000 + self.target_chans)
            self.correction_head = ZeroOutputEvidenceHead(
                6 * self.target_chans + int(self.uses_dc_evidence),
                self.target_chans,
                int(hidden_chans),
            )
        self.blur = FixedBinomialBlur(self.target_chans)
        self.register_buffer(
            "correction_beta",
            torch.tensor(float(QMAX_CORRECTION_BETA)),
            persistent=True,
        )

    def forward(
        self,
        target: torch.Tensor,
        auxiliary_u0: torch.Tensor,
        dc_evidence: torch.Tensor,
        dc_raw_rms: torch.Tensor,
        availability: torch.Tensor,
        alpha: torch.Tensor,
        reliability_override: Optional[float] = None,
        detail_neutral: bool = False,
        alignment_off: bool = False,
        correction_off: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if target.shape != auxiliary_u0.shape:
            raise RuntimeError(
                f"QMax features must align, got {target.shape}/{auxiliary_u0.shape}"
            )
        batch = target.shape[0]
        m = availability.to(target.dtype).view(batch, 1, 1, 1)

        # The reliability head must observe the untouched adapter output U0.
        q_logit, q_hat, _, _ = self.reliability(target, auxiliary_u0)
        if reliability_override is not None:
            q_hat = target.new_full((batch,), float(reliability_override))
        q = q_hat[:, None, None, None]

        stable = self.blur(auxiliary_u0)
        detail = auxiliary_u0 - stable
        alignment_evidence = torch.cat(
            [
                target,
                stable,
                detail,
                torch.abs(target - stable),
                target * stable,
            ],
            dim=1,
        )
        dc_scaled = F.interpolate(
            dc_evidence.detach(),
            size=target.shape[-2:],
            mode="area",
        )
        detail_evidence = alignment_evidence
        if self.uses_dc_evidence:
            detail_evidence = torch.cat([detail_evidence, dc_scaled], dim=1)

        if self.training and torch.is_grad_enabled():
            detail_logits = checkpoint(
                self.detail_head,
                detail_evidence,
                use_reentrant=False,
            )
        else:
            detail_logits = self.detail_head(detail_evidence)
        detail_gate = 2.0 * torch.sigmoid(detail_logits)
        if detail_neutral:
            detail_gate = torch.ones_like(detail_gate)

        if self.training and torch.is_grad_enabled():
            alignment = checkpoint(
                self.alignment_head,
                alignment_evidence,
                use_reentrant=False,
            )
        else:
            alignment = self.alignment_head(alignment_evidence)
        if alignment_off:
            alignment = torch.zeros_like(alignment)
        auxiliary = stable + detail_gate * detail + alignment

        direct = m * q * alpha * auxiliary
        direct_fused = target + direct
        correction_evidence = torch.cat(
            [
                direct_fused,
                target,
                direct,
                auxiliary,
                torch.abs(direct_fused - auxiliary),
                direct_fused * auxiliary,
            ],
            dim=1,
        )
        if self.uses_dc_evidence:
            correction_evidence = torch.cat(
                [correction_evidence, dc_scaled], dim=1
            )
        if self.training and torch.is_grad_enabled():
            correction_logits = checkpoint(
                self.correction_head,
                correction_evidence,
                use_reentrant=False,
            )
        else:
            correction_logits = self.correction_head(correction_evidence)
        correction_raw = torch.tanh(correction_logits)
        if correction_off:
            correction_raw = torch.zeros_like(correction_raw)
        correction = m * q * self.correction_beta * correction_raw
        total = direct + correction
        fused = target + total

        direct_rms = _relative_rms(direct, target)
        correction_rms = _relative_rms(correction, target)
        correction_to_direct = torch.where(
            direct_rms > 1e-8,
            correction_rms / direct_rms.clamp_min(1e-8),
            torch.full_like(direct_rms, -1.0),
        )
        diagnostics = {
            "q_logits": q_logit,
            "q_hat": q_hat,
            "q": availability.to(target.dtype) * q_hat,
            "alpha": alpha.detach().expand(batch),
            "detail_gate_mean": detail_gate.detach().mean((1, 2, 3)),
            "detail_gate_std": detail_gate.detach().flatten(1).std(dim=1),
            "detail_gate_min": detail_gate.detach().flatten(1).amin(dim=1),
            "detail_gate_max": detail_gate.detach().flatten(1).amax(dim=1),
            "alignment_to_target_rms": _relative_rms(alignment, target),
            "direct_to_target_rms": direct_rms,
            "correction_to_target_rms": correction_rms,
            "final_auxiliary_to_target_rms": _relative_rms(total, target),
            "cos_direct_correction": _feature_cosine(direct, correction),
            "dc_raw_rms": dc_raw_rms.detach().reshape(batch),
            "dc_normalized_rms": (
                dc_evidence.detach().square().mean((1, 2, 3)).sqrt()
            ),
            "raw_auxiliary_to_target_rms": _relative_rms(auxiliary_u0, target),
            "selected_auxiliary_to_target_rms": _relative_rms(auxiliary, target),
            "selected_minus_u0_max_abs": (
                (auxiliary - auxiliary_u0)
                .detach()
                .flatten(1)
                .abs()
                .amax(dim=1)
            ),
            "target_feature_rms": (
                target.detach().square().mean((1, 2, 3)).sqrt()
            ),
            "auxiliary_feature_rms": (
                auxiliary_u0.detach().square().mean((1, 2, 3)).sqrt()
            ),
            "target_auxiliary_cosine": _feature_cosine(target, auxiliary_u0),
            # Compatibility aliases for the existing quality-gain reporting.
            "gated_aux_to_target_rms": _relative_rms(total, target),
            "total_aux_to_target_rms": _relative_rms(total, target),
            "residual_to_target_rms": correction_rms,
            "raw_residual_to_target_rms": _relative_rms(
                correction_raw, target
            ),
            "residual_to_direct_rms_ratio": correction_to_direct,
            "residual_scale": self.correction_beta.detach().expand(batch),
            "need_mean": torch.ones_like(q_hat),
            "need_p05": torch.ones_like(q_hat),
            "need_p95": torch.ones_like(q_hat),
            "need_factor_mean": torch.ones_like(q_hat),
            "effective_weight_mean": (m * q).detach().mean((1, 2, 3)),
            "reliability_spatial_mean": torch.ones_like(q_hat),
            "reliability_channel_mean": torch.ones_like(q_hat),
            "local_spatial_gate_mean": detail_gate.detach().mean((1, 2, 3)),
            "local_channel_gate_mean": torch.ones_like(q_hat),
        }
        return fused, diagnostics


class QMaxFeatureFusion(nn.Module):
    """Per-cascade adapter and alpha; controller remains shared by scale."""

    def __init__(self, pd_chans: int, target_chans: int, initial_alpha: float):
        super().__init__()
        conv = nn.Conv2d(pd_chans, target_chans, 1, bias=False)
        nn.init.kaiming_normal_(conv.weight, a=0.2)
        self.adapter = nn.Sequential(
            conv,
            nn.InstanceNorm2d(target_chans, affine=False),
        )
        self.alpha = nn.Parameter(torch.tensor(float(initial_alpha)))

    def forward(
        self,
        target: torch.Tensor,
        pd_feature: torch.Tensor,
        controller: QMaxScaleController,
        dc_evidence: torch.Tensor,
        dc_raw_rms: torch.Tensor,
        availability: torch.Tensor,
        reliability_override: Optional[float],
        detail_neutral: bool,
        alignment_off: bool,
        correction_off: bool,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if pd_feature.shape[-2:] != target.shape[-2:]:
            pd_feature = F.interpolate(
                pd_feature,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        auxiliary_u0 = self.adapter(pd_feature)
        return controller(
            target=target,
            auxiliary_u0=auxiliary_u0,
            dc_evidence=dc_evidence,
            dc_raw_rms=dc_raw_rms,
            availability=availability,
            alpha=self.alpha,
            reliability_override=reliability_override,
            detail_neutral=detail_neutral,
            alignment_off=alignment_off,
            correction_off=correction_off,
        )


class QMaxRegulariser(nn.Module):
    def __init__(
        self,
        chans: int = 18,
        pools: int = 4,
        initial_aux_alpha: float = 0.1,
    ):
        super().__init__()
        self.chans, self.pools = int(chans), int(pools)
        down: List[nn.Module] = []
        in_chans = 2
        for level in range(self.pools):
            out_chans = self.chans * (2**level)
            down.append(ConvBlock(in_chans, out_chans))
            in_chans = out_chans
        self.target_down = nn.ModuleList(down)
        self.target_bottleneck = ConvBlock(
            self.chans * (2 ** (self.pools - 1)),
            self.chans * (2**self.pools),
        )
        self.fusions = nn.ModuleList(
            [
                QMaxFeatureFusion(
                    self.chans * (2**level),
                    self.chans * (2**level),
                    initial_aux_alpha,
                )
                for level in range(1, self.pools + 1)
            ]
        )
        up: List[nn.Module] = []
        up_chans = self.chans * (2**self.pools)
        for level in reversed(range(self.pools)):
            skip_chans = self.chans * (2**level)
            up.append(ConvBlock(up_chans + skip_chans, skip_chans))
            up_chans = skip_chans
        self.up = nn.ModuleList(up)
        self.out_conv = nn.Conv2d(self.chans, 2, 1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    @staticmethod
    def complex_to_channels(value: torch.Tensor) -> torch.Tensor:
        return value[:, 0].permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def channels_to_complex(value: torch.Tensor) -> torch.Tensor:
        return value.permute(0, 2, 3, 1).unsqueeze(1).contiguous()

    def forward(
        self,
        pdfs_image: torch.Tensor,
        pd_features: Sequence[torch.Tensor],
        controllers: Sequence[QMaxScaleController],
        dc_evidence: torch.Tensor,
        dc_raw_rms: torch.Tensor,
        availability: torch.Tensor,
        reliability_override: Optional[float],
        detail_neutral: bool,
        alignment_off: bool,
        correction_off: bool,
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        value = self.complex_to_channels(pdfs_image)
        original_hw = value.shape[-2:]
        value, target_std = _normalise_per_sample(value)
        value, pads = _pad_to_multiple(value, 2**self.pools)
        dc_evidence = _center_crop_or_pad(dc_evidence, original_hw)
        dc_evidence, _ = _pad_to_multiple(dc_evidence, 2**self.pools)

        skips: List[torch.Tensor] = []
        diagnostics: List[Dict[str, torch.Tensor]] = []
        out = value
        for level, block in enumerate(self.target_down):
            out = block(out)
            if level == 0:
                fused = out
            else:
                fused, diag = self.fusions[level - 1](
                    out,
                    pd_features[level],
                    controllers[level - 1],
                    dc_evidence,
                    dc_raw_rms,
                    availability,
                    reliability_override,
                    detail_neutral,
                    alignment_off,
                    correction_off,
                )
                diagnostics.append(diag)
            skips.append(fused)
            # Deeper encoding follows the target-only path.
            out = F.avg_pool2d(out, 2, 2)

        out = self.target_bottleneck(out)
        out, diag = self.fusions[-1](
            out,
            pd_features[-1],
            controllers[-1],
            dc_evidence,
            dc_raw_rms,
            availability,
            reliability_override,
            detail_neutral,
            alignment_off,
            correction_off,
        )
        diagnostics.append(diag)
        for block, skip in zip(self.up, reversed(skips)):
            out = F.interpolate(
                out,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            out = block(torch.cat([out, skip], dim=1))
        out = _unpad(self.out_conv(out), pads) * target_std
        if out.shape[-2:] != original_hw:
            raise RuntimeError("QMax regulariser changed target spatial shape")
        return self.channels_to_complex(out), diagnostics


class QMaxVarNetBlock(nn.Module):
    def __init__(
        self,
        chans: int = 18,
        pools: int = 4,
        initial_aux_alpha: float = 0.1,
        qmax_variant: str = "qmax_core",
    ):
        super().__init__()
        if qmax_variant not in QMAX_VARIANTS:
            raise ValueError(qmax_variant)
        self.qmax_variant = str(qmax_variant)
        self.regulariser = QMaxRegulariser(chans, pools, initial_aux_alpha)
        self.pdfs_dc_weight = nn.Parameter(torch.ones(1))

    @staticmethod
    def sens_expand(image: torch.Tensor, sens: torch.Tensor) -> torch.Tensor:
        return fastmri.fft2c(fastmri.complex_mul(image, sens))

    @staticmethod
    def sens_reduce(kspace: torch.Tensor, sens: torch.Tensor) -> torch.Tensor:
        return fastmri.complex_mul(
            fastmri.ifft2c(kspace), fastmri.complex_conj(sens)
        ).sum(dim=1, keepdim=True)

    def forward(
        self,
        current: torch.Tensor,
        reference: torch.Tensor,
        mask: torch.Tensor,
        sens: torch.Tensor,
        pd_features: Sequence[torch.Tensor],
        controllers: Sequence[QMaxScaleController],
        availability: torch.Tensor,
        reliability_override: Optional[float],
        detail_neutral: bool,
        alignment_off: bool,
        correction_off: bool,
        dc_zero: bool,
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        zero = torch.zeros(
            1, 1, 1, 1, 1, device=current.device, dtype=current.dtype
        )
        sampled_residual = torch.where(mask, current - reference, zero)
        soft_dc = sampled_residual * self.pdfs_dc_weight

        dc_image = self.sens_reduce(sampled_residual, sens)
        dc_magnitude = fastmri.complex_abs(dc_image).detach()
        dc_raw_rms = (
            dc_magnitude.square().mean((1, 2, 3)).add(1e-8).sqrt()
        )
        dc_evidence = dc_magnitude / dc_raw_rms[:, None, None, None]
        # DC-zero is defined after RMS normalisation.
        if dc_zero or self.qmax_variant == "qmax_core":
            dc_evidence = torch.zeros_like(dc_evidence)

        target_image = self.sens_reduce(current, sens)
        regularisation, diagnostics = self.regulariser(
            target_image,
            pd_features,
            controllers,
            dc_evidence.detach(),
            dc_raw_rms.detach(),
            availability,
            reliability_override,
            detail_neutral,
            alignment_off,
            correction_off,
        )
        next_current = (
            current
            - soft_dc
            - self.sens_expand(regularisation, sens)
        )
        return next_current, diagnostics


class QMaxAuxPDVarNet(nn.Module):
    """Twelve-cascade QMax-Core/QMax-Full VarNet."""

    def __init__(
        self,
        qmax_variant: str = "qmax_core",
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        mask_center: bool = True,
        controller_chans: int = 16,
        initial_aux_alpha: float = 0.1,
        initial_gate_probability: float = 0.95,
    ):
        super().__init__()
        if qmax_variant not in QMAX_VARIANTS:
            raise ValueError(
                f"Unknown QMax variant {qmax_variant}; "
                f"choose from {sorted(QMAX_VARIANTS)}"
            )
        if int(pools) != len(QMAX_SCALE_NAMES):
            raise ValueError("Frozen QMax requires exactly four scales")
        self.qmax_variant = str(qmax_variant)
        self.pools = int(pools)
        self.sens_net = SensitivityModel(
            chans=int(sens_chans),
            num_pools=int(sens_pools),
            mask_center=bool(mask_center),
        )
        self.pd_encoder = PDFeatureEncoder(chans=int(chans), pools=int(pools))
        self.cascades = nn.ModuleList(
            [
                QMaxVarNetBlock(
                    chans=int(chans),
                    pools=int(pools),
                    initial_aux_alpha=float(initial_aux_alpha),
                    qmax_variant=self.qmax_variant,
                )
                for _ in range(int(num_cascades))
            ]
        )
        self.controllers = nn.ModuleList(
            [
                QMaxScaleController(
                    target_chans=int(chans) * (2**level),
                    hidden_chans=int(controller_chans),
                    initial_gate_probability=float(initial_gate_probability),
                    qmax_variant=self.qmax_variant,
                )
                for level in range(1, int(pools) + 1)
            ]
        )

    def forward(
        self,
        pdfs_masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        pd_aux_image: torch.Tensor,
        pd_available: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        reliability_override: Optional[float] = None,
        detail_neutral: bool = False,
        alignment_off: bool = False,
        correction_off: bool = False,
        dc_zero: bool = False,
    ):
        if mask.dtype != torch.bool:
            mask = mask.bool()
        batch = pdfs_masked_kspace.shape[0]
        if pd_available is None:
            pd_available = torch.ones(
                batch, device=pdfs_masked_kspace.device
            )
        pd_available = pd_available.reshape(batch)
        if not bool(
            torch.logical_or(pd_available == 0, pd_available == 1).all()
        ):
            raise RuntimeError("pd_available must contain only zero or one")

        target_hw = (
            pdfs_masked_kspace.shape[-3],
            pdfs_masked_kspace.shape[-2],
        )
        if pd_aux_image.ndim == 3:
            pd_aux_image = pd_aux_image[:, None]
        pd_aux_image = _center_crop_or_pad(pd_aux_image.float(), target_hw)
        pd_features = self.pd_encoder(pd_aux_image)
        sensitivity = self.sens_net(pdfs_masked_kspace, mask)
        current = pdfs_masked_kspace.clone()
        all_diagnostics: List[List[Dict[str, torch.Tensor]]] = []
        for cascade in self.cascades:
            current, diagnostics = cascade(
                current=current,
                reference=pdfs_masked_kspace,
                mask=mask,
                sens=sensitivity,
                pd_features=pd_features,
                controllers=self.controllers,
                availability=pd_available,
                reliability_override=reliability_override,
                detail_neutral=bool(detail_neutral),
                alignment_off=bool(alignment_off),
                correction_off=bool(correction_off),
                dc_zero=bool(dc_zero),
            )
            all_diagnostics.append(diagnostics)

        # Preserve the audited fifth-arm VarNet update: every cascade combines
        # a learned soft-DC term and a regularisation term.  No final hard
        # projection is added here, and no auxiliary module runs after the
        # final cascade update.
        prediction = fastmri.rss(
            fastmri.complex_abs(fastmri.ifft2c(current)), dim=1
        )
        if not return_aux:
            return prediction
        keys = tuple(all_diagnostics[0][0].keys())
        auxiliary: Dict[str, torch.Tensor] = {}
        for key in keys:
            auxiliary[key] = torch.stack(
                [
                    torch.stack(
                        [scale[key] for scale in cascade_diagnostics],
                        dim=1,
                    )
                    for cascade_diagnostics in all_diagnostics
                ],
                dim=1,
            )
        auxiliary["scale_names"] = QMAX_SCALE_NAMES
        zero = torch.zeros(
            1, 1, 1, 1, 1, device=current.device, dtype=current.dtype
        )
        final_sampled_residual = torch.where(
            mask, current - pdfs_masked_kspace, zero
        )
        auxiliary["final_sampled_kspace_residual_max_abs"] = (
            fastmri.complex_abs(final_sampled_residual)
            .detach()
            .flatten(1)
            .amax(dim=1)
        )
        auxiliary["hard_final_dc_projection_applied"] = False
        auxiliary["post_final_cascade_auxiliary_module"] = False
        return prediction, auxiliary


def copy_matching_state(
    source: nn.Module,
    destination: nn.Module,
) -> Dict[str, object]:
    """Copy all same-name/same-shape tensors without using ``strict=False``."""

    source_state = source.state_dict()
    destination_state = destination.state_dict()
    copied: List[str] = []
    with torch.no_grad():
        for key, destination_value in destination_state.items():
            source_value = source_state.get(key)
            if source_value is None or source_value.shape != destination_value.shape:
                continue
            destination_value.copy_(source_value)
            copied.append(key)
    destination.load_state_dict(destination_state, strict=True)
    return {
        "copied_keys": copied,
        "num_copied": len(copied),
    }


def initialise_qmax_full_from_core(
    core: QMaxAuxPDVarNet,
    full: QMaxAuxPDVarNet,
) -> Dict[str, object]:
    """Make Full identical to Core at step 0, with zero DC input columns."""

    if core.qmax_variant != "qmax_core":
        raise ValueError("Source must be qmax_core")
    if full.qmax_variant != "qmax_full":
        raise ValueError("Destination must be qmax_full")
    core_state = core.state_dict()
    full_state = full.state_dict()
    copied: List[str] = []
    extended: List[str] = []
    unmatched: List[str] = []
    with torch.no_grad():
        for key, full_value in full_state.items():
            core_value = core_state.get(key)
            if core_value is None:
                unmatched.append(key)
                continue
            if core_value.shape == full_value.shape:
                full_value.copy_(core_value)
                copied.append(key)
                continue
            is_extended_input = (
                key.endswith("detail_head.in_proj.weight")
                or key.endswith("correction_head.in_proj.weight")
            )
            if (
                is_extended_input
                and full_value.ndim == 4
                and core_value.ndim == 4
                and full_value.shape[0] == core_value.shape[0]
                and full_value.shape[1] == core_value.shape[1] + 1
                and full_value.shape[2:] == core_value.shape[2:]
            ):
                full_value.zero_()
                full_value[:, : core_value.shape[1]].copy_(core_value)
                extended.append(key)
                continue
            unmatched.append(key)
    full.load_state_dict(full_state, strict=True)
    if unmatched:
        raise RuntimeError(
            "Unexpected Core/Full state mismatch: " + ", ".join(unmatched)
        )
    return {
        "copied_same_shape": copied,
        "zero_extended_inputs": extended,
        "num_same_shape": len(copied),
        "num_extended": len(extended),
    }


def qmax_shared_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    """State common to Core and Full, excluding the two extended input layers."""

    excluded_suffixes = (
        "detail_head.in_proj.weight",
        "correction_head.in_proj.weight",
    )
    return {
        key: value
        for key, value in model.state_dict().items()
        if not key.endswith(excluded_suffixes)
    }


def qmax_dc_input_columns(model: QMaxAuxPDVarNet) -> Dict[str, torch.Tensor]:
    if model.qmax_variant != "qmax_full":
        return {}
    output: Dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if name.endswith("detail_head.in_proj") or name.endswith(
            "correction_head.in_proj"
        ):
            if not isinstance(module, nn.Conv2d):
                raise RuntimeError(f"Unexpected DC input module {name}")
            output[f"{name}.weight"] = module.weight[:, -1:].detach()
    return output
