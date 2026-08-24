from __future__ import annotations

"""Unified, from-scratch auxiliary VarNet used by the final fair comparison.

The module vendors the audited low-level building blocks so its implementation
cannot silently drift with an external M2-GD file. It adds two
*architecturally separated* controls:

* pair reliability: may inspect target/auxiliary disagreement, but not DC;
* target need: may inspect target features and DC correction, but not PD.

Controllers are shared across cascades at each scale.  Per-cascade adapters
remain separate, matching the original M2-U regulariser design.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import fastmri
from fastmri.models.varnet import SensitivityModel

VALID_VARIANTS = {
    "m2u_clean",
    "m2u_augmented",
    "m2u_augcap_mask",
    "prnf_no_rel",
    "prnf_no_need",
    "prnf_full",
}


class ConvBlock(nn.Module):
    """Two-convolution block kept identical to the audited M2-U backbone."""

    def __init__(self, in_chans: int, out_chans: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_chans, out_chans, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def _pad_to_multiple(
    x: torch.Tensor, multiple: int
) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    height, width = x.shape[-2:]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    top, left = pad_h // 2, pad_w // 2
    bottom, right = pad_h - top, pad_w - left
    return F.pad(x, (left, right, top, bottom), mode="reflect"), (
        top, bottom, left, right
    )


def _unpad(x: torch.Tensor, pads: Tuple[int, int, int, int]) -> torch.Tensor:
    top, bottom, left, right = pads
    h_end = x.shape[-2] - bottom if bottom else x.shape[-2]
    w_end = x.shape[-1] - right if right else x.shape[-1]
    return x[..., top:h_end, left:w_end]


def _normalise_per_sample(
    x: torch.Tensor, eps: float = 1e-7
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch = x.shape[0]
    flat = x.reshape(batch, -1)
    mean = flat.mean(1).view(batch, 1, 1, 1)
    std = flat.std(1).view(batch, 1, 1, 1).clamp_min(eps)
    return (x - mean) / std, std


def _center_crop_or_pad(x: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    target_h, target_w = map(int, target_hw)
    height, width = x.shape[-2:]
    if height > target_h:
        top = (height - target_h) // 2
        x, height = x[..., top : top + target_h, :], target_h
    if width > target_w:
        left = (width - target_w) // 2
        x, width = x[..., left : left + target_w], target_w
    pad_h, pad_w = max(0, target_h - height), max(0, target_w - width)
    if pad_h or pad_w:
        top, left = pad_h // 2, pad_w // 2
        x = F.pad(
            x,
            (left, pad_w - left, top, pad_h - top),
            mode="constant",
            value=0.0,
        )
    return x


def _logit(probability: float) -> float:
    probability = min(1.0 - 1e-6, max(1e-6, float(probability)))
    return math.log(probability / (1.0 - probability))


class PDFeatureEncoder(nn.Module):
    """Shared M2-U PD encoder, computed once and reused across cascades."""

    def __init__(self, chans: int = 18, pools: int = 4):
        super().__init__()
        self.chans, self.pools = int(chans), int(pools)
        blocks, in_chans = [], 1
        for level in range(self.pools):
            out_chans = self.chans * (2 ** level)
            blocks.append(ConvBlock(in_chans, out_chans))
            in_chans = out_chans
        self.down = nn.ModuleList(blocks)
        self.bottleneck = ConvBlock(
            self.chans * (2 ** (self.pools - 1)),
            self.chans * (2 ** self.pools),
        )

    def forward(self, image: torch.Tensor) -> List[torch.Tensor]:
        if image.ndim == 3:
            image = image[:, None]
        if image.ndim != 4 or image.shape[1] != 1:
            raise RuntimeError(f"Expected PD [B,1,H,W], got {tuple(image.shape)}")
        out, _ = _normalise_per_sample(image.float())
        out, _ = _pad_to_multiple(out, 2 ** self.pools)
        features: List[torch.Tensor] = []
        for block in self.down:
            out = block(out)
            features.append(out)
            out = F.avg_pool2d(out, 2, 2)
        features.append(self.bottleneck(out))
        return features


def _normalised_dc_evidence(dc_magnitude: torch.Tensor) -> torch.Tensor:
    """Return bounded, intensity-invariant per-sample DC evidence [B,1,H,W]."""
    if dc_magnitude.ndim != 4 or dc_magnitude.shape[1] != 1:
        raise RuntimeError(
            f"Expected DC magnitude [B,1,H,W], got {tuple(dc_magnitude.shape)}"
        )
    mean = dc_magnitude.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    return torch.log1p(dc_magnitude / mean).clamp_(0.0, 6.0) / 6.0


class PairReliabilityHead(nn.Module):
    """Global, spatial and channel pair compatibility at one feature scale."""

    def __init__(
        self,
        target_chans: int,
        hidden_chans: int,
        initial_probability: float,
    ):
        super().__init__()
        self.target_proj = nn.Conv2d(target_chans, hidden_chans, 1, bias=False)
        self.aux_proj = nn.Conv2d(target_chans, hidden_chans, 1, bias=False)
        evidence_chans = 4 * hidden_chans + 1
        self.context = nn.Sequential(
            nn.Conv2d(evidence_chans, hidden_chans, 1, bias=False),
            nn.GroupNorm(4, hidden_chans),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                hidden_chans,
                hidden_chans,
                3,
                padding=1,
                groups=hidden_chans,
                bias=False,
            ),
            nn.Conv2d(hidden_chans, hidden_chans, 1, bias=False),
            nn.GroupNorm(4, hidden_chans),
            nn.SiLU(inplace=True),
        )
        self.global_out = nn.Linear(2 * hidden_chans, 1)
        self.spatial_out = nn.Conv2d(hidden_chans, 1, 1)
        self.channel_out = nn.Linear(hidden_chans, target_chans)
        bias = _logit(initial_probability)
        for layer in (self.global_out, self.spatial_out, self.channel_out):
            nn.init.zeros_(layer.weight)
            nn.init.constant_(layer.bias, bias)

    def forward(
        self, target: torch.Tensor, auxiliary: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Reliability labels must not rewrite reconstruction features directly.
        target = self.target_proj(target.detach())
        auxiliary = self.aux_proj(auxiliary.detach())
        t_norm = F.normalize(target, p=2, dim=1, eps=1e-6)
        a_norm = F.normalize(auxiliary, p=2, dim=1, eps=1e-6)
        correlation = (t_norm * a_norm).sum(dim=1, keepdim=True)
        evidence = torch.cat(
            [target, auxiliary, torch.abs(target - auxiliary), target * auxiliary,
             correlation],
            dim=1,
        )
        context = self.context(evidence)
        mean_pool = context.mean(dim=(-2, -1))
        max_pool = context.amax(dim=(-2, -1))
        q_logit = self.global_out(torch.cat([mean_pool, max_pool], dim=1))[:, 0]
        q_hat = torch.sigmoid(q_logit)
        spatial = torch.sigmoid(self.spatial_out(context))
        channel = torch.sigmoid(self.channel_out(mean_pool))[:, :, None, None]
        return q_logit, q_hat, spatial, channel


class TargetNeedHead(nn.Module):
    """Estimate whether target reconstruction needs external structure.

    This head cannot inspect the auxiliary feature.  It is trained only by the
    reconstruction objective, so its output is a need signal, not calibrated
    uncertainty.
    """

    def __init__(
        self,
        target_chans: int,
        hidden_chans: int,
        initial_probability: float,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(target_chans + 1, hidden_chans, 1, bias=False),
            nn.GroupNorm(4, hidden_chans),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                hidden_chans,
                hidden_chans,
                3,
                padding=1,
                groups=hidden_chans,
                bias=False,
            ),
            nn.Conv2d(hidden_chans, 1, 1),
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.constant_(self.layers[-1].bias, _logit(initial_probability))

    def forward(self, target: torch.Tensor, dc_evidence: torch.Tensor) -> torch.Tensor:
        dc_scaled = F.interpolate(
            dc_evidence,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return torch.sigmoid(self.layers(torch.cat([target, dc_scaled], dim=1)))


class CapacityMatchedAuxRefiner(nn.Module):
    """Extra capacity control that cannot compare target and auxiliary pairs."""

    def __init__(self, chans: int, hidden_chans: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(chans, hidden_chans, 1, bias=False),
            nn.GroupNorm(4, hidden_chans),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                hidden_chans,
                hidden_chans,
                3,
                padding=1,
                groups=hidden_chans,
                bias=False,
            ),
            nn.Conv2d(hidden_chans, chans, 1, bias=False),
        )
        # Start as a small residual refinement, not a replacement for M2-U.
        nn.init.zeros_(self.layers[-1].weight)

    def forward(self, auxiliary: torch.Tensor) -> torch.Tensor:
        return auxiliary + self.layers(auxiliary)


class SharedScaleController(nn.Module):
    """One scale-specific controller shared by all twelve cascades."""

    def __init__(
        self,
        target_chans: int,
        hidden_chans: int,
        initial_gate_probability: float,
        initial_need_probability: float,
        variant: str,
        capacity_hidden_chans: int,
        need_floor: float,
    ):
        super().__init__()
        # Variant-independent derived streams keep shared ablation modules at
        # identical initial weights without consuming the backbone RNG stream.
        base_seed = int(torch.initial_seed())
        if not 0.0 <= float(need_floor) < 1.0:
            raise ValueError("need_floor must lie in [0,1)")
        self.need_floor = float(need_floor)
        self.reliability = None
        if variant in {"prnf_full", "prnf_no_need"}:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(base_seed + 10_000 + int(target_chans))
                self.reliability = PairReliabilityHead(
                    target_chans, hidden_chans, initial_gate_probability
                )
        self.need = None
        if variant in {"prnf_full", "prnf_no_rel"}:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(base_seed + 20_000 + int(target_chans))
                self.need = TargetNeedHead(
                    target_chans, hidden_chans, initial_need_probability
                )
        # Scale-specific widths make this control parameter-matched to the
        # combined reliability+need heads without adding dormant parameters.
        self.capacity_refiner = None
        if variant == "m2u_augcap_mask":
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(base_seed + 30_000 + int(target_chans))
                self.capacity_refiner = CapacityMatchedAuxRefiner(
                    target_chans, capacity_hidden_chans
                )

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
        batch = target.shape[0]
        ones_scalar = target.new_ones(batch)
        ones_spatial = target.new_ones(batch, 1, *target.shape[-2:])
        ones_channel = target.new_ones(batch, target.shape[1], 1, 1)

        if variant in {"prnf_full", "prnf_no_need"}:
            if self.reliability is None:
                raise RuntimeError("Reliability module was not constructed")
            q_logit, q_hat, r_sp, r_ch = self.reliability(target, auxiliary)
        else:
            q_logit = target.new_zeros(batch)
            q_hat, r_sp, r_ch = ones_scalar, ones_spatial, ones_channel

        if reliability_override is not None:
            value = float(reliability_override)
            q_hat = target.new_full((batch,), value)
            # Override represents the complete pair-reliability factor; local
            # heads are neutralised so constant=0.5 means 0.5, not 0.5**3.
            r_sp = ones_spatial
            r_ch = ones_channel

        if variant in {"prnf_full", "prnf_no_rel"}:
            if self.need is None:
                raise RuntimeError("Need module was not constructed")
            # Reconstruction gradients train the need head through its output,
            # but cannot alter DC/backbone features to manufacture an easy cue.
            need = self.need(target.detach(), dc_evidence.detach())
        else:
            need = ones_spatial
        if need_override is not None:
            need = target.new_full(ones_spatial.shape, float(need_override))
        need_factor = self.need_floor + (1.0 - self.need_floor) * need

        if variant == "m2u_augcap_mask":
            if self.capacity_refiner is None:
                raise RuntimeError("Capacity refiner was not constructed")
            auxiliary_used = self.capacity_refiner(auxiliary)
            effective = availability.to(target.dtype).view(-1, 1, 1, 1)
        elif variant in {"m2u_clean", "m2u_augmented"}:
            auxiliary_used = auxiliary
            effective = ones_spatial
        else:
            auxiliary_used = auxiliary
            # Hard availability is part of PRNF, not the M2-U controls.
            m = availability.to(target.dtype).view(-1, 1, 1, 1)
            effective = m * q_hat[:, None, None, None] * r_sp * r_ch * need_factor

        term = alpha * effective * auxiliary_used
        fused = target + term
        target_rms = target.detach().square().mean((1, 2, 3)).sqrt().clamp_min(1e-8)
        term_rms = term.detach().square().mean((1, 2, 3)).sqrt() / target_rms
        diagnostics = {
            "q_logits": q_logit,
            "q_hat": q_hat,
            "q": availability.to(target.dtype) * q_hat,
            "reliability_spatial_mean": r_sp.detach().mean((1, 2, 3)),
            "reliability_channel_mean": r_ch.detach().mean((1, 2, 3)),
            "need_mean": need.detach().mean((1, 2, 3)),
            "need_p05": need.detach().flatten(1).quantile(0.05, dim=1),
            "need_p95": need.detach().flatten(1).quantile(0.95, dim=1),
            "need_factor_mean": need_factor.detach().mean((1, 2, 3)),
            "effective_weight_mean": effective.detach().mean((1, 2, 3)),
            "gated_aux_to_target_rms": term_rms,
            "alpha": alpha.detach().expand(batch),
        }
        return fused, diagnostics


class M2PRNFFeatureFusion(nn.Module):
    def __init__(self, pd_chans: int, target_chans: int, initial_alpha: float):
        super().__init__()
        conv = nn.Conv2d(pd_chans, target_chans, 1, bias=False)
        nn.init.kaiming_normal_(conv.weight, a=0.2)
        self.adapter = nn.Sequential(
            conv,
            nn.InstanceNorm2d(target_chans, affine=False),
        )
        # Preserve M2-U: alpha is independent for every cascade and scale.
        self.alpha = nn.Parameter(torch.tensor(float(initial_alpha)))

    def forward(
        self,
        target: torch.Tensor,
        pd_feature: torch.Tensor,
        controller: SharedScaleController,
        dc_evidence: torch.Tensor,
        availability: torch.Tensor,
        variant: str,
        reliability_override: Optional[float],
        need_override: Optional[float],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if pd_feature.shape[-2:] != target.shape[-2:]:
            pd_feature = F.interpolate(
                pd_feature, size=target.shape[-2:], mode="bilinear", align_corners=False
            )
        auxiliary = self.adapter(pd_feature)
        return controller(
            target,
            auxiliary,
            dc_evidence,
            availability,
            variant,
            self.alpha,
            reliability_override,
            need_override,
        )


class M2PRNFRegulariser(nn.Module):
    def __init__(
        self, chans: int = 18, pools: int = 4, initial_aux_alpha: float = 0.1
    ):
        super().__init__()
        self.chans, self.pools = int(chans), int(pools)
        down, in_chans = [], 2
        for level in range(self.pools):
            out_chans = self.chans * (2 ** level)
            down.append(ConvBlock(in_chans, out_chans))
            in_chans = out_chans
        self.target_down = nn.ModuleList(down)
        self.target_bottleneck = ConvBlock(
            self.chans * (2 ** (self.pools - 1)),
            self.chans * (2 ** self.pools),
        )
        self.fusions = nn.ModuleList(
            [
                M2PRNFFeatureFusion(
                    self.chans * (2 ** level),
                    self.chans * (2 ** level),
                    initial_aux_alpha,
                )
                for level in range(1, self.pools + 1)
            ]
        )
        up, up_chans = [], self.chans * (2 ** self.pools)
        for level in reversed(range(self.pools)):
            skip_chans = self.chans * (2 ** level)
            up.append(ConvBlock(up_chans + skip_chans, skip_chans))
            up_chans = skip_chans
        self.up = nn.ModuleList(up)
        self.out_conv = nn.Conv2d(self.chans, 2, 1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    @staticmethod
    def complex_to_channels(x: torch.Tensor) -> torch.Tensor:
        return x[:, 0].permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def channels_to_complex(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 3, 1).unsqueeze(1).contiguous()

    def forward(
        self,
        pdfs_image: torch.Tensor,
        pd_features: Sequence[torch.Tensor],
        controllers: Sequence[SharedScaleController],
        dc_evidence: torch.Tensor,
        availability: torch.Tensor,
        variant: str,
        reliability_override: Optional[float],
        need_override: Optional[float],
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        x = self.complex_to_channels(pdfs_image)
        original_hw = x.shape[-2:]
        x, target_std = _normalise_per_sample(x)
        x, pads = _pad_to_multiple(x, 2 ** self.pools)
        # DC evidence must follow the same padding geometry as target features.
        dc_evidence = _center_crop_or_pad(dc_evidence, original_hw)
        dc_evidence, _ = _pad_to_multiple(dc_evidence, 2 ** self.pools)

        skips: List[torch.Tensor] = []
        diagnostics: List[Dict[str, torch.Tensor]] = []
        out = x
        for level, block in enumerate(self.target_down):
            out = block(out)
            if level == 0:
                fused = out
            else:
                fused, diag = self.fusions[level - 1](
                    out, pd_features[level], controllers[level - 1], dc_evidence,
                    availability, variant, reliability_override, need_override
                )
                diagnostics.append(diag)
            skips.append(fused)
            # Preserve M2-U: the deeper target encoder receives the target path.
            out = F.avg_pool2d(out, 2, 2)

        out = self.target_bottleneck(out)
        out, diag = self.fusions[-1](
            out, pd_features[-1], controllers[-1], dc_evidence,
            availability, variant, reliability_override, need_override
        )
        diagnostics.append(diag)
        for block, skip in zip(self.up, reversed(skips)):
            out = F.interpolate(out, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            out = block(torch.cat([out, skip], dim=1))
        out = _unpad(self.out_conv(out), pads) * target_std
        if out.shape[-2:] != original_hw:
            raise RuntimeError("PRNF regulariser changed target spatial shape")
        return self.channels_to_complex(out), diagnostics


class M2PRNFVarNetBlock(nn.Module):
    def __init__(
        self, chans: int = 18, pools: int = 4, initial_aux_alpha: float = 0.1
    ):
        super().__init__()
        self.regulariser = M2PRNFRegulariser(chans, pools, initial_aux_alpha)
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
        controllers: Sequence[SharedScaleController],
        availability: torch.Tensor,
        variant: str,
        reliability_override: Optional[float],
        need_override: Optional[float],
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        zero = torch.zeros(1, 1, 1, 1, 1, device=current.device, dtype=current.dtype)
        soft_dc = torch.where(mask, current - reference, zero) * self.pdfs_dc_weight
        dc_image = self.sens_reduce(soft_dc, sens)
        dc_magnitude = fastmri.complex_abs(dc_image)
        dc_evidence = _normalised_dc_evidence(dc_magnitude)
        target_image = self.sens_reduce(current, sens)
        regularisation, diagnostics = self.regulariser(
            target_image, pd_features, controllers, dc_evidence, availability,
            variant, reliability_override, need_override
        )
        return current - soft_dc - self.sens_expand(regularisation, sens), diagnostics


class M2PRNFAuxPDVarNet(nn.Module):
    def __init__(
        self,
        variant: str = "prnf_full",
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        mask_center: bool = True,
        controller_chans: int = 16,
        initial_aux_alpha: float = 0.1,
        initial_gate_probability: float = 0.95,
        initial_need_probability: float = 0.95,
        need_floor: float = 0.25,
    ):
        super().__init__()
        if variant not in VALID_VARIANTS:
            raise ValueError(f"Unknown variant {variant}; choose from {sorted(VALID_VARIANTS)}")
        if controller_chans % 4:
            raise ValueError("controller_chans must be divisible by four")
        self.variant, self.pools = variant, int(pools)
        self.sens_net = SensitivityModel(
            chans=sens_chans, num_pools=sens_pools, mask_center=mask_center
        )
        self.pd_encoder = PDFeatureEncoder(chans=chans, pools=pools)
        # Construct every shared reconstruction parameter before any
        # variant-specific controller so seed-matched jobs start identically.
        self.cascades = nn.ModuleList(
            [
                M2PRNFVarNetBlock(chans, pools, initial_aux_alpha)
                for _ in range(num_cascades)
            ]
        )
        capacity_widths = [36 for _ in range(pools)]
        if capacity_widths:
            capacity_widths[-1] = 40
        self.controllers = nn.ModuleList(
            [
                SharedScaleController(
                    chans * (2 ** level),
                    controller_chans,
                    initial_gate_probability,
                    initial_need_probability,
                    variant,
                    capacity_widths[level - 1],
                    need_floor,
                )
                for level in range(1, pools + 1)
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
        need_override: Optional[float] = None,
    ):
        if mask.dtype != torch.bool:
            mask = mask.bool()
        batch = pdfs_masked_kspace.shape[0]
        if pd_available is None:
            pd_available = torch.ones(batch, device=pdfs_masked_kspace.device)
        pd_available = pd_available.reshape(batch)
        if not bool(torch.logical_or(pd_available == 0, pd_available == 1).all()):
            raise RuntimeError("pd_available must contain only zero or one")
        target_hw = (pdfs_masked_kspace.shape[-3], pdfs_masked_kspace.shape[-2])
        if pd_aux_image.ndim == 3:
            pd_aux_image = pd_aux_image[:, None]
        pd_aux_image = _center_crop_or_pad(pd_aux_image.float(), target_hw)
        pd_features = self.pd_encoder(pd_aux_image)
        sensitivity = self.sens_net(pdfs_masked_kspace, mask)
        current = pdfs_masked_kspace.clone()
        all_diagnostics: List[List[Dict[str, torch.Tensor]]] = []
        for cascade in self.cascades:
            current, diagnostics = cascade(
                current, pdfs_masked_kspace, mask, sensitivity, pd_features,
                self.controllers, pd_available, self.variant,
                reliability_override, need_override
            )
            all_diagnostics.append(diagnostics)
        prediction = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(current)), dim=1)
        if not return_aux:
            return prediction
        keys = tuple(all_diagnostics[0][0].keys())
        aux: Dict[str, torch.Tensor] = {}
        for key in keys:
            aux[key] = torch.stack(
                [torch.stack([scale[key] for scale in cascade], dim=1)
                 for cascade in all_diagnostics],
                dim=1,
            )
        aux["scale_names"] = tuple(f"H/{2 ** level}" for level in range(1, self.pools + 1))
        return prediction, aux
