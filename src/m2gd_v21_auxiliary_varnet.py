from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import fastmri
from fastmri.models.varnet import SensitivityModel


NEW_MODULE_FRAGMENTS = (
    "reliability_head",
    "channel_gate",
    "spatial_gate",
)


class ConvBlock(nn.Module):
    """Two-convolution feature block matching the pretrained M2-U backbone."""

    def __init__(self, in_chans: int, out_chans: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans, affine=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans, affine=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def _pad_to_multiple(
    x: torch.Tensor,
    multiple: int,
) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    _, _, height, width = x.shape
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    x = F.pad(
        x,
        (pad_left, pad_right, pad_top, pad_bottom),
        mode="reflect",
    )
    return x, (pad_top, pad_bottom, pad_left, pad_right)


def _unpad(
    x: torch.Tensor,
    pads: Tuple[int, int, int, int],
) -> torch.Tensor:
    pad_top, pad_bottom, pad_left, pad_right = pads
    h_end = x.shape[-2] - pad_bottom if pad_bottom else x.shape[-2]
    w_end = x.shape[-1] - pad_right if pad_right else x.shape[-1]
    return x[..., pad_top:h_end, pad_left:w_end]


def _normalise_per_sample(
    x: torch.Tensor,
    eps: float = 1e-7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = x.shape[0]
    flattened = x.reshape(batch_size, -1)
    mean = flattened.mean(dim=1).view(batch_size, 1, 1, 1)
    std = flattened.std(dim=1).view(batch_size, 1, 1, 1).clamp_min(eps)
    return (x - mean) / std, std


def _center_crop_or_pad(
    x: torch.Tensor,
    target_hw: Tuple[int, int],
) -> torch.Tensor:
    if x.ndim != 4:
        raise RuntimeError(f"Expected [B,C,H,W], got {tuple(x.shape)}")
    target_h, target_w = map(int, target_hw)
    _, _, height, width = x.shape

    if height > target_h:
        top = (height - target_h) // 2
        x = x[..., top:top + target_h, :]
        height = target_h
    if width > target_w:
        left = (width - target_w) // 2
        x = x[..., left:left + target_w]
        width = target_w

    pad_h = max(0, target_h - height)
    pad_w = max(0, target_w - width)
    if pad_h or pad_w:
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        x = F.pad(
            x,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=0.0,
        )
    return x


def _logit(probability: float) -> float:
    probability = min(1.0 - 1e-6, max(1e-6, float(probability)))
    return math.log(probability / (1.0 - probability))


def _central_roi(x: torch.Tensor, margin: int) -> torch.Tensor:
    margin = int(max(0, margin))
    if margin == 0:
        return x
    height, width = x.shape[-2:]
    if 2 * margin >= min(height, width):
        return x
    return x[..., margin:height - margin, margin:width - margin]


def _base_cross_contrast_evidence(
    target_feature: torch.Tensor,
    auxiliary_feature: torch.Tensor,
) -> torch.Tensor:
    if target_feature.shape != auxiliary_feature.shape:
        raise RuntimeError(
            "Cross-contrast evidence requires equal feature shapes, got "
            f"{tuple(target_feature.shape)} and {tuple(auxiliary_feature.shape)}."
        )
    target_norm = F.normalize(target_feature, p=2, dim=1, eps=1e-6)
    auxiliary_norm = F.normalize(auxiliary_feature, p=2, dim=1, eps=1e-6)
    correlation = (target_norm * auxiliary_norm).sum(dim=1, keepdim=True)
    return torch.cat(
        [
            target_feature,
            auxiliary_feature,
            torch.abs(target_feature - auxiliary_feature),
            target_feature * auxiliary_feature,
            correlation,
        ],
        dim=1,
    )


def _reliability_evidence_v21(
    target_feature: torch.Tensor,
    auxiliary_feature: torch.Tensor,
) -> torch.Tensor:
    """Retain local disagreement maps for the context-aware q head."""
    if target_feature.shape != auxiliary_feature.shape:
        raise RuntimeError(
            "Cross-contrast evidence requires equal feature shapes, got "
            f"{tuple(target_feature.shape)} and {tuple(auxiliary_feature.shape)}."
        )
    target_norm = F.normalize(target_feature, p=2, dim=1, eps=1e-6)
    auxiliary_norm = F.normalize(auxiliary_feature, p=2, dim=1, eps=1e-6)
    correlation = (target_norm * auxiliary_norm).sum(dim=1, keepdim=True)
    local_correlation = F.avg_pool2d(
        correlation,
        kernel_size=3,
        stride=1,
        padding=1,
    )
    local_absolute_difference = F.avg_pool2d(
        torch.abs(target_feature - auxiliary_feature).mean(
            dim=1,
            keepdim=True,
        ),
        kernel_size=3,
        stride=1,
        padding=1,
    )
    return torch.cat(
        [
            target_feature,
            auxiliary_feature,
            torch.abs(target_feature - auxiliary_feature),
            target_feature * auxiliary_feature,
            correlation,
            local_correlation,
            local_absolute_difference,
        ],
        dim=1,
    )


class PDFeatureEncoder(nn.Module):
    """Pretrained M2-U PD encoder, computed once and reused by all cascades."""

    def __init__(self, chans: int = 18, pools: int = 4):
        super().__init__()
        self.chans = int(chans)
        self.pools = int(pools)
        blocks = []
        in_chans = 1
        for level in range(self.pools):
            out_chans = self.chans * (2 ** level)
            blocks.append(ConvBlock(in_chans, out_chans))
            in_chans = out_chans
        self.down = nn.ModuleList(blocks)
        self.bottleneck = ConvBlock(
            self.chans * (2 ** (self.pools - 1)),
            self.chans * (2 ** self.pools),
        )

    def forward(self, pd_image: torch.Tensor) -> List[torch.Tensor]:
        if pd_image.ndim == 3:
            pd_image = pd_image.unsqueeze(1)
        elif pd_image.ndim != 4 or pd_image.shape[1] != 1:
            raise RuntimeError(
                "Expected PD auxiliary [B,H,W] or [B,1,H,W], "
                f"got {tuple(pd_image.shape)}"
            )
        pd_image = pd_image.float()
        pd_image, _ = _normalise_per_sample(pd_image)
        pd_image, _ = _pad_to_multiple(pd_image, 2 ** self.pools)

        features: List[torch.Tensor] = []
        out = pd_image
        for block in self.down:
            out = block(out)
            features.append(out)
            out = F.avg_pool2d(out, kernel_size=2, stride=2)
        out = self.bottleneck(out)
        features.append(out)
        return features


class ReliabilityHead(nn.Module):
    """Context-aware per-scale reliability from detached paired evidence."""

    def __init__(
        self,
        evidence_chans: int,
        hidden_chans: int,
        initial_probability: float,
    ):
        super().__init__()
        if hidden_chans % 8 != 0:
            raise ValueError("V2.1 reliability hidden channels must be divisible by 8.")
        self.context = nn.Sequential(
            nn.Conv2d(
                evidence_chans,
                hidden_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(8, hidden_chans),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                hidden_chans,
                hidden_chans,
                kernel_size=3,
                padding=2,
                dilation=2,
                groups=hidden_chans,
                bias=False,
            ),
            nn.Conv2d(hidden_chans, hidden_chans, kernel_size=1, bias=False),
            nn.GroupNorm(8, hidden_chans),
            nn.SiLU(inplace=True),
        )
        self.output = nn.Linear(2 * hidden_chans, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.constant_(self.output.bias, _logit(initial_probability))

    def forward(
        self,
        evidence: torch.Tensor,
        roi_margin: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features = _central_roi(self.context(evidence), roi_margin)
        mean_pool = features.mean(dim=(-2, -1))
        max_pool = features.amax(dim=(-2, -1))
        logits = self.output(torch.cat([mean_pool, max_pool], dim=1))[:, 0]
        return logits, torch.sigmoid(logits)


class ChannelGate(nn.Module):
    def __init__(
        self,
        evidence_chans: int,
        hidden_chans: int,
        target_chans: int,
        initial_probability: float,
    ):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Conv2d(evidence_chans, hidden_chans, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out = nn.Conv2d(hidden_chans, target_chans, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.constant_(self.out.bias, _logit(initial_probability))

    def forward(self, evidence: torch.Tensor, roi_margin: int) -> torch.Tensor:
        pooled = _central_roi(self.hidden(evidence), roi_margin).mean(
            dim=(-2, -1), keepdim=True
        )
        return torch.sigmoid(self.out(pooled))


class SpatialGate(nn.Module):
    def __init__(
        self,
        evidence_chans: int,
        hidden_chans: int,
        initial_probability: float,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(evidence_chans, hidden_chans, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_chans, 1, kernel_size=1),
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.constant_(self.layers[-1].bias, _logit(initial_probability))

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.layers(evidence))


class M2GDv21FeatureFusion(nn.Module):
    """M2-U adapter/alpha plus cross-contrast per-scale reliability control."""

    def __init__(
        self,
        pd_chans: int,
        target_chans: int,
        roi_margin: int,
        initial_alpha: float = 0.1,
        initial_gate_probability: float = 0.99,
    ):
        super().__init__()
        conv = nn.Conv2d(pd_chans, target_chans, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(conv.weight, a=0.2)
        # Keep names and shapes identical to M2-U for strict checkpoint transfer.
        self.adapter = nn.Sequential(
            conv,
            nn.InstanceNorm2d(target_chans, affine=False),
        )
        self.alpha = nn.Parameter(torch.tensor(float(initial_alpha)))
        self.roi_margin = int(roi_margin)

        reliability_evidence_chans = 4 * target_chans + 3
        gate_evidence_chans = 4 * target_chans + 1
        hidden_chans = max(
            16,
            min(64, 8 * math.ceil((target_chans / 2) / 8)),
        )
        self.reliability_head = ReliabilityHead(
            reliability_evidence_chans,
            hidden_chans,
            initial_gate_probability,
        )
        self.channel_gate = ChannelGate(
            gate_evidence_chans,
            max(8, target_chans // 2),
            target_chans,
            initial_gate_probability,
        )
        self.spatial_gate = SpatialGate(
            gate_evidence_chans,
            max(8, target_chans // 2),
            initial_gate_probability,
        )

    def forward(
        self,
        target_feature: torch.Tensor,
        pd_feature: torch.Tensor,
        pd_available: torch.Tensor,
        q_override: Optional[torch.Tensor] = None,
        detach_q_for_fusion: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if pd_feature.shape[-2:] != target_feature.shape[-2:]:
            pd_feature = F.interpolate(
                pd_feature,
                size=target_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        adapted = self.adapter(pd_feature)

        # BCE supervision must not rewrite pretrained reconstruction features.
        reliability_evidence = _reliability_evidence_v21(
            target_feature.detach(), adapted.detach()
        )
        q_logits, q_hat = self.reliability_head(
            reliability_evidence,
            self.roi_margin,
        )

        # Channel/spatial gates are reconstruction modules and may receive
        # gradients from the reconstruction objective after unfreezing.
        gate_evidence = _base_cross_contrast_evidence(target_feature, adapted)
        g_ch = self.channel_gate(gate_evidence, self.roi_margin)
        g_sp = self.spatial_gate(gate_evidence)

        availability = pd_available.to(
            device=target_feature.device,
            dtype=target_feature.dtype,
        ).view(-1)
        if q_override is None:
            q_control = q_hat.detach() if detach_q_for_fusion else q_hat
        else:
            q_control = torch.as_tensor(
                q_override,
                device=target_feature.device,
                dtype=target_feature.dtype,
            )
            if q_control.ndim == 0:
                q_control = q_control.expand_as(q_hat)
            elif q_control.ndim == 1 and q_control.shape[0] == q_hat.shape[0]:
                pass
            else:
                raise RuntimeError(
                    "q_override must be scalar or [B], got "
                    f"{tuple(q_control.shape)}."
                )
            q_control = q_control.clamp(0.0, 1.0)
        q = availability * q_control
        ungated_term = self.alpha * adapted
        effective_weight = q[:, None, None, None] * g_ch * g_sp
        gated_term = effective_weight * ungated_term
        fused = target_feature + gated_term

        target_rms = target_feature.detach().square().mean(
            dim=(1, 2, 3)
        ).sqrt().clamp_min(1e-8)
        ungated_ratio = ungated_term.detach().square().mean(
            dim=(1, 2, 3)
        ).sqrt() / target_rms
        gated_ratio = gated_term.detach().square().mean(
            dim=(1, 2, 3)
        ).sqrt() / target_rms

        diagnostics = {
            "q_logits": q_logits,
            "q_hat": q_hat,
            "q": q,
            "ungated_aux_to_target_rms": ungated_ratio,
            "gated_aux_to_target_rms": gated_ratio,
            "channel_gate_mean": g_ch.detach().mean(dim=(1, 2, 3)),
            "spatial_gate_mean": g_sp.detach().mean(dim=(1, 2, 3)),
            "effective_weight_mean": effective_weight.detach().mean(
                dim=(1, 2, 3)
            ),
            "alpha": self.alpha.detach().expand(target_feature.shape[0]),
        }
        return fused, diagnostics


class M2GDv21Regulariser(nn.Module):
    """M2-U regulariser with H/2--H/16 reliability-aware fusion."""

    def __init__(
        self,
        chans: int = 18,
        pools: int = 4,
        initial_aux_alpha: float = 0.1,
        initial_gate_probability: float = 0.99,
    ):
        super().__init__()
        self.chans = int(chans)
        self.pools = int(pools)
        target_down = []
        in_chans = 2
        for level in range(self.pools):
            out_chans = self.chans * (2 ** level)
            target_down.append(ConvBlock(in_chans, out_chans))
            in_chans = out_chans
        self.target_down = nn.ModuleList(target_down)
        self.target_bottleneck = ConvBlock(
            self.chans * (2 ** (self.pools - 1)),
            self.chans * (2 ** self.pools),
        )

        roi_margins = [4, 2, 1, 1]
        if self.pools != 4:
            roi_margins = [max(1, 8 // (2 ** level)) for level in range(1, self.pools + 1)]
        self.fusions = nn.ModuleList(
            [
                M2GDv21FeatureFusion(
                    pd_chans=self.chans * (2 ** level),
                    target_chans=self.chans * (2 ** level),
                    roi_margin=roi_margins[level - 1],
                    initial_alpha=initial_aux_alpha,
                    initial_gate_probability=initial_gate_probability,
                )
                for level in range(1, self.pools + 1)
            ]
        )
        self.fusion_scale_names = [
            f"H/{2 ** level}" for level in range(1, self.pools + 1)
        ]

        up_blocks = []
        up_chans = self.chans * (2 ** self.pools)
        for level in reversed(range(self.pools)):
            skip_chans = self.chans * (2 ** level)
            up_blocks.append(ConvBlock(up_chans + skip_chans, skip_chans))
            up_chans = skip_chans
        self.up = nn.ModuleList(up_blocks)
        self.out_conv = nn.Conv2d(self.chans, 2, kernel_size=1)
        nn.init.zeros_(self.out_conv.weight)
        if self.out_conv.bias is not None:
            nn.init.zeros_(self.out_conv.bias)

    @staticmethod
    def complex_to_channels(x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[1] != 1 or x.shape[-1] != 2:
            raise RuntimeError(
                "Expected image-domain tensor [B,1,H,W,2], "
                f"got {tuple(x.shape)}"
            )
        return x[:, 0].permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def channels_to_complex(x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 2:
            raise RuntimeError(
                "Expected channel tensor [B,2,H,W], "
                f"got {tuple(x.shape)}"
            )
        return x.permute(0, 2, 3, 1).unsqueeze(1).contiguous()

    def forward(
        self,
        pdfs_image: torch.Tensor,
        pd_features: Sequence[torch.Tensor],
        pd_available: torch.Tensor,
        q_override: Optional[torch.Tensor] = None,
        detach_q_for_fusion: bool = False,
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        if len(pd_features) != self.pools + 1:
            raise RuntimeError(
                f"Expected {self.pools + 1} PD scales, got {len(pd_features)}."
            )
        x = self.complex_to_channels(pdfs_image)
        original_hw = x.shape[-2:]
        x, pdfs_std = _normalise_per_sample(x)
        x, pads = _pad_to_multiple(x, 2 ** self.pools)

        target_features: List[torch.Tensor] = []
        fusion_diagnostics: List[Dict[str, torch.Tensor]] = []
        out = x
        for level, block in enumerate(self.target_down):
            out = block(out)
            if level == 0:
                fused = out
            else:
                fused, diagnostic = self.fusions[level - 1](
                    out,
                    pd_features[level],
                    pd_available,
                    q_override,
                    detach_q_for_fusion,
                )
                fusion_diagnostics.append(diagnostic)
            target_features.append(fused)
            # Preserve M2-U: deeper target encoder is target-only.
            out = F.avg_pool2d(out, kernel_size=2, stride=2)

        out = self.target_bottleneck(out)
        out, diagnostic = self.fusions[-1](
            out,
            pd_features[-1],
            pd_available,
            q_override,
            detach_q_for_fusion,
        )
        fusion_diagnostics.append(diagnostic)

        for block, skip in zip(self.up, reversed(target_features)):
            out = F.interpolate(
                out,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            out = block(torch.cat([out, skip], dim=1))
        out = self.out_conv(out)
        out = _unpad(out, pads)
        out = out * pdfs_std
        if out.shape[-2:] != original_hw:
            raise RuntimeError(
                f"M2-GD v2 output shape {tuple(out.shape[-2:])} "
                f"does not match input {tuple(original_hw)}."
            )
        return self.channels_to_complex(out), fusion_diagnostics


class M2GDv21VarNetBlock(nn.Module):
    def __init__(
        self,
        chans: int = 18,
        pools: int = 4,
        initial_aux_alpha: float = 0.1,
        initial_gate_probability: float = 0.99,
    ):
        super().__init__()
        self.regulariser = M2GDv21Regulariser(
            chans=chans,
            pools=pools,
            initial_aux_alpha=initial_aux_alpha,
            initial_gate_probability=initial_gate_probability,
        )
        self.pdfs_dc_weight = nn.Parameter(torch.ones(1))

    @staticmethod
    def sens_expand(image: torch.Tensor, sensitivity_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.fft2c(fastmri.complex_mul(image, sensitivity_maps))

    @staticmethod
    def sens_reduce(kspace: torch.Tensor, sensitivity_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.complex_mul(
            fastmri.ifft2c(kspace),
            fastmri.complex_conj(sensitivity_maps),
        ).sum(dim=1, keepdim=True)

    def forward(
        self,
        pdfs_current_kspace: torch.Tensor,
        pdfs_ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        pdfs_sens_maps: torch.Tensor,
        pd_features: Sequence[torch.Tensor],
        pd_available: torch.Tensor,
        q_override: Optional[torch.Tensor] = None,
        detach_q_for_fusion: bool = False,
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        zero = torch.zeros(
            1, 1, 1, 1, 1,
            device=pdfs_current_kspace.device,
            dtype=pdfs_current_kspace.dtype,
        )
        soft_dc = torch.where(
            mask,
            pdfs_current_kspace - pdfs_ref_kspace,
            zero,
        ) * self.pdfs_dc_weight
        pdfs_image = self.sens_reduce(pdfs_current_kspace, pdfs_sens_maps)
        regularisation_image, diagnostics = self.regulariser(
            pdfs_image,
            pd_features,
            pd_available,
            q_override,
            detach_q_for_fusion,
        )
        model_term = self.sens_expand(regularisation_image, pdfs_sens_maps)
        return pdfs_current_kspace - soft_dc - model_term, diagnostics


class M2GDv21AuxPDVarNet(nn.Module):
    """V2-branched, per-scale paired-discriminative reliability VarNet."""

    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        mask_center: bool = True,
        initial_aux_alpha: float = 0.1,
        initial_gate_probability: float = 0.99,
    ):
        super().__init__()
        self.pools = int(pools)
        self.num_cascades = int(num_cascades)
        self.sens_net = SensitivityModel(
            chans=sens_chans,
            num_pools=sens_pools,
            mask_center=mask_center,
        )
        self.pd_encoder = PDFeatureEncoder(chans=chans, pools=pools)
        self.cascades = nn.ModuleList(
            [
                M2GDv21VarNetBlock(
                    chans=chans,
                    pools=pools,
                    initial_aux_alpha=initial_aux_alpha,
                    initial_gate_probability=initial_gate_probability,
                )
                for _ in range(num_cascades)
            ]
        )

    @staticmethod
    def _prepare_pd(
        pd_aux_image: torch.Tensor,
        target_hw: Tuple[int, int],
    ) -> torch.Tensor:
        if pd_aux_image.ndim == 3:
            pd_aux_image = pd_aux_image.unsqueeze(1)
        elif pd_aux_image.ndim != 4 or pd_aux_image.shape[1] != 1:
            raise RuntimeError(
                "Expected PD auxiliary [B,H,W] or [B,1,H,W], "
                f"got {tuple(pd_aux_image.shape)}"
            )
        return _center_crop_or_pad(pd_aux_image.float(), target_hw)

    @staticmethod
    def _validate_availability(
        pd_available: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        if pd_available.ndim == 2 and pd_available.shape[1] == 1:
            pd_available = pd_available[:, 0]
        if pd_available.ndim != 1 or pd_available.shape[0] != batch_size:
            raise RuntimeError(
                f"Expected pd_available [B], got {tuple(pd_available.shape)}."
            )
        unique = torch.unique(pd_available.detach())
        if not bool(torch.logical_or(unique == 0, unique == 1).all().item()):
            raise RuntimeError(f"pd_available must be hard 0/1, got {unique.tolist()}.")
        return pd_available

    def forward(
        self,
        pdfs_masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        pd_aux_image: torch.Tensor,
        pd_available: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        q_override: Optional[torch.Tensor] = None,
        detach_q_for_fusion: bool = False,
    ):
        if pdfs_masked_kspace.ndim != 5 or pdfs_masked_kspace.shape[-1] != 2:
            raise RuntimeError(
                "Expected PDFS k-space [B,C,H,W,2], "
                f"got {tuple(pdfs_masked_kspace.shape)}"
            )
        if mask.dtype != torch.bool:
            mask = mask.bool()
        batch_size = pdfs_masked_kspace.shape[0]
        if pd_available is None:
            pd_available = torch.ones(
                batch_size,
                device=pdfs_masked_kspace.device,
                dtype=pdfs_masked_kspace.dtype,
            )
        pd_available = self._validate_availability(pd_available, batch_size)

        target_hw = (
            int(pdfs_masked_kspace.shape[-3]),
            int(pdfs_masked_kspace.shape[-2]),
        )
        pd_aux_image = self._prepare_pd(pd_aux_image, target_hw)
        pd_features = self.pd_encoder(pd_aux_image)
        sensitivity_maps = self.sens_net(pdfs_masked_kspace, mask)
        current_kspace = pdfs_masked_kspace.clone()

        cascade_diagnostics: List[List[Dict[str, torch.Tensor]]] = []
        for cascade in self.cascades:
            current_kspace, diagnostics = cascade(
                pdfs_current_kspace=current_kspace,
                pdfs_ref_kspace=pdfs_masked_kspace,
                mask=mask,
                pdfs_sens_maps=sensitivity_maps,
                pd_features=pd_features,
                pd_available=pd_available,
                q_override=q_override,
                detach_q_for_fusion=detach_q_for_fusion,
            )
            cascade_diagnostics.append(diagnostics)

        image = fastmri.ifft2c(current_kspace)
        magnitude = fastmri.complex_abs(image)
        prediction = fastmri.rss(magnitude, dim=1)
        if not return_aux:
            return prediction

        keys = tuple(cascade_diagnostics[0][0].keys())
        aux: Dict[str, torch.Tensor] = {}
        for key in keys:
            aux[key] = torch.stack(
                [
                    torch.stack(
                        [scale[key] for scale in cascade],
                        dim=1,
                    )
                    for cascade in cascade_diagnostics
                ],
                dim=1,
            )
            # Result shape: [B, cascades, scales]
        aux["scale_names"] = tuple(
            f"H/{2 ** level}" for level in range(1, self.pools + 1)
        )
        return prediction, aux

    def pretrained_parameter_names(self) -> List[str]:
        return [
            name
            for name, _ in self.named_parameters()
            if not any(fragment in name for fragment in NEW_MODULE_FRAGMENTS)
        ]

    def new_parameter_names(self) -> List[str]:
        return [
            name
            for name, _ in self.named_parameters()
            if any(fragment in name for fragment in NEW_MODULE_FRAGMENTS)
        ]

    def set_pretrained_trainable(self, trainable: bool) -> None:
        new_names = set(self.new_parameter_names())
        for name, parameter in self.named_parameters():
            if name not in new_names:
                parameter.requires_grad = bool(trainable)

    def parameter_groups(self) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        new_names = set(self.new_parameter_names())
        pretrained: List[nn.Parameter] = []
        new: List[nn.Parameter] = []
        for name, parameter in self.named_parameters():
            (new if name in new_names else pretrained).append(parameter)
        return pretrained, new


def _extract_state_dict(checkpoint: Mapping[str, object]) -> Mapping[str, torch.Tensor]:
    for key in ("model_state_dict", "model", "state_dict", "net", "network"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value  # type: ignore[return-value]
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint  # type: ignore[return-value]
    raise RuntimeError("Could not find a model state dict in the checkpoint.")


def load_m2u_backbone(
    model: M2GDv21AuxPDVarNet,
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> Dict[str, object]:
    """Strictly transfer every intended M2-U reconstruction parameter."""
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("M2-U checkpoint must be a mapping.")
    source = dict(_extract_state_dict(checkpoint))
    if source and all(key.startswith("module.") for key in source):
        source = {key[len("module."):]: value for key, value in source.items()}

    destination = model.state_dict()
    expected_backbone = {
        key
        for key in destination
        if not any(fragment in key for fragment in NEW_MODULE_FRAGMENTS)
    }
    missing = sorted(expected_backbone - set(source))
    shape_mismatch = sorted(
        key
        for key in expected_backbone & set(source)
        if tuple(source[key].shape) != tuple(destination[key].shape)
    )
    unexpected = sorted(set(source) - set(destination))
    if missing or shape_mismatch or unexpected:
        raise RuntimeError(
            "M2-U backbone transfer failed. "
            f"missing={missing[:10]}, shape_mismatch={shape_mismatch[:10]}, "
            f"unexpected={unexpected[:10]}"
        )

    transferred = {key: source[key] for key in expected_backbone}
    load_result = model.load_state_dict(transferred, strict=False)
    allowed_missing = {
        key
        for key in destination
        if any(fragment in key for fragment in NEW_MODULE_FRAGMENTS)
    }
    actual_missing = set(load_result.missing_keys)
    if actual_missing != allowed_missing or load_result.unexpected_keys:
        raise RuntimeError(
            "Unexpected state after M2-U transfer: "
            f"missing={sorted(actual_missing - allowed_missing)}, "
            f"unexpected={load_result.unexpected_keys}"
        )

    loaded_parameter_count = sum(
        destination[key].numel() for key in expected_backbone
    )
    total_parameter_count = sum(value.numel() for value in destination.values())
    report: Dict[str, object] = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_epoch": checkpoint.get("best_epoch"),
        "loaded_backbone_keys": len(expected_backbone),
        "loaded_backbone_parameter_count": int(loaded_parameter_count),
        "total_model_parameter_count": int(total_parameter_count),
        "new_keys": len(allowed_missing),
        "backbone_key_coverage": 1.0,
        "missing_expected_backbone_keys": [],
        "unexpected_checkpoint_keys": [],
    }
    return report


def load_m2gd_v2_for_v21(
    model: M2GDv21AuxPDVarNet,
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> Dict[str, object]:
    """Transfer the robust V2 model while replacing only reliability heads.

    Adapters, alpha, channel/spatial gates, VarNet, sensitivity estimation and
    the PD encoder are retained exactly.  V2 reliability-head parameters are
    intentionally excluded because V2.1 changes both their evidence channels
    and architecture.
    """
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("M2-GD V2 checkpoint must be a mapping.")
    source = dict(_extract_state_dict(checkpoint))
    if source and all(key.startswith("module.") for key in source):
        source = {key[len("module."):]: value for key, value in source.items()}

    destination = model.state_dict()
    transferable = {
        key
        for key in destination
        if "reliability_head" not in key
    }
    missing = sorted(transferable - set(source))
    shape_mismatch = sorted(
        key
        for key in transferable & set(source)
        if tuple(source[key].shape) != tuple(destination[key].shape)
    )
    unexpected_non_reliability = sorted(
        key
        for key in set(source) - set(destination)
        if "reliability_head" not in key
    )
    if missing or shape_mismatch or unexpected_non_reliability:
        raise RuntimeError(
            "M2-GD V2 to V2.1 transfer failed. "
            f"missing={missing[:10]}, shape_mismatch={shape_mismatch[:10]}, "
            f"unexpected={unexpected_non_reliability[:10]}"
        )

    transferred = {key: source[key] for key in transferable}
    load_result = model.load_state_dict(transferred, strict=False)
    allowed_missing = {
        key for key in destination if "reliability_head" in key
    }
    if set(load_result.missing_keys) != allowed_missing:
        raise RuntimeError(
            "Unexpected missing keys after V2 transfer: "
            f"{sorted(set(load_result.missing_keys) - allowed_missing)}"
        )
    if load_result.unexpected_keys:
        raise RuntimeError(
            f"Unexpected keys after V2 transfer: {load_result.unexpected_keys}"
        )

    loaded_parameter_count = sum(
        destination[key].numel() for key in transferable
    )
    checkpoint_config = checkpoint.get("config", {})
    if not isinstance(checkpoint_config, Mapping):
        checkpoint_config = {}
    hasher = hashlib.sha256()
    with Path(checkpoint_path).open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": hasher.hexdigest(),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_epoch": checkpoint.get("best_epoch"),
        "checkpoint_curriculum": checkpoint_config.get("curriculum"),
        "checkpoint_config": {
            key: checkpoint_config.get(key)
            for key in (
                "acceleration",
                "pd_aux_acceleration",
                "curriculum",
                "num_cascades",
                "pools",
                "chans",
            )
        },
        "loaded_non_reliability_keys": len(transferable),
        "loaded_non_reliability_parameter_count": int(loaded_parameter_count),
        "new_reliability_keys": len(allowed_missing),
        "non_reliability_key_coverage": 1.0,
    }
