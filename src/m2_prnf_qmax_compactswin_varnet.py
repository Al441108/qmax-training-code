from __future__ import annotations

"""Stage-B QMax-CompactSwin VarNet.

The auxiliary mechanism is imported unchanged from the shared QMax
implementation. This module has no dependency on trained Stage-A outputs.
The target regulariser uses:
H, H/2 and H/4 remain convolutional, while H/8 and H/16 use two alternating
window/shifted-window CompactSwin blocks.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import fastmri
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastmri.models.varnet import SensitivityModel
from torch.utils.checkpoint import checkpoint

from src.m2_prnf_qmax_varnet import (
    QMAX_VARIANTS,
    QMaxAuxPDVarNet,
    QMaxFeatureFusion,
    QMaxScaleController,
)
from src.m2_prnf_varnet import (
    ConvBlock,
    PDFeatureEncoder,
    _center_crop_or_pad,
    _normalise_per_sample,
    _pad_to_multiple,
    _unpad,
)


COMPACTSWIN_BACKBONE = "qmax_compactswin"
COMPACTSWIN_WINDOW_SIZE = 8
COMPACTSWIN_LAYER_SCALE_INIT = 1e-3
COMPACTSWIN_FFN_EXPANSION = 2
COMPACTSWIN_STAGE_SPECS = {
    "H/8": {"channels": 144, "heads": 6, "blocks": 2},
    "H/16": {"channels": 288, "heads": 8, "blocks": 2},
}


def _window_partition(value: torch.Tensor, window_size: int) -> torch.Tensor:
    batch, height, width, channels = value.shape
    if height % window_size or width % window_size:
        raise RuntimeError(
            f"Window partition requires multiples of {window_size}, "
            f"got {(height, width)}"
        )
    return (
        value.view(
            batch,
            height // window_size,
            window_size,
            width // window_size,
            window_size,
            channels,
        )
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(-1, window_size * window_size, channels)
    )


def _window_reverse(
    windows: torch.Tensor,
    window_size: int,
    height: int,
    width: int,
    batch: int,
) -> torch.Tensor:
    return (
        windows.view(
            batch,
            height // window_size,
            width // window_size,
            window_size,
            window_size,
            -1,
        )
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(batch, height, width, -1)
    )


def _stage_pad(
    value: torch.Tensor,
    window_size: int,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Reflect-pad NHWC stage features to a local window multiple."""

    _, height, width, _ = value.shape
    pad_h = (window_size - height % window_size) % window_size
    pad_w = (window_size - width % window_size) % window_size
    if pad_h or pad_w:
        nchw = value.permute(0, 3, 1, 2)
        nchw = F.pad(nchw, (0, pad_w, 0, pad_h), mode="reflect")
        value = nchw.permute(0, 2, 3, 1).contiguous()
    return value, (height, width)


def _shift_attention_mask(
    height: int,
    width: int,
    window_size: int,
    shift_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    if shift_size == 0:
        return None
    region = torch.zeros((1, height, width, 1), device=device)
    h_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    w_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    label = 0
    for h_slice in h_slices:
        for w_slice in w_slices:
            region[:, h_slice, w_slice, :] = label
            label += 1
    regions = _window_partition(region, window_size).squeeze(-1)
    difference = regions.unsqueeze(1) - regions.unsqueeze(2)
    return difference.ne(0).to(dtype=dtype) * -100.0


class WindowSelfAttention(nn.Module):
    def __init__(self, channels: int, heads: int, window_size: int):
        super().__init__()
        if channels % heads:
            raise ValueError(
                f"channels={channels} must be divisible by heads={heads}"
            )
        self.channels = int(channels)
        self.heads = int(heads)
        self.window_size = int(window_size)
        self.head_dim = self.channels // self.heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(self.channels, 3 * self.channels, bias=True)
        self.proj = nn.Linear(self.channels, self.channels, bias=True)

    def forward(
        self,
        windows: torch.Tensor,
        shift_mask: Optional[torch.Tensor],
        valid_windows: torch.Tensor,
    ) -> torch.Tensor:
        batch_windows, tokens, channels = windows.shape
        qkv = (
            self.qkv(windows)
            .reshape(
                batch_windows,
                tokens,
                3,
                self.heads,
                self.head_dim,
            )
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv.unbind(0)
        attention = (query * self.scale) @ key.transpose(-2, -1)
        windows_per_image = valid_windows.shape[0]
        batch = batch_windows // windows_per_image
        attention = attention.view(
            batch,
            windows_per_image,
            self.heads,
            tokens,
            tokens,
        )
        if shift_mask is not None:
            attention = attention + shift_mask[
                None, :, None, :, :
            ].to(attention.dtype)
        invalid_keys = ~valid_windows.bool()
        attention = attention.masked_fill(
            invalid_keys[None, :, None, None, :],
            -100.0,
        )
        attention = attention.view(
            batch_windows, self.heads, tokens, tokens
        )
        attention = attention.softmax(dim=-1)
        output = (
            (attention @ value)
            .transpose(1, 2)
            .reshape(batch_windows, tokens, channels)
        )
        return self.proj(output)


class CompactSwinBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        heads: int,
        window_size: int = COMPACTSWIN_WINDOW_SIZE,
        shifted: bool = False,
        ffn_expansion: int = COMPACTSWIN_FFN_EXPANSION,
        layer_scale_init: float = COMPACTSWIN_LAYER_SCALE_INIT,
    ):
        super().__init__()
        self.channels = int(channels)
        self.window_size = int(window_size)
        self.shift_size = self.window_size // 2 if shifted else 0
        hidden = self.channels * int(ffn_expansion)
        self.norm1 = nn.LayerNorm(self.channels)
        self.attention = WindowSelfAttention(
            self.channels, int(heads), self.window_size
        )
        self.layer_scale_attention = nn.Parameter(
            torch.full((self.channels,), float(layer_scale_init))
        )
        self.norm2 = nn.LayerNorm(self.channels)
        self.ffn_expand = nn.Conv2d(self.channels, hidden, 1)
        self.ffn_depthwise = nn.Conv2d(
            hidden, hidden, 3, padding=1, groups=hidden
        )
        self.ffn_project = nn.Conv2d(hidden, self.channels, 1)
        self.layer_scale_ffn = nn.Parameter(
            torch.full((self.channels,), float(layer_scale_init))
        )
        self._shift_mask_cache: Dict[tuple, Optional[torch.Tensor]] = {}
        self._valid_window_cache: Dict[tuple, torch.Tensor] = {}

    def _cached_shift_mask(
        self,
        height: int,
        width: int,
        value: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        key = (
            height,
            width,
            value.device.type,
            value.device.index,
            value.dtype,
            self.shift_size,
        )
        if key not in self._shift_mask_cache:
            self._shift_mask_cache[key] = _shift_attention_mask(
                height,
                width,
                self.window_size,
                self.shift_size,
                device=value.device,
                dtype=value.dtype,
            )
        return self._shift_mask_cache[key]

    def _cached_valid_windows(
        self,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        height, width = valid.shape[1:3]
        key = (
            height,
            width,
            valid.device.type,
            valid.device.index,
            valid.data_ptr(),
            self.shift_size,
        )
        if key not in self._valid_window_cache:
            selected = valid
            if self.shift_size:
                selected = torch.roll(
                    selected,
                    shifts=(-self.shift_size, -self.shift_size),
                    dims=(1, 2),
                )
            self._valid_window_cache[key] = _window_partition(
                selected.float(), self.window_size
            ).squeeze(-1).bool()
        return self._valid_window_cache[key]

    def forward(
        self,
        value: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        batch, height, width, channels = value.shape
        residual = value
        attended = self.norm1(value)
        if self.shift_size:
            attended = torch.roll(
                attended,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2),
            )
        windows = _window_partition(attended, self.window_size)
        valid_windows = self._cached_valid_windows(valid)
        shift_mask = self._cached_shift_mask(height, width, value)
        attended = self.attention(
            windows, shift_mask, valid_windows
        )
        attended = _window_reverse(
            attended,
            self.window_size,
            height,
            width,
            batch,
        )
        if self.shift_size:
            attended = torch.roll(
                attended,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2),
            )
        value = residual + (
            self.layer_scale_attention.view(1, 1, 1, -1) * attended
        )

        residual = value
        ffn = self.norm2(value).permute(0, 3, 1, 2).contiguous()
        ffn = F.gelu(self.ffn_expand(ffn))
        ffn = self.ffn_depthwise(ffn)
        ffn = self.ffn_project(ffn).permute(0, 2, 3, 1).contiguous()
        return residual + (
            self.layer_scale_ffn.view(1, 1, 1, -1) * ffn
        )


class CompactSwinStage(nn.Module):
    """Channel projection followed by alternating W-MSA/SW-MSA blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int,
        blocks: int = 2,
        window_size: int = COMPACTSWIN_WINDOW_SIZE,
        activation_checkpointing: bool = True,
        capture_padding_audit: bool = False,
    ):
        super().__init__()
        if int(blocks) not in (1, 2):
            raise ValueError("Frozen CompactSwin supports one or two blocks")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.window_size = int(window_size)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.capture_padding_audit = bool(capture_padding_audit)
        self._validity_cache: Dict[tuple, torch.Tensor] = {}
        self.input_projection = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                self.out_channels,
                1,
                bias=False,
            ),
            nn.InstanceNorm2d(self.out_channels, affine=False),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [
                CompactSwinBlock(
                    channels=self.out_channels,
                    heads=int(heads),
                    window_size=self.window_size,
                    shifted=bool(index % 2),
                )
                for index in range(int(blocks))
            ]
        )
        self.last_padding_audit: Dict[str, object] = {}

    def set_padding_audit(self, enabled: bool) -> None:
        self.capture_padding_audit = bool(enabled)

    def _cached_validity(
        self,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        padded_h = (
            math.ceil(height / self.window_size) * self.window_size
        )
        padded_w = (
            math.ceil(width / self.window_size) * self.window_size
        )
        key = (
            height,
            width,
            device.type,
            device.index,
        )
        if key not in self._validity_cache:
            valid = torch.zeros(
                1,
                padded_h,
                padded_w,
                1,
                dtype=torch.bool,
                device=device,
            )
            valid[:, :height, :width, :] = True
            self._validity_cache[key] = valid
        return self._validity_cache[key]

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.input_projection(value)
        original_hw = tuple(int(v) for v in value.shape[-2:])
        value = value.permute(0, 2, 3, 1).contiguous()
        value, unpadded_hw = _stage_pad(value, self.window_size)
        valid = self._cached_validity(
            unpadded_hw[0], unpadded_hw[1], value.device
        )
        padded_hw = tuple(int(v) for v in value.shape[1:3])
        if self.capture_padding_audit:
            self.last_padding_audit = {
                "original_hw": list(original_hw),
                "padded_hw": list(padded_hw),
                "valid_tokens": int(valid.sum().item()),
                "total_tokens": int(valid.numel()),
                "window_size": self.window_size,
            }
        for block in self.blocks:
            if (
                self.activation_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                value = checkpoint(
                    block, value, valid, use_reentrant=False
                )
            else:
                value = block(value, valid)
        height, width = unpadded_hw
        value = value[:, :height, :width, :]
        if value.shape[1:3] != original_hw:
            raise RuntimeError(
                "CompactSwin stage-local unpadding changed spatial size"
            )
        return value.permute(0, 3, 1, 2).contiguous()


class QMaxCompactSwinRegulariser(nn.Module):
    def __init__(
        self,
        chans: int = 18,
        pools: int = 4,
        initial_aux_alpha: float = 0.1,
        h8_blocks: int = 2,
        h16_blocks: int = 2,
        activation_checkpointing: bool = True,
    ):
        super().__init__()
        if int(chans) != 18 or int(pools) != 4:
            raise ValueError(
                "Frozen CompactSwin requires chans=18 and pools=4"
            )
        self.chans, self.pools = int(chans), int(pools)
        self.target_down = nn.ModuleList(
            [
                ConvBlock(2, 18),
                ConvBlock(18, 36),
                ConvBlock(36, 72),
                CompactSwinStage(
                    72,
                    144,
                    heads=6,
                    blocks=int(h8_blocks),
                    activation_checkpointing=activation_checkpointing,
                ),
            ]
        )
        self.target_bottleneck = CompactSwinStage(
            144,
            288,
            heads=8,
            blocks=int(h16_blocks),
            activation_checkpointing=activation_checkpointing,
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
            raise RuntimeError(
                "QMax-CompactSwin regulariser changed target spatial shape"
            )
        return self.channels_to_complex(out), diagnostics

    def padding_audit(self) -> Dict[str, Dict[str, object]]:
        return {
            "H/8": dict(self.target_down[3].last_padding_audit),
            "H/16": dict(self.target_bottleneck.last_padding_audit),
        }

    def set_padding_audit(self, enabled: bool) -> None:
        self.target_down[3].set_padding_audit(enabled)
        self.target_bottleneck.set_padding_audit(enabled)


class QMaxCompactSwinVarNetBlock(nn.Module):
    def __init__(
        self,
        chans: int = 18,
        pools: int = 4,
        initial_aux_alpha: float = 0.1,
        qmax_variant: str = "qmax_core",
        h8_blocks: int = 2,
        h16_blocks: int = 2,
        activation_checkpointing: bool = True,
    ):
        super().__init__()
        if qmax_variant not in QMAX_VARIANTS:
            raise ValueError(qmax_variant)
        self.qmax_variant = str(qmax_variant)
        self.regulariser = QMaxCompactSwinRegulariser(
            chans=chans,
            pools=pools,
            initial_aux_alpha=initial_aux_alpha,
            h8_blocks=h8_blocks,
            h16_blocks=h16_blocks,
            activation_checkpointing=activation_checkpointing,
        )
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
        return (
            current
            - soft_dc
            - self.sens_expand(regularisation, sens),
            diagnostics,
        )


class QMaxCompactSwinAuxPDVarNet(QMaxAuxPDVarNet):
    """QMax with CompactSwin only in the target H/8 and H/16 stages."""

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
        h8_blocks: int = 2,
        h16_blocks: int = 2,
        window_size: int = COMPACTSWIN_WINDOW_SIZE,
        activation_checkpointing: bool = True,
    ):
        nn.Module.__init__(self)
        if qmax_variant not in QMAX_VARIANTS:
            raise ValueError(qmax_variant)
        if int(pools) != 4 or int(chans) != 18:
            raise ValueError(
                "Frozen QMax-CompactSwin requires chans=18, pools=4"
            )
        if int(window_size) != COMPACTSWIN_WINDOW_SIZE:
            raise ValueError("Frozen CompactSwin window size is 8")
        self.qmax_variant = str(qmax_variant)
        self.pools = int(pools)
        self.backbone_variant = COMPACTSWIN_BACKBONE
        self.sens_net = SensitivityModel(
            chans=int(sens_chans),
            num_pools=int(sens_pools),
            mask_center=bool(mask_center),
        )
        self.pd_encoder = PDFeatureEncoder(chans=int(chans), pools=int(pools))
        self.cascades = nn.ModuleList(
            [
                QMaxCompactSwinVarNetBlock(
                    chans=int(chans),
                    pools=int(pools),
                    initial_aux_alpha=float(initial_aux_alpha),
                    qmax_variant=self.qmax_variant,
                    h8_blocks=int(h8_blocks),
                    h16_blocks=int(h16_blocks),
                    activation_checkpointing=bool(
                        activation_checkpointing
                    ),
                )
                for _ in range(int(num_cascades))
            ]
        )
        self.controllers = nn.ModuleList(
            [
                QMaxScaleController(
                    target_chans=int(chans) * (2**level),
                    hidden_chans=int(controller_chans),
                    initial_gate_probability=float(
                        initial_gate_probability
                    ),
                    qmax_variant=self.qmax_variant,
                )
                for level in range(1, int(pools) + 1)
            ]
        )
        self.compactswin_config = {
            "h8_blocks": int(h8_blocks),
            "h16_blocks": int(h16_blocks),
            "window_size": int(window_size),
            "activation_checkpointing": bool(
                activation_checkpointing
            ),
        }

    def architecture_audit(self) -> Dict[str, object]:
        layer_scales = []
        padding = {}
        for cascade_index, cascade in enumerate(self.cascades):
            padding[str(cascade_index)] = (
                cascade.regulariser.padding_audit()
            )
            for name, parameter in cascade.named_parameters():
                if "layer_scale_" in name:
                    layer_scales.append(parameter.detach())
        return {
            "backbone_variant": self.backbone_variant,
            "qmax_variant": self.qmax_variant,
            "num_cascades": len(self.cascades),
            "h8_blocks": self.compactswin_config["h8_blocks"],
            "h16_blocks": self.compactswin_config["h16_blocks"],
            "window_size": self.compactswin_config["window_size"],
            "activation_checkpointing": self.compactswin_config[
                "activation_checkpointing"
            ],
            "layer_scale_min": min(
                float(value.min().item()) for value in layer_scales
            ),
            "layer_scale_max": max(
                float(value.max().item()) for value in layer_scales
            ),
            "padding": padding,
        }

    def set_padding_audit(self, enabled: bool) -> None:
        for cascade in self.cascades:
            cascade.regulariser.set_padding_audit(enabled)


def copy_shared_b0_to_b1(
    b0: QMaxAuxPDVarNet,
    b1: QMaxCompactSwinAuxPDVarNet,
) -> Dict[str, object]:
    """Copy every same-name/same-shape tensor from CNN B0 to CompactSwin B1."""

    source = b0.state_dict()
    destination = b1.state_dict()
    copied: List[str] = []
    unmatched: List[str] = []
    with torch.no_grad():
        for key, value in destination.items():
            candidate = source.get(key)
            if candidate is not None and candidate.shape == value.shape:
                value.copy_(candidate)
                copied.append(key)
            else:
                unmatched.append(key)
    b1.load_state_dict(destination, strict=True)
    return {
        "copied_keys": copied,
        "unmatched_b1_keys": unmatched,
        "num_copied": len(copied),
        "num_unmatched_b1": len(unmatched),
    }


def shared_state_max_difference(
    b0: QMaxAuxPDVarNet,
    b1: QMaxCompactSwinAuxPDVarNet,
) -> Tuple[float, List[str]]:
    first = b0.state_dict()
    second = b1.state_dict()
    keys = sorted(
        key
        for key, value in first.items()
        if key in second and second[key].shape == value.shape
    )
    difference = max(
        (
            float((first[key] - second[key]).abs().max().item())
            for key in keys
        ),
        default=0.0,
    )
    return difference, keys
