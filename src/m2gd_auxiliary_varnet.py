from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import fastmri
from fastmri.models.varnet import SensitivityModel


TensorDict = Dict[str, torch.Tensor]


class ConvBlock(nn.Module):
    """Two-convolution feature block used by both contrast encoders."""

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
    """Independent per-sample standardisation. Returns normalised x and std."""
    batch_size = x.shape[0]
    mean = x.reshape(batch_size, -1).mean(dim=1).view(batch_size, 1, 1, 1)
    std = (
        x.reshape(batch_size, -1)
        .std(dim=1)
        .view(batch_size, 1, 1, 1)
        .clamp_min(eps)
    )
    return (x - mean) / std, std


def _center_crop_or_pad(
    x: torch.Tensor,
    target_hw: Tuple[int, int],
) -> torch.Tensor:
    """Centre-crop or zero-pad [B,C,H,W] to target spatial size."""
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
    if not 0.0 < probability < 1.0:
        raise ValueError(
            f"Probability for sigmoid-bias initialisation must be in (0,1), "
            f"got {probability}."
        )
    return math.log(probability / (1.0 - probability))


def _per_sample_rms(
    x: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return a finite-gradient RMS over non-batch dimensions, shape [B]."""
    if x.ndim < 2:
        raise RuntimeError(f"Expected batched tensor, got {tuple(x.shape)}")
    dims = tuple(range(1, x.ndim))
    mean_square = x.square().mean(dim=dims)
    # Subtracting sqrt(eps) preserves an exact zero for a zero tensor while
    # avoiding the undefined derivative of sqrt at zero.
    return (mean_square + eps).sqrt() - math.sqrt(eps)


def _prepare_availability(
    pd_available: Optional[torch.Tensor],
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert availability to [B,1,1,1] with values in [0,1]."""
    if pd_available is None:
        return torch.ones(batch_size, 1, 1, 1, device=device, dtype=dtype)

    availability = torch.as_tensor(pd_available, device=device)

    if availability.ndim == 0:
        availability = availability.expand(batch_size)

    if availability.shape[0] != batch_size:
        raise RuntimeError(
            "PD availability must have one value per sample. "
            f"Expected leading batch dimension {batch_size}, got "
            f"{tuple(availability.shape)}."
        )

    availability = availability.reshape(batch_size, -1)
    if availability.shape[1] != 1:
        raise RuntimeError(
            "PD availability must contain exactly one scalar per sample, "
            f"got shape {tuple(availability.shape)} after flattening."
        )

    availability = availability.to(dtype=dtype).clamp(0.0, 1.0)
    return availability.view(batch_size, 1, 1, 1)


class PDFeatureEncoder(nn.Module):
    """
    Independently encode the PD auxiliary magnitude image once per VarNet pass.

    Returned feature pyramid:
        p0: H
        p1: H/2
        ...
        pN: H/(2**N), where N=pools
    """

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


class GlobalReliabilityGate(nn.Module):
    """
    Predict a cascade-wise, sample-level PD reliability score from H/16.

    q_hat is the learned reliability prediction. The actual fusion gate is
    q = availability * q_hat. Keeping these quantities separate is essential:
    reliability supervision must act on q_hat, not on the hard-masked q.
    """

    def __init__(
        self,
        chans: int,
        hidden_chans: Optional[int] = None,
        initial_q: float = 0.8,
    ):
        super().__init__()
        hidden = int(hidden_chans or max(chans // 2, 16))

        self.network = nn.Sequential(
            nn.Conv2d(3 * chans, hidden, kernel_size=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
        )

        first = self.network[0]
        last = self.network[2]
        nn.init.kaiming_normal_(first.weight, a=0.2)
        nn.init.zeros_(first.bias)
        nn.init.normal_(last.weight, mean=0.0, std=1e-3)
        nn.init.constant_(last.bias, _logit(initial_q))

        self.last_q_hat_mean: Optional[torch.Tensor] = None
        self.last_q_mean: Optional[torch.Tensor] = None

    def forward(
        self,
        target_bottleneck: torch.Tensor,
        adapted_pd_bottleneck: torch.Tensor,
        availability: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if target_bottleneck.shape != adapted_pd_bottleneck.shape:
            raise RuntimeError(
                "Global reliability inputs must have matching shapes, got "
                f"target={tuple(target_bottleneck.shape)} and "
                f"PD={tuple(adapted_pd_bottleneck.shape)}."
            )

        disagreement = torch.abs(target_bottleneck - adapted_pd_bottleneck)
        descriptor = torch.cat(
            [target_bottleneck, adapted_pd_bottleneck, disagreement],
            dim=1,
        )
        descriptor = F.adaptive_avg_pool2d(descriptor, output_size=1)

        q_hat = torch.sigmoid(self.network(descriptor))
        q = availability * q_hat

        self.last_q_hat_mean = q_hat.detach().mean()
        self.last_q_mean = q.detach().mean()
        return q_hat, q


class M2GDFeatureFusion(nn.Module):
    """
    Factorised channel-spatial disagreement gate for one feature scale.

    Shapes:
        q:       [B,1,1,1]
        g_ch:    [B,C,1,1]
        g_sp:    [B,1,H,W]
        w:       [B,C,H,W]

    There is deliberately no additional learnable alpha. The conditional
    effective alpha is w = q * g_ch * g_sp.
    """

    def __init__(
        self,
        pd_chans: int,
        target_chans: int,
        initial_local_gate: float = 0.35,
    ):
        super().__init__()

        adapter_conv = nn.Conv2d(
            pd_chans,
            target_chans,
            kernel_size=1,
            bias=False,
        )
        nn.init.kaiming_normal_(adapter_conv.weight, a=0.2)
        self.adapter = nn.Sequential(
            adapter_conv,
            nn.InstanceNorm2d(target_chans, affine=False),
        )

        channel_hidden = max(target_chans // 2, 8)
        self.channel_gate = nn.Sequential(
            nn.Conv2d(3 * target_chans, channel_hidden, kernel_size=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(channel_hidden, target_chans, kernel_size=1, bias=True),
        )

        spatial_hidden = max(target_chans // 4, 8)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(3 * target_chans, spatial_hidden, kernel_size=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(spatial_hidden, 1, kernel_size=1, bias=True),
        )

        self._initialise_gate(self.channel_gate, initial_local_gate)
        self._initialise_gate(self.spatial_gate, initial_local_gate)

        self.last_diagnostics: Dict[str, float] = {}

    @staticmethod
    def _initialise_gate(gate: nn.Sequential, initial_value: float) -> None:
        first = gate[0]
        last = gate[2]
        nn.init.kaiming_normal_(first.weight, a=0.2)
        nn.init.zeros_(first.bias)
        nn.init.normal_(last.weight, mean=0.0, std=1e-3)
        nn.init.constant_(last.bias, _logit(initial_value))

    def adapt(
        self,
        pd_feature: torch.Tensor,
        target_feature: torch.Tensor,
    ) -> torch.Tensor:
        if pd_feature.shape[-2:] != target_feature.shape[-2:]:
            pd_feature = F.interpolate(
                pd_feature,
                size=target_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return self.adapter(pd_feature)

    def forward(
        self,
        target_feature: torch.Tensor,
        pd_feature: torch.Tensor,
        q: torch.Tensor,
        adapted_pd: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, TensorDict]:
        if adapted_pd is None:
            adapted_pd = self.adapt(pd_feature, target_feature)

        if adapted_pd.shape != target_feature.shape:
            raise RuntimeError(
                "Adapted PD and target feature shapes must match, got "
                f"target={tuple(target_feature.shape)} and "
                f"adapted_pd={tuple(adapted_pd.shape)}."
            )

        disagreement = torch.abs(target_feature - adapted_pd)
        gate_input = torch.cat(
            [target_feature, adapted_pd, disagreement],
            dim=1,
        )

        channel_descriptor = F.adaptive_avg_pool2d(gate_input, output_size=1)
        g_ch = torch.sigmoid(self.channel_gate(channel_descriptor))
        g_sp = torch.sigmoid(self.spatial_gate(gate_input))

        w = q * g_ch * g_sp
        auxiliary_term = w * adapted_pd
        fused = target_feature + auxiliary_term

        target_rms = _per_sample_rms(target_feature).detach()
        adapted_ratio = _per_sample_rms(adapted_pd) / target_rms.clamp_min(1e-8)
        contribution_ratio = (
            _per_sample_rms(auxiliary_term) / target_rms.clamp_min(1e-8)
        )

        diagnostics: TensorDict = {
            "g_ch_mean": g_ch.mean(dim=(1, 2, 3)),
            "g_ch_std": g_ch.flatten(1).std(dim=1, unbiased=False),
            "g_sp_mean": g_sp.mean(dim=(1, 2, 3)),
            "g_sp_std": g_sp.flatten(1).std(dim=1, unbiased=False),
            "w_mean": w.mean(dim=(1, 2, 3)),
            "w_std": w.flatten(1).std(dim=1, unbiased=False),
            "adapted_pd_to_target_rms": adapted_ratio,
            "gated_aux_to_target_rms": contribution_ratio,
        }

        self.last_diagnostics = {
            key: float(value.detach().mean().cpu())
            for key, value in diagnostics.items()
        }
        self.last_diagnostics.update(
            {
                "g_ch_min": float(g_ch.detach().min().cpu()),
                "g_ch_max": float(g_ch.detach().max().cpu()),
                "g_sp_min": float(g_sp.detach().min().cpu()),
                "g_sp_max": float(g_sp.detach().max().cpu()),
                "w_min": float(w.detach().min().cpu()),
                "w_max": float(w.detach().max().cpu()),
            }
        )
        return fused, diagnostics


class M2GDRegulariser(nn.Module):
    """
    PD-FS U-Net regulariser with reliability-aware multi-scale PD fusion.

    Important invariants:
      * the target encoder itself remains target-only;
      * PD is not fused at full resolution;
      * q_hat is predicted once per cascade from H/16;
      * the same q is shared by H/2, H/4, H/8 and H/16;
      * intermediate features never enter data consistency directly.
    """

    def __init__(
        self,
        chans: int = 18,
        pools: int = 4,
        initial_q: float = 0.8,
        initial_local_gate: float = 0.35,
        contribution_budgets: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        self.chans = int(chans)
        self.pools = int(pools)

        if self.pools < 1:
            raise ValueError(f"pools must be at least 1, got {self.pools}")

        if contribution_budgets is not None:
            if len(contribution_budgets) != self.pools:
                raise ValueError(
                    f"Expected {self.pools} contribution budgets for H/2 to "
                    f"H/{2 ** self.pools}, got {len(contribution_budgets)}."
                )
            budgets = torch.tensor(
                [float(value) for value in contribution_budgets],
                dtype=torch.float32,
            )
            if torch.any(budgets <= 0):
                raise ValueError("All contribution budgets must be positive.")
            self.register_buffer("contribution_budgets", budgets)
        else:
            self.register_buffer(
                "contribution_budgets",
                torch.empty(0, dtype=torch.float32),
            )

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

        # Feature 0 is full resolution and intentionally has no fusion module.
        self.fusions = nn.ModuleList(
            [
                M2GDFeatureFusion(
                    pd_chans=self.chans * (2 ** level),
                    target_chans=self.chans * (2 ** level),
                    initial_local_gate=initial_local_gate,
                )
                for level in range(1, self.pools + 1)
            ]
        )
        self.fusion_scale_names = [
            f"H/{2 ** level}" for level in range(1, self.pools + 1)
        ]

        bottleneck_chans = self.chans * (2 ** self.pools)
        self.global_reliability = GlobalReliabilityGate(
            chans=bottleneck_chans,
            initial_q=initial_q,
        )

        up_blocks = []
        up_chans = bottleneck_chans
        for level in reversed(range(self.pools)):
            skip_chans = self.chans * (2 ** level)
            up_blocks.append(ConvBlock(up_chans + skip_chans, skip_chans))
            up_chans = skip_chans

        self.up = nn.ModuleList(up_blocks)
        self.out_conv = nn.Conv2d(self.chans, 2, kernel_size=1)

        # Neutral initial model term, matching VarNet-style stable startup.
        nn.init.zeros_(self.out_conv.weight)
        if self.out_conv.bias is not None:
            nn.init.zeros_(self.out_conv.bias)

    @staticmethod
    def complex_to_channels(x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[1] != 1 or x.shape[-1] != 2:
            raise RuntimeError(
                "Expected image tensor [B,1,H,W,2], "
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

    def _budget_loss(
        self,
        contribution_ratios: torch.Tensor,
    ) -> torch.Tensor:
        """
        One-sided contribution budget.

        contribution_ratios has shape [B,S]. The target-feature denominator
        was detached inside the fusion module, preventing trivial evasion by
        inflating the target feature magnitude.
        """
        if self.contribution_budgets.numel() == 0:
            return contribution_ratios.new_zeros(())

        budgets = self.contribution_budgets.to(
            device=contribution_ratios.device,
            dtype=contribution_ratios.dtype,
        ).view(1, -1)
        exceedance = F.relu(contribution_ratios - budgets)
        return exceedance.square().mean()

    def forward(
        self,
        pdfs_image: torch.Tensor,
        pd_features: Sequence[torch.Tensor],
        availability: torch.Tensor,
    ) -> Tuple[torch.Tensor, TensorDict]:
        if len(pd_features) != self.pools + 1:
            raise RuntimeError(
                f"Expected {self.pools + 1} PD feature scales, "
                f"received {len(pd_features)}"
            )

        x = self.complex_to_channels(pdfs_image)
        original_hw = x.shape[-2:]

        # PD-FS and PD are normalised independently.
        x, pdfs_std = _normalise_per_sample(x)
        x, pads = _pad_to_multiple(x, 2 ** self.pools)

        # Target-only encoder. We defer every PD injection until q has been
        # computed from the H/16 bottleneck.
        target_features: List[torch.Tensor] = []
        out = x
        for block in self.target_down:
            out = block(out)
            target_features.append(out)
            out = F.avg_pool2d(out, kernel_size=2, stride=2)

        target_bottleneck = self.target_bottleneck(out)

        # Adapt the H/16 PD feature once and reuse it for q and fusion.
        bottleneck_fusion = self.fusions[-1]
        adapted_pd_bottleneck = bottleneck_fusion.adapt(
            pd_features[-1],
            target_bottleneck,
        )
        q_hat, q = self.global_reliability(
            target_bottleneck,
            adapted_pd_bottleneck,
            availability,
        )

        fused_features: List[torch.Tensor] = [target_features[0]]
        scale_diagnostics: List[TensorDict] = []

        # H/2 to H/(2**(pools-1)) skip fusion.
        for level in range(1, self.pools):
            fusion = self.fusions[level - 1]
            fused, diagnostics = fusion(
                target_features[level],
                pd_features[level],
                q,
            )
            fused_features.append(fused)
            scale_diagnostics.append(diagnostics)

        # H/(2**pools) bottleneck fusion.
        out, bottleneck_diagnostics = bottleneck_fusion(
            target_bottleneck,
            pd_features[-1],
            q,
            adapted_pd=adapted_pd_bottleneck,
        )
        scale_diagnostics.append(bottleneck_diagnostics)

        # Decoder receives target-only full-resolution skip and gated skips at
        # H/2 and below.
        for block, skip in zip(self.up, reversed(fused_features)):
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
                f"M2-GD output shape {tuple(out.shape[-2:])} "
                f"does not match input {tuple(original_hw)}"
            )

        # Stack all per-scale diagnostics as [B,S].
        diagnostics: TensorDict = {
            "q_hat": q_hat.flatten(1).squeeze(1),
            "q": q.flatten(1).squeeze(1),
        }
        for key in scale_diagnostics[0]:
            diagnostics[key] = torch.stack(
                [item[key] for item in scale_diagnostics],
                dim=1,
            )

        diagnostics["budget_loss"] = self._budget_loss(
            diagnostics["gated_aux_to_target_rms"]
        )
        return self.channels_to_complex(out), diagnostics

    def fusion_diagnostics(self) -> Dict[str, Dict[str, float]]:
        """Latest detached diagnostics for logging after a forward pass."""
        summary: Dict[str, Dict[str, float]] = {}
        for name, fusion in zip(self.fusion_scale_names, self.fusions):
            summary[name] = dict(fusion.last_diagnostics)

        q_hat = self.global_reliability.last_q_hat_mean
        q = self.global_reliability.last_q_mean
        summary["global"] = {
            "q_hat_mean": (
                float(q_hat.cpu()) if q_hat is not None else float("nan")
            ),
            "q_mean": float(q.cpu()) if q is not None else float("nan"),
        }
        return summary


class M2GDVarNetBlock(nn.Module):
    """One PD-FS VarNet cascade with M2-GD image-domain regularisation."""

    def __init__(
        self,
        chans: int = 18,
        pools: int = 4,
        initial_q: float = 0.8,
        initial_local_gate: float = 0.35,
        contribution_budgets: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        self.regulariser = M2GDRegulariser(
            chans=chans,
            pools=pools,
            initial_q=initial_q,
            initial_local_gate=initial_local_gate,
            contribution_budgets=contribution_budgets,
        )
        self.pdfs_dc_weight = nn.Parameter(torch.ones(1))

    @staticmethod
    def sens_expand(
        image: torch.Tensor,
        sensitivity_maps: torch.Tensor,
    ) -> torch.Tensor:
        return fastmri.fft2c(
            fastmri.complex_mul(image, sensitivity_maps)
        )

    @staticmethod
    def sens_reduce(
        kspace: torch.Tensor,
        sensitivity_maps: torch.Tensor,
    ) -> torch.Tensor:
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
        availability: torch.Tensor,
    ) -> Tuple[torch.Tensor, TensorDict]:
        zero = torch.zeros(
            1,
            1,
            1,
            1,
            1,
            device=pdfs_current_kspace.device,
            dtype=pdfs_current_kspace.dtype,
        )

        soft_dc = torch.where(
            mask,
            pdfs_current_kspace - pdfs_ref_kspace,
            zero,
        ) * self.pdfs_dc_weight

        pdfs_image = self.sens_reduce(
            pdfs_current_kspace,
            pdfs_sens_maps,
        )
        regularisation_image, diagnostics = self.regulariser(
            pdfs_image,
            pd_features,
            availability,
        )
        model_term = self.sens_expand(
            regularisation_image,
            pdfs_sens_maps,
        )

        updated_kspace = pdfs_current_kspace - soft_dc - model_term
        return updated_kspace, diagnostics


class M2GDAuxPDVarNet(nn.Module):
    """
    M2-GD: availability-constrained, reliability-supervised global-local
    disagreement-gated PD assistance for PD-FS VarNet reconstruction.

    Only the PD-FS stream has k-space data consistency. The PD feature pyramid
    is computed once and reused across every cascade.

    Forward compatibility:
      * return_aux=False returns only the reconstructed RSS image, matching M2-U.
      * return_aux=True returns (image, auxiliary_outputs) for reliability and
        contribution-budget training.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        mask_center: bool = True,
        initial_q: float = 0.8,
        initial_local_gate: float = 0.35,
        contribution_budgets: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        self.pools = int(pools)
        self.num_cascades = int(num_cascades)

        self.sens_net = SensitivityModel(
            chans=sens_chans,
            num_pools=sens_pools,
            mask_center=mask_center,
        )
        self.pd_encoder = PDFeatureEncoder(
            chans=chans,
            pools=pools,
        )
        self.cascades = nn.ModuleList(
            [
                M2GDVarNetBlock(
                    chans=chans,
                    pools=pools,
                    initial_q=initial_q,
                    initial_local_gate=initial_local_gate,
                    contribution_budgets=contribution_budgets,
                )
                for _ in range(num_cascades)
            ]
        )
        self.fusion_scale_names = [
            f"H/{2 ** level}" for level in range(1, self.pools + 1)
        ]

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

        return _center_crop_or_pad(
            pd_aux_image.float(),
            target_hw,
        )

    def forward(
        self,
        pdfs_masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        pd_aux_image: torch.Tensor,
        pd_available: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, TensorDict]]:
        if (
            pdfs_masked_kspace.ndim != 5
            or pdfs_masked_kspace.shape[-1] != 2
        ):
            raise RuntimeError(
                "Expected PDFS k-space [B,C,H,W,2], "
                f"got {tuple(pdfs_masked_kspace.shape)}"
            )

        if mask.dtype != torch.bool:
            mask = mask.bool()

        batch_size = int(pdfs_masked_kspace.shape[0])
        target_hw = (
            int(pdfs_masked_kspace.shape[-3]),
            int(pdfs_masked_kspace.shape[-2]),
        )
        pd_aux_image = self._prepare_pd(
            pd_aux_image,
            target_hw,
        )
        availability = _prepare_availability(
            pd_available=pd_available,
            batch_size=batch_size,
            device=pdfs_masked_kspace.device,
            dtype=pdfs_masked_kspace.dtype,
        )

        # Computed exactly once and reused by all cascades.
        pd_features = self.pd_encoder(pd_aux_image)

        sensitivity_maps = self.sens_net(
            pdfs_masked_kspace,
            mask,
        )
        current_kspace = pdfs_masked_kspace.clone()
        cascade_diagnostics: List[TensorDict] = []

        for cascade in self.cascades:
            current_kspace, diagnostics = cascade(
                pdfs_current_kspace=current_kspace,
                pdfs_ref_kspace=pdfs_masked_kspace,
                mask=mask,
                pdfs_sens_maps=sensitivity_maps,
                pd_features=pd_features,
                availability=availability,
            )
            cascade_diagnostics.append(diagnostics)

        image = fastmri.ifft2c(current_kspace)
        magnitude = fastmri.complex_abs(image)
        reconstruction = fastmri.rss(magnitude, dim=1)

        if not return_aux:
            return reconstruction

        auxiliary_outputs: TensorDict = {
            # [B,K]
            "q_hat": torch.stack(
                [item["q_hat"] for item in cascade_diagnostics],
                dim=1,
            ),
            "q": torch.stack(
                [item["q"] for item in cascade_diagnostics],
                dim=1,
            ),
            # [B,K,S]
            "g_ch_mean": torch.stack(
                [item["g_ch_mean"] for item in cascade_diagnostics],
                dim=1,
            ),
            "g_ch_std": torch.stack(
                [item["g_ch_std"] for item in cascade_diagnostics],
                dim=1,
            ),
            "g_sp_mean": torch.stack(
                [item["g_sp_mean"] for item in cascade_diagnostics],
                dim=1,
            ),
            "g_sp_std": torch.stack(
                [item["g_sp_std"] for item in cascade_diagnostics],
                dim=1,
            ),
            "w_mean": torch.stack(
                [item["w_mean"] for item in cascade_diagnostics],
                dim=1,
            ),
            "w_std": torch.stack(
                [item["w_std"] for item in cascade_diagnostics],
                dim=1,
            ),
            "adapted_pd_to_target_rms": torch.stack(
                [item["adapted_pd_to_target_rms"] for item in cascade_diagnostics],
                dim=1,
            ),
            "gated_aux_to_target_rms": torch.stack(
                [item["gated_aux_to_target_rms"] for item in cascade_diagnostics],
                dim=1,
            ),
            "availability": availability.flatten(1).squeeze(1),
            "budget_loss": torch.stack(
                [item["budget_loss"] for item in cascade_diagnostics]
            ).mean(),
        }
        return reconstruction, auxiliary_outputs

    def fusion_diagnostics(self) -> Dict[str, Dict[str, float]]:
        """Average latest detached diagnostics across VarNet cascades."""
        per_cascade = [
            cascade.regulariser.fusion_diagnostics()
            for cascade in self.cascades
        ]
        if not per_cascade:
            return {}

        summary: Dict[str, Dict[str, float]] = {}
        for scale in self.fusion_scale_names:
            keys = per_cascade[0][scale].keys()
            summary[scale] = {
                key: float(
                    sum(item[scale][key] for item in per_cascade)
                    / len(per_cascade)
                )
                for key in keys
            }

        summary["global"] = {
            key: float(
                sum(item["global"][key] for item in per_cascade)
                / len(per_cascade)
            )
            for key in ("q_hat_mean", "q_mean")
        }
        return summary
