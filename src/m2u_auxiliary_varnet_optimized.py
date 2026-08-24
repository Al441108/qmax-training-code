from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import fastmri
from fastmri.models.varnet import SensitivityModel


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


class PDFeatureEncoder(nn.Module):
    """
    Independently encodes the PD auxiliary magnitude image once per VarNet pass.

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


class M2UFeatureFusion(nn.Module):
    """Ungated but explicitly scaled PD-to-PD-FS feature adapter.

    ``alpha`` is deliberately separate from the adapter.  A tiny convolution
    followed by InstanceNorm is *not* a small perturbation: InstanceNorm
    rescales it to roughly unit variance.  The explicit scalar therefore
    provides a genuine, auditable near-identity start for M2-U.
    """

    def __init__(
        self,
        pd_chans: int,
        target_chans: int,
        initial_alpha: float = 0.1,
    ):
        super().__init__()
        conv = nn.Conv2d(
            pd_chans,
            target_chans,
            kernel_size=1,
            bias=False,
        )
        nn.init.kaiming_normal_(conv.weight, a=0.2)

        self.adapter = nn.Sequential(
            conv,
            # The residual scale is controlled only by alpha below.
            nn.InstanceNorm2d(target_chans, affine=False),
        )
        self.alpha = nn.Parameter(torch.tensor(float(initial_alpha)))
        self.last_aux_ratio: torch.Tensor | None = None

    def forward(
        self,
        target_feature: torch.Tensor,
        pd_feature: torch.Tensor,
    ) -> torch.Tensor:
        if pd_feature.shape[-2:] != target_feature.shape[-2:]:
            pd_feature = F.interpolate(
                pd_feature,
                size=target_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        pd_term = self.alpha * self.adapter(pd_feature)

        # Diagnostic only: how large is the injected auxiliary term relative
        # to the target feature?  Detaching avoids retaining an autograd graph.
        target_rms = target_feature.detach().square().mean().sqrt().clamp_min(1e-8)
        aux_rms = pd_term.detach().square().mean().sqrt()
        self.last_aux_ratio = aux_rms / target_rms
        return target_feature + pd_term


class M2URegulariser(nn.Module):
    """
    PD-FS U-Net regulariser with ungated multi-scale PD feature fusion.

    PD is not fused at full resolution. Fusion is applied at H/2 and below,
    including the bottleneck.
    """

    def __init__(
        self,
        chans: int = 18,
        pools: int = 4,
        initial_aux_alpha: float = 0.1,
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

        # Feature 0 is full resolution and intentionally has no fusion module.
        self.fusions = nn.ModuleList(
            [
                M2UFeatureFusion(
                    pd_chans=self.chans * (2 ** level),
                    target_chans=self.chans * (2 ** level),
                    initial_alpha=initial_aux_alpha,
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
            up_blocks.append(
                ConvBlock(up_chans + skip_chans, skip_chans)
            )
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

    def forward(
        self,
        pdfs_image: torch.Tensor,
        pd_features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
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

        target_features: List[torch.Tensor] = []
        out = x

        for level, block in enumerate(self.target_down):
            out = block(out)

            # No PD fusion at full resolution.
            if level == 0:
                fused = out
            else:
                fused = self.fusions[level - 1](
                    out,
                    pd_features[level],
                )

            target_features.append(fused)
            # Intentionally pool the target-only encoder state.  PD enters
            # through multi-scale skips and the bottleneck, not by repeatedly
            # rewriting deeper target-encoder representations.
            out = F.avg_pool2d(out, kernel_size=2, stride=2)

        out = self.target_bottleneck(out)
        out = self.fusions[-1](out, pd_features[-1])

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
                f"M2-U output shape {tuple(out.shape[-2:])} "
                f"does not match input {tuple(original_hw)}"
            )

        return self.channels_to_complex(out)

    def fusion_diagnostics(self) -> dict[str, dict[str, float]]:
        """Return per-scale alpha and relative auxiliary contribution."""
        diagnostics: dict[str, dict[str, float]] = {}
        for name, fusion in zip(self.fusion_scale_names, self.fusions):
            ratio = fusion.last_aux_ratio
            diagnostics[name] = {
                "alpha": float(fusion.alpha.detach().cpu()),
                "aux_to_target_rms": (
                    float(ratio.detach().cpu()) if ratio is not None else float("nan")
                ),
            }
        return diagnostics


class M2UVarNetBlock(nn.Module):
    """One PD-FS VarNet cascade with M2-U image-domain regularisation."""

    def __init__(
        self,
        chans: int = 18,
        pools: int = 4,
        initial_aux_alpha: float = 0.1,
    ):
        super().__init__()
        self.regulariser = M2URegulariser(
            chans=chans,
            pools=pools,
            initial_aux_alpha=initial_aux_alpha,
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
    ) -> torch.Tensor:
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
        regularisation_image = self.regulariser(
            pdfs_image,
            pd_features,
        )
        model_term = self.sens_expand(
            regularisation_image,
            pdfs_sens_maps,
        )

        return pdfs_current_kspace - soft_dc - model_term


class M2UAuxPDVarNet(nn.Module):
    """
    M2-U: independently encoded PD and PD-FS features with ungated,
    multi-scale additive fusion.

    Only the PD-FS stream has k-space data consistency. The PD feature
    pyramid is computed once and reused across every cascade.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        mask_center: bool = True,
        initial_aux_alpha: float = 0.1,
    ):
        super().__init__()
        self.pools = int(pools)

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
                M2UVarNetBlock(
                    chans=chans,
                    pools=pools,
                    initial_aux_alpha=initial_aux_alpha,
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

        return _center_crop_or_pad(
            pd_aux_image.float(),
            target_hw,
        )

    def forward(
        self,
        pdfs_masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        pd_aux_image: torch.Tensor,
    ) -> torch.Tensor:
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

        target_hw = (
            int(pdfs_masked_kspace.shape[-3]),
            int(pdfs_masked_kspace.shape[-2]),
        )
        pd_aux_image = self._prepare_pd(
            pd_aux_image,
            target_hw,
        )

        # Computed exactly once and reused by all cascades.
        pd_features = self.pd_encoder(pd_aux_image)

        sensitivity_maps = self.sens_net(
            pdfs_masked_kspace,
            mask,
        )
        current_kspace = pdfs_masked_kspace.clone()

        for cascade in self.cascades:
            current_kspace = cascade(
                pdfs_current_kspace=current_kspace,
                pdfs_ref_kspace=pdfs_masked_kspace,
                mask=mask,
                pdfs_sens_maps=sensitivity_maps,
                pd_features=pd_features,
            )

        image = fastmri.ifft2c(current_kspace)
        magnitude = fastmri.complex_abs(image)
        return fastmri.rss(magnitude, dim=1)

    def fusion_diagnostics(self) -> dict[str, dict[str, float]]:
        """Average the latest per-scale diagnostics across VarNet cascades."""
        per_cascade = [cascade.regulariser.fusion_diagnostics() for cascade in self.cascades]
        if not per_cascade:
            return {}

        summary: dict[str, dict[str, float]] = {}
        for scale in per_cascade[0]:
            summary[scale] = {
                key: float(sum(item[scale][key] for item in per_cascade) / len(per_cascade))
                for key in ("alpha", "aux_to_target_rms")
            }
        return summary
