from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import fastmri
from fastmri.models.varnet import SensitivityModel


class ConvBlock(nn.Module):
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

    def forward(self, x):
        return self.layers(x)


class AuxUNet3to2(nn.Module):
    """
    Auxiliary PD -> PD-FS U-Net regulariser.

    Input:
        [B, 3, H, W] = [PDFS_real, PDFS_imag, PD_aux_mag]

    Output:
        [B, 2, H, W] = [PDFS_update_real, PDFS_update_imag]

    This module predicts only a PD-FS model term/update. PD is an image-domain
    auxiliary input and is not reconstructed.
    """

    def __init__(
        self,
        in_chans: int = 3,
        out_chans: int = 2,
        chans: int = 18,
        pools: int = 4,
    ):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.pools = pools

        down = []
        ch = in_chans

        for i in range(pools):
            out_ch = chans * (2 ** i)
            down.append(ConvBlock(ch, out_ch))
            ch = out_ch

        self.down = nn.ModuleList(down)
        self.bottleneck = ConvBlock(ch, ch * 2)

        up = []
        up_ch = ch * 2

        for i in reversed(range(pools)):
            skip_ch = chans * (2 ** i)
            out_ch = skip_ch
            up.append(ConvBlock(up_ch + skip_ch, out_ch))
            up_ch = out_ch

        self.up = nn.ModuleList(up)
        self.out_conv = nn.Conv2d(chans, out_chans, kernel_size=1)

        # Stable VarNet-style start: zero regularisation update initially.
        nn.init.zeros_(self.out_conv.weight)
        if self.out_conv.bias is not None:
            nn.init.zeros_(self.out_conv.bias)

    @staticmethod
    def _pad_to_multiple(x, multiple: int):
        _, _, h, w = x.shape
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect")
        return x, (pad_top, pad_bottom, pad_left, pad_right)

    @staticmethod
    def _unpad(x, pads):
        pad_top, pad_bottom, pad_left, pad_right = pads

        h_end = x.shape[-2] - pad_bottom if pad_bottom > 0 else x.shape[-2]
        w_end = x.shape[-1] - pad_right if pad_right > 0 else x.shape[-1]

        return x[..., pad_top:h_end, pad_left:w_end]

    def forward(self, x):
        original_h, original_w = x.shape[-2:]
        multiple = 2 ** self.pools

        b = x.shape[0]
        mean = x.reshape(b, -1).mean(dim=1).view(b, 1, 1, 1)
        std = x.reshape(b, -1).std(dim=1).view(b, 1, 1, 1).clamp_min(1e-7)
        x = (x - mean) / std

        x, pads = self._pad_to_multiple(x, multiple)

        skips = []
        out = x

        for block in self.down:
            out = block(out)
            skips.append(out)
            out = F.avg_pool2d(out, kernel_size=2, stride=2)

        out = self.bottleneck(out)

        for block, skip in zip(self.up, reversed(skips)):
            out = F.interpolate(out, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            out = torch.cat([out, skip], dim=1)
            out = block(out)

        out = self.out_conv(out)
        out = self._unpad(out, pads)

        # Rescale update magnitude. Do not add mean, because zero output should
        # remain a neutral zero model term.
        out = out * std

        if out.shape[-2:] != (original_h, original_w):
            raise RuntimeError(
                f"AuxUNet3to2 output shape mismatch: got {tuple(out.shape[-2:])}, "
                f"expected {(original_h, original_w)}"
            )

        return out


class AuxRegulariser3to2(nn.Module):
    """
    Auxiliary PD -> PD-FS regulariser.

    PDFS complex image -> [B,2,H,W]
    PD auxiliary magnitude image -> [B,1,H,W], centre-cropped/padded to match PDFS.
    concat -> [B,3,H,W]
    U-Net -> [B,2,H,W]
    output -> [B,1,H,W,2]
    """

    def __init__(self, chans: int, pools: int):
        super().__init__()
        self.model = AuxUNet3to2(
            in_chans=3,
            out_chans=2,
            chans=chans,
            pools=pools,
        )

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
                "Expected real-channel tensor [B,2,H,W], "
                f"got {tuple(x.shape)}"
            )
        return x.permute(0, 2, 3, 1).unsqueeze(1).contiguous()

    @staticmethod
    def _center_crop_or_pad(x: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        """
        Centre-crop or zero-pad a [B,1,H,W] image to target_hw.
        This handles the common fastMRI case where target reconstruction is
        already centre-cropped while the VarNet internal image follows k-space FOV.
        """
        if x.ndim != 4 or x.shape[1] != 1:
            raise RuntimeError(f"Expected [B,1,H,W], got {tuple(x.shape)}")

        target_h, target_w = int(target_hw[0]), int(target_hw[1])
        _, _, h, w = x.shape

        # Centre crop if needed.
        if h > target_h:
            top = (h - target_h) // 2
            x = x[..., top:top + target_h, :]
            h = target_h
        if w > target_w:
            left = (w - target_w) // 2
            x = x[..., left:left + target_w]
            w = target_w

        # Centre pad if needed.
        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)
        if pad_h > 0 or pad_w > 0:
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0)

        return x

    @staticmethod
    def _prepare_pd_aux(pd_aux_image: torch.Tensor, pdfs_ch: torch.Tensor) -> torch.Tensor:
        """
        Convert PD auxiliary image to [B,1,H,W] and match PDFS regulariser FOV.
        Per-sample scale is normalised and then matched to current PDFS image scale.
        """
        if pd_aux_image.ndim == 3:
            pd = pd_aux_image.unsqueeze(1)
        elif pd_aux_image.ndim == 4 and pd_aux_image.shape[1] == 1:
            pd = pd_aux_image
        else:
            raise RuntimeError(
                "Expected PD auxiliary image [B,H,W] or [B,1,H,W], "
                f"got {tuple(pd_aux_image.shape)}"
            )

        pd = pd.float()
        pd = AuxRegulariser3to2._center_crop_or_pad(pd, pdfs_ch.shape[-2:])

        pd_scale = pd.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        pd = pd / pd_scale

        pdfs_scale = pdfs_ch.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)
        pd = pd * pdfs_scale

        return pd

    def forward(self, pdfs_img: torch.Tensor, pd_aux_image: torch.Tensor) -> torch.Tensor:
        pdfs_ch = self.complex_to_channels(pdfs_img)
        pd_ch = self._prepare_pd_aux(pd_aux_image, pdfs_ch)

        if pd_ch.shape[-2:] != pdfs_ch.shape[-2:]:
            raise RuntimeError(
                f"PD auxiliary shape {tuple(pd_ch.shape)} does not match "
                f"PDFS image channels {tuple(pdfs_ch.shape)}"
            )

        aux_in = torch.cat([pdfs_ch, pd_ch], dim=1)
        pdfs_out_ch = self.model(aux_in)
        return self.channels_to_complex(pdfs_out_ch)


class AuxVarNetBlock(nn.Module):
    def __init__(self, chans: int, pools: int):
        super().__init__()
        self.regulariser = AuxRegulariser3to2(chans=chans, pools=pools)
        self.pdfs_dc_weight = nn.Parameter(torch.ones(1))

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.fft2c(fastmri.complex_mul(x, sens_maps))

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.complex_mul(
            fastmri.ifft2c(x),
            fastmri.complex_conj(sens_maps),
        ).sum(dim=1, keepdim=True)

    def forward(
        self,
        pdfs_current_kspace: torch.Tensor,
        pdfs_ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        pdfs_sens_maps: torch.Tensor,
        pd_aux_image: torch.Tensor,
    ) -> torch.Tensor:

        zero = torch.zeros(
            1, 1, 1, 1, 1,
            device=pdfs_current_kspace.device,
            dtype=pdfs_current_kspace.dtype,
        )

        pdfs_soft_dc = torch.where(
            mask,
            pdfs_current_kspace - pdfs_ref_kspace,
            zero,
        ) * self.pdfs_dc_weight

        pdfs_img = self.sens_reduce(pdfs_current_kspace, pdfs_sens_maps)
        pdfs_reg_img = self.regulariser(pdfs_img, pd_aux_image)
        pdfs_model_term = self.sens_expand(pdfs_reg_img, pdfs_sens_maps)

        pdfs_next = pdfs_current_kspace - pdfs_soft_dc - pdfs_model_term
        return pdfs_next


class AuxPDVarNet(nn.Module):
    """
    Auxiliary PD-assisted PD-FS VarNet.

    Design:
        - only PD-FS has a k-space VarNet stream and data consistency
        - PD is provided as a full/high-quality image-domain auxiliary channel
        - output and loss are PD-FS only
    """

    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        mask_center: bool = True,
    ):
        super().__init__()

        self.sens_net = SensitivityModel(
            chans=sens_chans,
            num_pools=sens_pools,
            mask_center=mask_center,
        )

        self.cascades = nn.ModuleList(
            [AuxVarNetBlock(chans=chans, pools=pools) for _ in range(num_cascades)]
        )

    def forward(
        self,
        pdfs_masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        pd_aux_image: torch.Tensor,
    ) -> torch.Tensor:

        pdfs_sens_maps = self.sens_net(pdfs_masked_kspace, mask)
        pdfs_kspace_pred = pdfs_masked_kspace.clone()

        for cascade in self.cascades:
            pdfs_kspace_pred = cascade(
                pdfs_current_kspace=pdfs_kspace_pred,
                pdfs_ref_kspace=pdfs_masked_kspace,
                mask=mask,
                pdfs_sens_maps=pdfs_sens_maps,
                pd_aux_image=pd_aux_image,
            )

        pdfs_recon = fastmri.rss(
            fastmri.complex_abs(fastmri.ifft2c(pdfs_kspace_pred)),
            dim=1,
        )

        return pdfs_recon
