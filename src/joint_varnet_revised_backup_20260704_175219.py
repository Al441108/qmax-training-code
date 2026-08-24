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


class JointUNet4ch(nn.Module):
    """
    Real-valued 4-channel U-Net regulariser for jVN-style joint reconstruction.

    Input:
        [B, 4, H, W] = [PD_real, PD_imag, PDFS_real, PDFS_imag]

    Output:
        [B, 4, H, W]
    """

    def __init__(self, in_chans: int = 4, out_chans: int = 4, chans: int = 18, pools: int = 4):
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

        if out.shape[-2:] != (original_h, original_w):
            raise RuntimeError(
                f"JointUNet4ch output shape mismatch: got {tuple(out.shape[-2:])}, "
                f"expected {(original_h, original_w)}"
            )

        return out


class JointRegulariser4ch(nn.Module):
    """
    jVN-style joint regulariser.

    It stacks both contrasts along the channel dimension:
        PD complex image    -> [B, 2, H, W]
        PDFS complex image  -> [B, 2, H, W]
        concat              -> [B, 4, H, W]
        joint U-Net          -> [B, 4, H, W]
        split               -> two [B, 1, H, W, 2]
    """

    def __init__(self, chans: int, pools: int, cross_fusion: str = "concat"):
        super().__init__()

        if cross_fusion not in {"concat", "off"}:
            raise ValueError(f"cross_fusion must be 'concat' or 'off', got {cross_fusion}")

        self.cross_fusion = cross_fusion

        if cross_fusion == "concat":
            self.joint_model = JointUNet4ch(
                in_chans=4,
                out_chans=4,
                chans=chans,
                pools=pools,
            )
        else:
            self.pd_model = JointUNet4ch(
                in_chans=2,
                out_chans=2,
                chans=chans,
                pools=pools,
            )
            self.pdfs_model = JointUNet4ch(
                in_chans=2,
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

    def forward(self, pd_img: torch.Tensor, pdfs_img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if pd_img.shape != pdfs_img.shape:
            raise RuntimeError(
                "PD and PD-FS image-domain tensors must have identical shape, "
                f"got {tuple(pd_img.shape)} and {tuple(pdfs_img.shape)}"
            )

        pd_ch = self.complex_to_channels(pd_img)
        pdfs_ch = self.complex_to_channels(pdfs_img)

        if self.cross_fusion == "concat":
            joint_in = torch.cat([pd_ch, pdfs_ch], dim=1)
            joint_out = self.joint_model(joint_in)

            pd_out_ch = joint_out[:, 0:2]
            pdfs_out_ch = joint_out[:, 2:4]
        else:
            pd_out_ch = self.pd_model(pd_ch)
            pdfs_out_ch = self.pdfs_model(pdfs_ch)

        pd_out = self.channels_to_complex(pd_out_ch)
        pdfs_out = self.channels_to_complex(pdfs_out_ch)

        return pd_out, pdfs_out


class JointVarNetBlock(nn.Module):
    def __init__(self, chans: int, pools: int, cross_fusion: str = "concat"):
        super().__init__()

        self.regulariser = JointRegulariser4ch(
            chans=chans,
            pools=pools,
            cross_fusion=cross_fusion,
        )

        self.pd_dc_weight = nn.Parameter(torch.ones(1))
        self.pdfs_dc_weight = nn.Parameter(torch.ones(1))

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.fft2c(
            fastmri.complex_mul(x, sens_maps)
        )

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.complex_mul(
            fastmri.ifft2c(x),
            fastmri.complex_conj(sens_maps),
        ).sum(dim=1, keepdim=True)

    def forward(
        self,
        pd_current_kspace: torch.Tensor,
        pdfs_current_kspace: torch.Tensor,
        pd_ref_kspace: torch.Tensor,
        pdfs_ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        pd_sens_maps: torch.Tensor,
        pdfs_sens_maps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        zero = torch.zeros(1, 1, 1, 1, 1, device=pd_current_kspace.device, dtype=pd_current_kspace.dtype)

        pd_soft_dc = torch.where(
            mask,
            pd_current_kspace - pd_ref_kspace,
            zero,
        ) * self.pd_dc_weight

        pdfs_soft_dc = torch.where(
            mask,
            pdfs_current_kspace - pdfs_ref_kspace,
            zero,
        ) * self.pdfs_dc_weight

        pd_img = self.sens_reduce(pd_current_kspace, pd_sens_maps)
        pdfs_img = self.sens_reduce(pdfs_current_kspace, pdfs_sens_maps)

        pd_reg_img, pdfs_reg_img = self.regulariser(pd_img, pdfs_img)

        pd_model_term = self.sens_expand(pd_reg_img, pd_sens_maps)
        pdfs_model_term = self.sens_expand(pdfs_reg_img, pdfs_sens_maps)

        pd_next = pd_current_kspace - pd_soft_dc - pd_model_term
        pdfs_next = pdfs_current_kspace - pdfs_soft_dc - pdfs_model_term

        return pd_next, pdfs_next


class JointVarNet(nn.Module):
    """
    Revised M1 jVN-style JointVarNet.

    Design:
        - shared sensitivity estimation module, applied separately to PD and PD-FS
        - independent data consistency for each contrast
        - 4-channel joint U-Net regulariser inside each cascade
    """

    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        mask_center: bool = True,
        cross_fusion: str = "concat",
    ):
        super().__init__()

        self.cross_fusion = cross_fusion

        self.sens_net = SensitivityModel(
            chans=sens_chans,
            num_pools=sens_pools,
            mask_center=mask_center,
        )

        self.cascades = nn.ModuleList(
            [
                JointVarNetBlock(
                    chans=chans,
                    pools=pools,
                    cross_fusion=cross_fusion,
                )
                for _ in range(num_cascades)
            ]
        )

    def forward(
        self,
        pd_masked_kspace: torch.Tensor,
        pdfs_masked_kspace: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if pd_masked_kspace.shape != pdfs_masked_kspace.shape:
            raise RuntimeError(
                f"PD/PDFS k-space shape mismatch: "
                f"{tuple(pd_masked_kspace.shape)} vs {tuple(pdfs_masked_kspace.shape)}"
            )

        pd_sens_maps = self.sens_net(pd_masked_kspace, mask)
        pdfs_sens_maps = self.sens_net(pdfs_masked_kspace, mask)

        pd_kspace_pred = pd_masked_kspace.clone()
        pdfs_kspace_pred = pdfs_masked_kspace.clone()

        for cascade in self.cascades:
            pd_kspace_pred, pdfs_kspace_pred = cascade(
                pd_current_kspace=pd_kspace_pred,
                pdfs_current_kspace=pdfs_kspace_pred,
                pd_ref_kspace=pd_masked_kspace,
                pdfs_ref_kspace=pdfs_masked_kspace,
                mask=mask,
                pd_sens_maps=pd_sens_maps,
                pdfs_sens_maps=pdfs_sens_maps,
            )

        pd_recon = fastmri.rss(
            fastmri.complex_abs(
                fastmri.ifft2c(pd_kspace_pred)
            ),
            dim=1,
        )

        pdfs_recon = fastmri.rss(
            fastmri.complex_abs(
                fastmri.ifft2c(pdfs_kspace_pred)
            ),
            dim=1,
        )

        return pd_recon, pdfs_recon
