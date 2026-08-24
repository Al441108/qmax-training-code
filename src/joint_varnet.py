from typing import Optional, Tuple

import torch
import torch.nn as nn

import fastmri
from fastmri.models.varnet import NormUnet, SensitivityModel


class CoupledRegulariser(nn.Module):
    """
    Joint PD / PD-FS regulariser.

    Important implementation detail:
    fastMRI NormUnet in this version assumes real/imag grouping internally and
    is not safe as NormUnet(in_chans=4, out_chans=4). Therefore concat fusion is
    implemented as:

        PD complex image    -> [B, 2, H, W]
        PDFS complex image  -> [B, 2, H, W]
        concat              -> [B, 4, H, W]
        1x1 Conv mixing     -> [B, 4, H, W]
        split               -> two [B, 2, H, W]
        shared NormUnet     -> two [B, 1, H, W, 2]

    This keeps NormUnet at in_chans=2/out_chans=2 while still allowing
    cross-contrast information exchange.
    """

    def __init__(
        self,
        chans: int,
        pools: int,
        cross_fusion: str = "concat",
    ):
        super().__init__()

        if cross_fusion not in {"off", "concat"}:
            raise ValueError(
                f"cross_fusion must be 'off' or 'concat', got {cross_fusion}"
            )

        self.cross_fusion = cross_fusion

        self.shared_model = NormUnet(
            chans=chans,
            num_pools=pools,
            in_chans=2,
            out_chans=2,
        )

        if cross_fusion == "concat":
            self.mix = nn.Conv2d(
                in_channels=4,
                out_channels=4,
                kernel_size=1,
                bias=True,
            )
            self._init_identity_mix()
        else:
            self.mix = None

    def _init_identity_mix(self):
        with torch.no_grad():
            self.mix.weight.zero_()
            self.mix.bias.zero_()
            for i in range(4):
                self.mix.weight[i, i, 0, 0] = 1.0

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

    def forward(
        self,
        pd_img: torch.Tensor,
        pdfs_img: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if pd_img.shape != pdfs_img.shape:
            raise RuntimeError(
                "PD and PD-FS image-domain tensors must have identical shape "
                f"before fusion, got {tuple(pd_img.shape)} and {tuple(pdfs_img.shape)}"
            )

        if self.cross_fusion == "off":
            return self.shared_model(pd_img), self.shared_model(pdfs_img)

        pd_ch = self.complex_to_channels(pd_img)
        pdfs_ch = self.complex_to_channels(pdfs_img)

        mixed = self.mix(torch.cat([pd_ch, pdfs_ch], dim=1))

        pd_mixed = mixed[:, 0:2]
        pdfs_mixed = mixed[:, 2:4]

        pd_mixed_complex = self.channels_to_complex(pd_mixed)
        pdfs_mixed_complex = self.channels_to_complex(pdfs_mixed)

        pd_out = self.shared_model(pd_mixed_complex)
        pdfs_out = self.shared_model(pdfs_mixed_complex)

        return pd_out, pdfs_out


class JointVarNetBlock(nn.Module):
    """
    One joint VarNet cascade.

    PD and PD-FS keep independent data consistency:
        - own current k-space
        - own reference masked k-space
        - own sensitivity maps
        - own soft DC weight

    Coupling happens only in the image-domain regulariser.
    """

    def __init__(
        self,
        chans: int,
        pools: int,
        cross_fusion: str = "concat",
    ):
        super().__init__()

        self.regulariser = CoupledRegulariser(
            chans=chans,
            pools=pools,
            cross_fusion=cross_fusion,
        )

        self.pd_dc_weight = nn.Parameter(torch.ones(1))
        self.pdfs_dc_weight = nn.Parameter(torch.ones(1))

    def sens_expand(
        self,
        x: torch.Tensor,
        sens_maps: torch.Tensor,
    ) -> torch.Tensor:
        return fastmri.fft2c(
            fastmri.complex_mul(x, sens_maps)
        )

    def sens_reduce(
        self,
        x: torch.Tensor,
        sens_maps: torch.Tensor,
    ) -> torch.Tensor:
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
        zero = torch.zeros(1, 1, 1, 1, 1).to(pd_current_kspace)

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

        pd_img = self.sens_reduce(
            pd_current_kspace,
            pd_sens_maps,
        )

        pdfs_img = self.sens_reduce(
            pdfs_current_kspace,
            pdfs_sens_maps,
        )

        pd_reg_img, pdfs_reg_img = self.regulariser(
            pd_img=pd_img,
            pdfs_img=pdfs_img,
        )

        pd_model_term = self.sens_expand(
            pd_reg_img,
            pd_sens_maps,
        )

        pdfs_model_term = self.sens_expand(
            pdfs_reg_img,
            pdfs_sens_maps,
        )

        pd_next = pd_current_kspace - pd_soft_dc - pd_model_term
        pdfs_next = pdfs_current_kspace - pdfs_soft_dc - pdfs_model_term

        return pd_next, pdfs_next


class JointVarNet(nn.Module):
    """
    M1 deterministic joint PD / PD-FS VarNet.

    Design:
        - separate sensitivity maps for PD and PD-FS
        - separate data consistency for PD and PD-FS
        - shared / coupled regularisation branch inside each cascade
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
        num_low_frequencies: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if pd_masked_kspace.shape != pdfs_masked_kspace.shape:
            raise RuntimeError(
                "PD and PD-FS masked k-space must have identical shape for paired joint training, "
                f"got {tuple(pd_masked_kspace.shape)} and {tuple(pdfs_masked_kspace.shape)}"
            )

        pd_sens_maps = self.sens_net(
            pd_masked_kspace,
            mask,
            num_low_frequencies,
        )

        pdfs_sens_maps = self.sens_net(
            pdfs_masked_kspace,
            mask,
            num_low_frequencies,
        )

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
