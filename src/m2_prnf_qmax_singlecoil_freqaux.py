from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.m2_prnf_qmax_singlecoil import QMaxSinglecoilFull


def _as_image4d(value: torch.Tensor) -> torch.Tensor:
    """Convert [B,H,W] or [B,1,H,W] to [B,1,H,W]."""
    if value.ndim == 3:
        value = value.unsqueeze(1)
    if value.ndim != 4 or value.shape[1] != 1:
        raise RuntimeError(
            f"Expected [B,H,W] or [B,1,H,W], got {tuple(value.shape)}"
        )
    return value


def _center_crop_or_pad(
    value: torch.Tensor,
    target_hw: Tuple[int, int],
) -> torch.Tensor:
    target_h, target_w = (int(target_hw[0]), int(target_hw[1]))
    height, width = value.shape[-2:]

    if height > target_h:
        top = (height - target_h) // 2
        value = value[..., top : top + target_h, :]

    if width > target_w:
        left = (width - target_w) // 2
        value = value[..., :, left : left + target_w]

    pad_h = target_h - value.shape[-2]
    pad_w = target_w - value.shape[-1]

    if pad_h < 0 or pad_w < 0:
        raise RuntimeError("Center crop produced an invalid spatial shape")

    if pad_h or pad_w:
        value = F.pad(
            value,
            (
                pad_w // 2,
                pad_w - pad_w // 2,
                pad_h // 2,
                pad_h - pad_h // 2,
            ),
        )

    return value


class ConvPReLU(nn.Module):
    """FSMNet ConvBNReLU2D configuration used by its frequency branch."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        activate: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            bias=False,
        )
        self.activation = nn.PReLU() if activate else nn.Identity()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(value))


class InversePixelShuffle(nn.Module):
    def __init__(self, scale: int = 2):
        super().__init__()
        self.scale = int(scale)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = value.shape
        scale = self.scale

        if height % scale != 0 or width % scale != 0:
            raise RuntimeError(
                f"Shape {(height, width)} is not divisible by {scale}"
            )

        value = value.view(
            batch,
            channels,
            height // scale,
            scale,
            width // scale,
            scale,
        )
        value = value.permute(0, 1, 3, 5, 2, 4).contiguous()

        return value.view(
            batch,
            channels * scale * scale,
            height // scale,
            width // scale,
        )


class FSMDownSample(nn.Module):
    """FSMNet inverse-pixel-shuffle downsampling."""

    def __init__(self, channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            ConvPReLU(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                activate=False,
            ),
            InversePixelShuffle(scale=2),
            ConvPReLU(
                channels * 4,
                channels,
                kernel_size=1,
                activate=False,
            ),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class FSMUpSample(nn.Module):
    """FSMNet scale-2 pixel-shuffle upsampling."""

    def __init__(self, channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels * 4, kernel_size=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class FSMFreBlock9(nn.Module):
    """Amplitude/phase residual block adapted directly from FSMNet."""

    def __init__(self, channels: int):
        super().__init__()

        self.frequency_projection = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
        )
        self.amplitude_residual = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.phase_residual = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.output_projection = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        height, width = value.shape[-2:]
        epsilon = 1e-8

        spectrum = torch.fft.rfft2(
            self.frequency_projection(value) + epsilon,
            norm="backward",
        )

        amplitude = torch.abs(spectrum)
        phase = torch.angle(spectrum)

        amplitude = amplitude + self.amplitude_residual(amplitude)
        phase = phase + self.phase_residual(phase)

        reconstructed_spectrum = torch.complex(
            amplitude * torch.cos(phase) + epsilon,
            amplitude * torch.sin(phase) + epsilon,
        )

        reconstructed = torch.abs(
            torch.fft.irfft2(
                reconstructed_spectrum,
                s=(height, width),
                norm="backward",
            )
        )

        output = self.output_projection(reconstructed) + value

        return torch.nan_to_num(
            output,
            nan=1e-5,
            posinf=1e-5,
            neginf=1e-5,
        )


class FSMModalityFuseBlock6(nn.Module):
    """FSMNet gated fusion between PD and target frequency features."""

    def __init__(self, channels: int):
        super().__init__()

        self.pd_projection = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )
        self.target_projection = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )
        self.gates = nn.Sequential(
            nn.Conv2d(
                channels * 2,
                channels,
                kernel_size=3,
                padding=1,
            ),
            nn.Conv2d(
                channels,
                channels * 2,
                kernel_size=3,
                padding=1,
            ),
            nn.Sigmoid(),
        )

    def forward(
        self,
        pd_feature: torch.Tensor,
        target_feature: torch.Tensor,
    ) -> torch.Tensor:
        pd_feature = self.pd_projection(pd_feature)
        target_feature = self.target_projection(target_feature)

        pd_gate, target_gate = self.gates(
            torch.cat((pd_feature, target_feature), dim=1)
        ).chunk(2, dim=1)

        output = (
            pd_gate * pd_feature
            + target_gate * target_feature
        )

        return torch.nan_to_num(
            output,
            nan=1e-5,
            posinf=1e-5,
            neginf=1e-5,
        )


class FSMFrequencyEncoder(nn.Module):
    """Four-scale FSMNet-style frequency encoder."""

    def __init__(self, channels: int):
        super().__init__()

        self.head = ConvPReLU(
            1,
            channels,
            kernel_size=3,
            padding=1,
        )

        self.down1 = nn.Sequential(
            FSMDownSample(channels),
            FSMFreBlock9(channels),
        )
        self.refine1 = FSMFreBlock9(channels)

        self.down2 = nn.Sequential(
            FSMDownSample(channels),
            FSMFreBlock9(channels),
        )
        self.refine2 = FSMFreBlock9(channels)

        self.down3 = nn.Sequential(
            FSMDownSample(channels),
            FSMFreBlock9(channels),
        )
        self.refine3 = FSMFreBlock9(channels)

        self.neck = FSMFreBlock9(channels)
        self.neck_refine = FSMFreBlock9(channels)

    def forward(
        self,
        value: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        level0 = self.head(value)

        level1 = self.refine1(self.down1(level0))
        level2 = self.refine2(self.down2(level1))
        level3 = self.refine3(self.down3(level2))
        neck = self.neck_refine(self.neck(level3))

        return level0, level1, level2, level3, neck


class QMaxFSMFrequencyAuxiliary(nn.Module):
    """FSMNet-inspired frequency decoder specialised for QMax output."""

    def __init__(self, channels: int = 64):
        super().__init__()
        self.channels = int(channels)

        # PD uses a complete FSMNet frequency encoder.
        self.pd_encoder = FSMFrequencyEncoder(self.channels)

        # QMax output uses the same multiscale frequency topology.
        self.target_head = ConvPReLU(
            1,
            self.channels,
            kernel_size=3,
            padding=1,
        )

        self.target_down1 = nn.Sequential(
            FSMDownSample(self.channels),
            FSMFreBlock9(self.channels),
        )
        self.target_refine1 = FSMFreBlock9(self.channels)

        self.target_down2 = nn.Sequential(
            FSMDownSample(self.channels),
            FSMFreBlock9(self.channels),
        )
        self.target_refine2 = FSMFreBlock9(self.channels)

        self.target_down3 = nn.Sequential(
            FSMDownSample(self.channels),
            FSMFreBlock9(self.channels),
        )
        self.target_refine3 = FSMFreBlock9(self.channels)

        self.target_neck = FSMFreBlock9(self.channels)
        self.target_neck_refine = FSMFreBlock9(self.channels)

        # FSMNet uses five modality fusion blocks.
        self.modality_fusions = nn.ModuleList(
            [
                FSMModalityFuseBlock6(self.channels)
                for _ in range(5)
            ]
        )

        self.up1 = nn.Sequential(
            FSMUpSample(self.channels),
            FSMFreBlock9(self.channels),
        )
        self.up1_refine = FSMFreBlock9(self.channels)

        self.up2 = nn.Sequential(
            FSMUpSample(self.channels),
            FSMFreBlock9(self.channels),
        )
        self.up2_refine = FSMFreBlock9(self.channels)

        self.up3 = nn.Sequential(
            FSMUpSample(self.channels),
            FSMFreBlock9(self.channels),
        )
        self.up3_refine = FSMFreBlock9(self.channels)

        self.tail = ConvPReLU(
            self.channels,
            1,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self,
        qmax_image: torch.Tensor,
        pd_image: torch.Tensor,
    ) -> torch.Tensor:
        qmax_image = _as_image4d(qmax_image)
        pd_image = _as_image4d(pd_image)

        if qmax_image.shape != pd_image.shape:
            raise RuntimeError(
                "QMax and PD frequency inputs must have identical shapes: "
                f"{tuple(qmax_image.shape)} versus {tuple(pd_image.shape)}"
            )

        if (
            qmax_image.shape[-2] % 8 != 0
            or qmax_image.shape[-1] % 8 != 0
        ):
            raise RuntimeError(
                "Frequency input height and width must be divisible by 8"
            )

        pd0, pd1, pd2, pd3, pd_neck = self.pd_encoder(pd_image)

        target0 = self.target_head(qmax_image)
        target0 = self.modality_fusions[0](pd0, target0)

        target1 = self.target_refine1(
            self.target_down1(target0)
        )
        target1 = self.modality_fusions[1](pd1, target1)

        target2 = self.target_refine2(
            self.target_down2(target1)
        )
        target2 = self.modality_fusions[2](pd2, target2)

        target3 = self.target_refine3(
            self.target_down3(target2)
        )
        target3 = self.modality_fusions[3](pd3, target3)

        neck = self.target_neck_refine(
            self.target_neck(target3)
        )
        neck = self.modality_fusions[4](pd_neck, neck)

        # Same residual skip pattern as FSMNet's frequency decoder.
        decoder = neck + target3

        decoder = self.up1_refine(self.up1(decoder))
        decoder = decoder + target2

        decoder = self.up2_refine(self.up2(decoder))
        decoder = decoder + target1

        decoder = self.up3_refine(self.up3(decoder))
        decoder = decoder + target0

        frequency_residual = self.tail(decoder)

        # FSMNet definition: img_fre = res_fre + main.
        return frequency_residual + qmax_image


class QMaxSinglecoilFullFreqAux(nn.Module):
    """Original QMax-Full plus a training-time FSM frequency auxiliary path."""

    def __init__(
        self,
        frequency_channels: int = 64,
        crop_size: int = 320,
        **qmax_kwargs,
    ):
        super().__init__()

        # Construct QMax first. With the same global seed, the QMax
        # initialisation is identical to the original model.
        self.qmax = QMaxSinglecoilFull(**qmax_kwargs)
        self.frequency_auxiliary = QMaxFSMFrequencyAuxiliary(
            channels=frequency_channels
        )
        self.crop_size = int(crop_size)

    def forward(
        self,
        pdfs_masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        pd_aux_image: torch.Tensor,
        *args,
        frequency_mean: torch.Tensor | None = None,
        frequency_std: torch.Tensor | None = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        qmax_result = self.qmax(
            pdfs_masked_kspace,
            mask,
            pd_aux_image,
            *args,
            **kwargs,
        )

        qmax_auxiliary = None
        if isinstance(qmax_result, tuple):
            prediction_raw, qmax_auxiliary = qmax_result
        else:
            prediction_raw = qmax_result

        img_out_4d = _center_crop_or_pad(
            _as_image4d(prediction_raw),
            (self.crop_size, self.crop_size),
        )
        pd_image_4d = _center_crop_or_pad(
            _as_image4d(pd_aux_image.float()),
            (self.crop_size, self.crop_size),
        )

        batch_size = img_out_4d.shape[0]

        if frequency_mean is None or frequency_std is None:
            raise RuntimeError(
                "frequency_mean and frequency_std are required "
                "for the normalized frequency branch"
            )

        if frequency_mean.numel() != batch_size:
            raise RuntimeError(
                "frequency_mean must contain one value per sample"
            )

        if frequency_std.numel() != batch_size:
            raise RuntimeError(
                "frequency_std must contain one value per sample"
            )

        frequency_mean_4d = frequency_mean.to(
            device=img_out_4d.device,
            dtype=img_out_4d.dtype,
        ).reshape(batch_size, 1, 1, 1)

        frequency_std_4d = frequency_std.to(
            device=img_out_4d.device,
            dtype=img_out_4d.dtype,
        ).reshape(batch_size, 1, 1, 1).clamp_min(1e-11)

        # QMax stays in raw MRI units. Only the image-domain frequency
        # branch receives the target-modality normalization used by the
        # FSMNet loss protocol.
        qmax_frequency_input = (
            img_out_4d - frequency_mean_4d
        ) / frequency_std_4d

        # The clean PD auxiliary modality has its own intensity scale.
        # Instance-normalize it independently before feature extraction.
        pd_mean_4d = pd_image_4d.mean(
            dim=(-2, -1),
            keepdim=True,
        )
        pd_std_4d = pd_image_4d.std(
            dim=(-2, -1),
            keepdim=True,
        ).clamp_min(1e-11)
        pd_frequency_input = (
            (pd_image_4d - pd_mean_4d) / pd_std_4d
        ).clamp(-6.0, 6.0)

        # Deliberately not detached: the frequency loss must update QMax.
        # The auxiliary network predicts in normalized units; convert its
        # result back to raw MRI units for the existing output/loss API.
        img_fre_normalized_4d = self.frequency_auxiliary(
            qmax_frequency_input,
            pd_frequency_input,
        )
        img_fre_4d = (
            img_fre_normalized_4d * frequency_std_4d
            + frequency_mean_4d
        )

        output: Dict[str, torch.Tensor] = {
            "prediction_raw": prediction_raw,
            "img_out": img_out_4d[:, 0],
            "img_fre": img_fre_4d[:, 0],
        }

        if qmax_auxiliary is not None:
            output["qmax_auxiliary"] = qmax_auxiliary

        return output


class FSMAmplitudeLoss(nn.Module):
    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        prediction = _as_image4d(prediction)
        target = _as_image4d(target)

        prediction_fft = torch.fft.rfft2(
            prediction,
            norm="backward",
        )
        target_fft = torch.fft.rfft2(
            target,
            norm="backward",
        )

        return F.l1_loss(
            torch.abs(prediction_fft),
            torch.abs(target_fft),
        )


class FSMPhaseLoss(nn.Module):
    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        prediction = _as_image4d(prediction)
        target = _as_image4d(target)

        prediction_fft = torch.fft.rfft2(
            prediction,
            norm="backward",
        )
        target_fft = torch.fft.rfft2(
            target,
            norm="backward",
        )

        return F.l1_loss(
            torch.angle(prediction_fft),
            torch.angle(target_fft),
        )


def qmax_freqaux_loss(
    output: Dict[str, torch.Tensor],
    target: torch.Tensor,
    fft_weight: float = 0.01,
) -> Dict[str, torch.Tensor]:
    """Original QMax loss plus FSMNet-style frequency auxiliary loss."""

    target = _as_image4d(target)[:, 0]
    img_out = output["img_out"]
    img_fre = output["img_fre"]

    amplitude_loss = FSMAmplitudeLoss()
    phase_loss = FSMPhaseLoss()

    main_image = F.l1_loss(img_out, target)
    main_amplitude = amplitude_loss(img_out, target)
    main_phase = phase_loss(img_out, target)

    frequency_image = F.l1_loss(img_fre, target)
    frequency_amplitude = amplitude_loss(img_fre, target)
    frequency_phase = phase_loss(img_fre, target)

    main_total = (
        main_image
        + float(fft_weight) * main_amplitude
        + float(fft_weight) * main_phase
    )
    frequency_total = (
        frequency_image
        + float(fft_weight) * frequency_amplitude
        + float(fft_weight) * frequency_phase
    )

    return {
        "loss": main_total + frequency_total,
        "loss_main": main_total,
        "loss_main_image": main_image,
        "loss_main_amplitude": main_amplitude,
        "loss_main_phase": main_phase,
        "loss_frequency": frequency_total,
        "loss_frequency_image": frequency_image,
        "loss_frequency_amplitude": frequency_amplitude,
        "loss_frequency_phase": frequency_phase,
    }
