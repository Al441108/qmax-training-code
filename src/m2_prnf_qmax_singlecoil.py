from __future__ import annotations

"""
Single-coil physics adapter for frozen QMax-Full.

Only the acquisition physics are changed:

    coil dimension = 1
    sensitivity = 1
    A(x)  = mask * FFT2(x)
    AH(y) = IFFT2(mask * y)

All QMax mechanisms remain inherited from m2_prnf_qmax_varnet.py.
"""

from typing import Any

import fastmri
import torch
import torch.nn as nn

from src.m2_prnf_qmax_varnet import QMaxAuxPDVarNet


def _to_fastmri_complex(value: torch.Tensor) -> torch.Tensor:
    """
    Convert native PyTorch complex tensors to fastMRI real/imag layout.

    Accepted:
        native complex: [..., H, W]
        fastMRI layout: [..., H, W, 2]
    """
    if value.is_complex():
        return torch.view_as_real(value.contiguous())

    if value.ndim < 3 or value.shape[-1] != 2:
        raise ValueError(
            "Expected a native complex tensor or a tensor whose "
            f"last dimension is 2, got {tuple(value.shape)} "
            f"with dtype={value.dtype}"
        )

    return value.contiguous()


def _prepare_mask(
    mask: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """
    Convert a spatial mask to a boolean mask broadcastable to fastMRI
    real/imag tensors.

    Typical input:
        mask:      [B, 1, H, W]
        reference: [B, 1, H, W, 2]

    Output:
        [B, 1, H, W, 1]
    """
    if mask.is_complex():
        raise TypeError("Sampling mask cannot be complex")

    if mask.ndim == reference.ndim - 1:
        mask = mask.unsqueeze(-1)

    if mask.ndim != reference.ndim:
        raise ValueError(
            f"Mask/reference rank mismatch: "
            f"{tuple(mask.shape)} versus {tuple(reference.shape)}"
        )

    if mask.shape[-1] not in (1, reference.shape[-1]):
        raise ValueError(
            f"Unexpected mask complex dimension: {tuple(mask.shape)}"
        )

    for mask_size, reference_size in zip(
        mask.shape[:-1],
        reference.shape[:-1],
    ):
        if mask_size not in (1, reference_size):
            raise ValueError(
                f"Mask is not broadcastable: "
                f"{tuple(mask.shape)} versus "
                f"{tuple(reference.shape)}"
            )

    return mask.bool()


def singlecoil_forward_operator(
    image: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    A(x) = mask * FFT2(x).

    image may be native complex or fastMRI real/imag layout.
    The returned tensor uses fastMRI real/imag layout.
    """
    image_ri = _to_fastmri_complex(image)
    mask_bool = _prepare_mask(mask, image_ri)

    full_kspace = fastmri.fft2c(image_ri)
    zero = torch.zeros(
        1,
        dtype=full_kspace.dtype,
        device=full_kspace.device,
    )

    return torch.where(mask_bool, full_kspace, zero)


def singlecoil_adjoint_operator(
    kspace: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    AH(y) = IFFT2(mask * y).

    kspace may be native complex or fastMRI real/imag layout.
    The returned tensor uses fastMRI real/imag layout.
    """
    kspace_ri = _to_fastmri_complex(kspace)
    mask_bool = _prepare_mask(mask, kspace_ri)

    zero = torch.zeros(
        1,
        dtype=kspace_ri.dtype,
        device=kspace_ri.device,
    )

    sampled_kspace = torch.where(mask_bool, kspace_ri, zero)
    return fastmri.ifft2c(sampled_kspace)


class UnitSensitivityModel(nn.Module):
    """
    Exact unit sensitivity for a single-coil acquisition.

    Output representation is real/imag:
        real channel = 1
        imaginary channel = 0
    """

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Any = None,
    ) -> torch.Tensor:
        del mask, num_low_frequencies

        masked_kspace = _to_fastmri_complex(masked_kspace)

        if masked_kspace.ndim != 5:
            raise ValueError(
                "Expected masked k-space [B,1,H,W,2], got "
                f"{tuple(masked_kspace.shape)}"
            )

        if masked_kspace.shape[1] != 1:
            raise ValueError(
                "Single-coil QMax requires exactly one coil, got "
                f"{masked_kspace.shape[1]}"
            )

        sensitivity = torch.zeros_like(masked_kspace)
        sensitivity[..., 0] = 1.0
        return sensitivity


class QMaxSinglecoilFull(QMaxAuxPDVarNet):
    """
    QMax-Full with strict single-coil physics.

    The inherited QMax architecture, fusion, q, stable/detail split,
    alignment, direct path, correction and detached DC evidence are
    unchanged.
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs = dict(kwargs)

        requested_variant = kwargs.get(
            "qmax_variant",
            "qmax_full",
        )

        if requested_variant != "qmax_full":
            raise ValueError(
                "QMaxSinglecoilFull only supports "
                f"qmax_variant='qmax_full', got {requested_variant!r}"
            )

        kwargs["qmax_variant"] = "qmax_full"
        super().__init__(*args, **kwargs)

        # Remove learned multicoil sensitivity estimation.
        self.sens_net = UnitSensitivityModel()

    def forward(
        self,
        pdfs_masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        *args,
        **kwargs,
    ):
        pdfs_masked_kspace = _to_fastmri_complex(
            pdfs_masked_kspace
        )

        if pdfs_masked_kspace.ndim != 5:
            raise ValueError(
                "Expected [B,1,H,W,2] target k-space, got "
                f"{tuple(pdfs_masked_kspace.shape)}"
            )

        if pdfs_masked_kspace.shape[1] != 1:
            raise ValueError(
                "Single-coil QMax requires coil dimension 1, got "
                f"{pdfs_masked_kspace.shape[1]}"
            )

        mask = _prepare_mask(mask, pdfs_masked_kspace)

        return super().forward(
            pdfs_masked_kspace,
            mask,
            *args,
            **kwargs,
        )


# Explicit alias for training scripts.
QMaxSinglecoilVarNet = QMaxSinglecoilFull