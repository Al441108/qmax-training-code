from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch

from singlecoil_paired_dataset import (
    FSMNetSinglecoilPairedDataset,
    ifft2c_torch,
)


def center_crop_real(
    image: torch.Tensor,
    crop_size: int,
) -> torch.Tensor:
    height, width = image.shape[-2:]

    if crop_size > height or crop_size > width:
        raise ValueError(
            f"Cannot crop {height}x{width} to "
            f"{crop_size}x{crop_size}"
        )

    top = (height - crop_size) // 2
    left = (width - crop_size) // 2

    return image[
        ...,
        top:top + crop_size,
        left:left + crop_size,
    ]


class FSMNetSinglecoilRawGridDataset(
    FSMNetSinglecoilPairedDataset
):
    """
    FSMNet-aligned raw acquisition-grid dataset for QMax.

    The target mask is applied directly to the original fastMRI
    acquisition k-space. QMax therefore performs A/AH on [Hraw, Wraw].

    Loss and metrics must be computed after center-cropping the
    reconstructed magnitude image to 320x320.
    """

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.rows[index]

        pair_id = row["pair_id"]
        slice_index = int(row["slice_index"])
        pd_path = Path(row["pd_path"])
        fspd_path = Path(row["fspd_path"])

        _, pd_official_recon = self._load_slice(
            pd_path,
            slice_index,
        )

        fspd_raw_kspace_np, fspd_official_recon = (
            self._load_slice(
                fspd_path,
                slice_index,
            )
        )

        if fspd_raw_kspace_np.ndim != 2:
            raise RuntimeError(
                "Expected raw single-coil k-space [H,W], got "
                f"{fspd_raw_kspace_np.shape}"
            )

        encoded_height, encoded_width = (
            fspd_raw_kspace_np.shape
        )

        target_kspace = torch.from_numpy(
            np.asarray(
                fspd_raw_kspace_np,
                dtype=np.complex64,
            ).copy()
        ).to(torch.complex64)

        seed = self._mask_seed(fspd_path)

        # Exact FSMNet mask dimensions: [Hraw, Wraw, complex=2].
        raw_mask = self.mask_func(
            [encoded_height, encoded_width, 2],
            seed=seed,
        ).to(torch.float32)

        mask_1d = raw_mask.reshape(-1)

        if mask_1d.numel() != encoded_width:
            raise RuntimeError(
                f"Unexpected mask shape {tuple(raw_mask.shape)} "
                f"for raw k-space "
                f"{encoded_height}x{encoded_width}"
            )

        mask_2d = mask_1d.unsqueeze(0).expand(
            encoded_height,
            encoded_width,
        )

        masked_kspace = target_kspace * mask_2d

        full_complex_image = ifft2c_torch(target_kspace)
        zero_filled_complex = ifft2c_torch(masked_kspace)

        target_from_raw = center_crop_real(
            full_complex_image.abs(),
            self.crop_size,
        ).to(torch.float32)

        zero_filled_crop = center_crop_real(
            zero_filled_complex.abs(),
            self.crop_size,
        ).to(torch.float32)

        pd_image = torch.from_numpy(
            pd_official_recon.copy()
        ).to(torch.float32)

        target_image = torch.from_numpy(
            fspd_official_recon.copy()
        ).to(torch.float32)

        expected_shape = (
            self.crop_size,
            self.crop_size,
        )

        if tuple(pd_image.shape) != expected_shape:
            raise RuntimeError(
                f"PD reconstruction is {tuple(pd_image.shape)}, "
                f"expected {expected_shape}"
            )

        if tuple(target_image.shape) != expected_shape:
            raise RuntimeError(
                "Target reconstruction is "
                f"{tuple(target_image.shape)}, "
                f"expected {expected_shape}"
            )

        target_consistency = float(
            torch.max(
                torch.abs(target_from_raw - target_image)
            ).item()
        )

        acquired_lines = int(mask_1d.sum().item())
        actual_acceleration = (
            float(encoded_width) / float(acquired_lines)
        )

        sensitivity = torch.ones(
            (1, encoded_height, encoded_width),
            dtype=torch.complex64,
        )

        scale = max(
            float(target_image.max().item()),
            1e-8,
        )

        return {
            # Clean, fully sampled PD auxiliary.
            "pd_image": pd_image.unsqueeze(0),

            # Official 320x320 fastMRI metric/loss target.
            "target_image": target_image.unsqueeze(0),

            # Raw acquisition-grid target measurements.
            "kspace": target_kspace.unsqueeze(0),
            "masked_kspace": masked_kspace.unsqueeze(0),
            "mask": mask_2d.unsqueeze(0),
            "mask_1d": mask_1d,

            # Strict single-coil sensitivity.
            "sensitivity": sensitivity,

            # FSMNet-equivalent zero-filled image after raw masking.
            "zero_filled_crop": zero_filled_crop.unsqueeze(0),

            "scale": torch.tensor(
                scale,
                dtype=torch.float32,
            ),
            "actual_acceleration": torch.tensor(
                actual_acceleration,
                dtype=torch.float32,
            ),
            "acquired_lines": torch.tensor(
                acquired_lines,
                dtype=torch.int64,
            ),
            "encoded_height": torch.tensor(
                encoded_height,
                dtype=torch.int64,
            ),
            "encoded_width": torch.tensor(
                encoded_width,
                dtype=torch.int64,
            ),
            "target_fft_consistency_max_abs": torch.tensor(
                target_consistency,
                dtype=torch.float32,
            ),
            "pair_id": pair_id,
            "volume_id": row["volume_id"],
            "pd_volume_id": row["pd_volume_id"],
            "fspd_volume_id": row["fspd_volume_id"],
            "slice_index": slice_index,
            "pd_path": str(pd_path),
            "fspd_path": str(fspd_path),
            "split": row["split"],
        }