from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info


def _load_fsmnet_random_mask_func(fsmnet_root: Path):
    """
    Load RandomMaskFunc directly from the frozen FSMNet checkout.
    This avoids copying or silently changing the public mask implementation.
    """
    source = fsmnet_root / "dataloaders" / "subsample.py"
    if not source.is_file():
        raise FileNotFoundError(f"FSMNet mask source not found: {source}")

    module_name = "_fsmnet_subsample_exact"

    if module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        spec = importlib.util.spec_from_file_location(module_name, source)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import FSMNet mask source: {source}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    return module.RandomMaskFunc


def ifft2c_np(kspace: np.ndarray) -> np.ndarray:
    """Centered orthonormal 2-D inverse FFT."""
    image = np.fft.ifftshift(kspace, axes=(-2, -1))
    image = np.fft.ifft2(image, axes=(-2, -1), norm="ortho")
    image = np.fft.fftshift(image, axes=(-2, -1))
    return image.astype(np.complex64, copy=False)


def fft2c_np(image: np.ndarray) -> np.ndarray:
    """Centered orthonormal 2-D FFT."""
    kspace = np.fft.ifftshift(image, axes=(-2, -1))
    kspace = np.fft.fft2(kspace, axes=(-2, -1), norm="ortho")
    kspace = np.fft.fftshift(kspace, axes=(-2, -1))
    return kspace.astype(np.complex64, copy=False)


def ifft2c_torch(kspace: torch.Tensor) -> torch.Tensor:
    """Centered orthonormal 2-D inverse FFT for complex torch tensors."""
    image = torch.fft.ifftshift(kspace, dim=(-2, -1))
    image = torch.fft.ifft2(image, dim=(-2, -1), norm="ortho")
    image = torch.fft.fftshift(image, dim=(-2, -1))
    return image


def center_crop_complex(
    image: np.ndarray,
    output_shape: Tuple[int, int],
) -> np.ndarray:
    """Center-crop a complex 2-D image."""
    output_h, output_w = output_shape
    input_h, input_w = image.shape[-2:]

    if output_h > input_h or output_w > input_w:
        raise ValueError(
            f"Cannot crop {input_h}x{input_w} to "
            f"{output_h}x{output_w}"
        )

    top = (input_h - output_h) // 2
    left = (input_w - output_w) // 2

    return image[
        ...,
        top:top + output_h,
        left:left + output_w,
    ]


class FSMNetSinglecoilPairedDataset(Dataset):
    """
    Paired PD / FS-PD single-coil dataset for the QMax benchmark.

    Returned tensor layout:

        pd_image:          [1, 320, 320], float32
        target_image:      [1, 320, 320], float32
        target_complex:    [1, 320, 320], complex64
        kspace:            [1, 320, 320], complex64
        masked_kspace:     [1, 320, 320], complex64
        mask:              [1, 320, 320], float32
        mask_1d:           [320], float32
        sensitivity:       [1, 320, 320], complex64
        zero_filled:       [1, 320, 320], float32

    Protocol:

        mask type:         FSMNet RandomMaskFunc
        center fraction:   0.04
        acceleration:      8
        training mask:     stochastic
        evaluation mask:   deterministic from FS-PD filename
        auxiliary PD:      fully sampled and clean
        coil count:        1
    """

    def __init__(
        self,
        manifest_path: str | Path,
        fsmnet_root: str | Path,
        mode: str,
        crop_size: int = 320,
        center_fraction: float = 0.04,
        acceleration: int = 8,
        mask_rng_seed: int = 42,
        deterministic_train_mask: bool = False,
        limit: Optional[int] = None,
    ) -> None:
        super().__init__()

        if mode not in {"train", "val", "test"}:
            raise ValueError("mode must be train, val, or test")

        if crop_size <= 0:
            raise ValueError("crop_size must be positive")

        self.manifest_path = Path(manifest_path).resolve()
        self.fsmnet_root = Path(fsmnet_root).resolve()
        self.mode = mode
        self.crop_size = int(crop_size)
        self.center_fraction = float(center_fraction)
        self.acceleration = int(acceleration)
        self.mask_rng_seed = int(mask_rng_seed)
        self.deterministic_train_mask = bool(
            deterministic_train_mask
        )

        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)

        with self.manifest_path.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as handle:
            rows: List[Dict[str, str]] = list(
                csv.DictReader(handle)
            )

        required_columns = {
            "pair_id",
            "pd_volume_id",
            "fspd_volume_id",
            "slice_index",
            "pd_path",
            "fspd_path",
            "split",
        }

        if not rows:
            raise RuntimeError(
                f"Manifest contains no samples: {self.manifest_path}"
            )

        missing_columns = required_columns - set(rows[0])
        if missing_columns:
            raise RuntimeError(
                f"Manifest is missing columns: "
                f"{sorted(missing_columns)}"
            )

        if limit is not None:
            rows = rows[: int(limit)]

        self.rows = rows

        random_mask_func = _load_fsmnet_random_mask_func(
            self.fsmnet_root
        )

        self.mask_func = random_mask_func(
            center_fractions=[self.center_fraction],
            accelerations=[self.acceleration],
        )

        # FSMNet itself leaves the training mask stochastic. Seeding the
        # generator fixes the run-level RNG sequence without changing its
        # sampling distribution.
        self.mask_func.rng.seed(self.mask_rng_seed)

    def __len__(self) -> int:
        return len(self.rows)

    def reseed_mask_rng(self, seed: int) -> None:
        """Assign an independent reproducible RNG stream to a worker."""
        self.mask_func.rng.seed(int(seed) % (2**32))

    def _load_slice(
        self,
        path: Path,
        slice_index: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not path.is_file():
            raise FileNotFoundError(path)

        with h5py.File(path, "r") as handle:
            if "kspace" not in handle:
                raise KeyError(f"kspace missing from {path}")

            if "reconstruction_esc" not in handle:
                raise KeyError(
                    f"reconstruction_esc missing from {path}"
                )

            num_slices = int(handle["kspace"].shape[0])

            if not 0 <= slice_index < num_slices:
                raise IndexError(
                    f"slice {slice_index} is outside {path}; "
                    f"num_slices={num_slices}"
                )

            kspace = np.asarray(
                handle["kspace"][slice_index],
                dtype=np.complex64,
            )

            reconstruction = np.asarray(
                handle["reconstruction_esc"][slice_index],
                dtype=np.float32,
            )

        if kspace.ndim != 2:
            raise RuntimeError(
                f"Expected single-coil 2-D k-space, got "
                f"{kspace.shape} from {path}"
            )

        if reconstruction.ndim != 2:
            raise RuntimeError(
                f"Expected a 2-D reconstruction, got "
                f"{reconstruction.shape} from {path}"
            )

        return kspace, reconstruction

    def _prepare_complex_crop(
        self,
        raw_kspace: np.ndarray,
    ) -> np.ndarray:
        """
        Preserve phase while moving the acquisition to the fixed QMax grid.

        raw k-space -> complex image -> center crop -> 320-grid k-space
        """
        complex_image = ifft2c_np(raw_kspace)

        return center_crop_complex(
            complex_image,
            (self.crop_size, self.crop_size),
        ).astype(np.complex64, copy=False)

    def _mask_seed(self, fspd_path: Path):
        if self.mode == "train" and not self.deterministic_train_mask:
            # FSMNet uses seed=None, which is re-seeded from system
            # entropy inside temp_seed and therefore cannot be resumed.
            # Drawing an explicit seed preserves the same random-mask
            # distribution while making the sequence checkpointable.
            return int(
                self.mask_func.rng.randint(
                    0,
                    2**31 - 1,
                )
            )

        # Validation/test: one fixed mask per target volume.
        return tuple(map(ord, str(fspd_path)))

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.rows[index]

        pair_id = row["pair_id"]
        slice_index = int(row["slice_index"])
        pd_path = Path(row["pd_path"])
        fspd_path = Path(row["fspd_path"])

        pd_raw_kspace, pd_official_recon = self._load_slice(
            pd_path,
            slice_index,
        )

        fspd_raw_kspace, fspd_official_recon = self._load_slice(
            fspd_path,
            slice_index,
        )

        pd_complex = self._prepare_complex_crop(pd_raw_kspace)
        target_complex_np = self._prepare_complex_crop(
            fspd_raw_kspace
        )

        if pd_complex.shape != (
            self.crop_size,
            self.crop_size,
        ):
            raise RuntimeError(
                f"Unexpected PD crop shape: {pd_complex.shape}"
            )

        target_kspace_np = fft2c_np(target_complex_np)

        target_kspace = torch.from_numpy(
            target_kspace_np.copy()
        ).to(torch.complex64)

        seed = self._mask_seed(fspd_path)

        # Shape follows FSMNet convention: [height, width, complex=2].
        raw_mask = self.mask_func(
            [self.crop_size, self.crop_size, 2],
            seed=seed,
        ).to(torch.float32)

        mask_1d = raw_mask.reshape(-1)

        if mask_1d.numel() != self.crop_size:
            raise RuntimeError(
                f"Unexpected FSMNet mask shape: {tuple(raw_mask.shape)}"
            )

        mask_2d = mask_1d.unsqueeze(0).expand(
            self.crop_size,
            self.crop_size,
        )

        masked_kspace = target_kspace * mask_2d

        zero_filled_complex = ifft2c_torch(masked_kspace)
        zero_filled = zero_filled_complex.abs().to(torch.float32)

        target_complex = torch.from_numpy(
            target_complex_np.copy()
        ).to(torch.complex64)

        pd_from_kspace = np.abs(pd_complex).astype(np.float32)
        target_from_kspace = np.abs(
            target_complex_np
        ).astype(np.float32)

        # Use official fastMRI reconstruction_esc as the metric target and
        # clean auxiliary image. The consistency fields audit the FFT path.
        pd_image = torch.from_numpy(
            pd_official_recon.copy()
        ).to(torch.float32)

        target_image = torch.from_numpy(
            fspd_official_recon.copy()
        ).to(torch.float32)

        if pd_image.shape != (
            self.crop_size,
            self.crop_size,
        ):
            raise RuntimeError(
                f"Official PD reconstruction has shape "
                f"{tuple(pd_image.shape)}, expected "
                f"{self.crop_size}x{self.crop_size}"
            )

        if target_image.shape != (
            self.crop_size,
            self.crop_size,
        ):
            raise RuntimeError(
                f"Official target reconstruction has shape "
                f"{tuple(target_image.shape)}, expected "
                f"{self.crop_size}x{self.crop_size}"
            )

        pd_consistency = float(
            np.max(
                np.abs(pd_from_kspace - pd_official_recon)
            )
        )

        target_consistency = float(
            np.max(
                np.abs(
                    target_from_kspace - fspd_official_recon
                )
            )
        )

        acquired_lines = int(mask_1d.sum().item())
        actual_acceleration = (
            float(self.crop_size) / float(acquired_lines)
        )

        sensitivity = torch.ones(
            (1, self.crop_size, self.crop_size),
            dtype=torch.complex64,
        )

        scale = max(float(target_image.max().item()), 1e-8)

        return {
            "pd_image": pd_image.unsqueeze(0),
            "target_image": target_image.unsqueeze(0),
            "target_complex": target_complex.unsqueeze(0),
            "kspace": target_kspace.unsqueeze(0),
            "masked_kspace": masked_kspace.unsqueeze(0),
            "mask": mask_2d.unsqueeze(0),
            "mask_1d": mask_1d,
            "sensitivity": sensitivity,
            "zero_filled": zero_filled.unsqueeze(0),
            "scale": torch.tensor(scale, dtype=torch.float32),
            "actual_acceleration": torch.tensor(
                actual_acceleration,
                dtype=torch.float32,
            ),
            "acquired_lines": torch.tensor(
                acquired_lines,
                dtype=torch.int64,
            ),
            "pd_fft_consistency_max_abs": torch.tensor(
                pd_consistency,
                dtype=torch.float32,
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


def seed_singlecoil_worker(worker_id: int) -> None:
    """
    Reproducible DataLoader worker initialization.

    Use this as DataLoader(worker_init_fn=seed_singlecoil_worker).
    """
    worker_info = get_worker_info()
    worker_seed = int(torch.initial_seed() % (2**32))

    np.random.seed(worker_seed)

    if worker_info is not None:
        dataset = worker_info.dataset
        if hasattr(dataset, "reseed_mask_rng"):
            dataset.reseed_mask_rng(worker_seed)