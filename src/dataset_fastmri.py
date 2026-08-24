import h5py
import torch
from pathlib import Path
from torch.utils.data import Dataset

from .fft_utils import ifft2c, center_crop, complex_abs, normalize_instance
from .masks import random_cartesian_mask


class FastMRISinglecoilDataset(Dataset):
    def __init__(
        self,
        root,
        split="train",
        center_fraction=0.08,
        acceleration=4,
        sample_rate=1.0,
        use_seed=False,
    ):
        self.root = Path(root) / f"singlecoil_{split}"
        self.center_fraction = center_fraction
        self.acceleration = acceleration
        self.use_seed = use_seed

        files = sorted(self.root.glob("*.h5"))
        if len(files) == 0:
            raise RuntimeError(f"No .h5 files found in {self.root}")

        if sample_rate < 1.0:
            n = max(1, int(len(files) * sample_rate))
            files = files[:n]

        self.files = files
        self.examples = []

        for f in self.files:
            with h5py.File(f, "r") as hf:
                num_slices = hf["kspace"].shape[0]
            for sl in range(num_slices):
                self.examples.append((f, sl))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        fname, sl = self.examples[idx]

        with h5py.File(fname, "r") as hf:
            kspace_np = hf["kspace"][sl]                  # (640, 372), complex64
            target_np = hf["reconstruction_esc"][sl]     # (320, 320), float32

        kspace = torch.from_numpy(kspace_np)

        seed = sl if self.use_seed else None
        mask = random_cartesian_mask(
            kspace.shape,
            center_fraction=self.center_fraction,
            acceleration=self.acceleration,
            seed=seed,
        )

        masked_kspace = kspace * mask

        zf_img = ifft2c(masked_kspace)                  # complex image, (640, 372)
        zf_mag = complex_abs(zf_img)                    # magnitude
        zf_mag = center_crop(zf_mag, 320, 320)         # match target size

        target = torch.from_numpy(target_np).float()

        zf_mag, mean, std = normalize_instance(zf_mag)
        target = (target - mean) / (std + 1e-8)

        return {
            "input": zf_mag.unsqueeze(0),              # (1, 320, 320)
            "target": target.unsqueeze(0),             # (1, 320, 320)
            "fname": str(fname),
            "slice": sl,
        }
