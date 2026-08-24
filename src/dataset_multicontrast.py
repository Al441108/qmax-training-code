import json
import h5py
import torch
from pathlib import Path
from torch.utils.data import Dataset

from .fft_utils import ifft2c, center_crop, complex_abs, normalize_instance
from .masks import random_cartesian_mask


class FastMRIMultiContrastDataset(Dataset):
    """
    Multi-contrast fastMRI knee dataset using patient-level paired PD / PD-FS scans.

    Output:
        input  : (2, 320, 320) = [PD_zero_filled, PDFS_zero_filled]
        target : (2, 320, 320) = [PD_target, PDFS_target]

    Notes:
    - One sample = one paired patient volume + one matched slice index
    - Uses center crop to 320x320
    - Uses the same undersampling mask for PD and PD-FS
    """

    def __init__(
        self,
        pairs_json,
        split="singlecoil_train",
        center_fraction=0.08,
        acceleration=4,
        sample_rate=1.0,
        use_seed=False,
        match_mode="relative",   # "relative" or "middle"
    ):
        self.pairs_json = Path(pairs_json)
        self.split = split
        self.center_fraction = center_fraction
        self.acceleration = acceleration
        self.use_seed = use_seed
        self.match_mode = match_mode

        if not self.pairs_json.exists():
            raise FileNotFoundError(f"Pairs JSON not found: {self.pairs_json}")

        with open(self.pairs_json, "r", encoding="utf-8") as f:
            all_pairs = json.load(f)

        # 只保留当前 split
        pairs = [p for p in all_pairs if p["split"] == split]

        if len(pairs) == 0:
            raise RuntimeError(f"No pairs found for split={split}")

        if sample_rate < 1.0:
            n = max(1, int(len(pairs) * sample_rate))
            pairs = pairs[:n]

        self.pairs = pairs
        self.examples = []

        # 预生成 slice-level 样本索引
        for pair_idx, pair in enumerate(self.pairs):
            pd_path = pair["pd_path"]
            pdfs_path = pair["pdfs_path"]

            try:
                with h5py.File(pd_path, "r") as hf_pd, h5py.File(pdfs_path, "r") as hf_pdfs:
                    n_pd = hf_pd["kspace"].shape[0]
                    n_pdfs = hf_pdfs["kspace"].shape[0]
            except Exception as e:
                print(f"[WARN] Skip pair due to read error: {pd_path} | {pdfs_path} | {e}")
                continue

            # 只用两者共同可匹配的 slice 数
            n_common = min(n_pd, n_pdfs)
            if n_common <= 0:
                continue

            for i in range(n_common):
                self.examples.append((pair_idx, i, n_pd, n_pdfs))

        if len(self.examples) == 0:
            raise RuntimeError("No valid paired examples found.")

    def __len__(self):
        return len(self.examples)

    def _map_slice_idx(self, i, n_src):
        """
        Map common index i to actual slice index in a scan with n_src slices.
        """
        if self.match_mode == "middle":
            return n_src // 2

        # default: relative index mapping
        n_common = None  # not directly needed here since i already built from min(n_pd, n_pdfs)
        # Since i ranges in [0, min(n_pd, n_pdfs)-1], we map proportionally
        # using the current scan length.
        # If the scan lengths are equal, this reduces to identity.
        if len(self.examples) == 0:
            return i

        # simpler proportional fallback
        return i

    def __getitem__(self, idx):
        pair_idx, i, n_pd, n_pdfs = self.examples[idx]
        pair = self.pairs[pair_idx]

        pd_path = pair["pd_path"]
        pdfs_path = pair["pdfs_path"]

        # 当前实现：先用相同 i；如果后面你想更严格对齐，可再升级成比例映射
        pd_slice = i
        pdfs_slice = i

        with h5py.File(pd_path, "r") as hf_pd, h5py.File(pdfs_path, "r") as hf_pdfs:
            pd_kspace_np = hf_pd["kspace"][pd_slice]
            pd_target_np = hf_pd["reconstruction_esc"][pd_slice]

            pdfs_kspace_np = hf_pdfs["kspace"][pdfs_slice]
            pdfs_target_np = hf_pdfs["reconstruction_esc"][pdfs_slice]

        pd_kspace = torch.from_numpy(pd_kspace_np)
        pdfs_kspace = torch.from_numpy(pdfs_kspace_np)

        # 为了让两种对比使用同一个 mask，mask 维度必须一致
        # 如果宽度不一样，先各自做自己的 mask；如果你后面要更严格的 physics-consistency，
        # 可以再做 padding / custom collate
        same_shape = tuple(pd_kspace.shape) == tuple(pdfs_kspace.shape)

        if self.use_seed:
            seed = idx
        else:
            seed = None

        if same_shape:
            shared_mask = random_cartesian_mask(
                pd_kspace.shape,
                center_fraction=self.center_fraction,
                acceleration=self.acceleration,
                seed=seed,
            )
            pd_mask = shared_mask
            pdfs_mask = shared_mask
        else:
            pd_mask = random_cartesian_mask(
                pd_kspace.shape,
                center_fraction=self.center_fraction,
                acceleration=self.acceleration,
                seed=seed,
            )
            pdfs_mask = random_cartesian_mask(
                pdfs_kspace.shape,
                center_fraction=self.center_fraction,
                acceleration=self.acceleration,
                seed=seed,
            )

        pd_masked_kspace = pd_kspace * pd_mask
        pdfs_masked_kspace = pdfs_kspace * pdfs_mask

        pd_zf = ifft2c(pd_masked_kspace)
        pdfs_zf = ifft2c(pdfs_masked_kspace)

        pd_zf_mag = complex_abs(pd_zf)
        pdfs_zf_mag = complex_abs(pdfs_zf)

        pd_zf_mag = center_crop(pd_zf_mag, 320, 320)
        pdfs_zf_mag = center_crop(pdfs_zf_mag, 320, 320)

        pd_target = torch.from_numpy(pd_target_np).float()
        pdfs_target = torch.from_numpy(pdfs_target_np).float()

        # 各自独立 normalize
        pd_zf_mag, pd_mean, pd_std = normalize_instance(pd_zf_mag)
        pd_target = (pd_target - pd_mean) / (pd_std + 1e-8)

        pdfs_zf_mag, pdfs_mean, pdfs_std = normalize_instance(pdfs_zf_mag)
        pdfs_target = (pdfs_target - pdfs_mean) / (pdfs_std + 1e-8)

        # 2-channel input / target
        inp = torch.stack([pd_zf_mag, pdfs_zf_mag], dim=0)       # (2, 320, 320)
        target = torch.stack([pd_target, pdfs_target], dim=0)    # (2, 320, 320)

        return {
            "input": inp,
            "target": target,
            "patient_id": pair["patient_id"],
            "pd_path": pd_path,
            "pdfs_path": pdfs_path,
            "pd_slice": pd_slice,
            "pdfs_slice": pdfs_slice,
        }