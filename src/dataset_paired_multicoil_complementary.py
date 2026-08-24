from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .fft_utils import ifft2c, center_crop, rss_combine
from .masks import (
    stable_mask_seed,
    make_gaussian_vd_mask,
    apply_1d_mask_to_kspace,
)


class PairedMulticoilDataset(Dataset):
    """
    Paired PD / PD-FS multicoil dataset for controlled experiments.

    Each item corresponds to:
        same patient
        same slice index
        same acceleration factor
        same Gaussian VD mask

    It returns both masked multicoil k-space for future VarNet use and
    zero-filled RSS images for the Step-5 U-Net overfitting experiment.
    """

    def __init__(
        self,
        metadata_csv,
        split,
        acceleration=4,
        patient_ids=None,
        slices_per_patient=None,
        edge_weight=0.25,
        target_key="reconstruction_rss",
        crop_size=(320, 320),
        pd_mask_type="equispaced",
        pdfs_mask_type="gaussian_vd",
    ):
        self.metadata_csv = Path(metadata_csv)
        self.split = split
        self.acceleration = acceleration
        self.edge_weight = float(edge_weight)
        self.target_key = target_key
        self.crop_size = tuple(crop_size)
        self.pd_mask_type = str(pd_mask_type)
        self.pdfs_mask_type = str(pdfs_mask_type)

        valid_mask_types = {"gaussian_vd", "gaussian", "equispaced"}
        if self.pd_mask_type not in valid_mask_types:
            raise ValueError(
                f"Unknown pd_mask_type={self.pd_mask_type}. "
                f"Expected one of {sorted(valid_mask_types)}."
            )
        if self.pdfs_mask_type not in valid_mask_types:
            raise ValueError(
                f"Unknown pdfs_mask_type={self.pdfs_mask_type}. "
                f"Expected one of {sorted(valid_mask_types)}."
            )

        if split not in {"train", "val", "test"}:
            raise ValueError(
                f"split must be train, val or test; got {split}."
            )

        if acceleration not in {4, 6, 8}:
            raise ValueError(
                f"acceleration must be 4, 6 or 8; got {acceleration}."
            )

        if not self.metadata_csv.exists():
            raise FileNotFoundError(
                f"Metadata CSV not found: {self.metadata_csv}"
            )

        df = pd.read_csv(self.metadata_csv)

        required_columns = {
            "split",
            "patient_id",
            "pd_new_path",
            "pdfs_new_path",
        }
        missing = sorted(required_columns - set(df.columns))

        if missing:
            raise ValueError(
                f"Missing required metadata columns: {missing}"
            )

        df = df[df["split"] == split].copy()

        if patient_ids is not None:
            patient_ids = {str(x) for x in patient_ids}
            df = df[
                df["patient_id"].astype(str).isin(patient_ids)
            ].copy()

        if len(df) == 0:
            raise RuntimeError(
                f"No paired patients found for split={split}."
            )

        self.records = []
        self.patient_rows = []

        for _, row in df.iterrows():
            patient_id = str(row["patient_id"])
            pd_path = Path(row["pd_new_path"])
            pdfs_path = Path(row["pdfs_new_path"])

            if not pd_path.exists():
                raise FileNotFoundError(
                    f"PD file not found: {pd_path}"
                )

            if not pdfs_path.exists():
                raise FileNotFoundError(
                    f"PD-FS file not found: {pdfs_path}"
                )

            with h5py.File(pd_path, "r") as pd_hf, \
                 h5py.File(pdfs_path, "r") as pdfs_hf:

                for key in ("kspace", target_key):
                    if key not in pd_hf:
                        raise KeyError(
                            f"{key} missing from PD file {pd_path}"
                        )
                    if key not in pdfs_hf:
                        raise KeyError(
                            f"{key} missing from PD-FS file {pdfs_path}"
                        )

                n_pd = int(pd_hf["kspace"].shape[0])
                n_pdfs = int(pdfs_hf["kspace"].shape[0])

            n_common = min(n_pd, n_pdfs)

            if n_common <= 0:
                continue

            selected_slices = self._choose_slices(
                n_common,
                slices_per_patient,
            )

            self.patient_rows.append(
                {
                    "patient_id": patient_id,
                    "pd_path": str(pd_path),
                    "pdfs_path": str(pdfs_path),
                    "n_pd": n_pd,
                    "n_pdfs": n_pdfs,
                    "n_common": n_common,
                    "selected_slices": selected_slices,
                }
            )

            for slice_idx in selected_slices:
                self.records.append(
                    {
                        "patient_id": patient_id,
                        "pd_path": pd_path,
                        "pdfs_path": pdfs_path,
                        "slice_idx": int(slice_idx),
                        "num_slices": int(n_common),
                    }
                )

        if len(self.records) == 0:
            raise RuntimeError(
                "No valid paired slice records were created."
            )

    @staticmethod
    def _choose_slices(n_slices, slices_per_patient):
        if slices_per_patient is None:
            return list(range(n_slices))

        if slices_per_patient < 2:
            raise ValueError(
                "slices_per_patient must be at least 2 so that "
                "edge and central anatomy can both be included."
            )

        candidate_indices = [
            0,
            1,
            n_slices // 4,
            n_slices // 3,
            max(0, n_slices // 2 - 1),
            n_slices // 2,
            min(n_slices - 1, n_slices // 2 + 1),
            (2 * n_slices) // 3,
            (3 * n_slices) // 4,
            n_slices - 1,
        ]

        selected = []
        for idx in candidate_indices:
            idx = max(0, min(n_slices - 1, int(idx)))
            if idx not in selected:
                selected.append(idx)

        if len(selected) < slices_per_patient:
            evenly_spaced = torch.linspace(
                0,
                n_slices - 1,
                steps=slices_per_patient,
            ).round().int().tolist()

            for idx in evenly_spaced:
                if idx not in selected:
                    selected.append(idx)

        selected = selected[:slices_per_patient]
        return sorted(selected)

    @staticmethod
    def _make_equispaced_mask(
        num_cols,
        acceleration,
        seed=None,
    ):
        """Deterministic 1D Cartesian equispaced mask with fully sampled ACS."""
        if acceleration == 4:
            center_fraction = 0.08
        elif acceleration == 6:
            center_fraction = 0.06
        elif acceleration == 8:
            center_fraction = 0.04
        else:
            raise ValueError(
                f"Unsupported acceleration for equispaced mask: {acceleration}"
            )

        num_low_freqs = int(round(num_cols * center_fraction))
        num_low_freqs = max(4, min(num_low_freqs, num_cols))

        target_num_samples = int(round(num_cols / acceleration))
        target_num_samples = max(num_low_freqs, min(target_num_samples, num_cols))

        mask = np.zeros(num_cols, dtype=np.float32)
        center_start = (num_cols - num_low_freqs) // 2
        center_end = center_start + num_low_freqs
        mask[center_start:center_end] = 1.0

        num_outer = target_num_samples - num_low_freqs
        if num_outer > 0:
            outer_indices = np.concatenate(
                [
                    np.arange(0, center_start),
                    np.arange(center_end, num_cols),
                ]
            )
            if len(outer_indices) > 0:
                if num_outer >= len(outer_indices):
                    selected_outer = outer_indices
                else:
                    positions = np.linspace(
                        0,
                        len(outer_indices) - 1,
                        num=num_outer,
                    )
                    selected_outer = outer_indices[np.round(positions).astype(int)]
                mask[selected_outer] = 1.0

        num_sampled_lines = int(mask.sum())
        actual_R = float(num_cols) / float(num_sampled_lines)
        return mask, num_sampled_lines, actual_R

    @staticmethod
    def _make_mask_by_type(mask_type, num_cols, acceleration, seed):
        if mask_type in {"gaussian_vd", "gaussian"}:
            return make_gaussian_vd_mask(
                num_cols=num_cols,
                acceleration=acceleration,
                seed=seed,
            )
        if mask_type == "equispaced":
            return PairedMulticoilDataset._make_equispaced_mask(
                num_cols=num_cols,
                acceleration=acceleration,
                seed=seed,
            )
        raise ValueError(f"Unknown mask_type: {mask_type}")


    def __len__(self):
        return len(self.records)

    def _load_contrast(self, h5_path, slice_idx):
        with h5py.File(h5_path, "r") as hf:
            kspace_np = hf["kspace"][slice_idx]
            target_np = hf[self.target_key][slice_idx]

        kspace = torch.from_numpy(kspace_np)
        target = torch.from_numpy(target_np).float()

        return kspace, target

    def _zero_filled_rss(
        self,
        masked_kspace,
        target_shape,
    ):
        coil_images = ifft2c(masked_kspace)
        rss_image = rss_combine(
            coil_images,
            coil_dim=0,
        )

        rss_image = center_crop(
            rss_image,
            crop_h=target_shape[-2],
            crop_w=target_shape[-1],
        )

        return rss_image.float()

    @staticmethod
    def _normalise_input_and_target(
        input_image,
        target,
        eps=1e-8,
    ):
        mean = input_image.mean()
        std = input_image.std()

        input_normalised = (
            input_image - mean
        ) / (
            std + eps
        )

        target_normalised = (
            target - mean
        ) / (
            std + eps
        )

        return (
            input_normalised,
            target_normalised,
            mean,
            std,
        )

    def __getitem__(self, index):
        record = self.records[index]

        patient_id = record["patient_id"]
        slice_idx = record["slice_idx"]
        num_slices = record["num_slices"]

        pd_kspace, pd_target = self._load_contrast(
            record["pd_path"],
            slice_idx,
        )
        pdfs_kspace, pdfs_target = self._load_contrast(
            record["pdfs_path"],
            slice_idx,
        )

        if pd_kspace.shape[-2:] != pdfs_kspace.shape[-2:]:
            raise ValueError(
                "PD and PD-FS k-space spatial dimensions differ for "
                f"patient={patient_id}, slice={slice_idx}: "
                f"PD={tuple(pd_kspace.shape)}, "
                f"PD-FS={tuple(pdfs_kspace.shape)}."
            )

        base_seed = stable_mask_seed(
            patient_id=patient_id,
            slice_idx=slice_idx,
            acceleration=self.acceleration,
        )

        pd_seed = base_seed + 17
        pdfs_seed = base_seed + 31

        pd_mask_np, pd_num_sampled_lines, pd_actual_R = \
            self._make_mask_by_type(
                mask_type=self.pd_mask_type,
                num_cols=pd_kspace.shape[-1],
                acceleration=self.acceleration,
                seed=pd_seed,
            )

        pdfs_mask_np, pdfs_num_sampled_lines, pdfs_actual_R = \
            self._make_mask_by_type(
                mask_type=self.pdfs_mask_type,
                num_cols=pdfs_kspace.shape[-1],
                acceleration=self.acceleration,
                seed=pdfs_seed,
            )

        pd_masked_kspace = apply_1d_mask_to_kspace(
            pd_kspace,
            pd_mask_np,
        )
        pdfs_masked_kspace = apply_1d_mask_to_kspace(
            pdfs_kspace,
            pdfs_mask_np,
        )

        pd_zf = self._zero_filled_rss(
            pd_masked_kspace,
            pd_target.shape,
        )
        pdfs_zf = self._zero_filled_rss(
            pdfs_masked_kspace,
            pdfs_target.shape,
        )

        (
            pd_input,
            pd_target_normalised,
            pd_mean,
            pd_std,
        ) = self._normalise_input_and_target(
            pd_zf,
            pd_target,
        )

        (
            pdfs_input,
            pdfs_target_normalised,
            pdfs_mean,
            pdfs_std,
        ) = self._normalise_input_and_target(
            pdfs_zf,
            pdfs_target,
        )

        is_edge = slice_idx == 0
        sample_weight = (
            self.edge_weight if is_edge else 1.0
        )

        pd_mask = torch.from_numpy(pd_mask_np).float()
        pdfs_mask = torch.from_numpy(pdfs_mask_np).float()

        return {
            "pd_input": pd_input.unsqueeze(0),
            "pdfs_input": pdfs_input.unsqueeze(0),

            "pd_target": pd_target_normalised.unsqueeze(0),
            "pdfs_target": pdfs_target_normalised.unsqueeze(0),

            "pd_target_raw": pd_target.unsqueeze(0),
            "pdfs_target_raw": pdfs_target.unsqueeze(0),

            "pd_masked_kspace": pd_masked_kspace,
            "pdfs_masked_kspace": pdfs_masked_kspace,
            # Compatibility: existing joint code usually reads batch["mask"].
            # For the complementary setting, this should be the PD-FS target mask.
            "mask": pdfs_mask,
            "pd_mask": pd_mask,
            "pdfs_mask": pdfs_mask,

            "patient_id": patient_id,
            "slice_idx": slice_idx,
            "num_slices": num_slices,
            "is_edge": is_edge,
            "sample_weight": torch.tensor(
                sample_weight,
                dtype=torch.float32,
            ),

            "acceleration": self.acceleration,
            # Compatibility fields refer to the PD-FS target branch.
            "num_sampled_lines": pdfs_num_sampled_lines,
            "actual_R": torch.tensor(
                pdfs_actual_R,
                dtype=torch.float32,
            ),

            "pd_num_sampled_lines": pd_num_sampled_lines,
            "pdfs_num_sampled_lines": pdfs_num_sampled_lines,
            "pd_actual_R": torch.tensor(
                pd_actual_R,
                dtype=torch.float32,
            ),
            "pdfs_actual_R": torch.tensor(
                pdfs_actual_R,
                dtype=torch.float32,
            ),
            "pd_mask_type": self.pd_mask_type,
            "pdfs_mask_type": self.pdfs_mask_type,

            "pd_mean": pd_mean.float(),
            "pd_std": pd_std.float(),
            "pdfs_mean": pdfs_mean.float(),
            "pdfs_std": pdfs_std.float(),

            "pd_path": str(record["pd_path"]),
            "pdfs_path": str(record["pdfs_path"]),
        }
