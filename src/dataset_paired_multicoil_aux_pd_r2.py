from pathlib import Path

import h5py
import pandas as pd
import torch
from torch.utils.data import Dataset

from .fft_utils import ifft2c, center_crop, rss_combine
from .masks import (
    stable_mask_seed,
    make_gaussian_vd_mask,
    apply_1d_mask_to_kspace,
)


class PairedMulticoilAuxPDToPDFSDataset(Dataset):
    """
    Paired PD / PD-FS multicoil dataset for auxiliary PD-to-PD-FS experiments.

    Each item corresponds to:
        same patient
        same slice index
        PD-FS target acceleration and separate PD auxiliary acceleration
        separate Gaussian VD masks for PD-FS target and PD auxiliary

    It returns both masked multicoil k-space for future VarNet use and
    zero-filled RSS images for the Step-5 U-Net overfitting experiment.
    """

    def __init__(
        self,
        metadata_csv,
        split,
        pdfs_acceleration=8,
        pd_aux_acceleration=2,
        patient_ids=None,
        slices_per_patient=None,
        edge_weight=0.25,
        target_key="reconstruction_rss",
        crop_size=(320, 320),
    ):
        self.metadata_csv = Path(metadata_csv)
        self.split = split
        self.pdfs_acceleration = int(pdfs_acceleration)
        self.pd_aux_acceleration = int(pd_aux_acceleration)
        self.edge_weight = float(edge_weight)
        self.target_key = target_key
        self.crop_size = tuple(crop_size)

        if split not in {"train", "val", "test"}:
            raise ValueError(
                f"split must be train, val or test; got {split}."
            )

        if self.pdfs_acceleration not in {4, 6, 8}:
            raise ValueError(
                f"pdfs_acceleration must be 4, 6 or 8; got {self.pdfs_acceleration}."
            )

        if self.pd_aux_acceleration not in {2, 4, 6, 8}:
            raise ValueError(
                f"pd_aux_acceleration must be 2, 4, 6 or 8; got {self.pd_aux_acceleration}."
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
            pd_flip_lr = self._parse_bool(
                row.get("pd_flip_lr", False)
            )

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
                    "pd_flip_lr": pd_flip_lr,
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
                        "pd_flip_lr": pd_flip_lr,
                    }
                )

        if len(self.records) == 0:
            raise RuntimeError(
                "No valid paired slice records were created."
            )

    @staticmethod
    def _parse_bool(value) -> bool:
        """Parse metadata boolean values safely, including CSV strings."""
        if isinstance(value, bool):
            return value

        if value is None or pd.isna(value):
            return False

        return str(value).strip().lower() in {
            "true",
            "1",
            "yes",
            "y",
            "t",
        }

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

        seed = stable_mask_seed(
            patient_id=patient_id,
            slice_idx=slice_idx,
            acceleration=self.pdfs_acceleration,
        )

        pdfs_mask_np, pdfs_num_sampled_lines, pdfs_actual_R = \
            make_gaussian_vd_mask(
                num_cols=pdfs_kspace.shape[-1],
                acceleration=self.pdfs_acceleration,
                seed=seed,
            )

        pd_seed = stable_mask_seed(
            patient_id=patient_id,
            slice_idx=slice_idx,
            acceleration=self.pd_aux_acceleration,
        )

        pd_aux_mask_np, pd_aux_num_sampled_lines, pd_aux_actual_R = \
            make_gaussian_vd_mask(
                num_cols=pd_kspace.shape[-1],
                acceleration=self.pd_aux_acceleration,
                seed=pd_seed,
            )

        pd_masked_kspace = apply_1d_mask_to_kspace(
            pd_kspace,
            pd_aux_mask_np,
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

        mask = torch.from_numpy(pdfs_mask_np).float()
        pd_aux_mask = torch.from_numpy(pd_aux_mask_np).float()

        # For the clinical main setting, the auxiliary PD input is not the full target.
        # It is a zero-filled RSS image reconstructed from lightly undersampled PD k-space.
        pd_aux_image = pd_zf.float()

        # Apply a metadata-driven left-right orientation correction only
        # to the image-domain PD auxiliary input. The original H5 data,
        # PD-FS target, and PD-FS data-consistency pathway are unchanged.
        if record.get("pd_flip_lr", False):
            pd_aux_image = torch.flip(
                pd_aux_image,
                dims=[-1],
            )

        return {
            "pd_input": pd_input.unsqueeze(0),
            "pdfs_input": pdfs_input.unsqueeze(0),

            "pd_target": pd_target_normalised.unsqueeze(0),
            "pdfs_target": pdfs_target_normalised.unsqueeze(0),

            "pd_target_raw": pd_target.unsqueeze(0),
            "pdfs_target_raw": pdfs_target.unsqueeze(0),
            "pd_aux_image": pd_aux_image.unsqueeze(0),

            "pd_masked_kspace": pd_masked_kspace,
            "pdfs_masked_kspace": pdfs_masked_kspace,
            "mask": mask,
            "pd_aux_mask": pd_aux_mask,

            "patient_id": patient_id,
            "slice_idx": slice_idx,
            "num_slices": num_slices,
            "pd_flip_lr": bool(
                record.get("pd_flip_lr", False)
            ),
            "is_edge": is_edge,
            "sample_weight": torch.tensor(
                sample_weight,
                dtype=torch.float32,
            ),

            "acceleration": self.pdfs_acceleration,
            "pdfs_acceleration": self.pdfs_acceleration,
            "pd_aux_acceleration": self.pd_aux_acceleration,
            "num_sampled_lines": pdfs_num_sampled_lines,
            "pdfs_num_sampled_lines": pdfs_num_sampled_lines,
            "pd_aux_num_sampled_lines": pd_aux_num_sampled_lines,
            "actual_R": torch.tensor(
                pdfs_actual_R,
                dtype=torch.float32,
            ),
            "pdfs_actual_R": torch.tensor(
                pdfs_actual_R,
                dtype=torch.float32,
            ),
            "pd_aux_actual_R": torch.tensor(
                pd_aux_actual_R,
                dtype=torch.float32,
            ),

            "pd_mean": pd_mean.float(),
            "pd_std": pd_std.float(),
            "pdfs_mean": pdfs_mean.float(),
            "pdfs_std": pdfs_std.float(),

            "pd_path": str(record["pd_path"]),
            "pdfs_path": str(record["pdfs_path"]),
        }
