from __future__ import annotations

"""Shared audited utilities for QMax Stage-A scripts."""

import csv
import hashlib
import json
import math
import os
import platform
import random
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import h5py
import numpy as np
import torch
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader

from src.dataset_paired_multicoil_aux_pd_r2 import (
    PairedMulticoilAuxPDToPDFSDataset,
)
from src.fft_utils import center_crop
from src.m2_prnf_qmax_varnet import QMaxAuxPDVarNet


class IndexedDataset:
    def __init__(self, dataset: Any):
        self.dataset = dataset
        self.records = dataset.records
        self.patient_rows = dataset.patient_rows

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.dataset[index]
        item["sample_idx"] = int(index)
        return item


class ShapeBucketBatchSampler:
    """Epoch-addressable sampler; order does not depend on model RNG usage."""

    def __init__(
        self,
        dataset: Any,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        buckets: Dict[tuple, List[int]] = defaultdict(list)
        for index, record in enumerate(dataset.records):
            with h5py.File(record["pdfs_path"], "r") as handle:
                shape = tuple(
                    int(value) for value in handle["kspace"].shape[1:]
                )
            buckets[shape].append(index)
        self.buckets = dict(buckets)
        self._length = sum(
            math.ceil(len(indices) / self.batch_size)
            for indices in self.buckets.values()
        )

    def __len__(self) -> int:
        return self._length

    def __iter__(self):
        rng = random.Random(self.seed + int(self.epoch))
        batches: List[List[int]] = []
        for indices_value in self.buckets.values():
            indices = list(indices_value)
            if self.shuffle:
                rng.shuffle(indices)
            batches.extend(
                indices[start : start + self.batch_size]
                for start in range(0, len(indices), self.batch_size)
            )
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_dataset(
    metadata_csv: str,
    split: str,
    acceleration: int = 8,
    pd_aux_acceleration: int = 2,
    patient_ids: Optional[Sequence[str]] = None,
) -> PairedMulticoilAuxPDToPDFSDataset:
    return PairedMulticoilAuxPDToPDFSDataset(
        metadata_csv=metadata_csv,
        split=split,
        pdfs_acceleration=int(acceleration),
        pd_aux_acceleration=int(pd_aux_acceleration),
        patient_ids=patient_ids,
        slices_per_patient=None,
        edge_weight=1.0,
    )


def select_patient_ids(dataset: Any, limit: Optional[int]) -> List[str]:
    values = list(
        dict.fromkeys(str(row["patient_id"]) for row in dataset.patient_rows)
    )
    if limit is None:
        return values
    if int(limit) < 1 or int(limit) > len(values):
        raise ValueError(f"Invalid patient limit {limit}; available={len(values)}")
    return values[: int(limit)]


def prepare_batch(
    batch: Mapping[str, Any],
    device: torch.device,
):
    kspace = batch["pdfs_masked_kspace"].to(device, non_blocking=True)
    if not torch.is_complex(kspace):
        raise TypeError(f"Expected complex k-space, got {kspace.dtype}")
    kspace = torch.view_as_real(kspace).float()
    mask = batch["mask"].to(device, non_blocking=True).bool()
    if mask.ndim == 2:
        mask = mask[:, None, None, :, None]
    elif mask.ndim == 1:
        mask = mask[None, None, None, :, None]
    elif mask.ndim != 5:
        raise RuntimeError(f"Unexpected mask shape {tuple(mask.shape)}")
    pd = batch["pd_aux_image"].to(device, non_blocking=True).float()
    target = batch["pdfs_target_raw"].to(device, non_blocking=True).float()
    if pd.ndim == 4:
        pd = pd[:, 0]
    if target.ndim == 4:
        target = target[:, 0]
    indices = [int(value) for value in batch["sample_idx"]]
    return kspace, mask, pd, target, indices


def l1_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    scale = target.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    return torch.abs(prediction / scale - target / scale).mean(
        dim=(-2, -1)
    )


def slice_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    scale = float(target.max().clamp_min(1e-8).item())
    pred = prediction.detach().float().cpu().numpy() / scale
    truth = target.detach().float().cpu().numpy() / scale
    difference = pred - truth
    mse = float(np.mean(difference**2))
    return {
        "l1": float(np.mean(np.abs(difference))),
        "nmse": float(
            np.sum(difference**2) / max(np.sum(truth**2), 1e-12)
        ),
        "psnr": float(-10.0 * math.log10(max(mse, 1e-12))),
        "ssim": float(
            structural_similarity(truth, pred, data_range=1.0)
        ),
    }


def safe_mean(values: Iterable[float]) -> float:
    selected = [
        float(value)
        for value in values
        if math.isfinite(float(value))
    ]
    return float(np.mean(selected)) if selected else float("nan")


def patient_macro(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_patient: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_patient[str(row["patient_id"])].append(row)
    patient_rows = []
    for patient_id, values in sorted(by_patient.items()):
        patient_rows.append(
            {
                "patient_id": patient_id,
                "num_slices": len(values),
                **{
                    metric: safe_mean(row[metric] for row in values)
                    for metric in ("l1", "nmse", "psnr", "ssim")
                },
            }
        )
    return {
        "patient_rows": patient_rows,
        "num_patients": len(patient_rows),
        "num_slices": len(rows),
        **{
            f"patient_{metric}": safe_mean(
                row[metric] for row in patient_rows
            )
            for metric in ("l1", "nmse", "psnr", "ssim")
        },
    }


def autocast_context(device: torch.device, enabled: bool):
    if device.type != "cuda" or not enabled:
        return nullcontext()
    return torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=True
    )


def make_grad_scaler(enabled: bool):
    scaler_enabled = bool(enabled and torch.cuda.is_available())
    # PyTorch >=2.4 moved GradScaler to torch.amp. Keep the fallback only
    # for older fastMRI environments.
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(
            "cuda",
            enabled=scaler_enabled,
        )
    return torch.cuda.amp.GradScaler(enabled=scaler_enabled)


def capture_rng_state() -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else []
        ),
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(state) != required:
        raise RuntimeError(f"RNG-state keys mismatch: {sorted(state)}")

    def cpu_byte_state(value: Any, label: str) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(
                f"{label} must be a torch.Tensor, got {type(value)!r}"
            )
        if value.dtype != torch.uint8:
            raise RuntimeError(
                f"{label} must have dtype torch.uint8, got {value.dtype}"
            )
        return value.detach().to(device="cpu").contiguous()

    cuda_state = state["torch_cuda"]
    if not isinstance(cuda_state, (list, tuple)):
        raise RuntimeError(
            "torch_cuda RNG state must be a list or tuple of tensors"
        )

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(cpu_byte_state(state["torch_cpu"], "torch_cpu"))
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [
                cpu_byte_state(value, f"torch_cuda[{index}]")
                for index, value in enumerate(cuda_state)
            ]
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_hashes(project_root: Path) -> Dict[str, str]:
    paths = (
        "src/m2_prnf_varnet.py",
        "src/m2_prnf_corruptions.py",
        "src/m2_prnf_fusion_pilot_varnet.py",
        "src/m2_prnf_qmax_varnet.py",
        "src/qmax_deterministic_corruptions.py",
        "src/dataset_paired_multicoil_aux_pd_r2.py",
        "src/fft_utils.py",
        "src/masks.py",
        "scripts/generate_qmax_random_init.py",
        "scripts/qmax_common.py",
        "scripts/train_qmax_stage_a.py",
        "scripts/preflight_qmax_stage_a.py",
        "scripts/evaluate_qmax_counterfactuals.py",
        "scripts/compare_qmax_stage_a.py",
        "QMAX_STAGE_A_PROTOCOL_R8.json",
    )
    output = {}
    for relative in paths:
        path = project_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Locked QMax dependency missing: {path}")
        output[relative] = sha256_file(path)
    return output


def runtime_versions() -> Dict[str, Any]:
    import fastmri
    import scipy
    import skimage

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "numpy": np.__version__,
        "h5py": h5py.__version__,
        "scikit_image": skimage.__version__,
        "scipy": scipy.__version__,
        "fastmri_version": getattr(fastmri, "__version__", None),
        "fastmri_module": str(Path(fastmri.__file__).resolve()),
    }


def canonical_json(value: Any) -> Any:
    return json.loads(
        json.dumps(value, sort_keys=True, allow_nan=False)
    )


def validate_resume_config(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    immutable_keys: Sequence[str],
) -> None:
    mismatches = {}
    for key in immutable_keys:
        old = canonical_json(previous.get(key))
        new = canonical_json(current.get(key))
        if old != new:
            mismatches[key] = {"checkpoint": old, "current": new}
    if mismatches:
        raise RuntimeError(
            "Resume configuration mismatch:\n"
            + json.dumps(mismatches, indent=2)
        )


def atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def save_checkpoint(
    *,
    path: Path,
    model: QMaxAuxPDVarNet,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    epoch: int,
    best_epoch: int,
    best_val: float,
    config: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    corruption_audit: Mapping[str, Any],
) -> None:
    atomic_torch_save(
        {
            "epoch": int(epoch),
            "best_epoch": int(best_epoch),
            "best_val": float(best_val),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "grad_scaler_state_dict": scaler.state_dict(),
            "config": dict(config),
            "history": list(history),
            "rng_state": capture_rng_state(),
            "sampler_next_epoch": int(epoch),
            "code_hashes": config["code_hashes"],
            "run_corruption_audit": dict(corruption_audit),
        },
        path,
    )


class DiagnosticAccumulator:
    METRICS = (
        "q_hat",
        "alpha",
        "direct_to_target_rms",
        "detail_gate_mean",
        "detail_gate_std",
        "detail_gate_min",
        "detail_gate_max",
        "alignment_to_target_rms",
        "correction_to_target_rms",
        "final_auxiliary_to_target_rms",
        "cos_direct_correction",
        "dc_raw_rms",
        "dc_normalized_rms",
    )

    def __init__(self):
        self.sums: Dict[tuple, float] = defaultdict(float)
        self.counts: Dict[tuple, int] = defaultdict(int)

    def add(
        self,
        auxiliary: Mapping[str, torch.Tensor],
        sample_slice: slice,
        condition: str,
    ) -> None:
        for metric in self.METRICS:
            tensor = auxiliary[metric][sample_slice].detach().float().cpu()
            if tensor.ndim != 3:
                raise RuntimeError(
                    f"Expected [B,cascade,scale] diagnostic, "
                    f"got {metric}={tuple(tensor.shape)}"
                )
            for cascade in range(tensor.shape[1]):
                for scale in range(tensor.shape[2]):
                    values = tensor[:, cascade, scale]
                    key = (condition, cascade, scale, metric)
                    self.sums[key] += float(values.sum().item())
                    self.counts[key] += int(values.numel())

    def rows(self, epoch: int, scale_names: Sequence[str]):
        rows = []
        conditions = sorted({key[0] for key in self.sums})
        cascades = sorted({key[1] for key in self.sums})
        scales = sorted({key[2] for key in self.sums})
        for condition in conditions:
            for cascade in cascades:
                for scale in scales:
                    row: Dict[str, Any] = {
                        "epoch": int(epoch),
                        "condition": condition,
                        "cascade": int(cascade),
                        "scale": int(scale),
                        "scale_name": str(scale_names[scale]),
                    }
                    for metric in self.METRICS:
                        key = (condition, cascade, scale, metric)
                        count = self.counts.get(key, 0)
                        row[metric] = (
                            self.sums[key] / count
                            if count
                            else float("nan")
                        )
                    rows.append(row)
        return rows


class CorruptionAudit:
    def __init__(self):
        self.condition_counts: Dict[str, int] = defaultdict(int)
        self.fallback_counts: Dict[str, int] = defaultdict(int)
        self.missing_count = 0

    def add(self, records: Sequence[Mapping[str, Any]]) -> None:
        for record in records:
            self.condition_counts[str(record.get("condition_key"))] += 1
            fallback = record.get("fallback_from")
            if fallback:
                self.fallback_counts[str(fallback)] += 1
            self.missing_count += int(record.get("missing_mask", 0))

    def state_dict(self) -> Dict[str, Any]:
        return {
            "condition_counts": dict(self.condition_counts),
            "fallback_counts": dict(self.fallback_counts),
            "missing_count": int(self.missing_count),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.condition_counts.update(state.get("condition_counts", {}))
        self.fallback_counts.update(state.get("fallback_counts", {}))
        self.missing_count = int(state.get("missing_count", 0))

    def summary(self) -> Dict[str, Any]:
        state = self.state_dict()
        state["total"] = int(sum(self.condition_counts.values()))
        return state


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
