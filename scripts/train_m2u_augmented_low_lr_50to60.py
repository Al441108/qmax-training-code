#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.m2_prnf_corruptions import (  # noqa: E402
    CorruptionConfig,
    HardNegativeSampler,
    paired_discrimination_loss,
)
from src.dataset_paired_multicoil_aux_pd_r2 import (  # noqa: E402
    PairedMulticoilAuxPDToPDFSDataset,
)
from src.fft_utils import center_crop  # noqa: E402
from src.m2_prnf_corruptions import CORRUPT_MIXTURE, corrupt_batch_prnf  # noqa: E402
from src.m2_prnf_varnet import M2PRNFAuxPDVarNet, VALID_VARIANTS  # noqa: E402


class IndexedDataset:
    def __init__(self, dataset):
        self.dataset = dataset
        self.records = dataset.records
        self.patient_rows = dataset.patient_rows

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        item = self.dataset[index]
        item["sample_idx"] = int(index)
        return item


class ShapeBucketBatchSampler:
    def __init__(self, dataset, batch_size: int, shuffle: bool, seed: int):
        self.dataset, self.batch_size = dataset, int(batch_size)
        self.shuffle, self.seed, self.epoch = bool(shuffle), int(seed), 0
        buckets: Dict[tuple, List[int]] = defaultdict(list)
        for index, record in enumerate(dataset.records):
            with h5py.File(record["pdfs_path"], "r") as hf:
                shape = tuple(int(value) for value in hf["kspace"].shape[1:])
            buckets[shape].append(index)
        self.buckets = dict(buckets)
        self._length = sum(
            math.ceil(len(indices) / self.batch_size)
            for indices in self.buckets.values()
        )

    def __len__(self):
        return self._length

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        batches = []
        for indices in self.buckets.values():
            indices = list(indices)
            if self.shuffle:
                rng.shuffle(indices)
            batches.extend(
                indices[start : start + self.batch_size]
                for start in range(0, len(indices), self.batch_size)
            )
        if self.shuffle:
            rng.shuffle(batches)
        self.epoch += 1
        yield from batches


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_batch(batch: Dict[str, Any], device: torch.device):
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


def l1_per_sample(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    scale = target.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    return torch.abs(prediction / scale - target / scale).mean(dim=(-2, -1))


def safe_mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(values)) if values else float("nan")


class CorruptionAudit:
    def __init__(self):
        self.condition_counts: Dict[str, int] = defaultdict(int)
        self.fallback_counts: Dict[str, int] = defaultdict(int)
        self.padding_target: Dict[str, List[float]] = defaultdict(list)
        self.direction_counts: Dict[str, int] = defaultdict(int)
        self.shift_counts: Dict[str, int] = defaultdict(int)
        self.wrong_slice_delta_z: List[float] = []
        self.wrong_patient_replacements: Dict[int, set] = defaultdict(set)
        self.wrong_patient_replacement_patients: Dict[int, set] = defaultdict(set)

    def add(self, records, targets: torch.Tensor) -> None:
        targets = targets.detach().cpu()
        for index, record in enumerate(records):
            condition = str(record.get("condition_key", record.get("condition")))
            self.condition_counts[condition] += 1
            if record.get("fallback_from"):
                self.fallback_counts[str(record["fallback_from"])] += 1
            padding = record.get("padding_mode")
            if padding:
                key = f"{record.get('condition')}|{padding}"
                self.padding_target[key].append(float(targets[index].mean().item()))
            if record.get("direction_class"):
                self.direction_counts[str(record["direction_class"])] += 1
            if str(record.get("condition")) == "shift":
                self.shift_counts[f"shift{int(record.get('magnitude_linf', 0))}"] += 1
            if str(record.get("condition")) == "wrong_slice":
                self.wrong_slice_delta_z.append(float(record.get("delta_z_norm", 0.0)))
            if str(record.get("condition")) == "wrong_patient" and record.get("replacement_index") is not None:
                source = int(record["source_index"])
                self.wrong_patient_replacements[source].add(int(record["replacement_index"]))
                patient = record.get("replacement_patient_id")
                if patient is not None:
                    self.wrong_patient_replacement_patients[source].add(str(patient))

    def state_dict(self) -> Dict[str, Any]:
        return {
            "condition_counts": dict(self.condition_counts),
            "fallback_counts": dict(self.fallback_counts),
            "padding_target": {key: list(values) for key, values in self.padding_target.items()},
            "direction_counts": dict(self.direction_counts),
            "shift_counts": dict(self.shift_counts),
            "wrong_slice_delta_z": list(self.wrong_slice_delta_z),
            "wrong_patient_replacements": {
                str(key): sorted(values) for key, values in self.wrong_patient_replacements.items()
            },
            "wrong_patient_replacement_patients": {
                str(key): sorted(values)
                for key, values in self.wrong_patient_replacement_patients.items()
            },
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.condition_counts.update(state.get("condition_counts", {}))
        self.fallback_counts.update(state.get("fallback_counts", {}))
        self.padding_target.update({
            key: list(values) for key, values in state.get("padding_target", {}).items()
        })
        self.direction_counts.update(state.get("direction_counts", {}))
        self.shift_counts.update(state.get("shift_counts", {}))
        self.wrong_slice_delta_z.extend(state.get("wrong_slice_delta_z", []))
        for key, values in state.get("wrong_patient_replacements", {}).items():
            self.wrong_patient_replacements[int(key)].update(int(value) for value in values)
        for key, values in state.get("wrong_patient_replacement_patients", {}).items():
            self.wrong_patient_replacement_patients[int(key)].update(str(value) for value in values)

    def summary(self) -> Dict[str, Any]:
        total = sum(self.condition_counts.values())
        unique_counts = [len(values) for values in self.wrong_patient_replacements.values()]
        unique_patient_counts = [
            len(values) for values in self.wrong_patient_replacement_patients.values()
        ]
        return {
            "total_second_views": total,
            "condition_counts": dict(sorted(self.condition_counts.items())),
            "condition_fractions": {
                key: value / max(total, 1)
                for key, value in sorted(self.condition_counts.items())
            },
            "fallback_counts": dict(sorted(self.fallback_counts.items())),
            "fallback_fraction": sum(self.fallback_counts.values()) / max(total, 1),
            "shift_counts": dict(sorted(self.shift_counts.items())),
            "direction_counts": dict(sorted(self.direction_counts.items())),
            "padding_by_condition": {
                key: {"count": len(values), "mean_reliability_target": safe_mean(values)}
                for key, values in sorted(self.padding_target.items())
            },
            "wrong_slice_delta_z": {
                "count": len(self.wrong_slice_delta_z),
                "mean": safe_mean(self.wrong_slice_delta_z),
                "min": min(self.wrong_slice_delta_z) if self.wrong_slice_delta_z else None,
                "max": max(self.wrong_slice_delta_z) if self.wrong_slice_delta_z else None,
            },
            "wrong_patient_replacement_diversity": {
                "num_sources": len(unique_counts),
                "mean_unique_replacements_per_source": safe_mean(unique_counts),
                "min_unique_replacements_per_source": min(unique_counts) if unique_counts else None,
                "max_unique_replacements_per_source": max(unique_counts) if unique_counts else None,
                "mean_unique_patients_per_source": safe_mean(unique_patient_counts),
                "min_unique_patients_per_source": min(unique_patient_counts) if unique_patient_counts else None,
                "max_unique_patients_per_source": max(unique_patient_counts) if unique_patient_counts else None,
                "per_source": {
                    str(source): {
                        "replacement_indices": sorted(self.wrong_patient_replacements[source]),
                        "replacement_patient_ids": sorted(
                            self.wrong_patient_replacement_patients.get(source, set())
                        ),
                    }
                    for source in sorted(self.wrong_patient_replacements)
                },
            },
        }


def select_patient_ids(dataset, limit: Optional[int]) -> List[str]:
    ids = list(dict.fromkeys(str(row["patient_id"]) for row in dataset.patient_rows))
    if limit is None:
        return ids
    if limit < 1 or limit > len(ids):
        raise ValueError(f"Invalid patient limit {limit}; available={len(ids)}")
    return ids[:limit]


@torch.no_grad()
def evaluate_clean(model, loader, device, max_batches=None) -> Dict[str, Any]:
    model.eval()
    slice_rows, q_values, need_values, rms_values = [], [], [], []
    for batch_index, batch in enumerate(loader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break
        kspace, mask, pd, target, _ = prepare_batch(batch, device)
        prediction, aux = model(
            kspace, mask, pd, torch.ones(pd.shape[0], device=device), return_aux=True
        )
        prediction = center_crop(prediction, target.shape[-2], target.shape[-1])
        losses = l1_per_sample(prediction, target)
        q_values.extend(aux["q_hat"].mean((1, 2)).cpu().tolist())
        need_values.extend(aux["need_mean"].mean((1, 2)).cpu().tolist())
        rms_values.extend(aux["gated_aux_to_target_rms"].mean((1, 2)).cpu().tolist())
        for i in range(target.shape[0]):
            slice_rows.append(
                {
                    "patient_id": str(batch["patient_id"][i]),
                    "slice_idx": int(batch["slice_idx"][i]),
                    "l1": float(losses[i].item()),
                }
            )
    patient_values: Dict[str, List[float]] = defaultdict(list)
    for row in slice_rows:
        patient_values[row["patient_id"]].append(row["l1"])
    patient_rows = [
        {"patient_id": key, "num_slices": len(values), "l1": safe_mean(values)}
        for key, values in sorted(patient_values.items())
    ]
    model.train()
    return {
        "patient_l1": safe_mean(row["l1"] for row in patient_rows),
        "slice_l1": safe_mean(row["l1"] for row in slice_rows),
        "clean_q": safe_mean(q_values),
        "clean_need": safe_mean(need_values),
        "clean_gated_rms": safe_mean(rms_values),
        "num_patients": len(patient_rows),
        "num_slices": len(slice_rows),
        "patient_rows": patient_rows,
    }


def make_dataset(args, split: str, patient_ids=None):
    return PairedMulticoilAuxPDToPDFSDataset(
        metadata_csv=args.metadata_csv,
        split=split,
        pdfs_acceleration=args.acceleration,
        pd_aux_acceleration=args.pd_aux_acceleration,
        patient_ids=patient_ids,
        slices_per_patient=None,
        edge_weight=1.0,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locked_code_hashes() -> Dict[str, str]:
    relative_paths = [
        "src/m2_prnf_varnet.py",
        "src/m2_prnf_corruptions.py",
        "scripts/train_m2_prnf.py",
        "scripts/preflight_m2_prnf.py",
        "scripts/evaluate_m2_prnf_R8.py",
        "src/dataset_paired_multicoil_aux_pd_r2.py",
        "src/fft_utils.py",
        "src/masks.py",
        "FINAL_PROTOCOL_R8.json",
    ]
    result = {}
    for relative in relative_paths:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Locked dependency is missing: {path}")
        result[relative] = sha256_file(path)
    return result


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


def capture_rng_state() -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Dict[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(state) != required:
        raise RuntimeError(f"RNG-state keys mismatch: {sorted(state)}")

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])

    cpu_state = (
        state["torch_cpu"]
        .detach()
        .to(device="cpu", dtype=torch.uint8)
        .contiguous()
    )
    torch.set_rng_state(cpu_state)

    if torch.cuda.is_available():
        cuda_states = [
            item.detach()
            .to(device="cpu", dtype=torch.uint8)
            .contiguous()
            for item in state["torch_cuda"]
        ]
        torch.cuda.set_rng_state_all(cuda_states)


IMMUTABLE_RESUME_KEYS = (
    "variant", "metadata_csv", "acceleration", "pd_aux_acceleration",
    "epochs", "learning_rate", "batch_size", "grad_accum_steps", "num_workers",
    "num_train_patients", "num_val_patients", "max_train_batches",
    "max_val_batches", "num_cascades", "chans", "sens_chans", "pools",
    "sens_pools", "controller_chans", "initial_aux_alpha",
    "initial_gate_probability", "initial_need_probability", "need_floor",
    "lambda_rel", "lambda_rank", "aux_loss_ramp_epochs", "seed",
    "train_patient_ids", "val_patient_ids", "corrupt_view_mixture",
    "corruption_config", "full_clean_manifest", "full_clean_manifest_sha256",
    "robustness_manifest", "robustness_manifest_sha256",
    "optimizer", "gradient_clip_norm", "code_hashes", "runtime_versions",
)


def normalize_resume_value(value: Any) -> Any:
    """Canonicalize config containers before equality checks.

    Checkpoints may preserve tuples while JSON-derived current configs use
    lists.  They represent the same configuration but compare unequal in
    Python, and JSON error output hides the type difference.
    """
    if isinstance(value, dict):
        return {str(key): normalize_resume_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_resume_value(item) for item in value]
    return value


def validate_resume_config(previous: Dict[str, Any], current: Dict[str, Any]) -> None:
    """Validate the audited 50->60 low-learning-rate continuation.

    The only intentional protocol changes are total epochs (50 -> 60) and
    learning rate (3e-4 -> 3e-5). A checkpoint produced by this extension may
    also be resumed after a scheduler interruption.
    """
    mismatches = {}
    for key in IMMUTABLE_RESUME_KEYS:
        if key == "epochs" and int(previous.get(key, -1)) in {50, 60}:
            continue
        if key == "learning_rate" and float(previous.get(key, -1.0)) in {
            3e-4, 3e-5
        }:
            continue
        previous_value = normalize_resume_value(previous.get(key))
        current_value = normalize_resume_value(current.get(key))
        if previous_value != current_value:
            mismatches[key] = {
                "checkpoint": previous_value,
                "current": current_value,
            }
    if mismatches:
        raise RuntimeError(
            "Resume configuration mismatch:\n" + json.dumps(mismatches, indent=2)
        )


def save_checkpoint(
    path, model, optimizer, epoch, best_epoch, best_val, config, history,
    run_corruption_audit,
):
    temporary = Path(str(path) + ".tmp")
    torch.save(
        {
            "epoch": int(epoch),
            "best_epoch": int(best_epoch),
            "best_val": float(best_val),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "history": history,
            "rng_state": capture_rng_state(),
            "sampler_next_epoch": int(epoch),
            "code_hashes": config["code_hashes"],
            "run_corruption_audit": run_corruption_audit.state_dict(),
        },
        temporary,
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc low-LR continuation of M2-U Augmented from epoch 50 "
            "to epoch 60."
        )
    )
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--full_clean_manifest", required=True)
    parser.add_argument("--robustness_manifest", required=True)
    parser.add_argument("--variant", required=True, choices=sorted(VALID_VARIANTS))
    parser.add_argument("--acceleration", type=int, default=8, choices=[4, 6, 8])
    parser.add_argument("--pd_aux_acceleration", type=int, default=2, choices=[2, 4, 6, 8])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=2, help="Anatomical pairs; forward batch is 2x")
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_train_patients", type=int, default=None)
    parser.add_argument("--num_val_patients", type=int, default=None)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--num_cascades", type=int, default=12)
    parser.add_argument("--chans", type=int, default=18)
    parser.add_argument("--sens_chans", type=int, default=8)
    parser.add_argument("--pools", type=int, default=4)
    parser.add_argument("--sens_pools", type=int, default=4)
    parser.add_argument("--controller_chans", type=int, default=16)
    parser.add_argument("--initial_aux_alpha", type=float, default=0.1)
    parser.add_argument("--initial_gate_probability", type=float, default=0.95)
    parser.add_argument("--initial_need_probability", type=float, default=0.95)
    parser.add_argument("--need_floor", type=float, default=0.25)
    parser.add_argument("--lambda_rel", type=float, default=0.05)
    parser.add_argument("--lambda_rank", type=float, default=0.02)
    parser.add_argument("--aux_loss_ramp_epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    if args.variant != "m2u_augmented":
        raise ValueError("This continuation is locked to variant=m2u_augmented")
    if args.epochs != 60:
        raise ValueError("This continuation is locked to total epochs=60")
    if args.learning_rate != 3e-5:
        raise ValueError("This continuation is locked to learning_rate=3e-5")
    if not args.resume:
        raise ValueError("An epoch-50 model_last.pt checkpoint is required")
    if min(args.epochs, args.batch_size, args.grad_accum_steps) < 1:
        raise ValueError("epochs, batch_size and grad_accum_steps must be positive")
    if args.learning_rate <= 0 or min(args.lambda_rel, args.lambda_rank) < 0:
        raise ValueError("Invalid learning rate or auxiliary loss weight")
    args.metadata_csv = str(Path(args.metadata_csv).resolve())
    if not Path(args.metadata_csv).is_file():
        raise FileNotFoundError(args.metadata_csv)
    for name in ("full_clean_manifest", "robustness_manifest"):
        value = str(Path(getattr(args, name)).resolve())
        setattr(args, name, value)
        if not Path(value).is_file():
            raise FileNotFoundError(value)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    full_train, full_val = make_dataset(args, "train"), make_dataset(args, "val")
    train_ids = select_patient_ids(full_train, args.num_train_patients)
    val_ids = select_patient_ids(full_val, args.num_val_patients)
    if set(train_ids) & set(val_ids):
        raise RuntimeError("Patient leakage detected")
    train_base = make_dataset(args, "train", train_ids)
    val_base = make_dataset(args, "val", val_ids)
    train_dataset, val_dataset = IndexedDataset(train_base), IndexedDataset(val_base)
    train_sampler = ShapeBucketBatchSampler(train_dataset, args.batch_size, True, args.seed)
    val_sampler = ShapeBucketBatchSampler(val_dataset, args.batch_size, False, args.seed)
    train_loader = DataLoader(
        train_dataset, batch_sampler=train_sampler, num_workers=args.num_workers,
        pin_memory=device.type == "cuda"
    )
    val_loader = DataLoader(
        val_dataset, batch_sampler=val_sampler, num_workers=args.num_workers,
        pin_memory=device.type == "cuda"
    )
    negative_sampler = HardNegativeSampler(train_dataset)
    corruption_config = CorruptionConfig()

    # Dataset indexing/auditing must not perturb the shared model initialisation.
    set_seed(args.seed)
    model = M2PRNFAuxPDVarNet(
        variant=args.variant,
        num_cascades=args.num_cascades,
        sens_chans=args.sens_chans,
        sens_pools=args.sens_pools,
        chans=args.chans,
        pools=args.pools,
        controller_chans=args.controller_chans,
        initial_aux_alpha=args.initial_aux_alpha,
        initial_gate_probability=args.initial_gate_probability,
        initial_need_probability=args.initial_need_probability,
        need_floor=args.need_floor,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    uses_reliability = args.variant in {"prnf_full", "prnf_no_need"}
    code_hashes = locked_code_hashes()

    config = vars(args).copy()
    # Resume location is mutable execution state, not part of model identity.
    config["resume"] = None
    config.update(
        {
            "parameter_count": parameter_count,
            "trainable_parameter_count": trainable_count,
            "random_initialisation": True,
            "optimizer": {
                "name": "Adam", "betas": [0.9, 0.999],
                "eps": 1e-8, "weight_decay": 0.0,
            },
            "gradient_clip_norm": 10.0,
            "correct_corrupt_loss_weights": {"correct": 0.7, "second_view": 0.3},
            "forward_view_proportions": {"correct": 0.5, "second_view": 0.5},
            "corrupt_view_mixture": (
                {} if args.variant == "m2u_clean" else dict(CORRUPT_MIXTURE)
            ),
            "effective_training_exposure": (
                {"correct": 1.0}
                if args.variant == "m2u_clean"
                else {
                    "correct_objective_weight": 0.70,
                    "border_control_objective_weight": 0.03,
                    "shift_objective_weight": 0.09,
                    "wrong_slice_objective_weight": 0.0675,
                    "wrong_patient_objective_weight": 0.0675,
                    "missing_objective_weight": 0.045,
                }
            ),
            "corruption_config": asdict(corruption_config),
            "missing_participates_in_reconstruction": True,
            "missing_participates_in_reliability_bce_or_ranking": False,
            "checkpoint_selection_metric": "patient-level clean validation L1",
            "target_need_definition": "target/DC evidence; reconstruction gradients only",
            "reliability_definition": "detached target-auxiliary paired evidence",
            "image_view_forwards_per_pair": 2,
            "train_patient_ids": train_ids,
            "val_patient_ids": val_ids,
            "full_clean_manifest_sha256": sha256_file(Path(args.full_clean_manifest)),
            "robustness_manifest_sha256": sha256_file(Path(args.robustness_manifest)),
            "code_hashes": code_hashes,
            "runtime_versions": runtime_versions(),
            "protocol_version": (
                "M2-PRNF-R8-posthoc-lowLR-50to60-three-arm-v1"
            ),
            "posthoc_study": True,
            "continuation_source_total_epochs": 50,
            "continuation_target_total_epochs": 60,
            "continuation_learning_rate": 3e-5,
            "continuation_semantics": (
                "Resume epoch-50 model_last with optimizer/RNG/sampler state; "
                "set every Adam parameter-group learning rate to 3e-5."
            ),
        }
    )

    start_epoch, best_epoch, best_val, history = 1, 0, float("inf"), []
    run_corruption_audit = CorruptionAudit()
    if args.resume:
        resume_path = Path(args.resume).resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        for key in (
            "model_state_dict", "optimizer_state_dict", "config", "rng_state",
            "code_hashes", "epoch", "sampler_next_epoch", "history",
            "run_corruption_audit",
        ):
            if key not in checkpoint:
                raise RuntimeError(f"Resume checkpoint is missing {key}")
        source_epoch = int(checkpoint["epoch"])
        if source_epoch < 50 or source_epoch >= 60:
            raise RuntimeError(
                f"Expected a continuation checkpoint from epoch 50-59, got {source_epoch}"
            )
        if source_epoch == 50:
            source = checkpoint["config"]
            expected = {
                "variant": "m2u_augmented",
                "epochs": 50,
                "learning_rate": 3e-4,
                "seed": 42,
                "acceleration": 8,
                "pd_aux_acceleration": 2,
                "batch_size": 4,
                "grad_accum_steps": 1,
            }
            observed = {key: source.get(key) for key in expected}
            if observed != expected:
                raise RuntimeError(
                    "Invalid M2-U Augmented epoch-50 source:\n"
                    + json.dumps(
                        {"expected": expected, "observed": observed}, indent=2
                    )
                )
            config["source_checkpoint_sha256"] = sha256_file(resume_path)
        else:
            if checkpoint["config"].get("protocol_version") != config[
                "protocol_version"
            ]:
                raise RuntimeError("Resume checkpoint is not from this extension")
            config["source_checkpoint_sha256"] = checkpoint["config"].get(
                "source_checkpoint_sha256"
            )
        validate_resume_config(checkpoint["config"], config)
        if checkpoint["code_hashes"] != code_hashes:
            raise RuntimeError("Checkpoint code hashes differ from the installed code")
        config_path = output_dir / "config.json"
        if not config_path.is_file():
            raise RuntimeError("Resume output directory has no config.json")
        installed_config = json.loads(config_path.read_text(encoding="utf-8"))
        validate_resume_config(installed_config, config)
        for filename, expected in (
            ("train_patient_ids.txt", train_ids), ("val_patient_ids.txt", val_ids)
        ):
            path = output_dir / filename
            observed = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
            if observed != expected:
                raise RuntimeError(f"Resume patient manifest mismatch: {filename}")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = args.learning_rate
        observed_lrs = sorted(
            {float(group["lr"]) for group in optimizer.param_groups}
        )
        if observed_lrs != [3e-5]:
            raise RuntimeError(f"Failed to apply continuation LR: {observed_lrs}")
        start_epoch = int(checkpoint["epoch"]) + 1
        best_epoch = int(checkpoint.get("best_epoch", 0))
        best_val = float(checkpoint.get("best_val", float("inf")))
        history = list(checkpoint.get("history", []))
        if int(checkpoint["sampler_next_epoch"]) != int(checkpoint["epoch"]):
            raise RuntimeError("Checkpoint sampler epoch is inconsistent")
        existing_log = output_dir / "training_log.csv"
        if not existing_log.is_file():
            raise RuntimeError("Resume output directory has no training_log.csv")
        with open(existing_log, newline="", encoding="utf-8") as file:
            logged_rows = list(csv.DictReader(file))
        if len(logged_rows) != int(checkpoint["epoch"]):
            raise RuntimeError(
                "Resume log/checkpoint epoch mismatch: "
                f"rows={len(logged_rows)}, checkpoint={checkpoint['epoch']}"
            )
        if len(history) != int(checkpoint["epoch"]):
            raise RuntimeError("Resume history/checkpoint epoch mismatch")
        run_corruption_audit.load_state_dict(checkpoint["run_corruption_audit"])
        restore_rng_state(checkpoint["rng_state"])
        extension_path = output_dir / "low_lr_extension_config.json"
        if extension_path.is_file():
            installed_extension = json.loads(
                extension_path.read_text(encoding="utf-8")
            )
            validate_resume_config(installed_extension, config)
        else:
            extension_path.write_text(
                json.dumps(config, indent=2), encoding="utf-8"
            )
        if start_epoch > args.epochs:
            raise RuntimeError("The requested run has already completed")
    else:
        with open(output_dir / "config.json", "x", encoding="utf-8") as file:
            json.dump(config, file, indent=2)
        (output_dir / "train_patient_ids.txt").write_text(
            "\n".join(train_ids), encoding="utf-8"
        )
        (output_dir / "val_patient_ids.txt").write_text(
            "\n".join(val_ids), encoding="utf-8"
        )

    log_path = output_dir / "training_log.csv"
    fieldnames = [
        "epoch", "train_total_loss", "train_recon_loss", "train_correct_l1",
        "train_corrupt_l1", "train_reliability_bce", "train_rank_loss",
        "aux_loss_ramp", "train_q_clean", "train_q_corrupt", "train_need_clean",
        "train_need_corrupt", "train_gated_rms_clean", "train_gated_rms_corrupt",
        "train_need_p05_clean", "train_need_p95_clean",
        "val_patient_l1", "val_slice_l1", "val_clean_q", "val_clean_need",
        "val_clean_gated_rms", "gradient_norm_mean", "epoch_seconds",
        "peak_gpu_memory_gb",
    ]
    if start_epoch == 1:
        with open(log_path, "w", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=fieldnames).writeheader()

    print(json.dumps({
        "variant": args.variant, "device": str(device), "parameters": parameter_count,
        "train_patients": len(train_ids), "val_patients": len(val_ids),
        "train_slices": len(train_dataset), "val_slices": len(val_dataset),
    }, indent=2))

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        train_sampler.epoch = epoch - 1
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        stats: Dict[str, List[float]] = defaultdict(list)
        corruption_audit = CorruptionAudit()
        optimizer.zero_grad(set_to_none=True)
        accumulation = 0

        for batch_index, batch in enumerate(train_loader, start=1):
            if args.max_train_batches and batch_index > args.max_train_batches:
                break
            kspace, mask, pd, target, indices = prepare_batch(batch, device)
            if args.variant == "m2u_clean":
                corrupt = SimpleNamespace(
                    image=pd.clone(),
                    availability=torch.ones(pd.shape[0], device=device),
                    reliability_target=torch.ones(
                        pd.shape[0], args.pools, device=device
                    ),
                    records=[
                        {"condition": "clean", "condition_key": "clean"}
                        for _ in range(pd.shape[0])
                    ],
                )
            else:
                corrupt = corrupt_batch_prnf(
                    pd, indices, train_dataset, negative_sampler, epoch,
                    batch_index, args.seed, corruption_config
                )
            corruption_audit.add(corrupt.records, corrupt.reliability_target)
            run_corruption_audit.add(corrupt.records, corrupt.reliability_target)
            base = pd.shape[0]
            paired_kspace = torch.cat([kspace, kspace], dim=0)
            paired_mask = torch.cat([mask, mask], dim=0)
            paired_pd = torch.cat([pd, corrupt.image], dim=0)
            paired_available = torch.cat(
                [torch.ones_like(corrupt.availability), corrupt.availability], dim=0
            )
            paired_target = torch.cat([target, target], dim=0)
            prediction, aux = model(
                paired_kspace, paired_mask, paired_pd, paired_available, return_aux=True
            )
            prediction = center_crop(prediction, target.shape[-2], target.shape[-1])
            clean_l1 = l1_per_sample(prediction[:base], target).mean()
            corrupt_l1 = l1_per_sample(prediction[base:], target).mean()
            recon_loss = 0.7 * clean_l1 + 0.3 * corrupt_l1

            zero = recon_loss * 0.0
            rel_bce, rank_loss = zero, zero
            if uses_reliability:
                logits_clean = aux["q_logits"][:base]
                logits_corrupt = aux["q_logits"][base:]
                targets_clean = torch.ones_like(logits_clean)
                targets_corrupt = corrupt.reliability_target[:, None, :].expand_as(logits_corrupt)
                clean_bce = F.binary_cross_entropy_with_logits(
                    logits_clean, targets_clean
                )
                reliability_mask = torch.tensor(
                    [record.get("condition") != "missing" for record in corrupt.records],
                    device=device,
                    dtype=torch.bool,
                )
                if bool(reliability_mask.any()):
                    corrupt_bce = F.binary_cross_entropy_with_logits(
                        logits_corrupt[reliability_mask],
                        targets_corrupt[reliability_mask],
                    )
                    rel_bce = 0.5 * (clean_bce + corrupt_bce)
                else:
                    rel_bce = clean_bce
                rank_loss, _ = paired_discrimination_loss(
                    aux["q_hat"][:base], aux["q_hat"][base:],
                    corrupt.reliability_target, corrupt.records
                )
            ramp = min(1.0, epoch / max(1, args.aux_loss_ramp_epochs))
            total_loss = recon_loss + ramp * (
                args.lambda_rel * rel_bce + args.lambda_rank * rank_loss
            )
            if not torch.isfinite(total_loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}, batch {batch_index}")
            (total_loss / args.grad_accum_steps).backward()
            accumulation += 1
            should_step = accumulation == args.grad_accum_steps or batch_index == len(train_loader)
            if args.max_train_batches and batch_index == args.max_train_batches:
                should_step = True
            if should_step:
                if accumulation < args.grad_accum_steps:
                    correction = args.grad_accum_steps / accumulation
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                if not torch.isfinite(norm):
                    raise RuntimeError("Non-finite gradient norm")
                stats["gradient"].append(float(norm.item()))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulation = 0

            for key, value in {
                "total": total_loss, "recon": recon_loss, "clean": clean_l1,
                "corrupt": corrupt_l1, "bce": rel_bce, "rank": rank_loss,
            }.items():
                stats[key].append(float(value.detach().item()))
            for key, tensor in {
                "q_clean": aux["q_hat"][:base], "q_corrupt": aux["q_hat"][base:],
                "need_clean": aux["need_mean"][:base], "need_corrupt": aux["need_mean"][base:],
                "need_p05_clean": aux["need_p05"][:base],
                "need_p95_clean": aux["need_p95"][:base],
                "rms_clean": aux["gated_aux_to_target_rms"][:base],
                "rms_corrupt": aux["gated_aux_to_target_rms"][base:],
            }.items():
                stats[key].append(float(tensor.detach().mean().item()))
            if batch_index == 1 or batch_index % 25 == 0:
                print(
                    f"epoch={epoch:02d} batch={batch_index:04d}/{len(train_loader)} "
                    f"clean={clean_l1.item():.6f} corrupt={corrupt_l1.item():.6f} "
                    f"q={stats['q_clean'][-1]:.3f}/{stats['q_corrupt'][-1]:.3f} "
                    f"need={stats['need_clean'][-1]:.3f}/{stats['need_corrupt'][-1]:.3f}",
                    flush=True,
                )

        val = evaluate_clean(model, val_loader, device, args.max_val_batches)
        seconds = time.time() - epoch_start
        row = {
            "epoch": epoch,
            "train_total_loss": safe_mean(stats["total"]),
            "train_recon_loss": safe_mean(stats["recon"]),
            "train_correct_l1": safe_mean(stats["clean"]),
            "train_corrupt_l1": safe_mean(stats["corrupt"]),
            "train_reliability_bce": safe_mean(stats["bce"]),
            "train_rank_loss": safe_mean(stats["rank"]),
            "aux_loss_ramp": min(1.0, epoch / max(1, args.aux_loss_ramp_epochs)),
            "train_q_clean": safe_mean(stats["q_clean"]),
            "train_q_corrupt": safe_mean(stats["q_corrupt"]),
            "train_need_clean": safe_mean(stats["need_clean"]),
            "train_need_corrupt": safe_mean(stats["need_corrupt"]),
            "train_need_p05_clean": safe_mean(stats["need_p05_clean"]),
            "train_need_p95_clean": safe_mean(stats["need_p95_clean"]),
            "train_gated_rms_clean": safe_mean(stats["rms_clean"]),
            "train_gated_rms_corrupt": safe_mean(stats["rms_corrupt"]),
            "val_patient_l1": val["patient_l1"],
            "val_slice_l1": val["slice_l1"],
            "val_clean_q": val["clean_q"],
            "val_clean_need": val["clean_need"],
            "val_clean_gated_rms": val["clean_gated_rms"],
            "gradient_norm_mean": safe_mean(stats["gradient"]),
            "epoch_seconds": seconds,
            "peak_gpu_memory_gb": (
                torch.cuda.max_memory_allocated() / 1024 ** 3 if device.type == "cuda" else 0.0
            ),
        }
        history.append(row)
        with open(log_path, "a", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=fieldnames).writerow(row)
        if val["patient_l1"] < best_val:
            best_val, best_epoch = val["patient_l1"], epoch
            save_checkpoint(
                output_dir / "model_best.pt", model, optimizer, epoch,
                best_epoch, best_val, config, history, run_corruption_audit
            )
        save_checkpoint(
            output_dir / "model_last.pt", model, optimizer, epoch,
            best_epoch, best_val, config, history, run_corruption_audit
        )
        with open(output_dir / f"epoch_{epoch:02d}_clean_audit.json", "w", encoding="utf-8") as file:
            json.dump(val, file, indent=2)
        with open(output_dir / f"epoch_{epoch:02d}_corruption_audit.json", "w", encoding="utf-8") as file:
            json.dump(corruption_audit.summary(), file, indent=2)
        with open(output_dir / "run_corruption_audit.json", "w", encoding="utf-8") as file:
            json.dump(run_corruption_audit.summary(), file, indent=2)
        print(json.dumps(row, indent=2), flush=True)

    summary = {
        "variant": args.variant,
        "completed_epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_val_patient_l1": best_val,
        "parameter_count": parameter_count,
        "checkpoint_selection_metric": "patient-level clean validation L1",
        "protocol_version": config["protocol_version"],
        "continued_from_epoch": 50,
        "continuation_learning_rate": 3e-5,
        "source_checkpoint_sha256": config["source_checkpoint_sha256"],
    }
    with open(output_dir / "final_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
