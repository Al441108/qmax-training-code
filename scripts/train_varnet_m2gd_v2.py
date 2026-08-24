#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.auxiliary_corruptions_v2 import (
    CorruptionConfig,
    HardNegativeSampler,
    SCALE_NAMES,
    corrupt_batch,
    curriculum_mixture,
)
from src.dataset_paired_multicoil_aux_pd_r2 import (
    PairedMulticoilAuxPDToPDFSDataset,
)
from src.fft_utils import center_crop
from src.m2gd_v2_auxiliary_varnet import (
    M2GDv2AuxPDVarNet,
    load_m2u_backbone,
)


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
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        buckets: Dict[tuple, List[int]] = defaultdict(list)
        for index, record in enumerate(dataset.records):
            with h5py.File(record["pdfs_path"], "r") as hf:
                key = tuple(int(value) for value in hf["kspace"].shape[1:])
            buckets[key].append(index)
        self.buckets = dict(buckets)
        self._num_batches = sum(
            math.ceil(len(indices) / self.batch_size)
            for indices in self.buckets.values()
        )
        print(
            f"ShapeBucketBatchSampler: {len(self.buckets)} buckets, "
            f"{self._num_batches} batches, batch_size={self.batch_size}, "
            f"shuffle={self.shuffle}"
        )

    def __len__(self):
        return self._num_batches

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        batches: List[List[int]] = []
        for indices in self.buckets.values():
            indices = list(indices)
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batches.append(indices[start:start + self.batch_size])
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
    pdfs_kspace = batch["pdfs_masked_kspace"].to(device, non_blocking=True)
    if not torch.is_complex(pdfs_kspace):
        raise TypeError(f"Expected complex k-space, got {pdfs_kspace.dtype}")
    pdfs_kspace = torch.view_as_real(pdfs_kspace).float()

    mask = batch["mask"].to(device, non_blocking=True).bool()
    if mask.ndim == 2:
        mask = mask[:, None, None, :, None]
    elif mask.ndim == 1:
        mask = mask[None, None, None, :, None]
    elif mask.ndim != 5:
        raise RuntimeError(f"Unexpected mask shape: {tuple(mask.shape)}")

    pd_aux = batch["pd_aux_image"].to(device, non_blocking=True).float()
    target = batch["pdfs_target_raw"].to(device, non_blocking=True).float()
    if pd_aux.ndim == 4 and pd_aux.shape[1] == 1:
        pd_aux = pd_aux[:, 0]
    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]
    if pd_aux.ndim != 3 or target.ndim != 3:
        raise RuntimeError(
            f"Expected PD/target [B,H,W], got {tuple(pd_aux.shape)} / {tuple(target.shape)}"
        )
    sample_indices = [int(value) for value in batch["sample_idx"]]
    return pdfs_kspace, mask, pd_aux, target, sample_indices


def l1_per_sample(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    scale = target.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    return torch.abs(prediction / scale - target / scale).mean(dim=(-2, -1))


def safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else float("nan")


def select_patient_ids(dataset, limit: Optional[int]) -> List[str]:
    patient_ids = list(
        dict.fromkeys(str(row["patient_id"]) for row in dataset.patient_rows)
    )
    if limit is None:
        return patient_ids
    if limit < 1 or len(patient_ids) < limit:
        raise ValueError(
            f"Invalid patient limit {limit}; available patients={len(patient_ids)}"
        )
    return patient_ids[:limit]


def make_model(args, device: torch.device) -> M2GDv2AuxPDVarNet:
    return M2GDv2AuxPDVarNet(
        num_cascades=args.num_cascades,
        sens_chans=args.sens_chans,
        sens_pools=args.sens_pools,
        chans=args.chans,
        pools=args.pools,
        mask_center=True,
        initial_aux_alpha=args.initial_aux_alpha,
        initial_gate_probability=args.initial_gate_probability,
    ).to(device)


def make_corruption_config(args) -> CorruptionConfig:
    return CorruptionConfig(
        padding_prob_reflect=args.padding_prob_reflect,
        padding_prob_replicate=args.padding_prob_replicate,
        padding_prob_zero=args.padding_prob_zero,
        border_padding_prob_reflect=args.border_padding_prob_reflect,
        border_padding_prob_replicate=args.border_padding_prob_replicate,
        border_padding_prob_zero=args.border_padding_prob_zero,
    )


def parameter_gradient_norm(model: torch.nn.Module, fragments: Sequence[str]) -> float:
    total = 0.0
    found = False
    for name, parameter in model.named_parameters():
        if not any(fragment in name for fragment in fragments):
            continue
        if parameter.grad is None:
            continue
        found = True
        total += float(parameter.grad.detach().square().sum().item())
    return math.sqrt(total) if found else 0.0


class EpochAudit:
    def __init__(self, scale_names: Sequence[str]):
        self.scale_names = list(scale_names)
        self.conditions: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "count": 0,
                "target": [],
                "target_by_scale": defaultdict(list),
                "bce": [],
                "q_hat": defaultdict(list),
                "q": defaultdict(list),
                "gated_rms": defaultdict(list),
                "channel_gate": defaultdict(list),
                "spatial_gate": defaultdict(list),
            }
        )
        self.padding_counts: Dict[str, int] = defaultdict(int)
        self.direction_counts: Dict[str, int] = defaultdict(int)
        self.fallback_counts: Dict[str, int] = defaultdict(int)

    def add(
        self,
        records: Sequence[Dict[str, Any]],
        targets: torch.Tensor,
        aux: Dict[str, torch.Tensor],
    ) -> None:
        q_hat = aux["q_hat"].detach()
        q = aux["q"].detach()
        gated_rms = aux["gated_aux_to_target_rms"].detach()
        channel_gate = aux["channel_gate_mean"].detach()
        spatial_gate = aux["spatial_gate_mean"].detach()
        if targets.ndim != 2 or targets.shape[1] != len(self.scale_names):
            raise RuntimeError(
                "Expected scale-aware reliability targets [B,S], got "
                f"{tuple(targets.shape)}."
            )
        expanded_target = targets[:, None, :].expand_as(q_hat)
        bce = F.binary_cross_entropy(q_hat, expanded_target, reduction="none").mean(
            dim=(1, 2)
        )

        for sample_index, record in enumerate(records):
            key = str(record["condition_key"])
            bucket = self.conditions[key]
            bucket["count"] += 1
            bucket["target"].append(float(targets[sample_index].mean().item()))
            bucket["bce"].append(float(bce[sample_index].item()))
            for scale_index, scale_name in enumerate(self.scale_names):
                bucket["target_by_scale"][scale_name].append(
                    float(targets[sample_index, scale_index].item())
                )
                bucket["q_hat"][scale_name].append(
                    float(q_hat[sample_index, :, scale_index].mean().item())
                )
                bucket["q"][scale_name].append(
                    float(q[sample_index, :, scale_index].mean().item())
                )
                bucket["gated_rms"][scale_name].append(
                    float(gated_rms[sample_index, :, scale_index].mean().item())
                )
                bucket["channel_gate"][scale_name].append(
                    float(channel_gate[sample_index, :, scale_index].mean().item())
                )
                bucket["spatial_gate"][scale_name].append(
                    float(spatial_gate[sample_index, :, scale_index].mean().item())
                )
            if record.get("padding_mode"):
                self.padding_counts[str(record["padding_mode"])] += 1
            if record.get("direction_class"):
                self.direction_counts[str(record["direction_class"])] += 1
            if record.get("fallback_from"):
                self.fallback_counts[str(record["fallback_from"])] += 1

    def summary(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "padding_counts": dict(sorted(self.padding_counts.items())),
            "direction_counts": dict(sorted(self.direction_counts.items())),
            "fallback_counts": dict(sorted(self.fallback_counts.items())),
            "conditions": {},
        }
        for condition, values in sorted(self.conditions.items()):
            result["conditions"][condition] = {
                "count": int(values["count"]),
                "target_mean": safe_mean(values["target"]),
                "target_by_scale": {
                    scale: safe_mean(values["target_by_scale"][scale])
                    for scale in self.scale_names
                },
                "bce_mean": safe_mean(values["bce"]),
                "q_hat_by_scale": {
                    scale: safe_mean(values["q_hat"][scale])
                    for scale in self.scale_names
                },
                "q_by_scale": {
                    scale: safe_mean(values["q"][scale])
                    for scale in self.scale_names
                },
                "gated_rms_by_scale": {
                    scale: safe_mean(values["gated_rms"][scale])
                    for scale in self.scale_names
                },
                "channel_gate_by_scale": {
                    scale: safe_mean(values["channel_gate"][scale])
                    for scale in self.scale_names
                },
                "spatial_gate_by_scale": {
                    scale: safe_mean(values["spatial_gate"][scale])
                    for scale in self.scale_names
                },
            }
        return result


@torch.no_grad()
def evaluate_clean(
    model: M2GDv2AuxPDVarNet,
    loader,
    device: torch.device,
    max_batches: Optional[int],
) -> Dict[str, Any]:
    model.eval()
    slice_rows: List[Dict[str, Any]] = []
    q_values: List[float] = []
    rms_values: List[float] = []

    for batch_index, batch in enumerate(loader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break
        kspace, mask, pd_aux, target, _ = prepare_batch(batch, device)
        availability = torch.ones(pd_aux.shape[0], device=device)
        prediction, aux = model(
            pdfs_masked_kspace=kspace,
            mask=mask,
            pd_aux_image=pd_aux,
            pd_available=availability,
            return_aux=True,
        )
        prediction = center_crop(
            prediction,
            crop_h=target.shape[-2],
            crop_w=target.shape[-1],
        )
        losses = l1_per_sample(prediction, target)
        q_values.extend(aux["q_hat"].mean(dim=(1, 2)).cpu().tolist())
        rms_values.extend(
            aux["gated_aux_to_target_rms"].mean(dim=(1, 2)).cpu().tolist()
        )
        for sample_index in range(target.shape[0]):
            slice_rows.append(
                {
                    "patient_id": str(batch["patient_id"][sample_index]),
                    "slice_idx": int(batch["slice_idx"][sample_index]),
                    "pdfs_l1": float(losses[sample_index].item()),
                }
            )

    per_patient: Dict[str, List[float]] = defaultdict(list)
    for row in slice_rows:
        per_patient[row["patient_id"]].append(row["pdfs_l1"])
    patient_rows = [
        {
            "patient_id": patient_id,
            "num_slices": len(values),
            "pdfs_patient_l1": safe_mean(values),
        }
        for patient_id, values in sorted(per_patient.items())
    ]
    result = {
        "pdfs_patient_l1": safe_mean(
            row["pdfs_patient_l1"] for row in patient_rows
        ),
        "pdfs_slice_l1": safe_mean(row["pdfs_l1"] for row in slice_rows),
        "clean_q_hat": safe_mean(q_values),
        "clean_gated_rms": safe_mean(rms_values),
        "num_patients": len(patient_rows),
        "num_slices": len(slice_rows),
        "patient_rows": patient_rows,
        "slice_rows": slice_rows,
    }
    model.train()
    return result


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    epoch: int,
    best_epoch: int,
    best_val: float,
    config: Dict[str, Any],
    history: List[Dict[str, Any]],
) -> None:
    torch.save(
        {
            "epoch": int(epoch),
            "best_epoch": int(best_epoch),
            "best_val": float(best_val),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "history": history,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train M2-GD v2 from a pretrained M2-U Epoch-50 backbone."
    )
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--m2u_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--acceleration", type=int, default=8, choices=[4, 6, 8])
    parser.add_argument("--pd_aux_acceleration", type=int, default=2, choices=[2, 4, 6, 8])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--curriculum", choices=["smoke5", "formal15"], default="smoke5")
    parser.add_argument("--batch_size", type=int, default=4)
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
    parser.add_argument("--initial_aux_alpha", type=float, default=0.1)
    parser.add_argument("--initial_gate_probability", type=float, default=0.99)
    parser.add_argument("--pretrained_lr", type=float, default=1e-5)
    parser.add_argument("--new_module_lr", type=float, default=1e-4)
    parser.add_argument("--lambda_rel", type=float, default=0.05)
    parser.add_argument("--freeze_epochs", type=int, default=2)
    parser.add_argument("--padding_prob_reflect", type=float, default=0.70)
    parser.add_argument("--padding_prob_replicate", type=float, default=0.20)
    parser.add_argument("--padding_prob_zero", type=float, default=0.10)
    parser.add_argument("--border_padding_prob_reflect", type=float, default=0.25)
    parser.add_argument("--border_padding_prob_replicate", type=float, default=0.25)
    parser.add_argument("--border_padding_prob_zero", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.grad_accum_steps < 1:
        raise ValueError("epochs, batch_size and grad_accum_steps must be positive.")
    if args.curriculum == "smoke5" and args.epochs != 5:
        raise ValueError("smoke5 curriculum is pre-registered for exactly five epochs.")
    if args.curriculum == "formal15" and args.epochs < 9:
        raise ValueError("formal15 curriculum requires at least nine epochs.")
    if args.pools != len(SCALE_NAMES):
        raise ValueError(
            f"This scale-aware protocol requires {len(SCALE_NAMES)} pools; "
            f"received {args.pools}."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    common = {
        "metadata_csv": args.metadata_csv,
        "pdfs_acceleration": args.acceleration,
        "pd_aux_acceleration": args.pd_aux_acceleration,
        "slices_per_patient": None,
        "edge_weight": 1.0,
    }
    full_train = PairedMulticoilAuxPDToPDFSDataset(split="train", **common)
    full_val = PairedMulticoilAuxPDToPDFSDataset(split="val", **common)
    train_ids = select_patient_ids(full_train, args.num_train_patients)
    val_ids = select_patient_ids(full_val, args.num_val_patients)
    if set(train_ids) & set(val_ids):
        raise RuntimeError("Patient leakage detected.")

    train_base = PairedMulticoilAuxPDToPDFSDataset(
        split="train", patient_ids=train_ids, **common
    )
    val_base = PairedMulticoilAuxPDToPDFSDataset(
        split="val", patient_ids=val_ids, **common
    )
    train_dataset = IndexedDataset(train_base)
    val_dataset = IndexedDataset(val_base)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=ShapeBucketBatchSampler(
            train_dataset, args.batch_size, True, args.seed
        ),
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=ShapeBucketBatchSampler(
            val_dataset, args.batch_size, False, args.seed
        ),
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    print(
        f"Train: {len(train_ids)} patients, {len(train_dataset)} slices, "
        f"{len(train_loader)} batches"
    )
    print(
        f"Validation: {len(val_ids)} patients, {len(val_dataset)} slices, "
        f"{len(val_loader)} batches"
    )

    negative_sampler = HardNegativeSampler(train_dataset)
    corruption_config = make_corruption_config(args)
    corruption_config.validate()

    model = make_model(args, device)
    transfer_report = load_m2u_backbone(
        model,
        args.m2u_checkpoint,
        map_location=device,
    )
    print("M2-U transfer report:")
    print(json.dumps(transfer_report, indent=2, default=str))

    pretrained_parameters, new_parameters = model.parameter_groups()
    optimizer = torch.optim.Adam(
        [
            {"params": pretrained_parameters, "lr": args.pretrained_lr, "name": "pretrained"},
            {"params": new_parameters, "lr": args.new_module_lr, "name": "new"},
        ]
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    new_parameter_count = sum(parameter.numel() for parameter in new_parameters)

    config = vars(args).copy()
    config.update(
        {
            "train_patient_ids": train_ids,
            "val_patient_ids": val_ids,
            "parameter_count": parameter_count,
            "new_parameter_count": new_parameter_count,
            "effective_batch_size": args.batch_size * args.grad_accum_steps,
            "checkpoint_selection_metric": (
                "patient-level mean clean validation PDFS L1 among checkpoints "
                "eligible only after the complete corruption curriculum"
            ),
            "reliability_name": "severity-supervised cross-contrast reliability score",
            "m2u_transfer_report": transfer_report,
            "corruption_config": corruption_config.__dict__,
        }
    )
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "train_patient_ids.txt").write_text(
        "\n".join(train_ids) + "\n", encoding="utf-8"
    )
    (output_dir / "val_patient_ids.txt").write_text(
        "\n".join(val_ids) + "\n", encoding="utf-8"
    )

    csv_path = output_dir / "training_log.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "backbone_frozen",
                "checkpoint_eligible",
                "train_total_loss",
                "train_recon_l1",
                "train_reliability_bce",
                "train_weighted_reliability",
                "val_pdfs_patient_l1",
                "val_pdfs_slice_l1",
                "val_clean_q_hat",
                "val_clean_gated_rms",
                "gradient_norm_mean",
                "reliability_gradient_norm_mean",
                "epoch_seconds",
                "peak_gpu_memory_gb",
            ],
        ).writeheader()

    best_val = float("inf")
    best_epoch = 0
    best_clean_val = float("inf")
    best_clean_epoch = 0
    history: List[Dict[str, Any]] = []
    first_eligible_epoch = args.epochs if args.curriculum == "smoke5" else 9

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        backbone_frozen = epoch <= args.freeze_epochs
        model.set_pretrained_trainable(not backbone_frozen)
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        mixture = curriculum_mixture(epoch, args.curriculum)
        print("=" * 90)
        print(
            f"Epoch {epoch}/{args.epochs} | backbone_frozen={backbone_frozen} | "
            f"mixture={mixture}"
        )
        total_losses: List[float] = []
        recon_losses: List[float] = []
        reliability_losses: List[float] = []
        gradient_norms: List[float] = []
        reliability_gradient_norms: List[float] = []
        epoch_audit = EpochAudit(SCALE_NAMES)

        optimizer.zero_grad(set_to_none=True)
        processed_batches = 0
        accumulation_counter = 0
        for batch_index, batch in enumerate(train_loader, start=1):
            if args.max_train_batches is not None and batch_index > args.max_train_batches:
                break
            kspace, mask, pd_aux, target, sample_indices = prepare_batch(batch, device)
            corrupted = corrupt_batch(
                pd_aux=pd_aux,
                sample_indices=sample_indices,
                dataset=train_dataset,
                negative_sampler=negative_sampler,
                epoch=epoch,
                batch_index=batch_index,
                seed=args.seed,
                curriculum=args.curriculum,
                config=corruption_config,
            )
            prediction, aux = model(
                pdfs_masked_kspace=kspace,
                mask=mask,
                pd_aux_image=corrupted.image,
                pd_available=corrupted.availability,
                return_aux=True,
            )
            prediction = center_crop(
                prediction,
                crop_h=target.shape[-2],
                crop_w=target.shape[-1],
            )
            loss_recon = l1_per_sample(prediction, target).mean()
            q_hat = aux["q_hat"]
            reliability_target = corrupted.reliability_target[:, None, :].expand_as(q_hat)
            loss_reliability = F.binary_cross_entropy(q_hat, reliability_target)
            loss_total = loss_recon + args.lambda_rel * loss_reliability
            if not torch.isfinite(loss_total):
                raise RuntimeError(
                    f"Non-finite loss at epoch={epoch}, batch={batch_index}."
                )

            (loss_total / args.grad_accum_steps).backward()
            processed_batches += 1
            accumulation_counter += 1
            should_step = (
                processed_batches % args.grad_accum_steps == 0
                or batch_index == len(train_loader)
                or (
                    args.max_train_batches is not None
                    and batch_index == args.max_train_batches
                )
            )
            if should_step:
                if accumulation_counter < args.grad_accum_steps:
                    correction = args.grad_accum_steps / accumulation_counter
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)
                reliability_gradient_norms.append(
                    parameter_gradient_norm(
                        model,
                        ("reliability_head", "channel_gate", "spatial_gate"),
                    )
                )
                parameters_with_grad = [
                    parameter
                    for parameter in model.parameters()
                    if parameter.grad is not None
                ]
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    parameters_with_grad,
                    max_norm=10.0,
                )
                gradient_norms.append(float(gradient_norm.item()))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulation_counter = 0

            total_losses.append(float(loss_total.item()))
            recon_losses.append(float(loss_recon.item()))
            reliability_losses.append(float(loss_reliability.item()))
            epoch_audit.add(
                corrupted.records,
                corrupted.reliability_target,
                aux,
            )
            if batch_index == 1 or batch_index % 25 == 0:
                print(
                    f"Epoch {epoch:02d} batch {batch_index:04d}/{len(train_loader)} | "
                    f"total={loss_total.item():.6f} | recon={loss_recon.item():.6f} | "
                    f"rel={loss_reliability.item():.6f}",
                    flush=True,
                )

        validation = evaluate_clean(
            model,
            val_loader,
            device,
            args.max_val_batches,
        )
        epoch_seconds = time.time() - epoch_start
        peak_gpu_memory = (
            torch.cuda.max_memory_allocated() / 1024 ** 3
            if device.type == "cuda"
            else 0.0
        )
        row = {
            "epoch": epoch,
            "backbone_frozen": int(backbone_frozen),
            "checkpoint_eligible": int(epoch >= first_eligible_epoch),
            "train_total_loss": safe_mean(total_losses),
            "train_recon_l1": safe_mean(recon_losses),
            "train_reliability_bce": safe_mean(reliability_losses),
            "train_weighted_reliability": args.lambda_rel * safe_mean(reliability_losses),
            "val_pdfs_patient_l1": validation["pdfs_patient_l1"],
            "val_pdfs_slice_l1": validation["pdfs_slice_l1"],
            "val_clean_q_hat": validation["clean_q_hat"],
            "val_clean_gated_rms": validation["clean_gated_rms"],
            "gradient_norm_mean": safe_mean(gradient_norms),
            "reliability_gradient_norm_mean": safe_mean(reliability_gradient_norms),
            "epoch_seconds": epoch_seconds,
            "peak_gpu_memory_gb": peak_gpu_memory,
        }
        audit_summary = epoch_audit.summary()
        history_entry = {**row, "curriculum_mixture": mixture, "audit": audit_summary}
        history.append(history_entry)

        with csv_path.open("a", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=list(row.keys())).writerow(row)
        with (output_dir / "training_audit.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(history_entry, default=str) + "\n")
        with (output_dir / f"epoch_{epoch:02d}_audit.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(history_entry, file, indent=2, default=str)
        with (output_dir / f"epoch_{epoch:02d}_val_patient_l1.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["patient_id", "num_slices", "pdfs_patient_l1"],
            )
            writer.writeheader()
            writer.writerows(validation["patient_rows"])

        current_val = float(validation["pdfs_patient_l1"])
        if current_val < best_clean_val:
            best_clean_val = current_val
            best_clean_epoch = epoch
            save_checkpoint(
                output_dir / "model_best_clean.pt",
                model,
                optimizer,
                epoch,
                best_clean_epoch,
                best_clean_val,
                config,
                history,
            )

        checkpoint_eligible = epoch >= first_eligible_epoch
        if checkpoint_eligible and current_val < best_val:
            best_val = current_val
            best_epoch = epoch
            save_checkpoint(
                output_dir / "model_best.pt",
                model,
                optimizer,
                epoch,
                best_epoch,
                best_val,
                config,
                history,
            )
        save_checkpoint(
            output_dir / "model_last.pt",
            model,
            optimizer,
            epoch,
            best_epoch,
            best_val,
            config,
            history,
        )
        print(json.dumps(row, indent=2))
        print("Condition audit:")
        print(json.dumps(audit_summary, indent=2, default=str))

    if best_epoch == 0 or not math.isfinite(best_val):
        raise RuntimeError("No curriculum-eligible checkpoint was saved.")

    final_summary = {
        "status": "completed",
        "completed_epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_val_pdfs_patient_l1": best_val,
        "best_clean_epoch": best_clean_epoch,
        "best_clean_val_pdfs_patient_l1": best_clean_val,
        "parameter_count": parameter_count,
        "new_parameter_count": new_parameter_count,
        "checkpoint_selection_metric": (
            "patient-level mean clean validation PDFS L1 among curriculum-eligible epochs"
        ),
        "m2u_checkpoint": args.m2u_checkpoint,
        "curriculum": args.curriculum,
        "model_best": str(output_dir / "model_best.pt"),
        "model_best_clean": str(output_dir / "model_best_clean.pt"),
        "model_last": str(output_dir / "model_last.pt"),
    }
    (output_dir / "final_summary.json").write_text(
        json.dumps(final_summary, indent=2), encoding="utf-8"
    )
    print("=" * 90)
    print(json.dumps(final_summary, indent=2))


if __name__ == "__main__":
    main()
