#!/usr/bin/env python3
import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_paired_multicoil_aux_pd_r2 import (
    PairedMulticoilAuxPDToPDFSDataset,
)
from src.fft_utils import center_crop
from src.m2u_auxiliary_varnet_optimized import M2UAuxPDVarNet


class ShapeBucketBatchSampler:
    def __init__(self, dataset, batch_size, shuffle=False, seed=42):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

        buckets = defaultdict(list)
        for idx, record in enumerate(dataset.records):
            with h5py.File(record["pdfs_path"], "r") as hf:
                shape_key = tuple(hf["kspace"].shape[1:])
            buckets[shape_key].append(idx)

        self.buckets = dict(buckets)
        self._num_batches = sum(
            (len(indices) + self.batch_size - 1) // self.batch_size
            for indices in self.buckets.values()
        )

        print(
            f"ShapeBucketBatchSampler: {len(self.buckets)} shape buckets, "
            f"{self._num_batches} batches, batch_size={self.batch_size}, "
            f"shuffle={self.shuffle}"
        )

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        all_batches = []

        for indices in self.buckets.values():
            indices = list(indices)
            if self.shuffle:
                rng.shuffle(indices)

            for start in range(0, len(indices), self.batch_size):
                all_batches.append(indices[start:start + self.batch_size])

        if self.shuffle:
            rng.shuffle(all_batches)

        self.epoch += 1
        yield from all_batches

    def __len__(self):
        return self._num_batches


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_aux_batch(batch, device: torch.device):
    required = [
        "pdfs_masked_kspace",
        "pd_aux_image",
        "pdfs_target_raw",
        "mask",
    ]
    for key in required:
        if key not in batch:
            raise KeyError(f"Missing batch key: {key}")

    pdfs_kspace = batch["pdfs_masked_kspace"].to(
        device,
        non_blocking=True,
    )
    if not torch.is_complex(pdfs_kspace):
        raise TypeError(
            f"Expected complex PDFS k-space, got {pdfs_kspace.dtype}"
        )
    pdfs_kspace = torch.view_as_real(pdfs_kspace).float()

    mask = batch["mask"].to(device, non_blocking=True).bool()
    mask = mask[:, None, None, :, None]

    pd_aux = batch["pd_aux_image"].to(
        device,
        non_blocking=True,
    ).float()
    pdfs_target = batch["pdfs_target_raw"].to(
        device,
        non_blocking=True,
    ).float()

    if pd_aux.ndim == 4 and pd_aux.shape[1] == 1:
        pd_aux = pd_aux.squeeze(1)
    if pdfs_target.ndim == 4 and pdfs_target.shape[1] == 1:
        pdfs_target = pdfs_target.squeeze(1)

    if pd_aux.ndim != 3:
        raise RuntimeError(
            f"Expected PD auxiliary [B,H,W], got {tuple(pd_aux.shape)}"
        )
    if pdfs_target.ndim != 3:
        raise RuntimeError(
            f"Expected PDFS target [B,H,W], got {tuple(pdfs_target.shape)}"
        )

    return pdfs_kspace, mask, pd_aux, pdfs_target


def l1_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise RuntimeError(
            f"Prediction/target mismatch: "
            f"{tuple(prediction.shape)} vs {tuple(target.shape)}"
        )

    scale = target.amax(
        dim=(-2, -1),
        keepdim=True,
    ).clamp_min(1e-8)

    return torch.abs(
        prediction / scale - target / scale
    ).mean(dim=(-2, -1))


def safe_mean(values):
    return float(np.mean(values)) if values else float("nan")


def patient_level_metrics(rows):
    """Aggregate slice losses so every validation volume has equal weight."""
    per_patient = defaultdict(list)
    for row in rows:
        per_patient[row["patient_id"]].append(row["pdfs_l1"])

    patient_rows = [
        {
            "patient_id": patient_id,
            "num_slices": len(losses),
            "pdfs_volume_l1": safe_mean(losses),
        }
        for patient_id, losses in sorted(per_patient.items())
    ]
    return patient_rows


def average_fusion_diagnostics(diagnostics):
    """Mean nested fusion diagnostics across all validation batches."""
    if not diagnostics:
        return {}

    return {
        scale: {
            metric: safe_mean(
                [diagnostic[scale][metric] for diagnostic in diagnostics]
            )
            for metric in diagnostics[0][scale]
        }
        for scale in diagnostics[0]
    }


@torch.no_grad()
def evaluate(model, loader, device, max_val_batches=None):
    model.eval()

    overall = []
    edge = []
    central = []
    rows = []
    fusion_diagnostics = []

    for batch_index, batch in enumerate(loader, start=1):
        if (
            max_val_batches is not None
            and batch_index > max_val_batches
        ):
            break

        pdfs_kspace, mask, pd_aux, target = prepare_aux_batch(
            batch,
            device,
        )

        prediction = model(
            pdfs_masked_kspace=pdfs_kspace,
            mask=mask,
            pd_aux_image=pd_aux,
        )
        fusion_diagnostics.append(model.fusion_diagnostics())
        prediction = center_crop(
            prediction,
            crop_h=target.shape[-2],
            crop_w=target.shape[-1],
        )

        if not torch.isfinite(prediction).all():
            raise RuntimeError(
                "Non-finite PD-FS prediction during validation"
            )

        losses = l1_per_sample(prediction, target)
        if not torch.isfinite(losses).all():
            raise RuntimeError(
                "Non-finite PD-FS validation loss"
            )

        for sample_idx in range(target.shape[0]):
            value = float(losses[sample_idx].item())

            is_edge_value = batch["is_edge"][sample_idx]
            is_edge = bool(
                is_edge_value.item()
                if torch.is_tensor(is_edge_value)
                else is_edge_value
            )

            slice_value = batch["slice_idx"][sample_idx]
            slice_idx = int(
                slice_value.item()
                if torch.is_tensor(slice_value)
                else slice_value
            )

            patient_id = str(batch["patient_id"][sample_idx])

            overall.append(value)
            (edge if is_edge else central).append(value)
            rows.append(
                {
                    "patient_id": patient_id,
                    "slice_idx": slice_idx,
                    "is_edge": is_edge,
                    "pdfs_l1": value,
                }
            )

    patient_rows = patient_level_metrics(rows)
    volume_losses = [row["pdfs_volume_l1"] for row in patient_rows]

    results = {
        "pdfs_slice_mean_l1": safe_mean(overall),
        "pdfs_volume_mean_l1": safe_mean(volume_losses),
        "pdfs_volume_std_l1": (
            float(np.std(volume_losses, ddof=1))
            if len(volume_losses) > 1
            else 0.0
        ),
        "pdfs_edge_l1": safe_mean(edge),
        "pdfs_central_l1": safe_mean(central),
        "num_slices": len(overall),
        "num_volumes": len(patient_rows),
        "num_edge_slices": len(edge),
        "num_central_slices": len(central),
        "fusion_diagnostics": average_fusion_diagnostics(
            fusion_diagnostics
        ),
    }

    model.train()
    return results, rows, patient_rows


def make_model(args, device):
    return M2UAuxPDVarNet(
        num_cascades=args.num_cascades,
        sens_chans=args.sens_chans,
        sens_pools=args.sens_pools,
        chans=args.chans,
        pools=args.pools,
        mask_center=True,
        initial_aux_alpha=args.initial_aux_alpha,
    ).to(device)


def select_patient_ids(dataset, limit: Optional[int]):
    patient_ids = list(
        dict.fromkeys(
            str(row["patient_id"])
            for row in dataset.patient_rows
        )
    )

    if limit is None:
        return patient_ids
    if limit < 1:
        raise ValueError(
            "Patient limit must be at least 1 or omitted"
        )

    selected = patient_ids[:limit]
    if len(selected) != limit:
        raise RuntimeError(
            f"Requested {limit} patients, found {len(selected)}"
        )
    return selected


TRAINING_LOG_COLUMNS = [
    "epoch",
    "train_pdfs_l1",
    "val_pdfs_volume_l1",
    "val_pdfs_slice_l1",
    "val_pdfs_edge_l1",
    "val_pdfs_central_l1",
    "gradient_norm_mean",
    "epoch_seconds",
    "peak_gpu_memory_gb",
    "learning_rate",
    "fusion_diagnostics_json",
]


def initialise_training_log(path):
    with open(path, "w", newline="", encoding="utf-8") as file:
        csv.DictWriter(
            file,
            fieldnames=TRAINING_LOG_COLUMNS,
        ).writeheader()


def append_training_log(path, row):
    with open(path, "a", newline="", encoding="utf-8") as file:
        csv.DictWriter(
            file,
            fieldnames=TRAINING_LOG_COLUMNS,
        ).writerow(row)


def save_slice_metrics(path, rows):
    columns = [
        "patient_id",
        "slice_idx",
        "is_edge",
        "pdfs_l1",
    ]
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_patient_metrics(path, rows):
    columns = ["patient_id", "num_slices", "pdfs_volume_l1"]
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_patient_ids(path, patient_ids):
    with open(path, "w", encoding="utf-8") as file:
        for patient_id in sorted(patient_ids):
            file.write(f"{patient_id}\n")


def verify_resume_config(
    checkpoint_config: dict,
    current_config: dict,
) -> None:
    fields = [
        "metadata_csv",
        "acceleration",
        "pd_aux_acceleration",
        "num_cascades",
        "chans",
        "sens_chans",
        "pools",
        "sens_pools",
        "initial_aux_alpha",
        "batch_size",
        "seed",
        "train_patient_ids",
        "val_patient_ids",
        "fusion_type",
        "fusion_scales",
        "target_contrast",
    ]

    mismatches = []
    for field in fields:
        previous = checkpoint_config.get(field)
        current = current_config.get(field)
        if previous != current:
            mismatches.append(
                f"{field}: checkpoint={previous!r}, current={current!r}"
            )

    if mismatches:
        raise RuntimeError(
            "Resume configuration does not match checkpoint:\n"
            + "\n".join(mismatches)
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "M2-U ungated multi-scale auxiliary PD "
            "to PD-FS VarNet training."
        )
    )
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument(
        "--acceleration",
        type=int,
        choices=[4, 6, 8],
        required=True,
    )
    parser.add_argument(
        "--pd_aux_acceleration",
        type=int,
        choices=[2, 4, 6, 8],
        default=2,
    )
    parser.add_argument(
        "--num_train_patients",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--num_val_patients",
        type=int,
        default=None,
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        "--num_cascades",
        type=int,
        default=12,
    )
    parser.add_argument("--chans", type=int, default=18)
    parser.add_argument(
        "--initial_aux_alpha",
        type=float,
        default=0.1,
        help=(
            "Initial residual scale for each ungated PD feature injection. "
            "Use the same value for all formal M2-U runs."
        ),
    )
    parser.add_argument(
        "--sens_chans",
        type=int,
        default=8,
    )
    parser.add_argument("--pools", type=int, default=4)
    parser.add_argument(
        "--sens_pools",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_train_batches",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max_val_batches",
        type=int,
        default=None,
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to M2-U model_last.pt.",
    )

    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.learning_rate <= 0:
        raise ValueError("--learning_rate must be positive")
    if args.initial_aux_alpha < 0:
        raise ValueError("--initial_aux_alpha must be non-negative")

    metadata_path = Path(args.metadata_csv).resolve()
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata CSV does not exist: {metadata_path}"
        )
    args.metadata_csv = str(metadata_path)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    resume_path = (
        Path(args.resume).resolve()
        if args.resume is not None
        else None
    )
    if resume_path is not None and not resume_path.exists():
        raise FileNotFoundError(
            f"Resume checkpoint does not exist: {resume_path}"
        )

    set_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 80)
    print("M2-U ungated multi-scale auxiliary PD VarNet training")
    print("=" * 80)
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Metadata:", args.metadata_csv)
    print("PD-FS target acceleration:", args.acceleration)
    print("PD auxiliary acceleration:", args.pd_aux_acceleration)
    print("Epochs:", args.epochs)
    print("Max train batches:", args.max_train_batches)
    print("Max val batches:", args.max_val_batches)
    print("Output directory:", output_dir)
    print("Resume checkpoint:", resume_path)
    print("=" * 80)

    common_dataset_args = {
        "metadata_csv": args.metadata_csv,
        "pdfs_acceleration": args.acceleration,
        "pd_aux_acceleration": args.pd_aux_acceleration,
        "slices_per_patient": None,
        "edge_weight": 1.0,
    }

    full_train = PairedMulticoilAuxPDToPDFSDataset(
        split="train",
        **common_dataset_args,
    )
    full_val = PairedMulticoilAuxPDToPDFSDataset(
        split="val",
        **common_dataset_args,
    )

    train_patient_ids = select_patient_ids(
        full_train,
        args.num_train_patients,
    )
    val_patient_ids = select_patient_ids(
        full_val,
        args.num_val_patients,
    )

    overlap = set(train_patient_ids) & set(val_patient_ids)
    if overlap:
        raise RuntimeError(
            f"Patient leakage detected: {sorted(overlap)}"
        )

    train_dataset = PairedMulticoilAuxPDToPDFSDataset(
        split="train",
        patient_ids=train_patient_ids,
        **common_dataset_args,
    )
    val_dataset = PairedMulticoilAuxPDToPDFSDataset(
        split="val",
        patient_ids=val_patient_ids,
        **common_dataset_args,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=ShapeBucketBatchSampler(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            seed=args.seed,
        ),
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=ShapeBucketBatchSampler(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            seed=args.seed,
        ),
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print("Train patients:", len(train_patient_ids))
    print("Validation patients:", len(val_patient_ids))
    print("Train slices:", len(train_dataset))
    print("Validation slices:", len(val_dataset))
    print("Train batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("Patient leakage check: PASSED")

    save_patient_ids(
        output_dir / "train_patient_ids.txt",
        train_patient_ids,
    )
    save_patient_ids(
        output_dir / "val_patient_ids.txt",
        val_patient_ids,
    )

    model = make_model(args, device)
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    print("Model parameters:", parameter_count)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    config = vars(args).copy()
    config.update(
        {
            "metadata_csv": args.metadata_csv,
            "output_dir": str(output_dir),
            "resume": (
                str(resume_path)
                if resume_path is not None
                else None
            ),
            "train_patient_ids": train_patient_ids,
            "val_patient_ids": val_patient_ids,
            "train_patients": len(train_patient_ids),
            "val_patients": len(val_patient_ids),
            "train_slices": len(train_dataset),
            "val_slices": len(val_dataset),
            "parameter_count": parameter_count,
            "loss": "pdfs_l1 only, target-scaled",
            "mask": (
                "Gaussian variable-density 1D Cartesian, "
                "PD-FS target stream only"
            ),
            "model": (
                "M2-U: independently encoded PD and PD-FS "
                "features with ungated multi-scale additive fusion"
            ),
            "target_contrast": "PD-FS",
            "auxiliary_contrast": "PD",
            "auxiliary_full_PD_access": False,
            "auxiliary_source": (
                "zero-filled RSS from undersampled PD k-space"
            ),
            "fusion_type": "ungated_additive",
            "initial_aux_alpha": args.initial_aux_alpha,
            "fusion_scales": [
                "H/2",
                "H/4",
                "H/8",
                "H/16",
            ][:args.pools],
            "full_resolution_pd_fusion": False,
            "pd_encoder_shared_across_cascades": True,
            "pd_auxiliary_flip_correction": True,
            "checkpoint_selection_metric": (
                "validation patient/volume-level mean PDFS L1"
            ),
        }
    )

    consistency_config = config.copy()
    consistency_config["resume"] = None

    start_epoch = 1
    best_val = float("inf")
    best_epoch = 0
    epoch_times = []
    history = []

    if resume_path is not None:
        checkpoint = torch.load(
            resume_path,
            map_location=device,
            weights_only=False,
        )
        for key in [
            "epoch",
            "model_state_dict",
            "optimizer_state_dict",
            "config",
        ]:
            if key not in checkpoint:
                raise KeyError(
                    f"Resume checkpoint missing key: {key}"
                )

        verify_resume_config(
            checkpoint["config"],
            consistency_config,
        )
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        start_epoch = int(checkpoint["epoch"]) + 1
        best_val = float(
            checkpoint.get("best_val", float("inf"))
        )
        best_epoch = int(
            checkpoint.get("best_epoch", 0)
        )
        epoch_times = list(
            checkpoint.get("epoch_times", [])
        )
        history = list(
            checkpoint.get("history", [])
        )

        print("Resume loaded:", resume_path)
        print("Starting epoch:", start_epoch)

        if start_epoch > args.epochs:
            raise RuntimeError(
                "Checkpoint has already completed the requested "
                "number of epochs."
            )

    with open(
        output_dir / "config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(config, file, indent=2)

    training_log_path = output_dir / "training_log.csv"
    if resume_path is None:
        initialise_training_log(training_log_path)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        model.train()

        train_losses = []
        gradient_norms = []

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        for batch_index, batch in enumerate(
            train_loader,
            start=1,
        ):
            if (
                args.max_train_batches is not None
                and batch_index > args.max_train_batches
            ):
                break

            pdfs_kspace, mask, pd_aux, target = (
                prepare_aux_batch(batch, device)
            )

            prediction = model(
                pdfs_masked_kspace=pdfs_kspace,
                mask=mask,
                pd_aux_image=pd_aux,
            )
            prediction = center_crop(
                prediction,
                crop_h=target.shape[-2],
                crop_w=target.shape[-1],
            )

            if not torch.isfinite(prediction).all():
                raise RuntimeError(
                    f"Non-finite prediction at epoch {epoch}, "
                    f"batch {batch_index}"
                )

            loss = l1_per_sample(
                prediction,
                target,
            ).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at epoch {epoch}, "
                    f"batch {batch_index}"
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            gradient_norm = (
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=10.0,
                )
            )
            if not torch.isfinite(gradient_norm):
                raise RuntimeError(
                    f"Non-finite gradient at epoch {epoch}, "
                    f"batch {batch_index}"
                )

            optimizer.step()

            train_losses.append(float(loss.item()))
            gradient_norms.append(
                float(gradient_norm.item())
            )

            if batch_index == 1 or batch_index % 10 == 0:
                print(
                    f"Epoch {epoch:02d}/{args.epochs} | "
                    f"Batch {batch_index:04d}/{len(train_loader)} | "
                    f"pdfs={loss.item():.6f}",
                    flush=True,
                )

        val_results, slice_rows, patient_rows = evaluate(
            model,
            val_loader,
            device,
            args.max_val_batches,
        )
        fusion_diagnostics = val_results["fusion_diagnostics"]

        epoch_seconds = time.time() - epoch_start
        epoch_times.append(epoch_seconds)

        peak_gpu_memory = (
            torch.cuda.max_memory_allocated() / 1024**3
            if device.type == "cuda"
            else 0.0
        )

        epoch_row = {
            "epoch": epoch,
            "train_pdfs_l1": safe_mean(train_losses),
            "val_pdfs_volume_l1": val_results[
                "pdfs_volume_mean_l1"
            ],
            "val_pdfs_slice_l1": val_results[
                "pdfs_slice_mean_l1"
            ],
            "val_pdfs_edge_l1": val_results[
                "pdfs_edge_l1"
            ],
            "val_pdfs_central_l1": val_results[
                "pdfs_central_l1"
            ],
            "gradient_norm_mean": safe_mean(
                gradient_norms
            ),
            "epoch_seconds": epoch_seconds,
            "peak_gpu_memory_gb": peak_gpu_memory,
            "learning_rate": float(
                optimizer.param_groups[0]["lr"]
            ),
            "fusion_diagnostics_json": json.dumps(
                fusion_diagnostics,
                sort_keys=True,
            ),
        }

        history.append(epoch_row)
        append_training_log(
            training_log_path,
            epoch_row,
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} completed | "
            f"train={epoch_row['train_pdfs_l1']:.6f} | "
            f"val_volume={epoch_row['val_pdfs_volume_l1']:.6f} | "
            f"val_slice={epoch_row['val_pdfs_slice_l1']:.6f} | "
            f"time={epoch_seconds:.1f}s | "
            f"peak_gpu={peak_gpu_memory:.2f}GB",
            flush=True,
        )

        improved = (
            val_results["pdfs_volume_mean_l1"] < best_val
        )
        if improved:
            best_val = float(
                val_results["pdfs_volume_mean_l1"]
            )
            best_epoch = epoch

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_results": val_results,
            "best_val": best_val,
            "best_epoch": best_epoch,
            "epoch_times": epoch_times,
            "history": history,
            "config": consistency_config,
        }

        torch.save(
            checkpoint,
            output_dir / "model_last.pt",
        )

        if improved:
            torch.save(
                checkpoint,
                output_dir / "model_best.pt",
            )
            save_slice_metrics(
                output_dir / "best_val_per_slice_metrics.csv",
                slice_rows,
            )
            save_patient_metrics(
                output_dir / "best_val_per_patient_metrics.csv",
                patient_rows,
            )
            print(
                f"New best checkpoint at epoch {epoch}: "
                f"{best_val:.6f}",
                flush=True,
            )

    summary = {
        "best_epoch": best_epoch,
        "best_val_pdfs_volume_l1": best_val,
        "completed_epochs": args.epochs,
        "mean_epoch_seconds": safe_mean(epoch_times),
        "parameter_count": parameter_count,
    }

    with open(
        output_dir / "training_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    print("=" * 80)
    print("M2-U training finished")
    print("Best epoch:", best_epoch)
    print("Best validation PDFS L1:", best_val)
    print("Output:", output_dir)
    print("=" * 80)


if __name__ == "__main__":
    main()
