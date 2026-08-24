import argparse
import csv
import json
import random
from collections import defaultdict
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from fastmri.models import VarNet
from src.dataset_paired_multicoil import PairedMulticoilDataset


class ShapeBucketBatchSampler:
    """Batch sampler that groups samples with identical k-space shape.

    This avoids default_collate failures for fastMRI cases with different
    matrix widths, e.g. [15, 640, 372] vs [15, 640, 368], without padding.
    """

    def __init__(self, dataset, batch_size, shuffle=False, seed=42):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

        buckets = defaultdict(list)

        for idx, record in enumerate(dataset.records):
            path = record["pd_path"]
            with h5py.File(path, "r") as hf:
                # kspace shape is usually [num_slices, num_coils, height, width].
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

        for _, indices in self.buckets.items():
            indices = list(indices)
            if self.shuffle:
                rng.shuffle(indices)

            for start in range(0, len(indices), self.batch_size):
                all_batches.append(indices[start:start + self.batch_size])

        if self.shuffle:
            rng.shuffle(all_batches)

        self.epoch += 1

        for batch in all_batches:
            yield batch

    def __len__(self):
        return self._num_batches



def _pad_tensor_list(tensors):
    """Pad tensors in a batch to the maximum shape before stacking.

    This is needed because fastMRI knee multicoil samples can have slightly
    different matrix widths, e.g. [15, 640, 372] vs [15, 640, 368].
    """
    first = tensors[0]

    if first.ndim == 0:
        return torch.stack(tensors, dim=0)

    shapes = [tuple(t.shape) for t in tensors]
    if all(s == shapes[0] for s in shapes):
        return torch.stack(tensors, dim=0)

    max_shape = tuple(max(s[d] for s in shapes) for d in range(first.ndim))

    out_shape = (len(tensors),) + max_shape
    out = first.new_zeros(out_shape)

    for i, t in enumerate(tensors):
        slices = (i,) + tuple(slice(0, size) for size in t.shape)
        out[slices] = t

    return out


def pad_collate_fn(batch):
    """Collate function for variable-size fastMRI samples.

    Tensor fields are zero-padded to the largest shape within the batch.
    String/path fields are kept as Python lists.
    Numeric metadata is converted to tensors where appropriate.
    """
    if len(batch) == 0:
        return batch

    output = {}
    keys = batch[0].keys()

    for key in keys:
        values = [sample[key] for sample in batch]
        first = values[0]

        if torch.is_tensor(first):
            output[key] = _pad_tensor_list(values)
        elif isinstance(first, bool):
            output[key] = torch.tensor(values, dtype=torch.bool)
        elif isinstance(first, int):
            output[key] = torch.tensor(values, dtype=torch.long)
        elif isinstance(first, float):
            output[key] = torch.tensor(values, dtype=torch.float32)
        else:
            output[key] = values

    return output

from src.fft_utils import center_crop


# -------------------------------------------------------------------------
# Reproducibility
# -------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------------------------------------------------------
# Batch preparation
# -------------------------------------------------------------------------
def prepare_batch(
    batch,
    contrast: str,
    device: torch.device,
):
    kspace_key = f"{contrast}_masked_kspace"
    target_key = f"{contrast}_target_raw"

    if kspace_key not in batch:
        raise KeyError(f"Missing batch key: {kspace_key}")

    if target_key not in batch:
        raise KeyError(f"Missing batch key: {target_key}")

    kspace = batch[kspace_key].to(
        device,
        non_blocking=True,
    )

    if not torch.is_complex(kspace):
        raise TypeError(
            f"Expected complex k-space for {kspace_key}, "
            f"but received dtype={kspace.dtype}"
        )

    # fastMRI VarNet expects the final dimension to contain real/imaginary parts.
    kspace = torch.view_as_real(kspace).float()

    mask = batch["mask"].to(
        device,
        non_blocking=True,
    ).bool()

    # Dataset mask: [B, W]
    # VarNet mask:   [B, 1, 1, W, 1]
    mask = mask[:, None, None, :, None]

    target = batch[target_key].to(
        device,
        non_blocking=True,
    )

    # Dataset target is normally [B, 1, H, W].
    if target.ndim == 4 and target.shape[1] == 1:
        target = target.squeeze(1)

    if target.ndim != 3:
        raise RuntimeError(
            f"Expected target shape [B,H,W], got {tuple(target.shape)}"
        )

    return kspace, mask, target


# -------------------------------------------------------------------------
# Target-scaled unweighted L1
# -------------------------------------------------------------------------
def l1_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise RuntimeError(
            f"Prediction/target shape mismatch: "
            f"{tuple(prediction.shape)} vs {tuple(target.shape)}"
        )

    # Use one target-derived scale for both prediction and target.
    # Do not independently normalise prediction.
    scale = target.amax(
        dim=(-2, -1),
        keepdim=True,
    ).clamp_min(1e-8)

    prediction_scaled = prediction / scale
    target_scaled = target / scale

    return torch.abs(
        prediction_scaled - target_scaled
    ).mean(dim=(-2, -1))


# -------------------------------------------------------------------------
# Validation
# -------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    contrast: str,
    device: torch.device,
):
    model.eval()

    all_losses = []
    edge_losses = []
    central_losses = []
    slice_rows = []

    for batch in loader:
        kspace, mask, target = prepare_batch(
            batch=batch,
            contrast=contrast,
            device=device,
        )

        prediction = model(kspace, mask)

        prediction = center_crop(
            prediction,
            crop_h=target.shape[-2],
            crop_w=target.shape[-1],
        )

        if not torch.isfinite(prediction).all():
            raise RuntimeError(
                "Non-finite prediction encountered during validation"
            )

        losses = l1_per_sample(
            prediction=prediction,
            target=target,
        )

        if not torch.isfinite(losses).all():
            raise RuntimeError(
                "Non-finite validation loss encountered"
            )

        batch_size = target.shape[0]

        for index in range(batch_size):
            value = float(losses[index].item())

            is_edge_value = batch["is_edge"][index]

            if torch.is_tensor(is_edge_value):
                is_edge = bool(is_edge_value.item())
            else:
                is_edge = bool(is_edge_value)

            patient_id = batch["patient_id"][index]

            if torch.is_tensor(patient_id):
                patient_id = patient_id.item()

            patient_id = str(patient_id)

            slice_idx_value = batch["slice_idx"][index]

            if torch.is_tensor(slice_idx_value):
                slice_idx = int(slice_idx_value.item())
            else:
                slice_idx = int(slice_idx_value)

            all_losses.append(value)

            if is_edge:
                edge_losses.append(value)
            else:
                central_losses.append(value)

            slice_rows.append(
                {
                    "patient_id": patient_id,
                    "slice_idx": slice_idx,
                    "is_edge": is_edge,
                    "prediction_l1": value,
                }
            )

    def safe_mean(values):
        if not values:
            return float("nan")
        return float(np.mean(values))

    results = {
        "overall_l1": safe_mean(all_losses),
        "edge_l1": safe_mean(edge_losses),
        "central_l1": safe_mean(central_losses),
        "num_slices": len(all_losses),
        "num_edge_slices": len(edge_losses),
        "num_central_slices": len(central_losses),
    }

    model.train()
    return results, slice_rows


# -------------------------------------------------------------------------
# Model
# -------------------------------------------------------------------------
def make_model(
    args,
    device: torch.device,
) -> torch.nn.Module:
    model = VarNet(
        num_cascades=args.num_cascades,
        sens_chans=args.sens_chans,
        sens_pools=args.sens_pools,
        chans=args.chans,
        pools=args.pools,
        mask_center=True,
    )

    return model.to(device)


# -------------------------------------------------------------------------
# Patient selection
# -------------------------------------------------------------------------
def select_patient_ids(
    dataset: PairedMulticoilDataset,
    limit: Optional[int],
):
    patient_ids = [
        str(row["patient_id"])
        for row in dataset.patient_rows
    ]

    # Preserve dataset order but remove any accidental duplicates.
    patient_ids = list(dict.fromkeys(patient_ids))

    if limit is None:
        return patient_ids

    if limit < 1:
        raise ValueError(
            "Patient limit must be at least 1 or omitted"
        )

    selected = patient_ids[:limit]

    if len(selected) != limit:
        raise RuntimeError(
            f"Requested {limit} patients, "
            f"but only found {len(selected)}"
        )

    return selected


# -------------------------------------------------------------------------
# CSV helpers
# -------------------------------------------------------------------------
TRAINING_LOG_COLUMNS = [
    "epoch",
    "train_l1",
    "val_overall_l1",
    "val_edge_l1",
    "val_central_l1",
    "gradient_norm_mean",
    "epoch_seconds",
    "peak_gpu_memory_gb",
    "learning_rate",
]


def initialise_training_log(
    log_path: Path,
    resume: bool,
) -> None:
    if resume and log_path.exists():
        return

    with open(log_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=TRAINING_LOG_COLUMNS,
        )
        writer.writeheader()


def append_training_log(
    log_path: Path,
    row: dict,
) -> None:
    with open(log_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=TRAINING_LOG_COLUMNS,
        )
        writer.writerow(row)


def save_slice_metrics(
    path: Path,
    slice_rows,
) -> None:
    columns = [
        "patient_id",
        "slice_idx",
        "is_edge",
        "prediction_l1",
    ]

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns,
        )
        writer.writeheader()
        writer.writerows(slice_rows)


def save_patient_ids(
    path: Path,
    patient_ids,
) -> None:
    with open(path, "w", encoding="utf-8") as file:
        for patient_id in sorted(patient_ids):
            file.write(f"{patient_id}\n")


# -------------------------------------------------------------------------
# Resume consistency checks
# -------------------------------------------------------------------------
def verify_resume_config(
    checkpoint_config: dict,
    current_config: dict,
) -> None:
    fields_to_match = [
        "metadata_csv",
        "contrast",
        "acceleration",
        "num_cascades",
        "chans",
        "sens_chans",
        "pools",
        "sens_pools",
        "batch_size",
        "seed",
        "train_patient_ids",
        "val_patient_ids",
    ]

    mismatches = []

    for field in fields_to_match:
        previous = checkpoint_config.get(field)
        current = current_config.get(field)

        if previous != current:
            mismatches.append(
                f"{field}: checkpoint={previous!r}, current={current!r}"
            )

    if mismatches:
        mismatch_text = "\n".join(mismatches)

        raise RuntimeError(
            "Resume configuration does not match the checkpoint:\n"
            f"{mismatch_text}"
        )


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Formal single-contrast fastMRI VarNet training "
            "for paired PD/PD-FS knee data."
        )
    )

    parser.add_argument(
        "--metadata_csv",
        required=True,
        help="Frozen patient-level split CSV.",
    )

    parser.add_argument(
        "--contrast",
        choices=["pd", "pdfs"],
        required=True,
    )

    parser.add_argument(
        "--acceleration",
        type=int,
        choices=[4, 6, 8],
        required=True,
    )

    parser.add_argument(
        "--num_train_patients",
        type=int,
        default=None,
        help=(
            "Optional debugging limit. "
            "Omit for all frozen training patients."
        ),
    )

    parser.add_argument(
        "--num_val_patients",
        type=int,
        default=None,
        help=(
            "Optional debugging limit. "
            "Omit for all frozen validation patients."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--num_cascades",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--chans",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--sens_chans",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--pools",
        type=int,
        default=4,
    )

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

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output_dir",
        required=True,
    )

    parser.add_argument(
        "--resume",
        default=None,
        help="Path to model_last.pt for resuming training.",
    )

    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")

    if args.learning_rate <= 0:
        raise ValueError("--learning_rate must be positive")

    metadata_path = Path(args.metadata_csv).resolve()

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata CSV does not exist: {metadata_path}"
        )

    if not metadata_path.is_file():
        raise RuntimeError(
            f"Metadata path is not a file: {metadata_path}"
        )

    args.metadata_csv = str(metadata_path)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    resume_path = None

    if args.resume is not None:
        resume_path = Path(args.resume).resolve()

        if not resume_path.exists():
            raise FileNotFoundError(
                f"Resume checkpoint does not exist: {resume_path}"
            )

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 80)
    print("Formal single-contrast VarNet training")
    print("=" * 80)
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("Metadata:", args.metadata_csv)
    print("Contrast:", args.contrast.upper())
    print("Acceleration:", args.acceleration)
    print("Epoch target:", args.epochs)
    print("Learning rate:", args.learning_rate)
    print("Output directory:", output_dir)
    print("Resume checkpoint:", resume_path)

    # ------------------------------------------------------------------
    # Load frozen splits
    # ------------------------------------------------------------------
    full_train = PairedMulticoilDataset(
        metadata_csv=args.metadata_csv,
        split="train",
        acceleration=args.acceleration,
        slices_per_patient=None,
        edge_weight=1.0,
    )

    full_val = PairedMulticoilDataset(
        metadata_csv=args.metadata_csv,
        split="val",
        acceleration=args.acceleration,
        slices_per_patient=None,
        edge_weight=1.0,
    )

    train_patient_ids = select_patient_ids(
        dataset=full_train,
        limit=args.num_train_patients,
    )

    val_patient_ids = select_patient_ids(
        dataset=full_val,
        limit=args.num_val_patients,
    )

    overlap = set(train_patient_ids) & set(val_patient_ids)

    if overlap:
        raise RuntimeError(
            "Patient leakage detected between train and validation: "
            f"{sorted(overlap)}"
        )

    train_dataset = PairedMulticoilDataset(
        metadata_csv=args.metadata_csv,
        split="train",
        acceleration=args.acceleration,
        patient_ids=train_patient_ids,
        slices_per_patient=None,
        edge_weight=1.0,
    )

    val_dataset = PairedMulticoilDataset(
        metadata_csv=args.metadata_csv,
        split="val",
        acceleration=args.acceleration,
        patient_ids=val_patient_ids,
        slices_per_patient=None,
        edge_weight=1.0,
    )

    if len(train_dataset) == 0:
        raise RuntimeError("Training dataset is empty")

    if len(val_dataset) == 0:
        raise RuntimeError("Validation dataset is empty")

    generator = torch.Generator()
    generator.manual_seed(args.seed)

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

    # ------------------------------------------------------------------
    # Model and optimiser
    # ------------------------------------------------------------------
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
            "loss": "unweighted target-scaled L1",
            "mask": "Gaussian variable-density 1D Cartesian",
            "model": "fastMRI VarNet 0.3.0",
            "all_slices_retained": True,
            "edge_weight": 1.0,
            "checkpoint_selection_metric": "validation overall L1",
        }
    )

    # Resume path itself should not be used in consistency comparison.
    consistency_config = config.copy()
    consistency_config["resume"] = None

    start_epoch = 1
    best_val = float("inf")
    best_epoch = 0
    epoch_times = []
    history = []

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    if resume_path is not None:
        checkpoint = torch.load(
            resume_path,
            map_location=device,
            weights_only=False,
        )

        required_keys = [
            "epoch",
            "model_state_dict",
            "optimizer_state_dict",
            "config",
        ]

        for key in required_keys:
            if key not in checkpoint:
                raise KeyError(
                    f"Resume checkpoint is missing key: {key}"
                )

        verify_resume_config(
            checkpoint_config=checkpoint["config"],
            current_config=consistency_config,
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

        print("=" * 80)
        print("Resume loaded successfully")
        print("Checkpoint:", resume_path)
        print("Completed epoch:", checkpoint["epoch"])
        print("Starting epoch:", start_epoch)
        print("Previous best epoch:", best_epoch)
        print("Previous best validation L1:", best_val)
        print("=" * 80)

        if start_epoch > args.epochs:
            raise RuntimeError(
                f"Checkpoint already completed epoch "
                f"{checkpoint['epoch']}, but --epochs={args.epochs}. "
                "Set --epochs to a larger total epoch target."
            )

    # Save the current run configuration.
    with open(
        output_dir / "config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(config, file, indent=2)

    training_log_path = output_dir / "training_log.csv"

    initialise_training_log(
        log_path=training_log_path,
        resume=(resume_path is not None),
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.time()
        model.train()

        train_losses = []
        gradient_norms = []

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        for batch_index, batch in enumerate(
            train_loader,
            start=1,
        ):
            kspace, mask, target = prepare_batch(
                batch=batch,
                contrast=args.contrast,
                device=device,
            )

            prediction = model(kspace, mask)

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
                prediction=prediction,
                target=target,
            ).mean()

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at epoch {epoch}, "
                    f"batch {batch_index}"
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=10.0,
            )

            if not torch.isfinite(gradient_norm):
                raise RuntimeError(
                    f"Non-finite gradient at epoch {epoch}, "
                    f"batch {batch_index}"
                )

            optimizer.step()

            train_losses.append(
                float(loss.item())
            )

            gradient_norms.append(
                float(gradient_norm.item())
            )

            if batch_index == 1 or batch_index % 50 == 0:
                print(
                    f"Epoch {epoch:02d}/{args.epochs} | "
                    f"Batch {batch_index:04d}/{len(train_loader)} | "
                    f"loss={loss.item():.6f}",
                    flush=True,
                )

        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------
        val_results, slice_rows = evaluate(
            model=model,
            loader=val_loader,
            contrast=args.contrast,
            device=device,
        )

        mean_train = float(
            np.mean(train_losses)
        )

        mean_gradient = float(
            np.mean(gradient_norms)
        )

        epoch_seconds = (
            time.time() - epoch_start_time
        )

        epoch_times.append(epoch_seconds)

        if device.type == "cuda":
            peak_gpu_memory = (
                torch.cuda.max_memory_allocated() / 1024**3
            )
        else:
            peak_gpu_memory = 0.0

        current_learning_rate = float(
            optimizer.param_groups[0]["lr"]
        )

        epoch_row = {
            "epoch": epoch,
            "train_l1": mean_train,
            "val_overall_l1": val_results["overall_l1"],
            "val_edge_l1": val_results["edge_l1"],
            "val_central_l1": val_results["central_l1"],
            "gradient_norm_mean": mean_gradient,
            "epoch_seconds": epoch_seconds,
            "peak_gpu_memory_gb": peak_gpu_memory,
            "learning_rate": current_learning_rate,
        }

        history.append(epoch_row)

        append_training_log(
            log_path=training_log_path,
            row=epoch_row,
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} completed | "
            f"train={mean_train:.6f} | "
            f"val={val_results['overall_l1']:.6f} | "
            f"edge={val_results['edge_l1']:.6f} | "
            f"central={val_results['central_l1']:.6f} | "
            f"grad={mean_gradient:.6f} | "
            f"time={epoch_seconds:.1f}s | "
            f"peak_gpu={peak_gpu_memory:.2f}GB",
            flush=True,
        )

        improved = (
            val_results["overall_l1"] < best_val
        )

        if improved:
            best_val = float(
                val_results["overall_l1"]
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

        # Always save the latest recoverable state.
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

            print(
                f"Saved best checkpoint: "
                f"epoch={epoch}, val={best_val:.6f}",
                flush=True,
            )

    # ------------------------------------------------------------------
    # Reload best checkpoint
    # ------------------------------------------------------------------
    best_checkpoint_path = output_dir / "model_best.pt"

    if not best_checkpoint_path.exists():
        raise FileNotFoundError(
            f"Best checkpoint not found: {best_checkpoint_path}"
        )

    best_checkpoint = torch.load(
        best_checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    reload_model = make_model(args, device)

    reload_model.load_state_dict(
        best_checkpoint["model_state_dict"]
    )

    reload_results, _ = evaluate(
        model=reload_model,
        loader=val_loader,
        contrast=args.contrast,
        device=device,
    )

    saved_best_value = float(
        best_checkpoint["validation_results"]["overall_l1"]
    )

    reloaded_best_value = float(
        reload_results["overall_l1"]
    )

    reload_matches = bool(
        np.isclose(
            saved_best_value,
            reloaded_best_value,
            rtol=1e-6,
            atol=1e-8,
        )
    )

    if not reload_matches:
        raise RuntimeError(
            "Reloaded checkpoint validation result does not "
            "match the saved validation result"
        )

    mean_epoch_seconds = float(
        np.mean(epoch_times)
    )

    total_training_seconds = float(
        np.sum(epoch_times)
    )

    summary = {
        "status": "completed",
        "contrast": args.contrast,
        "acceleration": args.acceleration,
        "completed_epochs": args.epochs,
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_validation_results": reload_results,
        "checkpoint_reload_matches": reload_matches,
        "mean_epoch_seconds": mean_epoch_seconds,
        "total_training_seconds": total_training_seconds,
        "total_training_hours": total_training_seconds / 3600,
        "train_patients": len(train_patient_ids),
        "validation_patients": len(val_patient_ids),
        "train_slices": len(train_dataset),
        "validation_slices": len(val_dataset),
        "parameter_count": parameter_count,
        "peak_gpu_memory_gb": max(
            row["peak_gpu_memory_gb"]
            for row in history
        ),
        "model_best": str(best_checkpoint_path),
        "model_last": str(output_dir / "model_last.pt"),
    }

    with open(
        output_dir / "final_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    print("=" * 80)
    print("Formal VarNet training completed")
    print("=" * 80)
    print("Best epoch:", summary["best_epoch"])
    print(
        "Best validation L1:",
        reload_results["overall_l1"],
    )
    print(
        "Best edge L1:",
        reload_results["edge_l1"],
    )
    print(
        "Best central L1:",
        reload_results["central_l1"],
    )
    print(
        "Checkpoint reload matches:",
        reload_matches,
    )
    print(
        "Mean epoch time:",
        f"{mean_epoch_seconds:.1f} seconds",
    )
    print(
        "Total training time:",
        f"{summary['total_training_hours']:.2f} hours",
    )
    print(
        "Peak GPU memory:",
        f"{summary['peak_gpu_memory_gb']:.2f} GB",
    )


if __name__ == "__main__":
    main()

