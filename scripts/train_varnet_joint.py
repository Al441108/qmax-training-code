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

from src.dataset_paired_multicoil import PairedMulticoilDataset
from src.fft_utils import center_crop
from src.joint_varnet import JointVarNet


class ShapeBucketBatchSampler:
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_joint_batch(batch, device: torch.device):
    required = [
        "pd_masked_kspace",
        "pdfs_masked_kspace",
        "pd_target_raw",
        "pdfs_target_raw",
        "mask",
    ]

    for key in required:
        if key not in batch:
            raise KeyError(f"Missing batch key: {key}")

    pd_kspace = batch["pd_masked_kspace"].to(device, non_blocking=True)
    pdfs_kspace = batch["pdfs_masked_kspace"].to(device, non_blocking=True)

    if not torch.is_complex(pd_kspace):
        raise TypeError(f"Expected complex PD k-space, got {pd_kspace.dtype}")

    if not torch.is_complex(pdfs_kspace):
        raise TypeError(f"Expected complex PDFS k-space, got {pdfs_kspace.dtype}")

    if pd_kspace.shape != pdfs_kspace.shape:
        raise RuntimeError(
            f"PD/PDFS k-space shape mismatch: {pd_kspace.shape} vs {pdfs_kspace.shape}"
        )

    pd_kspace = torch.view_as_real(pd_kspace).float()
    pdfs_kspace = torch.view_as_real(pdfs_kspace).float()

    mask = batch["mask"].to(device, non_blocking=True).bool()
    mask = mask[:, None, None, :, None]

    pd_target = batch["pd_target_raw"].to(device, non_blocking=True)
    pdfs_target = batch["pdfs_target_raw"].to(device, non_blocking=True)

    if pd_target.ndim == 4 and pd_target.shape[1] == 1:
        pd_target = pd_target.squeeze(1)

    if pdfs_target.ndim == 4 and pdfs_target.shape[1] == 1:
        pdfs_target = pdfs_target.squeeze(1)

    if pd_target.ndim != 3:
        raise RuntimeError(f"Expected PD target [B,H,W], got {pd_target.shape}")

    if pdfs_target.ndim != 3:
        raise RuntimeError(f"Expected PDFS target [B,H,W], got {pdfs_target.shape}")

    if pd_target.shape != pdfs_target.shape:
        raise RuntimeError(
            f"PD/PDFS target shape mismatch: {pd_target.shape} vs {pdfs_target.shape}"
        )

    return pd_kspace, pdfs_kspace, mask, pd_target, pdfs_target


def l1_per_sample(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise RuntimeError(
            f"Prediction/target shape mismatch: {prediction.shape} vs {target.shape}"
        )

    scale = target.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    prediction_scaled = prediction / scale
    target_scaled = target / scale

    return torch.abs(prediction_scaled - target_scaled).mean(dim=(-2, -1))


def safe_mean(values):
    if not values:
        return float("nan")
    return float(np.mean(values))


@torch.no_grad()
def evaluate(model, loader, device, max_val_batches=None):
    model.eval()

    pd_all = []
    pdfs_all = []
    joint_all = []

    pd_edge = []
    pd_central = []
    pdfs_edge = []
    pdfs_central = []
    joint_edge = []
    joint_central = []

    slice_rows = []

    for batch_index, batch in enumerate(loader, start=1):
        if max_val_batches is not None and batch_index > max_val_batches:
            break

        pd_kspace, pdfs_kspace, mask, pd_target, pdfs_target = prepare_joint_batch(
            batch=batch,
            device=device,
        )

        pd_pred, pdfs_pred = model(
            pd_masked_kspace=pd_kspace,
            pdfs_masked_kspace=pdfs_kspace,
            mask=mask,
        )

        pd_pred = center_crop(
            pd_pred,
            crop_h=pd_target.shape[-2],
            crop_w=pd_target.shape[-1],
        )

        pdfs_pred = center_crop(
            pdfs_pred,
            crop_h=pdfs_target.shape[-2],
            crop_w=pdfs_target.shape[-1],
        )

        if not torch.isfinite(pd_pred).all():
            raise RuntimeError("Non-finite PD prediction during validation")

        if not torch.isfinite(pdfs_pred).all():
            raise RuntimeError("Non-finite PDFS prediction during validation")

        pd_losses = l1_per_sample(pd_pred, pd_target)
        pdfs_losses = l1_per_sample(pdfs_pred, pdfs_target)
        joint_losses = pd_losses + pdfs_losses

        if not torch.isfinite(joint_losses).all():
            raise RuntimeError("Non-finite joint validation loss")

        batch_size = pd_target.shape[0]

        for i in range(batch_size):
            pd_value = float(pd_losses[i].item())
            pdfs_value = float(pdfs_losses[i].item())
            joint_value = float(joint_losses[i].item())

            is_edge_value = batch["is_edge"][i]
            if torch.is_tensor(is_edge_value):
                is_edge = bool(is_edge_value.item())
            else:
                is_edge = bool(is_edge_value)

            patient_id = str(batch["patient_id"][i])

            slice_idx_value = batch["slice_idx"][i]
            if torch.is_tensor(slice_idx_value):
                slice_idx = int(slice_idx_value.item())
            else:
                slice_idx = int(slice_idx_value)

            pd_all.append(pd_value)
            pdfs_all.append(pdfs_value)
            joint_all.append(joint_value)

            if is_edge:
                pd_edge.append(pd_value)
                pdfs_edge.append(pdfs_value)
                joint_edge.append(joint_value)
            else:
                pd_central.append(pd_value)
                pdfs_central.append(pdfs_value)
                joint_central.append(joint_value)

            slice_rows.append(
                {
                    "patient_id": patient_id,
                    "slice_idx": slice_idx,
                    "is_edge": is_edge,
                    "pd_l1": pd_value,
                    "pdfs_l1": pdfs_value,
                    "joint_l1": joint_value,
                }
            )

    results = {
        "pd_overall_l1": safe_mean(pd_all),
        "pd_edge_l1": safe_mean(pd_edge),
        "pd_central_l1": safe_mean(pd_central),
        "pdfs_overall_l1": safe_mean(pdfs_all),
        "pdfs_edge_l1": safe_mean(pdfs_edge),
        "pdfs_central_l1": safe_mean(pdfs_central),
        "joint_overall_l1": safe_mean(joint_all),
        "joint_edge_l1": safe_mean(joint_edge),
        "joint_central_l1": safe_mean(joint_central),
        "num_slices": len(joint_all),
        "num_edge_slices": len(joint_edge),
        "num_central_slices": len(joint_central),
    }

    model.train()
    return results, slice_rows


def make_model(args, device):
    model = JointVarNet(
        num_cascades=args.num_cascades,
        sens_chans=args.sens_chans,
        sens_pools=args.sens_pools,
        chans=args.chans,
        pools=args.pools,
        mask_center=True,
        cross_fusion=args.cross_fusion,
    )
    return model.to(device)


def select_patient_ids(dataset, limit: Optional[int]):
    patient_ids = [str(row["patient_id"]) for row in dataset.patient_rows]
    patient_ids = list(dict.fromkeys(patient_ids))

    if limit is None:
        return patient_ids

    if limit < 1:
        raise ValueError("Patient limit must be at least 1 or omitted")

    selected = patient_ids[:limit]

    if len(selected) != limit:
        raise RuntimeError(f"Requested {limit} patients, found {len(selected)}")

    return selected


TRAINING_LOG_COLUMNS = [
    "epoch",
    "train_joint_l1",
    "train_pd_l1",
    "train_pdfs_l1",
    "val_joint_l1",
    "val_pd_l1",
    "val_pdfs_l1",
    "val_joint_edge_l1",
    "val_joint_central_l1",
    "gradient_norm_mean",
    "epoch_seconds",
    "peak_gpu_memory_gb",
    "learning_rate",
]


def initialise_training_log(path):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=TRAINING_LOG_COLUMNS)
        writer.writeheader()


def append_training_log(path, row):
    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=TRAINING_LOG_COLUMNS)
        writer.writerow(row)


def save_slice_metrics(path, rows):
    columns = [
        "patient_id",
        "slice_idx",
        "is_edge",
        "pd_l1",
        "pdfs_l1",
        "joint_l1",
    ]

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_patient_ids(path, patient_ids):
    with open(path, "w", encoding="utf-8") as file:
        for patient_id in sorted(patient_ids):
            file.write(f"{patient_id}\n")




def verify_resume_config(checkpoint_config: dict, current_config: dict) -> None:
    fields_to_match = [
        "metadata_csv",
        "acceleration",
        "num_cascades",
        "chans",
        "sens_chans",
        "pools",
        "sens_pools",
        "batch_size",
        "seed",
        "cross_fusion",
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
        raise RuntimeError(
            "Resume configuration does not match checkpoint:\n"
            + "\n".join(mismatches)
        )

def main():
    parser = argparse.ArgumentParser(
        description="M1 joint PD/PD-FS VarNet training."
    )

    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--acceleration", type=int, choices=[4, 6, 8], required=True)
    parser.add_argument("--num_train_patients", type=int, default=None)
    parser.add_argument("--num_val_patients", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--num_cascades", type=int, default=12)
    parser.add_argument("--chans", type=int, default=18)
    parser.add_argument("--sens_chans", type=int, default=8)
    parser.add_argument("--pools", type=int, default=4)
    parser.add_argument("--sens_pools", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cross_fusion", choices=["off", "concat"], default="concat")
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument(
        "--resume",
        default=None,
        help="Path to model_last.pt for resuming joint training.",
    )

    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")

    if args.learning_rate <= 0:
        raise ValueError("--learning_rate must be positive")

    metadata_path = Path(args.metadata_csv).resolve()
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata CSV does not exist: {metadata_path}")

    args.metadata_csv = str(metadata_path)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    resume_path = None
    if args.resume is not None:
        resume_path = Path(args.resume).resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("M1 joint PD/PD-FS VarNet training")
    print("=" * 80)
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Metadata:", args.metadata_csv)
    print("Acceleration:", args.acceleration)
    print("Cross fusion:", args.cross_fusion)
    print("Epochs:", args.epochs)
    print("Max train batches:", args.max_train_batches)
    print("Max val batches:", args.max_val_batches)
    print("Output directory:", output_dir)
    print("Resume checkpoint:", resume_path)

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

    train_patient_ids = select_patient_ids(full_train, args.num_train_patients)
    val_patient_ids = select_patient_ids(full_val, args.num_val_patients)

    overlap = set(train_patient_ids) & set(val_patient_ids)
    if overlap:
        raise RuntimeError(f"Patient leakage detected: {sorted(overlap)}")

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

    save_patient_ids(output_dir / "train_patient_ids.txt", train_patient_ids)
    save_patient_ids(output_dir / "val_patient_ids.txt", val_patient_ids)

    model = make_model(args, device)

    parameter_count = sum(p.numel() for p in model.parameters())
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
            "resume": str(resume_path) if resume_path is not None else None,
            "train_patient_ids": train_patient_ids,
            "val_patient_ids": val_patient_ids,
            "train_patients": len(train_patient_ids),
            "val_patients": len(val_patient_ids),
            "train_slices": len(train_dataset),
            "val_slices": len(val_dataset),
            "parameter_count": parameter_count,
            "loss": "pd_l1 + pdfs_l1, target-scaled per contrast",
            "mask": "Gaussian variable-density 1D Cartesian, shared between PD and PD-FS",
            "model": "M1 JointVarNet: independent DC + coupled regulariser",
            "checkpoint_selection_metric": "validation joint overall L1",
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

        required_keys = [
            "epoch",
            "model_state_dict",
            "optimizer_state_dict",
            "config",
        ]

        for key in required_keys:
            if key not in checkpoint:
                raise KeyError(f"Resume checkpoint is missing key: {key}")

        verify_resume_config(
            checkpoint_config=checkpoint["config"],
            current_config=consistency_config,
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = int(checkpoint["epoch"]) + 1
        best_val = float(checkpoint.get("best_val", float("inf")))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        epoch_times = list(checkpoint.get("epoch_times", []))
        history = list(checkpoint.get("history", []))

        print("=" * 80)
        print("Resume loaded successfully")
        print("Checkpoint:", resume_path)
        print("Completed epoch:", checkpoint["epoch"])
        print("Starting epoch:", start_epoch)
        print("Previous best epoch:", best_epoch)
        print("Previous best validation joint L1:", best_val)
        print("=" * 80)

        if start_epoch > args.epochs:
            raise RuntimeError(
                f"Checkpoint already completed epoch {checkpoint['epoch']}, "
                f"but --epochs={args.epochs}. Set --epochs to a larger total epoch target."
            )

    with open(output_dir / "config.json", "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    training_log_path = output_dir / "training_log.csv"
    if resume_path is None:
        initialise_training_log(training_log_path)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        model.train()

        train_joint_losses = []
        train_pd_losses = []
        train_pdfs_losses = []
        gradient_norms = []

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        for batch_index, batch in enumerate(train_loader, start=1):
            if args.max_train_batches is not None and batch_index > args.max_train_batches:
                break

            pd_kspace, pdfs_kspace, mask, pd_target, pdfs_target = prepare_joint_batch(
                batch=batch,
                device=device,
            )

            pd_pred, pdfs_pred = model(
                pd_masked_kspace=pd_kspace,
                pdfs_masked_kspace=pdfs_kspace,
                mask=mask,
            )

            pd_pred = center_crop(
                pd_pred,
                crop_h=pd_target.shape[-2],
                crop_w=pd_target.shape[-1],
            )

            pdfs_pred = center_crop(
                pdfs_pred,
                crop_h=pdfs_target.shape[-2],
                crop_w=pdfs_target.shape[-1],
            )

            if not torch.isfinite(pd_pred).all():
                raise RuntimeError(f"Non-finite PD prediction at epoch {epoch}, batch {batch_index}")

            if not torch.isfinite(pdfs_pred).all():
                raise RuntimeError(f"Non-finite PDFS prediction at epoch {epoch}, batch {batch_index}")

            pd_loss = l1_per_sample(pd_pred, pd_target).mean()
            pdfs_loss = l1_per_sample(pdfs_pred, pdfs_target).mean()
            loss = pd_loss + pdfs_loss

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite joint loss at epoch {epoch}, batch {batch_index}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=10.0,
            )

            if not torch.isfinite(gradient_norm):
                raise RuntimeError(f"Non-finite gradient at epoch {epoch}, batch {batch_index}")

            optimizer.step()

            train_joint_losses.append(float(loss.item()))
            train_pd_losses.append(float(pd_loss.item()))
            train_pdfs_losses.append(float(pdfs_loss.item()))
            gradient_norms.append(float(gradient_norm.item()))

            if batch_index == 1 or batch_index % 10 == 0:
                print(
                    f"Epoch {epoch:02d}/{args.epochs} | "
                    f"Batch {batch_index:04d}/{len(train_loader)} | "
                    f"joint={loss.item():.6f} | "
                    f"pd={pd_loss.item():.6f} | "
                    f"pdfs={pdfs_loss.item():.6f}",
                    flush=True,
                )

        val_results, slice_rows = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            max_val_batches=args.max_val_batches,
        )

        epoch_seconds = time.time() - epoch_start
        epoch_times.append(epoch_seconds)

        if device.type == "cuda":
            peak_gpu_memory = torch.cuda.max_memory_allocated() / 1024**3
        else:
            peak_gpu_memory = 0.0

        current_lr = float(optimizer.param_groups[0]["lr"])

        epoch_row = {
            "epoch": epoch,
            "train_joint_l1": safe_mean(train_joint_losses),
            "train_pd_l1": safe_mean(train_pd_losses),
            "train_pdfs_l1": safe_mean(train_pdfs_losses),
            "val_joint_l1": val_results["joint_overall_l1"],
            "val_pd_l1": val_results["pd_overall_l1"],
            "val_pdfs_l1": val_results["pdfs_overall_l1"],
            "val_joint_edge_l1": val_results["joint_edge_l1"],
            "val_joint_central_l1": val_results["joint_central_l1"],
            "gradient_norm_mean": safe_mean(gradient_norms),
            "epoch_seconds": epoch_seconds,
            "peak_gpu_memory_gb": peak_gpu_memory,
            "learning_rate": current_lr,
        }

        history.append(epoch_row)
        append_training_log(training_log_path, epoch_row)

        print(
            f"Epoch {epoch:02d}/{args.epochs} completed | "
            f"train_joint={epoch_row['train_joint_l1']:.6f} | "
            f"val_joint={val_results['joint_overall_l1']:.6f} | "
            f"val_pd={val_results['pd_overall_l1']:.6f} | "
            f"val_pdfs={val_results['pdfs_overall_l1']:.6f} | "
            f"time={epoch_seconds:.1f}s | "
            f"peak_gpu={peak_gpu_memory:.2f}GB",
            flush=True,
        )

        improved = val_results["joint_overall_l1"] < best_val

        if improved:
            best_val = float(val_results["joint_overall_l1"])
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

        torch.save(checkpoint, output_dir / "model_last.pt")

        if improved:
            torch.save(checkpoint, output_dir / "model_best.pt")
            save_slice_metrics(
                output_dir / "best_val_per_slice_metrics.csv",
                slice_rows,
            )
            print(
                f"Saved best checkpoint: epoch={epoch}, val_joint={best_val:.6f}",
                flush=True,
            )

    best_checkpoint_path = output_dir / "model_best.pt"

    if not best_checkpoint_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {best_checkpoint_path}")

    best_checkpoint = torch.load(
        best_checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    reload_model = make_model(args, device)
    reload_model.load_state_dict(best_checkpoint["model_state_dict"])

    reload_results, _ = evaluate(
        model=reload_model,
        loader=val_loader,
        device=device,
        max_val_batches=args.max_val_batches,
    )

    saved_best_value = float(
        best_checkpoint["validation_results"]["joint_overall_l1"]
    )

    reloaded_best_value = float(
        reload_results["joint_overall_l1"]
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
            "Reloaded checkpoint validation result does not match saved validation result"
        )

    total_seconds = float(np.sum(epoch_times))
    mean_epoch_seconds = float(np.mean(epoch_times))

    summary = {
        "status": "completed",
        "acceleration": args.acceleration,
        "cross_fusion": args.cross_fusion,
        "completed_epochs": args.epochs,
        "best_epoch": int(best_epoch),
        "best_validation_results": reload_results,
        "checkpoint_reload_matches": reload_matches,
        "mean_epoch_seconds": mean_epoch_seconds,
        "total_training_seconds": total_seconds,
        "total_training_hours": total_seconds / 3600,
        "train_patients": len(train_patient_ids),
        "validation_patients": len(val_patient_ids),
        "train_slices": len(train_dataset),
        "validation_slices": len(val_dataset),
        "parameter_count": parameter_count,
        "peak_gpu_memory_gb": max(row["peak_gpu_memory_gb"] for row in history),
        "model_best": str(best_checkpoint_path),
        "model_last": str(output_dir / "model_last.pt"),
    }

    with open(output_dir / "final_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("=" * 80)
    print("M1 joint VarNet training completed")
    print("=" * 80)
    print("Best epoch:", summary["best_epoch"])
    print("Best validation joint L1:", reload_results["joint_overall_l1"])
    print("Best validation PD L1:", reload_results["pd_overall_l1"])
    print("Best validation PDFS L1:", reload_results["pdfs_overall_l1"])
    print("Checkpoint reload matches:", reload_matches)
    print("Mean epoch time:", f"{mean_epoch_seconds:.1f} seconds")
    print("Total training time:", f"{summary['total_training_hours']:.2f} hours")
    print("Peak GPU memory:", f"{summary['peak_gpu_memory_gb']:.2f} GB")


if __name__ == "__main__":
    main()
