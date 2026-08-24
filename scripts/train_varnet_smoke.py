import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from fastmri.models import VarNet
from src.dataset_paired_multicoil import PairedMulticoilDataset
from src.fft_utils import center_crop


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_batch(batch, contrast: str, device: torch.device):
    kspace = batch[f"{contrast}_masked_kspace"].to(device, non_blocking=True)
    if not torch.is_complex(kspace):
        raise TypeError(f"Expected complex k-space, got {kspace.dtype}")

    kspace = torch.view_as_real(kspace).float()

    mask = batch["mask"].to(device, non_blocking=True).bool()
    mask = mask[:, None, None, :, None]

    target = batch[f"{contrast}_target_raw"].to(
        device, non_blocking=True
    ).squeeze(1)

    return kspace, mask, target


def l1_per_sample(prediction: torch.Tensor, target: torch.Tensor):
    scale = target.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    prediction = prediction / scale
    target = target / scale
    return torch.abs(prediction - target).mean(dim=(-2, -1))


@torch.no_grad()
def evaluate(model, loader, contrast: str, device: torch.device):
    model.eval()

    all_losses = []
    edge_losses = []
    central_losses = []
    slice_rows = []

    for batch in loader:
        kspace, mask, target = prepare_batch(batch, contrast, device)
        prediction = model(kspace, mask)
        prediction = center_crop(
            prediction,
            crop_h=target.shape[-2],
            crop_w=target.shape[-1],
        )

        losses = l1_per_sample(prediction, target)

        for index in range(target.shape[0]):
            value = float(losses[index].item())
            is_edge = bool(batch["is_edge"][index].item())

            all_losses.append(value)
            if is_edge:
                edge_losses.append(value)
            else:
                central_losses.append(value)

            slice_rows.append(
                {
                    "patient_id": batch["patient_id"][index],
                    "slice_idx": int(batch["slice_idx"][index].item()),
                    "is_edge": is_edge,
                    "prediction_l1": value,
                }
            )

    def safe_mean(values):
        return float(np.mean(values)) if values else float("nan")

    results = {
        "overall_l1": safe_mean(all_losses),
        "edge_l1": safe_mean(edge_losses),
        "central_l1": safe_mean(central_losses),
        "num_slices": len(all_losses),
    }

    model.train()
    return results, slice_rows


def make_model(args, device: torch.device):
    return VarNet(
        num_cascades=args.num_cascades,
        sens_chans=args.sens_chans,
        sens_pools=args.sens_pools,
        chans=args.chans,
        pools=args.pools,
    ).to(device)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--contrast", choices=["pd", "pdfs"], default="pd")
    parser.add_argument("--acceleration", type=int, choices=[4, 6, 8], default=4)
    parser.add_argument("--num_train_patients", type=int, default=10)
    parser.add_argument("--num_val_patients", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--num_cascades", type=int, default=6)
    parser.add_argument("--chans", type=int, default=12)
    parser.add_argument("--sens_chans", type=int, default=8)
    parser.add_argument("--pools", type=int, default=4)
    parser.add_argument("--sens_pools", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()

    if args.num_train_patients < 1:
        raise ValueError("--num_train_patients must be at least 1")
    if args.num_val_patients < 1:
        raise ValueError("--num_val_patients must be at least 1")

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("VarNet medium-scale smoke test")
    print("=" * 80)
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Contrast:", args.contrast.upper())
    print("Acceleration:", args.acceleration)
    print("Epochs:", args.epochs)
    print("Learning rate:", args.learning_rate)

    full_train = PairedMulticoilDataset(
        metadata_csv=args.metadata_csv,
        split="train",
        acceleration=args.acceleration,
        slices_per_patient=None,
        edge_weight=1.0,
    )

    train_patient_ids = [
        row["patient_id"]
        for row in full_train.patient_rows[: args.num_train_patients]
    ]

    full_val = PairedMulticoilDataset(
        metadata_csv=args.metadata_csv,
        split="val",
        acceleration=args.acceleration,
        slices_per_patient=None,
        edge_weight=1.0,
    )

    val_patient_ids = [
        row["patient_id"]
        for row in full_val.patient_rows[: args.num_val_patients]
    ]

    if len(train_patient_ids) != args.num_train_patients:
        raise RuntimeError(
            f"Requested {args.num_train_patients} train patients, "
            f"but found {len(train_patient_ids)}"
        )

    if len(val_patient_ids) != args.num_val_patients:
        raise RuntimeError(
            f"Requested {args.num_val_patients} val patients, "
            f"but found {len(val_patient_ids)}"
        )

    overlap = set(train_patient_ids) & set(val_patient_ids)
    if overlap:
        raise RuntimeError(f"Patient leakage detected: {overlap}")

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

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
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

    model = make_model(args, device)
    parameter_count = sum(p.numel() for p in model.parameters())
    print("Model parameters:", parameter_count)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    config = vars(args).copy()
    config.update(
        {
            "train_patient_ids": train_patient_ids,
            "val_patient_ids": val_patient_ids,
            "train_slices": len(train_dataset),
            "val_slices": len(val_dataset),
            "parameter_count": parameter_count,
            "loss": "target-max-normalised L1",
            "mask": "Gaussian VD",
            "model": "fastMRI VarNet 0.3.0",
        }
    )

    with open(output_dir / "config.json", "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    log_path = output_dir / "training_log.csv"
    with open(log_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "epoch",
                "train_l1",
                "val_overall_l1",
                "val_edge_l1",
                "val_central_l1",
                "gradient_norm_mean",
                "epoch_seconds",
                "peak_gpu_memory_gb",
            ]
        )

    best_val = float("inf")
    epoch_times = []

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        model.train()

        train_losses = []
        gradient_norms = []

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        for batch_index, batch in enumerate(train_loader, start=1):
            kspace, mask, target = prepare_batch(batch, args.contrast, device)

            prediction = model(kspace, mask)
            prediction = center_crop(
                prediction,
                crop_h=target.shape[-2],
                crop_w=target.shape[-1],
            )

            loss = l1_per_sample(prediction, target).mean()

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at epoch {epoch}, batch {batch_index}"
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=10.0
            )

            if not torch.isfinite(gradient_norm):
                raise RuntimeError(
                    f"Non-finite gradient at epoch {epoch}, batch {batch_index}"
                )

            optimizer.step()

            train_losses.append(float(loss.item()))
            gradient_norms.append(float(gradient_norm.item()))

            if batch_index == 1 or batch_index % 50 == 0:
                print(
                    f"Epoch {epoch:02d}/{args.epochs} | "
                    f"Batch {batch_index:04d}/{len(train_loader)} | "
                    f"loss={loss.item():.6f}",
                    flush=True,
                )

        val_results, slice_rows = evaluate(
            model, val_loader, args.contrast, device
        )

        mean_train = float(np.mean(train_losses))
        mean_gradient = float(np.mean(gradient_norms))
        epoch_seconds = time.time() - start_time
        epoch_times.append(epoch_seconds)

        if device.type == "cuda":
            peak_gpu_memory = torch.cuda.max_memory_allocated() / 1024**3
        else:
            peak_gpu_memory = 0.0

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

        with open(log_path, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    epoch,
                    mean_train,
                    val_results["overall_l1"],
                    val_results["edge_l1"],
                    val_results["central_l1"],
                    mean_gradient,
                    epoch_seconds,
                    peak_gpu_memory,
                ]
            )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_results": val_results,
            "config": config,
        }

        torch.save(checkpoint, output_dir / "model_last.pt")

        if val_results["overall_l1"] < best_val:
            best_val = val_results["overall_l1"]
            torch.save(checkpoint, output_dir / "model_best.pt")

            with open(
                output_dir / "best_val_per_slice_metrics.csv",
                "w",
                newline="",
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "patient_id",
                        "slice_idx",
                        "is_edge",
                        "prediction_l1",
                    ],
                )
                writer.writeheader()
                writer.writerows(slice_rows)

            print(
                f"Saved best checkpoint: epoch={epoch}, val={best_val:.6f}",
                flush=True,
            )

    best_checkpoint = torch.load(
        output_dir / "model_best.pt",
        map_location=device,
        weights_only=False,
    )

    reload_model = make_model(args, device)
    reload_model.load_state_dict(best_checkpoint["model_state_dict"])

    reload_results, _ = evaluate(
        reload_model, val_loader, args.contrast, device
    )

    reload_matches = bool(
        np.isclose(
            best_checkpoint["validation_results"]["overall_l1"],
            reload_results["overall_l1"],
            rtol=1e-6,
            atol=1e-8,
        )
    )

    mean_epoch_seconds = float(np.mean(epoch_times))
    estimated_full_epoch_seconds = (
        mean_epoch_seconds * 110 / args.num_train_patients
    )
    estimated_30_epoch_hours = (
        estimated_full_epoch_seconds * 30 / 3600
    )

    summary = {
        "best_epoch": best_checkpoint["epoch"],
        "best_validation_results": reload_results,
        "checkpoint_reload_matches": reload_matches,
        "mean_smoke_epoch_seconds": mean_epoch_seconds,
        "estimated_110_patient_epoch_hours": (
            estimated_full_epoch_seconds / 3600
        ),
        "estimated_110_patient_30_epoch_hours": estimated_30_epoch_hours,
        "train_patients": len(train_patient_ids),
        "validation_patients": len(val_patient_ids),
        "train_slices": len(train_dataset),
        "validation_slices": len(val_dataset),
        "parameter_count": parameter_count,
    }

    with open(
        output_dir / "final_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    print("=" * 80)
    print("VarNet smoke test completed")
    print("=" * 80)
    print("Best epoch:", best_checkpoint["epoch"])
    print("Best validation L1:", reload_results["overall_l1"])
    print("Checkpoint reload matches:", reload_matches)
    print("Mean epoch time:", f"{mean_epoch_seconds:.1f} seconds")
    print(
        "Estimated 110-patient, 30-epoch time:",
        f"{estimated_30_epoch_hours:.2f} hours",
    )


if __name__ == "__main__":
    main()
