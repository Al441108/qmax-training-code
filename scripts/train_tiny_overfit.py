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

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

from src.dataset_paired_multicoil import PairedMulticoilDataset
from src.model_unet import SmallUNet


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Reproducibility is more important than speed for this tiny pilot.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_per_sample_l1(prediction, target):
    """
    Return one unweighted L1 value per sample.

    prediction / target:
        [batch, channel, height, width]

    output:
        [batch]
    """
    return torch.abs(
        prediction - target
    ).mean(dim=(1, 2, 3))


def aggregate_loss(per_sample_loss, sample_weights, weighted):
    """
    Optimisation loss.

    If weighted=False:
        ordinary average across samples.

    If weighted=True:
        sum(loss_i * weight_i) / sum(weight_i)
    """
    if weighted:
        return (
            per_sample_loss * sample_weights
        ).sum() / sample_weights.sum().clamp_min(1e-8)

    return per_sample_loss.mean()


@torch.no_grad()
def evaluate_on_tiny_set(model, loader, device, contrast):
    """
    Evaluate the model on the same tiny subset used for training.

    This is intentional: the purpose is an overfitting sanity check,
    not validation of generalisation.
    """
    model.eval()

    all_losses = []
    edge_losses = []
    central_losses = []

    input_losses = []
    predicted_losses = []

    per_slice_rows = []

    input_key = f"{contrast}_input"
    target_key = f"{contrast}_target"

    for batch in loader:
        x = batch[input_key].to(
            device,
            non_blocking=True,
        )
        y = batch[target_key].to(
            device,
            non_blocking=True,
        )

        prediction = model(x)

        pred_per_sample = compute_per_sample_l1(
            prediction,
            y,
        )
        input_per_sample = compute_per_sample_l1(
            x,
            y,
        )

        patient_ids = batch["patient_id"]
        slice_indices = batch["slice_idx"]
        edge_flags = batch["is_edge"]

        for sample_index in range(x.shape[0]):
            pred_loss = float(
                pred_per_sample[sample_index].item()
            )
            input_loss = float(
                input_per_sample[sample_index].item()
            )
            is_edge = bool(
                edge_flags[sample_index].item()
            )

            all_losses.append(pred_loss)
            input_losses.append(input_loss)
            predicted_losses.append(pred_loss)

            if is_edge:
                edge_losses.append(pred_loss)
            else:
                central_losses.append(pred_loss)

            per_slice_rows.append(
                {
                    "patient_id": patient_ids[sample_index],
                    "slice_idx": int(
                        slice_indices[sample_index].item()
                    ),
                    "is_edge": is_edge,
                    "input_l1": input_loss,
                    "prediction_l1": pred_loss,
                    "l1_improvement": input_loss - pred_loss,
                }
            )

    def safe_mean(values):
        return float(np.mean(values)) if values else float("nan")

    results = {
        "overall_l1": safe_mean(all_losses),
        "edge_l1": safe_mean(edge_losses),
        "central_l1": safe_mean(central_losses),
        "input_l1": safe_mean(input_losses),
        "prediction_l1": safe_mean(predicted_losses),
        "l1_improvement": (
            safe_mean(input_losses)
            - safe_mean(predicted_losses)
        ),
    }

    model.train()

    return results, per_slice_rows


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    config,
    selected_patient_ids,
    evaluation_results,
):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
            "selected_patient_ids": selected_patient_ids,
            "evaluation_results": evaluation_results,
        },
        path,
    )


def verify_checkpoint(
    checkpoint_path,
    device,
    model_base_channels,
):
    """
    Reload a checkpoint into a new model instance to verify that
    checkpoint saving is valid.
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    reloaded_model = SmallUNet(
        in_ch=1,
        out_ch=1,
        base_ch=model_base_channels,
    ).to(device)

    reloaded_model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    reloaded_model.eval()

    return checkpoint, reloaded_model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--metadata_csv",
        required=True,
    )
    parser.add_argument(
        "--contrast",
        default="pd",
        choices=["pd", "pdfs"],
    )
    parser.add_argument(
        "--acceleration",
        type=int,
        default=4,
        choices=[4, 6, 8],
    )
    parser.add_argument(
        "--num_patients",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--slices_per_patient",
        type=int,
        default=9,
    )
    parser.add_argument(
        "--edge_weight",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--weighted",
        action="store_true",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--base_channels",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--output_dir",
        required=True,
    )

    args = parser.parse_args()

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 80)
    print("Tiny overfitting experiment")
    print("=" * 80)
    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print("Contrast:", args.contrast.upper())
    print("Acceleration:", args.acceleration)
    print("Weighted loss:", args.weighted)
    print("Epochs:", args.epochs)
    print("Batch size:", args.batch_size)
    print("Learning rate:", args.learning_rate)
    print("Seed:", args.seed)
    print("Output:", output_dir)
    print("=" * 80)

    # First build the split-level dataset so patient selection is explicit.
    all_train_dataset = PairedMulticoilDataset(
        metadata_csv=args.metadata_csv,
        split="train",
        acceleration=args.acceleration,
        slices_per_patient=args.slices_per_patient,
        edge_weight=args.edge_weight,
    )

    selected_patient_ids = [
        row["patient_id"]
        for row in all_train_dataset.patient_rows[
            :args.num_patients
        ]
    ]

    tiny_dataset = PairedMulticoilDataset(
        metadata_csv=args.metadata_csv,
        split="train",
        acceleration=args.acceleration,
        patient_ids=selected_patient_ids,
        slices_per_patient=args.slices_per_patient,
        edge_weight=args.edge_weight,
    )

    # Separate generators ensure deterministic training order.
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)

    train_loader = DataLoader(
        tiny_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        generator=train_generator,
    )

    eval_loader = DataLoader(
        tiny_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print("Selected patients:")
    for patient_id in selected_patient_ids:
        print("  ", patient_id)

    print("Tiny samples:", len(tiny_dataset))

    for row in tiny_dataset.patient_rows:
        print(
            f"patient={row['patient_id']} | "
            f"n_common={row['n_common']} | "
            f"slices={row['selected_slices']}"
        )

    model = SmallUNet(
        in_ch=1,
        out_ch=1,
        base_ch=args.base_channels,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    config = {
        "metadata_csv": args.metadata_csv,
        "contrast": args.contrast,
        "acceleration": args.acceleration,
        "num_patients": args.num_patients,
        "slices_per_patient": args.slices_per_patient,
        "edge_weight": args.edge_weight,
        "weighted": args.weighted,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "base_channels": args.base_channels,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "selected_patient_ids": selected_patient_ids,
        "model": "SmallUNet",
        "loss": "per-sample L1",
        "mask": "Gaussian VD",
    }

    with open(
        output_dir / "config.json",
        "w",
        encoding="utf-8",
    ) as config_file:
        json.dump(
            config,
            config_file,
            indent=2,
        )

    log_path = output_dir / "training_log.csv"

    with open(
        log_path,
        "w",
        newline="",
    ) as log_file:
        writer = csv.writer(log_file)
        writer.writerow(
            [
                "epoch",
                "optimisation_loss",
                "unweighted_overall_l1",
                "edge_l1",
                "central_l1",
                "input_l1",
                "prediction_l1",
                "l1_improvement",
                "seconds",
            ]
        )

    input_key = f"{args.contrast}_input"
    target_key = f"{args.contrast}_target"

    best_overall_l1 = float("inf")
    initial_results = None

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        model.train()

        optimisation_loss_sum = 0.0
        batches_seen = 0

        for batch in train_loader:
            x = batch[input_key].to(
                device,
                non_blocking=True,
            )
            y = batch[target_key].to(
                device,
                non_blocking=True,
            )

            sample_weights = batch[
                "sample_weight"
            ].to(
                device,
                non_blocking=True,
            )

            prediction = model(x)

            per_sample_loss = compute_per_sample_l1(
                prediction,
                y,
            )

            loss = aggregate_loss(
                per_sample_loss=per_sample_loss,
                sample_weights=sample_weights,
                weighted=args.weighted,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss detected at epoch {epoch}."
                )

            optimizer.zero_grad(
                set_to_none=True
            )
            loss.backward()

            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=10.0,
            )

            if not torch.isfinite(gradient_norm):
                raise RuntimeError(
                    f"Non-finite gradient norm at epoch {epoch}."
                )

            optimizer.step()

            optimisation_loss_sum += float(
                loss.item()
            )
            batches_seen += 1

        mean_optimisation_loss = (
            optimisation_loss_sum
            / max(1, batches_seen)
        )

        evaluation_results, per_slice_rows = \
            evaluate_on_tiny_set(
                model=model,
                loader=eval_loader,
                device=device,
                contrast=args.contrast,
            )

        if initial_results is None:
            initial_results = evaluation_results.copy()

        elapsed_seconds = time.time() - start_time

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"optim={mean_optimisation_loss:.6f} | "
            f"overall={evaluation_results['overall_l1']:.6f} | "
            f"edge={evaluation_results['edge_l1']:.6f} | "
            f"central={evaluation_results['central_l1']:.6f} | "
            f"input={evaluation_results['input_l1']:.6f} | "
            f"improvement={evaluation_results['l1_improvement']:.6f} | "
            f"time={elapsed_seconds:.1f}s"
        )

        with open(
            log_path,
            "a",
            newline="",
        ) as log_file:
            writer = csv.writer(log_file)
            writer.writerow(
                [
                    epoch,
                    mean_optimisation_loss,
                    evaluation_results["overall_l1"],
                    evaluation_results["edge_l1"],
                    evaluation_results["central_l1"],
                    evaluation_results["input_l1"],
                    evaluation_results["prediction_l1"],
                    evaluation_results["l1_improvement"],
                    elapsed_seconds,
                ]
            )

        save_checkpoint(
            path=output_dir / "model_last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            config=config,
            selected_patient_ids=selected_patient_ids,
            evaluation_results=evaluation_results,
        )

        if (
            evaluation_results["overall_l1"]
            < best_overall_l1
        ):
            best_overall_l1 = \
                evaluation_results["overall_l1"]

            save_checkpoint(
                path=output_dir / "model_best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=config,
                selected_patient_ids=selected_patient_ids,
                evaluation_results=evaluation_results,
            )

            with open(
                output_dir / "best_per_slice_metrics.csv",
                "w",
                newline="",
            ) as metrics_file:
                writer = csv.DictWriter(
                    metrics_file,
                    fieldnames=[
                        "patient_id",
                        "slice_idx",
                        "is_edge",
                        "input_l1",
                        "prediction_l1",
                        "l1_improvement",
                    ],
                )
                writer.writeheader()
                writer.writerows(per_slice_rows)

    checkpoint, reloaded_model = verify_checkpoint(
        checkpoint_path=output_dir / "model_best.pt",
        device=device,
        model_base_channels=args.base_channels,
    )

    reload_results, _ = evaluate_on_tiny_set(
        model=reloaded_model,
        loader=eval_loader,
        device=device,
        contrast=args.contrast,
    )

    final_summary = {
        "initial_epoch_results": initial_results,
        "best_checkpoint_epoch": checkpoint["epoch"],
        "best_checkpoint_saved_results": checkpoint[
            "evaluation_results"
        ],
        "reloaded_checkpoint_results": reload_results,
        "checkpoint_reload_matches": bool(
            np.isclose(
                checkpoint["evaluation_results"]["overall_l1"],
                reload_results["overall_l1"],
                rtol=1e-6,
                atol=1e-8,
            )
        ),
    }

    with open(
        output_dir / "final_summary.json",
        "w",
        encoding="utf-8",
    ) as summary_file:
        json.dump(
            final_summary,
            summary_file,
            indent=2,
        )

    print("\n" + "=" * 80)
    print("Training completed")
    print("=" * 80)
    print("Best epoch:", checkpoint["epoch"])
    print(
        "Reloaded overall L1:",
        reload_results["overall_l1"],
    )
    print(
        "Checkpoint reload matches:",
        final_summary["checkpoint_reload_matches"],
    )
    print("Log:", log_path)
    print(
        "Best checkpoint:",
        output_dir / "model_best.pt",
    )
    print(
        "Final summary:",
        output_dir / "final_summary.json",
    )


if __name__ == "__main__":
    main()
