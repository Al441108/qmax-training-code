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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_varnet_batch(batch, contrast, device):
    kspace_key = f"{contrast}_masked_kspace"
    target_key = f"{contrast}_target_raw"

    native_kspace = batch[kspace_key].to(
        device,
        non_blocking=True,
    )

    if not torch.is_complex(native_kspace):
        raise TypeError(
            f"Expected complex k-space, got {native_kspace.dtype}"
        )

    masked_kspace = torch.view_as_real(
        native_kspace
    ).float()

    mask = batch["mask"].to(
        device,
        non_blocking=True,
    ).bool()

    # [B, W] -> [B, 1, 1, W, 1]
    mask = mask[:, None, None, :, None]

    # [B, 1, H, W] -> [B, H, W]
    target = batch[target_key].to(
        device,
        non_blocking=True,
    ).squeeze(1)

    return masked_kspace, mask, target


def normalised_l1_per_sample(prediction, target):
    """
    Prediction and target are [B, H, W].

    The prediction and target share the same target-derived scale.
    """
    scale = target.amax(
        dim=(-2, -1),
        keepdim=True,
    ).clamp_min(1e-8)

    prediction_norm = prediction / scale
    target_norm = target / scale

    return torch.abs(
        prediction_norm - target_norm
    ).mean(dim=(-2, -1))


@torch.no_grad()
def evaluate(model, loader, contrast, device):
    model.eval()

    all_losses = []
    edge_losses = []
    central_losses = []

    per_slice_rows = []

    for batch in loader:
        masked_kspace, mask, target = prepare_varnet_batch(
            batch=batch,
            contrast=contrast,
            device=device,
        )

        output = model(
            masked_kspace,
            mask,
        )

        output_crop = center_crop(
            output,
            crop_h=target.shape[-2],
            crop_w=target.shape[-1],
        )

        per_sample_loss = normalised_l1_per_sample(
            output_crop,
            target,
        )

        for index in range(target.shape[0]):
            loss_value = float(
                per_sample_loss[index].item()
            )
            is_edge = bool(
                batch["is_edge"][index].item()
            )

            all_losses.append(loss_value)

            if is_edge:
                edge_losses.append(loss_value)
            else:
                central_losses.append(loss_value)

            per_slice_rows.append(
                {
                    "patient_id": batch["patient_id"][index],
                    "slice_idx": int(
                        batch["slice_idx"][index].item()
                    ),
                    "is_edge": is_edge,
                    "prediction_l1": loss_value,
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
    results,
):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
            "selected_patient_ids": selected_patient_ids,
            "evaluation_results": results,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--metadata_csv", required=True)

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
        "--epochs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--num_cascades",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--chans",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--sens_chans",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--pools",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--sens_pools",
        type=int,
        default=2,
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
    print("VarNet tiny overfit")
    print("=" * 80)
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("Contrast:", args.contrast.upper())
    print("Acceleration:", args.acceleration)
    print("Epochs:", args.epochs)
    print("Learning rate:", args.learning_rate)
    print("Output:", output_dir)
    print("=" * 80)

    full_dataset = PairedMulticoilDataset(
        metadata_csv=args.metadata_csv,
        split="train",
        acceleration=args.acceleration,
        slices_per_patient=args.slices_per_patient,
        edge_weight=1.0,
    )

    selected_patient_ids = [
        row["patient_id"]
        for row in full_dataset.patient_rows[
            :args.num_patients
        ]
    ]

    tiny_dataset = PairedMulticoilDataset(
        metadata_csv=args.metadata_csv,
        split="train",
        acceleration=args.acceleration,
        patient_ids=selected_patient_ids,
        slices_per_patient=args.slices_per_patient,
        edge_weight=1.0,
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)

    train_loader = DataLoader(
        tiny_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        generator=train_generator,
    )

    eval_loader = DataLoader(
        tiny_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    print("Selected patients:")
    for patient_id in selected_patient_ids:
        print("  ", patient_id)

    print("Samples:", len(tiny_dataset))

    model = VarNet(
        num_cascades=args.num_cascades,
        sens_chans=args.sens_chans,
        sens_pools=args.sens_pools,
        chans=args.chans,
        pools=args.pools,
    ).to(device)

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
    config["selected_patient_ids"] = selected_patient_ids
    config["model"] = "fastMRI VarNet 0.3.0"
    config["loss"] = "target-max-normalised L1"
    config["mask"] = "Gaussian VD"
    config["batch_size"] = 1

    with open(
        output_dir / "config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config,
            file,
            indent=2,
        )

    log_path = output_dir / "training_log.csv"

    with open(
        log_path,
        "w",
        newline="",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "epoch",
                "train_l1",
                "eval_overall_l1",
                "eval_edge_l1",
                "eval_central_l1",
                "gradient_norm_mean",
                "seconds",
            ]
        )

    best_l1 = float("inf")

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        model.train()

        train_losses = []
        gradient_norms = []

        for batch in train_loader:
            masked_kspace, mask, target = prepare_varnet_batch(
                batch=batch,
                contrast=args.contrast,
                device=device,
            )

            output = model(
                masked_kspace,
                mask,
            )

            output_crop = center_crop(
                output,
                crop_h=target.shape[-2],
                crop_w=target.shape[-1],
            )

            per_sample_loss = normalised_l1_per_sample(
                output_crop,
                target,
            )

            loss = per_sample_loss.mean()

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at epoch {epoch}."
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
                    f"Non-finite gradient at epoch {epoch}."
                )

            optimizer.step()

            train_losses.append(
                float(loss.item())
            )
            gradient_norms.append(
                float(gradient_norm.item())
            )

            del (
                masked_kspace,
                mask,
                target,
                output,
                output_crop,
                loss,
            )

        results, per_slice_rows = evaluate(
            model=model,
            loader=eval_loader,
            contrast=args.contrast,
            device=device,
        )

        mean_train_loss = float(
            np.mean(train_losses)
        )
        mean_gradient_norm = float(
            np.mean(gradient_norms)
        )

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train={mean_train_loss:.6f} | "
            f"eval={results['overall_l1']:.6f} | "
            f"edge={results['edge_l1']:.6f} | "
            f"central={results['central_l1']:.6f} | "
            f"grad={mean_gradient_norm:.6f} | "
            f"time={elapsed:.1f}s"
        )

        with open(
            log_path,
            "a",
            newline="",
        ) as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    epoch,
                    mean_train_loss,
                    results["overall_l1"],
                    results["edge_l1"],
                    results["central_l1"],
                    mean_gradient_norm,
                    elapsed,
                ]
            )

        save_checkpoint(
            path=output_dir / "model_last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            config=config,
            selected_patient_ids=selected_patient_ids,
            results=results,
        )

        if results["overall_l1"] < best_l1:
            best_l1 = results["overall_l1"]

            save_checkpoint(
                path=output_dir / "model_best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=config,
                selected_patient_ids=selected_patient_ids,
                results=results,
            )

            with open(
                output_dir / "best_per_slice_metrics.csv",
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
                writer.writerows(per_slice_rows)

    checkpoint = torch.load(
        output_dir / "model_best.pt",
        map_location=device,
        weights_only=False,
    )

    reload_model = VarNet(
        num_cascades=args.num_cascades,
        sens_chans=args.sens_chans,
        sens_pools=args.sens_pools,
        chans=args.chans,
        pools=args.pools,
    ).to(device)

    reload_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    reload_results, _ = evaluate(
        model=reload_model,
        loader=eval_loader,
        contrast=args.contrast,
        device=device,
    )

    reload_matches = bool(
        np.isclose(
            checkpoint["evaluation_results"]["overall_l1"],
            reload_results["overall_l1"],
            rtol=1e-6,
            atol=1e-8,
        )
    )

    summary = {
        "best_epoch": checkpoint["epoch"],
        "saved_results": checkpoint["evaluation_results"],
        "reloaded_results": reload_results,
        "checkpoint_reload_matches": reload_matches,
    }

    with open(
        output_dir / "final_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print("\n" + "=" * 80)
    print("VarNet tiny overfit completed")
    print("=" * 80)
    print("Best epoch:", checkpoint["epoch"])
    print("Best overall L1:", reload_results["overall_l1"])
    print("Checkpoint reload matches:", reload_matches)
    print("Log:", log_path)
    print("Summary:", output_dir / "final_summary.json")


if __name__ == "__main__":
    main()
