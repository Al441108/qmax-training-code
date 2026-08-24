from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data._utils.collate import default_collate

from preflight_singlecoil_qmax_raw import (
    AmplitudeLoss,
    PhaseLoss,
    normalized_loss_inputs,
)
from raw_shape_batch_sampler import RawShapeBatchSampler
from singlecoil_paired_dataset_raw import (
    FSMNetSinglecoilRawGridDataset,
    center_crop_real,
)
from src.m2_prnf_qmax_singlecoil import QMaxSinglecoilFull


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    


def capture_rng_state() -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def torch_load(path: Path):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(path, map_location="cpu")


def atomic_torch_save(
    state: Dict[str, Any],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def build_checkpoint(
    model,
    optimizer,
    scheduler,
    dataset,
    args,
    global_update: int,
    epoch: int,
    batch_position: int,
) -> Dict[str, Any]:
    return {
        "format_version": 1,
        "model_name": "QMaxSinglecoilFull",
        "qmax_variant": "qmax_full",
        "precision": "fp32",
        "seed": args.seed,
        "global_update": global_update,
        "epoch": epoch,
        "batch_position": batch_position,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "rng_state": capture_rng_state(),
        "mask_rng_state": dataset.mask_func.rng.get_state(),
        "arguments": vars(args),
    }


def save_training_state(
    model,
    optimizer,
    scheduler,
    dataset,
    args,
    global_update: int,
    epoch: int,
    batch_position: int,
) -> None:
    output_dir = Path(args.output_dir)

    state = build_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataset=dataset,
        args=args,
        global_update=global_update,
        epoch=epoch,
        batch_position=batch_position,
    )

    update_path = output_dir / (
        f"checkpoint_update{global_update:06d}.pt"
    )
    last_path = output_dir / "model_last.pt"

    atomic_torch_save(state, update_path)
    atomic_torch_save(state, last_path)

    print(f"checkpoint={update_path}", flush=True)


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--bench-root", required=True)
    parser.add_argument("--fsmnet-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", default=None)

    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-updates", type=int, default=100000)
    parser.add_argument("--stop-at-update", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=20000)
    parser.add_argument("--log-every", type=int, default=100)

    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-step-size", type=int, default=20000)
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=0.01)

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    if args.batch_size != 4:
        raise ValueError(
            "Formal FSMNet-aligned training requires batch size 4"
        )

    target_update = (
        args.max_updates
        if args.stop_at_update is None
        else min(args.stop_at_update, args.max_updates)
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.resume is None and (
        output_dir / "model_last.pt"
    ).exists():
        raise RuntimeError(
            f"Refusing to overwrite existing run: {output_dir}"
        )

    seed_everything(args.seed)
    device = torch.device("cuda")

    dataset = FSMNetSinglecoilRawGridDataset(
        manifest_path=Path(args.bench_root)
        / "manifests"
        / "train.csv",
        fsmnet_root=args.fsmnet_root,
        mode="train",
        mask_rng_seed=args.seed,
        deterministic_train_mask=False,
    )

    sampler = RawShapeBatchSampler(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        seed=args.seed,
    )

    model = QMaxSinglecoilFull(
        qmax_variant="qmax_full",
        num_cascades=12,
        chans=18,
        pools=4,
        controller_chans=16,
        initial_aux_alpha=0.1,
        initial_gate_probability=0.95,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma,
    )

    image_l1 = nn.L1Loss()
    amplitude_loss = AmplitudeLoss()
    phase_loss = PhaseLoss()

    global_update = 0
    epoch = 0
    batch_position = 0

    if args.resume is not None:
        checkpoint = torch_load(Path(args.resume))

        if checkpoint["seed"] != args.seed:
            raise RuntimeError(
                "Resume checkpoint seed does not match"
            )

        if checkpoint["precision"] != "fp32":
            raise RuntimeError(
                "Resume checkpoint precision does not match"
            )

        model.load_state_dict(
            checkpoint["model_state"],
            strict=True,
        )
        optimizer.load_state_dict(
            checkpoint["optimizer_state"]
        )
        scheduler.load_state_dict(
            checkpoint["scheduler_state"]
        )

        global_update = int(checkpoint["global_update"])
        epoch = int(checkpoint["epoch"])
        batch_position = int(
            checkpoint["batch_position"]
        )

        dataset.mask_func.rng.set_state(
            checkpoint["mask_rng_state"]
        )

        # Restore last, after model/optimizer construction.
        restore_rng_state(checkpoint["rng_state"])

        print(
            f"resumed={args.resume} "
            f"update={global_update} "
            f"epoch={epoch} "
            f"batch_position={batch_position}",
            flush=True,
        )

    if global_update >= target_update:
        raise RuntimeError(
            f"Checkpoint is already at update {global_update}, "
            f"target is {target_update}"
        )

    log_path = output_dir / "train_metrics.jsonl"

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("device:", torch.cuda.get_device_name(0))
    print("trainable parameters:", trainable_parameters)
    print("samples:", len(dataset))
    print("batches per epoch:", len(sampler))
    print("starting update:", global_update)
    print("target update:", target_update)
    print("precision: fp32")
    print("seed:", args.seed)

    model.train()
    torch.cuda.reset_peak_memory_stats(device)

    while global_update < target_update:
        current_epoch = epoch
        sampler.set_epoch(current_epoch)
        epoch_batches = list(iter(sampler))

        if batch_position >= len(epoch_batches):
            epoch += 1
            batch_position = 0
            continue

        for position in range(
            batch_position,
            len(epoch_batches),
        ):
            step_start = time.perf_counter()

            indices = epoch_batches[position]
            batch = default_collate(
                [dataset[index] for index in indices]
            )

            # Minority shape buckets can produce a final short batch.
            current_batch_size = len(indices)

            masked_kspace = batch[
                "masked_kspace"
            ].to(device)
            mask = batch["mask"].to(device)
            pd_image = batch["pd_image"].to(device)

            target = batch["target_image"].to(
                device
            ).squeeze(1)

            zero_filled = batch[
                "zero_filled_crop"
            ].to(device).squeeze(1)

            optimizer.zero_grad(set_to_none=True)

            prediction_raw = model(
                pdfs_masked_kspace=masked_kspace,
                mask=mask,
                pd_aux_image=pd_image,
                pd_available=torch.ones(
                    current_batch_size,
                    device=device,
                ),
            )

            prediction = center_crop_real(
                prediction_raw,
                320,
            )

            prediction_norm, target_norm = (
                normalized_loss_inputs(
                    prediction,
                    target,
                    zero_filled,
                )
            )

            loss_image = image_l1(
                prediction_norm,
                target_norm,
            )
            loss_amplitude = amplitude_loss(
                prediction_norm,
                target_norm,
            )
            loss_phase = phase_loss(
                prediction_norm,
                target_norm,
            )

            # Shared portion of the public FSMNet objective.
            # SSIM is evaluation-only and is not included.
            loss = (
                loss_image
                + 0.01 * loss_amplitude
                + 0.01 * loss_phase
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at update "
                    f"{global_update + 1}"
                )

            loss.backward()

            grad_before_clip = clip_grad_norm_(
                model.parameters(),
                max_norm=args.grad_clip,
            )

            if not torch.isfinite(grad_before_clip):
                raise RuntimeError(
                    f"Non-finite gradient at update "
                    f"{global_update + 1}"
                )

            learning_rate = optimizer.param_groups[0]["lr"]

            optimizer.step()
            scheduler.step()

            global_update += 1
            batch_position = position + 1

            if batch_position == len(epoch_batches):
                epoch = current_epoch + 1
                batch_position = 0

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - step_start

            record = {
                "update": global_update,
                "epoch": current_epoch,
                "batch_position": position,
                "batch_size": current_batch_size,
                "raw_height": int(
                    masked_kspace.shape[-2]
                ),
                "raw_width": int(
                    masked_kspace.shape[-1]
                ),
                "loss": float(loss.detach()),
                "loss_image": float(loss_image.detach()),
                "loss_amplitude": float(
                    loss_amplitude.detach()
                ),
                "loss_phase": float(loss_phase.detach()),
                "grad_before_clip": float(
                    grad_before_clip
                ),
                "learning_rate": float(learning_rate),
                "seconds": float(elapsed),
                "peak_memory_gib": float(
                    torch.cuda.max_memory_allocated(device)
                    / (1024**3)
                ),
            }

            append_jsonl(log_path, record)

            if (
                global_update == 1
                or global_update % args.log_every == 0
                or global_update == target_update
            ):
                print(json.dumps(record), flush=True)

            should_save = (
                global_update % args.save_every == 0
                or global_update == target_update
            )

            if should_save:
                save_training_state(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    dataset=dataset,
                    args=args,
                    global_update=global_update,
                    epoch=epoch,
                    batch_position=batch_position,
                )

            if global_update >= target_update:
                break

    summary = {
        "status": "complete",
        "global_update": global_update,
        "epoch": epoch,
        "batch_position": batch_position,
        "precision": "fp32",
        "seed": args.seed,
        "batch_size": args.batch_size,
        "peak_memory_gib": (
            torch.cuda.max_memory_allocated(device)
            / (1024**3)
        ),
    }

    with (
        output_dir / "training_summary.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print("TRAINING SEGMENT COMPLETE")


if __name__ == "__main__":
    main()