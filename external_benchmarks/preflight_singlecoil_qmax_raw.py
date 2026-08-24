from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data._utils.collate import default_collate

from raw_shape_batch_sampler import RawShapeBatchSampler
from singlecoil_paired_dataset_raw import (
    FSMNetSinglecoilRawGridDataset,
    center_crop_real,
)
from src.m2_prnf_qmax_singlecoil import QMaxSinglecoilFull


class AmplitudeLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        prediction_fft = torch.fft.rfft2(
            prediction,
            norm="backward",
        )
        target_fft = torch.fft.rfft2(
            target,
            norm="backward",
        )
        return self.l1(
            torch.abs(prediction_fft),
            torch.abs(target_fft),
        )


class PhaseLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        prediction_fft = torch.fft.rfft2(
            prediction,
            norm="backward",
        )
        target_fft = torch.fft.rfft2(
            target,
            norm="backward",
        )
        return self.l1(
            torch.angle(prediction_fft),
            torch.angle(target_fft),
        )


def normalized_loss_inputs(
    prediction: torch.Tensor,
    target: torch.Tensor,
    zero_filled: torch.Tensor,
):
    """
    Match FSMNet target normalization as closely as possible.

    FSMNet derives mean/std from the target zero-filled input, clamps the
    normalized target to [-6,6], and leaves model output unclamped.
    """
    mean = zero_filled.mean(
        dim=(-2, -1),
        keepdim=True,
    )
    std = zero_filled.std(
        dim=(-2, -1),
        keepdim=True,
    ).clamp_min(1e-11)

    prediction_norm = (prediction - mean) / std
    target_norm = ((target - mean) / std).clamp(-6, 6)

    return prediction_norm, target_norm


def grad_summary(model: nn.Module, prefix: str):
    finite = True
    nonzero_tensors = 0
    total_tensors = 0
    squared_norm = 0.0

    for name, parameter in model.named_parameters():
        if not name.startswith(prefix):
            continue
        if parameter.grad is None:
            continue

        total_tensors += 1
        grad = parameter.grad.detach()

        if not torch.isfinite(grad).all():
            finite = False

        if bool((grad != 0).any()):
            nonzero_tensors += 1

        squared_norm += float(
            grad.float().square().sum().item()
        )

    return {
        "finite": finite,
        "nonzero": nonzero_tensors,
        "total": total_tensors,
        "norm": squared_norm ** 0.5,
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bench-root",
        default=os.environ.get("BENCH_ROOT"),
    )
    parser.add_argument(
        "--fsmnet-root",
        default=os.environ.get("FSMNET"),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--raw-height", type=int, default=640)
    parser.add_argument("--raw-width", type=int, default=368)
    parser.add_argument("--seed", type=int, default=1337)

    args = parser.parse_args()

    if not args.bench_root or not args.fsmnet_root:
        raise ValueError(
            "BENCH_ROOT and FSMNET must be provided"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required. Run this preflight on a compute node."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda")

    dataset = FSMNetSinglecoilRawGridDataset(
        manifest_path=os.path.join(
            args.bench_root,
            "manifests/train.csv",
        ),
        fsmnet_root=args.fsmnet_root,
        mode="train",
        mask_rng_seed=args.seed,
        deterministic_train_mask=True,
    )

    sampler = RawShapeBatchSampler(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        seed=args.seed,
    )

    requested_shape = (
        args.raw_height,
        args.raw_width,
    )

    if requested_shape not in sampler._buckets:
        raise RuntimeError(
            f"Raw shape {requested_shape} is unavailable; "
            f"choices={sorted(sampler._buckets)}"
        )

    indices = sampler._buckets[requested_shape][
        :args.batch_size
    ]

    if len(indices) != args.batch_size:
        raise RuntimeError("Not enough samples for requested batch")

    batch = default_collate(
        [dataset[index] for index in indices]
    )

    masked_kspace = batch["masked_kspace"].to(device)
    mask = batch["mask"].to(device)
    pd_image = batch["pd_image"].to(device)
    target = batch["target_image"].to(device).squeeze(1)
    zero_filled = batch["zero_filled_crop"].to(
        device
    ).squeeze(1)

    model = QMaxSinglecoilFull(
        qmax_variant="qmax_full",
        num_cascades=12,
        chans=18,
        pools=4,
        controller_chans=16,
        initial_aux_alpha=0.1,
        initial_gate_probability=0.95,
    ).to(device)

    if sum(p.numel() for p in model.sens_net.parameters()) != 0:
        raise RuntimeError(
            "Unit sensitivity model must have zero parameters"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
    )

    image_l1 = nn.L1Loss()
    amplitude_loss = AmplitudeLoss()
    phase_loss = PhaseLoss()

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("device:", torch.cuda.get_device_name(0))
    print("batch size:", args.batch_size)
    print("raw shape:", tuple(masked_kspace.shape))
    print("PD shape:", tuple(pd_image.shape))
    print("target shape:", tuple(target.shape))
    print("trainable parameters:", trainable_parameters)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    model.train()

    for step in range(2):
        optimizer.zero_grad(set_to_none=True)

        prediction_raw = model(
            pdfs_masked_kspace=masked_kspace,
            mask=mask,
            pd_aux_image=pd_image,
            pd_available=torch.ones(
                args.batch_size,
                device=device,
            ),
        )

        prediction = center_crop_real(
            prediction_raw,
            320,
        )

        if prediction.shape != target.shape:
            raise RuntimeError(
                f"Prediction/target mismatch: "
                f"{tuple(prediction.shape)} versus "
                f"{tuple(target.shape)}"
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

        # Common part of the FSMNet objective. QMax has no img_fre branch.
        loss = (
            loss_image
            + 0.01 * loss_amplitude
            + 0.01 * loss_phase
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at step {step}"
            )

        loss.backward()

        total_grad_norm = clip_grad_norm_(
            model.parameters(),
            max_norm=0.01,
        )

        if not torch.isfinite(total_grad_norm):
            raise RuntimeError(
                f"Non-finite gradient norm at step {step}"
            )

        if step == 1:
            groups = {
                "pd_encoder": grad_summary(
                    model,
                    "pd_encoder",
                ),
                "controllers": grad_summary(
                    model,
                    "controllers",
                ),
                "cascades": grad_summary(
                    model,
                    "cascades",
                ),
            }

            for name, result in groups.items():
                print("gradient", name, result)

            if not all(
                result["finite"]
                for result in groups.values()
            ):
                raise RuntimeError(
                    "Non-finite module gradient detected"
                )

            if groups["cascades"]["nonzero"] == 0:
                raise RuntimeError(
                    "All cascade gradients are zero"
                )

        optimizer.step()
        torch.cuda.synchronize()

        print(
            f"step={step + 1}",
            f"loss={float(loss.detach()):.6f}",
            f"image={float(loss_image.detach()):.6f}",
            f"amplitude={float(loss_amplitude.detach()):.6f}",
            f"phase={float(loss_phase.detach()):.6f}",
            f"grad_before_clip={float(total_grad_norm):.6f}",
        )

    peak_bytes = torch.cuda.max_memory_allocated(device)
    peak_gib = peak_bytes / (1024**3)

    print("prediction raw:", tuple(prediction_raw.shape))
    print("prediction crop:", tuple(prediction.shape))
    print(f"peak allocated GPU memory: {peak_gib:.3f} GiB")
    print("QMAX RAW-GRID FORWARD/BACKWARD PREFLIGHT PASSED")


if __name__ == "__main__":
    main()