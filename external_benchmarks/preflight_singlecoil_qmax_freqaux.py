from __future__ import annotations

import argparse
import os
import random
from typing import Dict, Iterable

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
from src.m2_prnf_qmax_singlecoil_freqaux import (
    QMaxSinglecoilFullFreqAux,
)


def gradient_summary(
    model: nn.Module,
    prefixes: Iterable[str],
) -> Dict[str, float | int | bool]:
    prefixes = tuple(prefixes)

    finite = True
    nonzero_tensors = 0
    total_tensors = 0
    squared_norm = 0.0

    for name, parameter in model.named_parameters():
        if not name.startswith(prefixes):
            continue

        if parameter.grad is None:
            continue

        total_tensors += 1
        gradient = parameter.grad.detach()

        if not torch.isfinite(gradient).all():
            finite = False

        if bool((gradient != 0).any()):
            nonzero_tensors += 1

        squared_norm += float(
            gradient.float().square().sum().item()
        )

    return {
        "finite": finite,
        "nonzero": nonzero_tensors,
        "total": total_tensors,
        "norm": squared_norm**0.5,
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--raw-height",
        type=int,
        default=640,
    )
    parser.add_argument(
        "--raw-width",
        type=int,
        default=368,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
    )
    parser.add_argument(
        "--frequency-channels",
        type=int,
        default=64,
    )

    args = parser.parse_args()

    if not args.bench_root or not args.fsmnet_root:
        raise ValueError(
            "BENCH_ROOT and FSMNET must be provided"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

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
        : args.batch_size
    ]

    if len(indices) != args.batch_size:
        raise RuntimeError(
            "Not enough samples for requested batch"
        )

    batch = default_collate(
        [dataset[index] for index in indices]
    )

    masked_kspace = batch["masked_kspace"].to(device)
    mask = batch["mask"].to(device)
    pd_image = batch["pd_image"].to(device)

    target = (
        batch["target_image"]
        .to(device)
        .squeeze(1)
    )

    zero_filled = (
        batch["zero_filled_crop"]
        .to(device)
        .squeeze(1)
    )

    model = QMaxSinglecoilFullFreqAux(
        frequency_channels=args.frequency_channels,
        crop_size=320,
        qmax_variant="qmax_full",
        num_cascades=12,
        chans=18,
        pools=4,
        controller_chans=16,
        initial_aux_alpha=0.1,
        initial_gate_probability=0.95,
    ).to(device)

    if (
        sum(
            parameter.numel()
            for parameter in model.qmax.sens_net.parameters()
        )
        != 0
    ):
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

    qmax_parameters = sum(
        parameter.numel()
        for parameter in model.qmax.parameters()
        if parameter.requires_grad
    )
    frequency_parameters = sum(
        parameter.numel()
        for parameter in model.frequency_auxiliary.parameters()
        if parameter.requires_grad
    )
    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    if qmax_parameters != 32840882:
        raise RuntimeError(
            f"Unexpected QMax parameters: {qmax_parameters}"
        )

    if total_parameters != (
        qmax_parameters + frequency_parameters
    ):
        raise RuntimeError(
            "Parameter accounting mismatch"
        )

    print("device:", torch.cuda.get_device_name(0))
    print("batch size:", args.batch_size)
    print("raw shape:", tuple(masked_kspace.shape))
    print("PD shape:", tuple(pd_image.shape))
    print("target shape:", tuple(target.shape))
    print("QMax parameters:", qmax_parameters)
    print("frequency parameters:", frequency_parameters)
    print("total parameters:", total_parameters)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    model.train()

    frequency_mean = zero_filled.mean(
        dim=(-2, -1),
        keepdim=True,
    )
    frequency_std = zero_filled.std(
        dim=(-2, -1),
        keepdim=True,
    ).clamp_min(1e-11)

    print(
        "frequency normalization mean:",
        float(frequency_mean.mean()),
    )
    print(
        "frequency normalization std min/max:",
        float(frequency_std.min()),
        float(frequency_std.max()),
    )

    for step in range(2):
        optimizer.zero_grad(set_to_none=True)

        output = model(
            pdfs_masked_kspace=masked_kspace,
            mask=mask,
            pd_aux_image=pd_image,
            pd_available=torch.ones(
                args.batch_size,
                device=device,
            ),
            frequency_mean=frequency_mean,
            frequency_std=frequency_std,
        )

        prediction_raw = output["prediction_raw"]
        img_out = output["img_out"]
        img_fre = output["img_fre"]

        expected_crop = center_crop_real(
            prediction_raw,
            320,
        )

        crop_difference = float(
            (img_out - expected_crop)
            .detach()
            .abs()
            .max()
        )

        if crop_difference != 0.0:
            raise RuntimeError(
                "FreqAux wrapper changed the QMax image: "
                f"maximum difference={crop_difference}"
            )

        if img_out.shape != target.shape:
            raise RuntimeError(
                f"img_out/target mismatch: "
                f"{tuple(img_out.shape)} versus "
                f"{tuple(target.shape)}"
            )

        if img_fre.shape != target.shape:
            raise RuntimeError(
                f"img_fre/target mismatch: "
                f"{tuple(img_fre.shape)} versus "
                f"{tuple(target.shape)}"
            )

        if not torch.isfinite(img_out).all():
            raise RuntimeError(
                "Non-finite QMax output"
            )

        if not torch.isfinite(img_fre).all():
            raise RuntimeError(
                "Non-finite frequency output"
            )

        img_out_norm, target_norm = (
            normalized_loss_inputs(
                img_out,
                target,
                zero_filled,
            )
        )

        img_fre_norm, frequency_target_norm = (
            normalized_loss_inputs(
                img_fre,
                target,
                zero_filled,
            )
        )

        if not torch.equal(
            target_norm,
            frequency_target_norm,
        ):
            raise RuntimeError(
                "Main and frequency targets differ"
            )

        loss_main_image = image_l1(
            img_out_norm,
            target_norm,
        )
        loss_main_amplitude = amplitude_loss(
            img_out_norm,
            target_norm,
        )
        loss_main_phase = phase_loss(
            img_out_norm,
            target_norm,
        )

        loss_frequency_image = image_l1(
            img_fre_norm,
            target_norm,
        )
        loss_frequency_amplitude = amplitude_loss(
            img_fre_norm,
            target_norm,
        )
        loss_frequency_phase = phase_loss(
            img_fre_norm,
            target_norm,
        )

        loss_main = (
            loss_main_image
            + 0.01 * loss_main_amplitude
            + 0.01 * loss_main_phase
        )

        loss_frequency = (
            loss_frequency_image
            + 0.01 * loss_frequency_amplitude
            + 0.01 * loss_frequency_phase
        )

        frequency_to_main_ratio = (
            loss_frequency.detach()
            / loss_main.detach().clamp_min(1e-12)
        )

        if not torch.isfinite(frequency_to_main_ratio):
            raise RuntimeError(
                "Non-finite frequency/main loss ratio"
            )

        if float(frequency_to_main_ratio) > 10.0:
            raise RuntimeError(
                "Frequency loss remains badly scaled: "
                "frequency/main="
                f"{float(frequency_to_main_ratio):.6f}"
            )

        loss = loss_main + loss_frequency

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at step {step + 1}"
            )

        loss.backward()

        groups = {
            "qmax_pd_encoder": gradient_summary(
                model,
                ("qmax.pd_encoder",),
            ),
            "qmax_controllers": gradient_summary(
                model,
                ("qmax.controllers",),
            ),
            "qmax_cascades": gradient_summary(
                model,
                ("qmax.cascades",),
            ),
            "frequency_pd_encoder": gradient_summary(
                model,
                (
                    "frequency_auxiliary.pd_encoder",
                ),
            ),
            "frequency_target_encoder": gradient_summary(
                model,
                (
                    "frequency_auxiliary.target_head",
                    "frequency_auxiliary.target_down",
                    "frequency_auxiliary.target_refine",
                    "frequency_auxiliary.target_neck",
                ),
            ),
            "frequency_fusions": gradient_summary(
                model,
                (
                    "frequency_auxiliary.modality_fusions",
                ),
            ),
            "frequency_decoder": gradient_summary(
                model,
                (
                    "frequency_auxiliary.up",
                    "frequency_auxiliary.tail",
                ),
            ),
        }

        if step == 1:
            for name, result in groups.items():
                print("gradient", name, result)

        delayed_qmax_groups = {
            "qmax_pd_encoder",
            "qmax_controllers",
        }

        for name, result in groups.items():
            if not result["finite"]:
                raise RuntimeError(
                    f"Non-finite gradient in {name}"
                )

            if result["total"] == 0:
                raise RuntimeError(
                    f"No gradients found in {name}"
                )

            if result["nonzero"] == 0:
                if step == 0 and name in delayed_qmax_groups:
                    print(
                        f"gradient {name} is zero at initial step; "
                        "checking again after one optimizer update"
                    )
                    continue

                raise RuntimeError(
                    f"All gradients are zero in {name} "
                    f"at step {step + 1}"
                )

        grad_before_clip = clip_grad_norm_(
            model.parameters(),
            max_norm=0.01,
        )

        if not torch.isfinite(grad_before_clip):
            raise RuntimeError(
                f"Non-finite total gradient at step {step + 1}"
            )

        optimizer.step()
        torch.cuda.synchronize()

        print(
            f"step={step + 1}",
            f"loss={float(loss.detach()):.6f}",
            f"main={float(loss_main.detach()):.6f}",
            f"frequency={float(loss_frequency.detach()):.6f}",
            f"main_image={float(loss_main_image.detach()):.6f}",
            f"frequency_image={float(loss_frequency_image.detach()):.6f}",
            "frequency_to_main="
            f"{float(frequency_to_main_ratio):.6f}",
            f"grad_before_clip={float(grad_before_clip):.6f}",
        )

    peak_gib = (
        torch.cuda.max_memory_allocated(device)
        / (1024**3)
    )

    frequency_delta = float(
        (img_fre - img_out)
        .detach()
        .abs()
        .mean()
    )

    print("prediction raw:", tuple(prediction_raw.shape))
    print("img_out:", tuple(img_out.shape))
    print("img_fre:", tuple(img_fre.shape))
    print(
        "mean absolute frequency residual:",
        frequency_delta,
    )
    print(
        f"peak allocated GPU memory: {peak_gib:.3f} GiB"
    )
    print(
        "QMAX FREQUENCY-AUX FORWARD/BACKWARD "
        "PREFLIGHT PASSED"
    )


if __name__ == "__main__":
    main()
