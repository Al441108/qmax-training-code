from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np
import torch

from singlecoil_paired_dataset_raw import (
    FSMNetSinglecoilRawGridDataset,
    center_crop_real,
)
from src.m2_prnf_qmax_singlecoil import (
    QMaxSinglecoilFull,
)
from src.m2_prnf_qmax_singlecoil_freqaux import (
    QMaxSinglecoilFullFreqAux,
)


MODEL_QMAX = "qmax_full"
MODEL_FREQAUX = "qmax_freqaux"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def torch_load(path: Path):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_fsmnet_metric(metric_path: Path):
    if not metric_path.is_file():
        raise FileNotFoundError(
            f"FSMNet metric file is missing: {metric_path}"
        )

    specification = importlib.util.spec_from_file_location(
        "fsmnet_public_metric",
        metric_path,
    )

    if (
        specification is None
        or specification.loader is None
    ):
        raise RuntimeError(
            f"Cannot import FSMNet metric file: {metric_path}"
        )

    module = importlib.util.module_from_spec(
        specification
    )
    specification.loader.exec_module(module)

    for name in ("nmse", "psnr", "ssim"):
        if not callable(getattr(module, name, None)):
            raise RuntimeError(
                f"FSMNet metric.py does not provide {name}()"
            )

    return module


def build_model(
    model_type: str,
    checkpoint: Mapping[str, Any],
    device: torch.device,
):
    if model_type == MODEL_QMAX:
        expected_name = "QMaxSinglecoilFull"

        model = QMaxSinglecoilFull(
            qmax_variant="qmax_full",
            num_cascades=12,
            chans=18,
            pools=4,
            controller_chans=16,
            initial_aux_alpha=0.1,
            initial_gate_probability=0.95,
        )
    elif model_type == MODEL_FREQAUX:
        expected_name = "QMaxSinglecoilFullFreqAux"

        model = QMaxSinglecoilFullFreqAux(
            frequency_channels=64,
            crop_size=320,
            qmax_variant="qmax_full",
            num_cascades=12,
            chans=18,
            pools=4,
            controller_chans=16,
            initial_aux_alpha=0.1,
            initial_gate_probability=0.95,
        )
    else:
        raise ValueError(
            f"Unsupported model type: {model_type}"
        )

    actual_name = checkpoint.get("model_name")

    if actual_name != expected_name:
        raise RuntimeError(
            "Checkpoint/model mismatch: "
            f"expected={expected_name!r}, "
            f"actual={actual_name!r}"
        )

    if checkpoint.get("qmax_variant") != "qmax_full":
        raise RuntimeError(
            "Checkpoint is not qmax_full"
        )

    if checkpoint.get("precision") != "fp32":
        raise RuntimeError(
            "Checkpoint precision is not fp32"
        )

    state = checkpoint.get("model_state")

    if not isinstance(state, Mapping):
        raise RuntimeError(
            "Checkpoint has no model_state dictionary"
        )

    model.load_state_dict(
        state,
        strict=True,
    )

    model = model.to(device)
    model.eval()

    return model


def batched_tensor(
    sample: Mapping[str, Any],
    key: str,
    device: torch.device,
) -> torch.Tensor:
    if key not in sample:
        raise KeyError(
            f"Dataset sample is missing {key!r}; "
            f"keys={sorted(sample)}"
        )

    value = sample[key]

    if not torch.is_tensor(value):
        value = torch.as_tensor(value)

    return value.unsqueeze(0).to(
        device,
        non_blocking=True,
    )


def squeeze_image(value: torch.Tensor) -> torch.Tensor:
    while value.ndim > 2 and value.shape[0] == 1:
        value = value.squeeze(0)

    if value.ndim != 2:
        raise RuntimeError(
            f"Expected a 2D image, got {tuple(value.shape)}"
        )

    return value


def python_value(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()

    if isinstance(value, np.generic):
        return value.item()

    return value


def record_value(
    record: Any,
    key: str,
):
    if isinstance(record, Mapping):
        return record.get(key)

    return getattr(record, key, None)


def find_metadata(
    sample: Mapping[str, Any],
    record: Any,
    names: Iterable[str],
):
    for name in names:
        if name in sample:
            value = python_value(sample[name])

            if value not in (None, ""):
                return value

        value = record_value(record, name)

        if value not in (None, ""):
            return python_value(value)

    return None


def sample_identity(
    sample: Mapping[str, Any],
    record: Any,
    index: int,
):
    volume_id = find_metadata(
        sample,
        record,
        (
            "volume_id",
            "pair_id",
            "target_volume_id",
            "fspd_volume_id",
            "pdfs_volume_id",
            "fname",
        ),
    )

    if volume_id is None:
        target_path = find_metadata(
            sample,
            record,
            (
                "fspd_path",
                "pdfs_path",
                "target_path",
            ),
        )

        if target_path is not None:
            volume_id = Path(
                str(target_path)
            ).stem

    if volume_id is None:
        volume_id = f"unknown_volume_{index:06d}"

    slice_index = find_metadata(
        sample,
        record,
        (
            "slice_index",
            "slice_idx",
            "slice_num",
        ),
    )

    if slice_index is None:
        slice_index = index

    return str(volume_id), int(slice_index)


def write_csv(
    path: Path,
    rows: List[Dict[str, Any]],
) -> None:
    if not rows:
        raise RuntimeError(
            f"Refusing to write empty CSV: {path}"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def finite_metric(
    name: str,
    value: float,
) -> float:
    value = float(value)

    if not np.isfinite(value):
        raise RuntimeError(
            f"Non-finite {name}: {value}"
        )

    return value


def mean_metric(
    rows: List[Dict[str, Any]],
    name: str,
) -> float:
    return float(
        np.mean(
            [float(row[name]) for row in rows]
        )
    )


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-type",
        required=True,
        choices=(
            MODEL_QMAX,
            MODEL_FREQAUX,
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--fsmnet-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--dataset-mode",
        choices=("train", "val"),
        default="train",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
    )
    parser.add_argument(
        "--max-slices",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--require-update",
        type=int,
        default=100000,
    )
    parser.add_argument(
        "--allow-held-out",
        action="store_true",
    )

    args = parser.parse_args()

    checkpoint_path = args.checkpoint.resolve()
    manifest_path = args.manifest.resolve()
    fsmnet_root = args.fsmnet_root.resolve()
    output_dir = args.output_dir.resolve()
    metric_path = fsmnet_root / "metric.py"

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required"
        )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            checkpoint_path
        )

    if not manifest_path.is_file():
        raise FileNotFoundError(
            manifest_path
        )

    held_out = (
        manifest_path.name == "test_locked.csv"
    )

    if held_out and not args.allow_held_out:
        raise RuntimeError(
            "Refusing held-out evaluation without "
            "--allow-held-out"
        )

    if (
        not held_out
        and args.allow_held_out
    ):
        raise RuntimeError(
            "--allow-held-out was supplied for a "
            "non-held-out manifest"
        )

    if (
        output_dir / "summary.json"
    ).exists():
        raise RuntimeError(
            f"Output already exists: {output_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    seed_everything(args.seed)

    device = torch.device("cuda")
    checkpoint = torch_load(checkpoint_path)

    if int(
        checkpoint.get("global_update", -1)
    ) != args.require_update:
        raise RuntimeError(
            "Unexpected checkpoint update: "
            f"{checkpoint.get('global_update')}"
        )

    if int(
        checkpoint.get("seed", -1)
    ) != args.seed:
        raise RuntimeError(
            "Unexpected checkpoint seed: "
            f"{checkpoint.get('seed')}"
        )

    model = build_model(
        model_type=args.model_type,
        checkpoint=checkpoint,
        device=device,
    )

    metric = load_fsmnet_metric(
        metric_path
    )

    dataset = FSMNetSinglecoilRawGridDataset(
        manifest_path=manifest_path,
        fsmnet_root=fsmnet_root,
        mode=args.dataset_mode,
        mask_rng_seed=args.seed,
        deterministic_train_mask=True,
    )

    total_available = len(dataset)

    if total_available <= 0:
        raise RuntimeError(
            "Evaluation dataset is empty"
        )

    if args.max_slices > 0:
        num_slices = min(
            args.max_slices,
            total_available,
        )
    else:
        num_slices = total_available

    indices = range(num_slices)

    slice_rows: List[Dict[str, Any]] = []
    volume_arrays = defaultdict(list)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
        flush=True,
    )
    print(
        "model type:",
        args.model_type,
        flush=True,
    )
    print(
        "checkpoint:",
        checkpoint_path,
        flush=True,
    )
    print(
        "checkpoint update:",
        checkpoint["global_update"],
        flush=True,
    )
    print(
        "manifest:",
        manifest_path,
        flush=True,
    )
    print(
        "held out:",
        held_out,
        flush=True,
    )
    print(
        "available slices:",
        total_available,
        flush=True,
    )
    print(
        "evaluated slices:",
        num_slices,
        flush=True,
    )

    evaluation_started = time.perf_counter()

    records = getattr(
        dataset,
        "records",
        None,
    )

    for position, index in enumerate(
        indices,
        start=1,
    ):
        sample = dataset[index]

        record = (
            records[index]
            if records is not None
            else {}
        )

        volume_id, slice_index = sample_identity(
            sample,
            record,
            index,
        )

        masked_kspace = batched_tensor(
            sample,
            "masked_kspace",
            device,
        )
        mask = batched_tensor(
            sample,
            "mask",
            device,
        )
        pd_image = batched_tensor(
            sample,
            "pd_image",
            device,
        )
        target = batched_tensor(
            sample,
            "target_image",
            device,
        )

        start = time.perf_counter()

        if args.model_type == MODEL_QMAX:
            prediction_raw = model(
                pdfs_masked_kspace=masked_kspace,
                mask=mask,
                pd_aux_image=pd_image,
                pd_available=torch.ones(
                    1,
                    device=device,
                ),
            )

            prediction = center_crop_real(
                prediction_raw,
                320,
            )
        else:
            output = model(
                pdfs_masked_kspace=masked_kspace,
                mask=mask,
                pd_aux_image=pd_image,
                pd_available=torch.ones(
                    1,
                    device=device,
                ),
            )

            prediction = output["img_out"]

        torch.cuda.synchronize(device)
        inference_seconds = (
            time.perf_counter() - start
        )

        prediction = squeeze_image(
            prediction
        )
        target = squeeze_image(
            target
        )

        if prediction.shape != target.shape:
            raise RuntimeError(
                "Prediction/target mismatch: "
                f"{tuple(prediction.shape)} versus "
                f"{tuple(target.shape)}"
            )

        if not torch.isfinite(
            prediction
        ).all():
            raise RuntimeError(
                f"Non-finite prediction at index {index}"
            )

        prediction_np = (
            prediction.detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        target_np = (
            target.detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        slice_nmse = finite_metric(
            "slice NMSE",
            metric.nmse(
                target_np,
                prediction_np,
            ),
        )
        slice_psnr = finite_metric(
            "slice PSNR",
            metric.psnr(
                target_np,
                prediction_np,
            ),
        )
        slice_ssim = finite_metric(
            "slice SSIM",
            metric.ssim(
                target_np[None, ...],
                prediction_np[None, ...],
            ),
        )

        slice_rows.append(
            {
                "model_type": args.model_type,
                "volume_id": volume_id,
                "slice_index": slice_index,
                "nmse": slice_nmse,
                "psnr": slice_psnr,
                "ssim": slice_ssim,
                "inference_seconds": float(
                    inference_seconds
                ),
            }
        )

        volume_arrays[volume_id].append(
            (
                slice_index,
                target_np,
                prediction_np,
            )
        )

        if (
            position == 1
            or position % 10 == 0
            or position == num_slices
        ):
            print(
                f"evaluated={position}/{num_slices} "
                f"volume={volume_id} "
                f"slice={slice_index} "
                f"PSNR={slice_psnr:.6f} "
                f"SSIM={slice_ssim:.6f}",
                flush=True,
            )

    volume_rows: List[Dict[str, Any]] = []

    for volume_id in sorted(volume_arrays):
        ordered = sorted(
            volume_arrays[volume_id],
            key=lambda item: item[0],
        )

        target_volume = np.stack(
            [item[1] for item in ordered],
            axis=0,
        )
        prediction_volume = np.stack(
            [item[2] for item in ordered],
            axis=0,
        )

        volume_rows.append(
            {
                "model_type": args.model_type,
                "volume_id": volume_id,
                "num_slices": len(ordered),
                "nmse": finite_metric(
                    "volume NMSE",
                    metric.nmse(
                        target_volume,
                        prediction_volume,
                    ),
                ),
                "psnr": finite_metric(
                    "volume PSNR",
                    metric.psnr(
                        target_volume,
                        prediction_volume,
                    ),
                ),
                "ssim": finite_metric(
                    "volume SSIM",
                    metric.ssim(
                        target_volume,
                        prediction_volume,
                    ),
                ),
            }
        )

    evaluation_seconds = (
        time.perf_counter()
        - evaluation_started
    )

    slice_metrics_path = (
        output_dir / "slice_metrics.csv"
    )
    volume_metrics_path = (
        output_dir / "volume_metrics.csv"
    )

    write_csv(
        slice_metrics_path,
        slice_rows,
    )
    write_csv(
        volume_metrics_path,
        volume_rows,
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    summary = {
        "status": "complete",
        "model_type": args.model_type,
        "checkpoint_model_name": checkpoint[
            "model_name"
        ],
        "checkpoint_update": int(
            checkpoint["global_update"]
        ),
        "checkpoint_seed": int(
            checkpoint["seed"]
        ),
        "precision": checkpoint["precision"],
        "held_out_accessed": bool(held_out),
        "manifest_name": manifest_path.name,
        "dataset_mode": args.dataset_mode,
        "available_slices": int(total_available),
        "evaluated_slices": int(num_slices),
        "evaluated_volumes": int(
            len(volume_rows)
        ),
        "complete_manifest_evaluation": bool(
            num_slices == total_available
        ),
        "volume_nmse_mean": mean_metric(
            volume_rows,
            "nmse",
        ),
        "volume_psnr_mean": mean_metric(
            volume_rows,
            "psnr",
        ),
        "volume_ssim_mean": mean_metric(
            volume_rows,
            "ssim",
        ),
        "slice_nmse_mean": mean_metric(
            slice_rows,
            "nmse",
        ),
        "slice_psnr_mean": mean_metric(
            slice_rows,
            "psnr",
        ),
        "slice_ssim_mean": mean_metric(
            slice_rows,
            "ssim",
        ),
        "mean_inference_seconds_per_slice": (
            mean_metric(
                slice_rows,
                "inference_seconds",
            )
        ),
        "evaluation_seconds": float(
            evaluation_seconds
        ),
        "trainable_parameters": int(
            trainable_parameters
        ),
        "peak_gpu_memory_gib": float(
            torch.cuda.max_memory_allocated(
                device
            )
            / (1024**3)
        ),
        "checkpoint_sha256": sha256_file(
            checkpoint_path
        ),
        "manifest_sha256": sha256_file(
            manifest_path
        ),
        "fsmnet_metric_sha256": sha256_file(
            metric_path
        ),
    }

    summary_path = (
        output_dir / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    evaluation_manifest = {
        "status": "complete",
        "created_at_unix": time.time(),
        "model_type": args.model_type,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": summary[
            "checkpoint_sha256"
        ],
        "manifest": str(manifest_path),
        "manifest_sha256": summary[
            "manifest_sha256"
        ],
        "fsmnet_metric_file": str(metric_path),
        "fsmnet_metric_sha256": summary[
            "fsmnet_metric_sha256"
        ],
        "held_out_accessed": bool(held_out),
        "seed": int(args.seed),
        "max_slices": int(args.max_slices),
        "outputs": {
            "slice_metrics": str(
                slice_metrics_path
            ),
            "volume_metrics": str(
                volume_metrics_path
            ),
            "summary": str(
                summary_path
            ),
        },
    }

    (
        output_dir
        / "evaluation_manifest.json"
    ).write_text(
        json.dumps(
            evaluation_manifest,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    print(
        "SHARED SINGLECOIL EVALUATION COMPLETE",
        flush=True,
    )


if __name__ == "__main__":
    main()