from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data._utils.collate import default_collate

from raw_shape_batch_sampler import RawShapeBatchSampler
from singlecoil_paired_dataset_raw import (
    FSMNetSinglecoilRawGridDataset,
)
from src.m2_prnf_qmax_singlecoil import QMaxSinglecoilFull
from train_singlecoil_qmax import seed_everything


def tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def model_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()

    for name, tensor in sorted(
        model.state_dict().items()
    ):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())

    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-root", required=True)
    parser.add_argument("--fsmnet-root", required=True)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    seed_everything(args.seed)

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
        batch_size=4,
        shuffle=True,
        drop_last=False,
        seed=args.seed,
    )
    sampler.set_epoch(0)

    first_indices = list(iter(sampler))[0]

    model = QMaxSinglecoilFull(
        qmax_variant="qmax_full",
        num_cascades=12,
        chans=18,
        pools=4,
        controller_chans=16,
        initial_aux_alpha=0.1,
        initial_gate_probability=0.95,
    )

    batch = default_collate(
        [dataset[index] for index in first_indices]
    )

    result = {
        "seed": args.seed,
        "first_indices": first_indices,
        "pair_ids": batch["pair_id"],
        "slice_indices": batch["slice_index"].tolist(),
        "fspd_volume_ids": batch["fspd_volume_id"],
        "model_sha256": model_hash(model),
        "mask_sha256": tensor_hash(batch["mask"]),
        "masked_kspace_sha256": tensor_hash(
            batch["masked_kspace"]
        ),
        "pd_image_sha256": tensor_hash(
            batch["pd_image"]
        ),
        "target_image_sha256": tensor_hash(
            batch["target_image"]
        ),
        "zero_filled_sha256": tensor_hash(
            batch["zero_filled_crop"]
        ),
        "acquired_lines": (
            batch["acquired_lines"].tolist()
        ),
        "actual_acceleration": (
            batch["actual_acceleration"].tolist()
        ),
        "raw_shape": list(
            batch["masked_kspace"].shape
        ),
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()