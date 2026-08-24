from __future__ import annotations

"""Per-sample deterministic corruption streams for QMax Stage A."""

import hashlib
import json
import math
import random
from typing import Any, Dict, List, Optional, Sequence

import torch

from src.m2_prnf_corruptions import (
    CorruptedBatch,
    CorruptionConfig,
    HardNegativeSampler,
    _sample_condition,
    border_only,
    border_reliability_target,
    load_pd_auxiliary,
    sample_direction,
    sample_padding_mode,
    scale_targets_from_base,
    shift_reliability_target,
    translate_nonwrapping,
    wrong_slice_reliability_target,
)


DETERMINISTIC_CORRUPTION_PROTOCOL = (
    "QMax-R8-R2-per-sample-corruption-v1"
)


def local_corruption_seed(
    *,
    global_seed: int,
    epoch: int,
    patient_id: str,
    slice_index: int,
    view_index: int,
    occurrence_index: int,
    stream_id: str,
) -> int:
    """Stable 63-bit seed independent of Python hash randomisation."""

    payload = {
        "protocol": DETERMINISTIC_CORRUPTION_PROTOCOL,
        "global_seed": int(global_seed),
        "epoch": int(epoch),
        "patient_id": str(patient_id),
        "slice_index": int(slice_index),
        "view_index": int(view_index),
        "occurrence_index": int(occurrence_index),
        "stream_id": str(stream_id),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    digest = hashlib.blake2b(encoded, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) & (
        (1 << 63) - 1
    )


def _record_identity(dataset: Any, source_index: int) -> Dict[str, Any]:
    record = dataset.records[int(source_index)]
    return {
        "patient_id": str(record["patient_id"]),
        "slice_index": int(record["slice_idx"]),
    }


def corrupt_batch_qmax(
    pd_aux: torch.Tensor,
    sample_indices: Sequence[int],
    dataset: Any,
    negative_sampler: HardNegativeSampler,
    epoch: int,
    global_seed: int,
    config: CorruptionConfig,
    *,
    view_index: int = 1,
    occurrence_indices: Optional[Sequence[int]] = None,
    stream_id: str = "qmax_train_corrupt",
) -> CorruptedBatch:
    """Create one corruption using a private RNG for every anatomical sample."""

    config.validate()
    if pd_aux.ndim != 3:
        raise RuntimeError(f"Expected [B,H,W], got {tuple(pd_aux.shape)}")
    if len(sample_indices) != pd_aux.shape[0]:
        raise RuntimeError("sample_indices does not match anatomical batch")
    if occurrence_indices is None:
        occurrence_indices = [0] * len(sample_indices)
    if len(occurrence_indices) != len(sample_indices):
        raise RuntimeError("occurrence_indices length mismatch")

    output = pd_aux.clone()
    availability = torch.ones(
        pd_aux.shape[0], device=pd_aux.device, dtype=pd_aux.dtype
    )
    targets = torch.ones(
        pd_aux.shape[0],
        4,
        device=pd_aux.device,
        dtype=pd_aux.dtype,
    )
    records: List[Dict[str, Any]] = []

    for position, source_index_value in enumerate(sample_indices):
        source_index = int(source_index_value)
        identity = _record_identity(dataset, source_index)
        occurrence_index = int(occurrence_indices[position])
        seed = local_corruption_seed(
            global_seed=int(global_seed),
            epoch=int(epoch),
            patient_id=identity["patient_id"],
            slice_index=identity["slice_index"],
            view_index=int(view_index),
            occurrence_index=occurrence_index,
            stream_id=str(stream_id),
        )
        rng = random.Random(seed)
        condition = _sample_condition(rng)
        image = output[position]
        record: Dict[str, Any] = {
            "protocol": DETERMINISTIC_CORRUPTION_PROTOCOL,
            "global_seed": int(global_seed),
            "epoch": int(epoch),
            "patient_id": identity["patient_id"],
            "slice_idx": identity["slice_index"],
            "view_index": int(view_index),
            "occurrence_index": occurrence_index,
            "stream_id": str(stream_id),
            "local_seed": int(seed),
            "condition": condition,
            "condition_key": condition,
            "source_index": source_index,
            "replacement_index": None,
            "replacement_patient_id": None,
            "replacement_slice_idx": None,
            "padding_mode": None,
            "dx": 0,
            "dy": 0,
            "magnitude_linf": 0,
            "magnitude_l2": 0.0,
            "direction": None,
            "direction_class": None,
            "delta_z_norm": 0.0,
            "fallback_from": None,
            "missing_mask": 0,
        }

        if condition == "border":
            width = int(rng.choice(config.shift_magnitudes))
            padding_mode = sample_padding_mode(config, rng)
            output[position] = border_only(image, width, padding_mode)
            base_target = border_reliability_target(width, config)
            record.update(
                {
                    "condition_key": f"border{width}",
                    "padding_mode": padding_mode,
                    "magnitude_linf": width,
                    "magnitude_l2": float(width),
                }
            )
        elif condition == "missing":
            output[position].zero_()
            availability[position] = 0.0
            base_target = config.reliability_missing
            record["missing_mask"] = 1
        elif condition == "shift":
            magnitude = int(rng.choice(config.shift_magnitudes))
            padding_mode = sample_padding_mode(config, rng)
            dy, dx, direction, direction_class = sample_direction(
                magnitude,
                rng,
                allow_cardinal=True,
                allow_diagonal=True,
            )
            output[position] = translate_nonwrapping(
                image, dy, dx, padding_mode
            )
            base_target = shift_reliability_target(magnitude, config)
            record.update(
                {
                    "condition_key": f"shift{magnitude}",
                    "padding_mode": padding_mode,
                    "dx": int(dx),
                    "dy": int(dy),
                    "magnitude_linf": magnitude,
                    "magnitude_l2": float(math.hypot(dx, dy)),
                    "direction": direction,
                    "direction_class": direction_class,
                }
            )
        elif condition == "wrong_slice":
            candidate = negative_sampler.same_patient_wrong_slice(
                source_index, rng
            )
            if candidate is None:
                magnitude = 8
                dy, dx, direction, direction_class = sample_direction(
                    magnitude, rng, True, True
                )
                output[position] = translate_nonwrapping(
                    image, dy, dx, "reflect"
                )
                base_target = shift_reliability_target(magnitude, config)
                record.update(
                    {
                        "fallback_from": "wrong_slice",
                        "condition": "shift",
                        "condition_key": "shift8",
                        "padding_mode": "reflect",
                        "dx": int(dx),
                        "dy": int(dy),
                        "magnitude_linf": magnitude,
                        "magnitude_l2": float(math.hypot(dx, dy)),
                        "direction": direction,
                        "direction_class": direction_class,
                    }
                )
            else:
                replacement_index, delta_z = candidate
                replacement = load_pd_auxiliary(
                    dataset, replacement_index, pd_aux.device
                )
                if replacement.shape != image.shape:
                    raise RuntimeError(
                        "Wrong-slice replacement shape mismatch"
                    )
                output[position] = replacement
                base_target = wrong_slice_reliability_target(delta_z)
                replacement_record = dataset.records[int(replacement_index)]
                record.update(
                    {
                        "replacement_index": int(replacement_index),
                        "replacement_patient_id": str(
                            replacement_record["patient_id"]
                        ),
                        "replacement_slice_idx": int(
                            replacement_record["slice_idx"]
                        ),
                        "delta_z_norm": float(delta_z),
                    }
                )
        elif condition == "wrong_patient":
            candidate = negative_sampler.wrong_patient_matched_level(
                source_index,
                tuple(int(value) for value in image.shape),
                rng,
                config.wrong_patient_top_k,
            )
            if candidate is None:
                magnitude = 8
                dy, dx, direction, direction_class = sample_direction(
                    magnitude, rng, True, True
                )
                output[position] = translate_nonwrapping(
                    image, dy, dx, "reflect"
                )
                base_target = shift_reliability_target(magnitude, config)
                record.update(
                    {
                        "fallback_from": "wrong_patient",
                        "condition": "shift",
                        "condition_key": "shift8",
                        "padding_mode": "reflect",
                        "dx": int(dx),
                        "dy": int(dy),
                        "magnitude_linf": magnitude,
                        "magnitude_l2": float(math.hypot(dx, dy)),
                        "direction": direction,
                        "direction_class": direction_class,
                    }
                )
            else:
                replacement_index, delta_z = candidate
                replacement = load_pd_auxiliary(
                    dataset, replacement_index, pd_aux.device
                )
                if replacement.shape != image.shape:
                    raise RuntimeError(
                        "Wrong-patient replacement shape mismatch"
                    )
                output[position] = replacement
                base_target = config.reliability_wrong_patient
                replacement_record = dataset.records[int(replacement_index)]
                record.update(
                    {
                        "replacement_index": int(replacement_index),
                        "replacement_patient_id": str(
                            replacement_record["patient_id"]
                        ),
                        "replacement_slice_idx": int(
                            replacement_record["slice_idx"]
                        ),
                        "delta_z_norm": float(delta_z),
                    }
                )
        else:
            raise RuntimeError(f"Unsupported condition {condition}")

        allow_coarse_relief = str(record["condition"]) in {
            "shift",
            "wrong_slice",
        }
        targets[position] = torch.tensor(
            scale_targets_from_base(
                float(base_target),
                config,
                allow_coarse_relief,
            ),
            device=targets.device,
            dtype=targets.dtype,
        )
        records.append(record)

    return CorruptedBatch(
        image=output,
        availability=availability,
        reliability_target=targets,
        records=records,
    )


def manifest_rows(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compact preflight manifest with all required random decisions."""

    keys = (
        "protocol",
        "global_seed",
        "epoch",
        "patient_id",
        "slice_idx",
        "view_index",
        "occurrence_index",
        "stream_id",
        "local_seed",
        "condition",
        "condition_key",
        "dx",
        "dy",
        "magnitude_linf",
        "direction",
        "direction_class",
        "padding_mode",
        "replacement_index",
        "replacement_patient_id",
        "replacement_slice_idx",
        "missing_mask",
    )
    return [{key: record.get(key) for key in keys} for record in records]
