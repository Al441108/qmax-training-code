from __future__ import annotations

"""Pre-registered corrupt view sampler for the final PRNF comparison."""

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import h5py
import torch
import torch.nn.functional as F

SCALE_NAMES = ("H/2", "H/4", "H/8", "H/16")
CARDINAL_DIRECTIONS = ((1, 0, "+y"), (-1, 0, "-y"), (0, 1, "+x"), (0, -1, "-x"))
DIAGONAL_DIRECTIONS = ((1, 1, "+y+x"), (1, -1, "+y-x"), (-1, 1, "-y+x"), (-1, -1, "-y-x"))


@dataclass(frozen=True)
class CorruptionConfig:
    shift_magnitudes: Tuple[int, ...] = (2, 4, 8)
    padding_prob_reflect: float = 0.60
    padding_prob_replicate: float = 0.20
    padding_prob_zero: float = 0.20
    scale_relief: Tuple[float, ...] = (0.00, 0.15, 0.30, 0.45)
    reliability_clean: float = 1.00
    reliability_shift_2: float = 0.65
    reliability_shift_4: float = 0.35
    reliability_shift_8: float = 0.05
    reliability_border_2: float = 0.95
    reliability_border_4: float = 0.90
    reliability_border_8: float = 0.85
    reliability_wrong_patient: float = 0.05
    reliability_missing: float = 0.00
    wrong_patient_top_k: int = 8

    def validate(self) -> None:
        probabilities = (
            self.padding_prob_reflect,
            self.padding_prob_replicate,
            self.padding_prob_zero,
        )
        if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-8):
            raise ValueError("Padding probabilities must sum to one")
        if any(value < 0 for value in probabilities):
            raise ValueError("Padding probabilities cannot be negative")
        if len(self.scale_relief) != len(SCALE_NAMES):
            raise ValueError("scale_relief does not match the number of scales")
        if self.wrong_patient_top_k < 1:
            raise ValueError("wrong_patient_top_k must be positive")


@dataclass
class CorruptedBatch:
    image: torch.Tensor
    availability: torch.Tensor
    reliability_target: torch.Tensor
    records: List[Dict[str, Any]]


def _z_norm(record: Mapping[str, Any]) -> float:
    slices = max(int(record.get("num_slices", 1)), 1)
    return float(record["slice_idx"]) / float(max(slices - 1, 1))


class HardNegativeSampler:
    """Shape-matched hard negatives with reproducible top-k randomisation."""

    def __init__(self, dataset: Any):
        self.dataset, self.records = dataset, dataset.records
        self.patient_ids = [str(record["patient_id"]) for record in self.records]
        self.z_norm = [_z_norm(record) for record in self.records]
        self.by_patient: Dict[str, List[int]] = defaultdict(list)
        self.by_shape: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        shape_cache: Dict[str, Tuple[int, int]] = {}
        for index, record in enumerate(self.records):
            self.by_patient[self.patient_ids[index]].append(index)
            path = str(record["pdfs_path"])
            if path not in shape_cache:
                with h5py.File(path, "r") as hf:
                    source = hf["reconstruction_rss"] if "reconstruction_rss" in hf else hf["kspace"]
                    shape_cache[path] = tuple(int(value) for value in source.shape[-2:])
            self.by_shape[shape_cache[path]].append(index)

    def same_patient_wrong_slice(
        self, source_index: int, rng: random.Random
    ) -> Optional[Tuple[int, float]]:
        patient, source_z = self.patient_ids[source_index], self.z_norm[source_index]
        candidates = [
            index for index in self.by_patient[patient] if index != source_index
        ]
        if not candidates:
            return None
        preferred = [
            index for index in candidates
            if 0.05 <= abs(self.z_norm[index] - source_z) <= 0.25
        ]
        replacement = rng.choice(preferred if preferred else candidates)
        return replacement, abs(self.z_norm[replacement] - source_z)

    def wrong_patient_matched_level(
        self,
        source_index: int,
        source_shape: Tuple[int, int],
        rng: random.Random,
        top_k: int,
    ) -> Optional[Tuple[int, float]]:
        patient, source_z = self.patient_ids[source_index], self.z_norm[source_index]
        candidates = [
            index for index in self.by_shape.get(tuple(source_shape), [])
            if self.patient_ids[index] != patient
        ]
        if not candidates:
            return None
        # Use at most one matched slice per candidate patient.  A slice-level
        # top-k can otherwise be dominated by a single patient and give little
        # cross-epoch negative diversity.
        best_by_patient: Dict[str, int] = {}
        for index in candidates:
            candidate_patient = self.patient_ids[index]
            previous = best_by_patient.get(candidate_patient)
            if previous is None or (
                abs(self.z_norm[index] - source_z), index
            ) < (
                abs(self.z_norm[previous] - source_z), previous
            ):
                best_by_patient[candidate_patient] = index
        ranked = sorted(
            best_by_patient.values(),
            key=lambda index: (
                abs(self.z_norm[index] - source_z), self.patient_ids[index], index
            ),
        )
        pool = ranked[: min(int(top_k), len(ranked))]
        replacement = rng.choice(pool)
        return replacement, abs(self.z_norm[replacement] - source_z)


def load_pd_auxiliary(dataset: Any, index: int, device: torch.device) -> torch.Tensor:
    image = dataset[int(index)]["pd_aux_image"]
    image = image if torch.is_tensor(image) else torch.as_tensor(image)
    if image.ndim == 3 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 2:
        raise RuntimeError(f"Alternative PD has invalid shape {tuple(image.shape)}")
    return image.float().to(device, non_blocking=True)


def sample_padding_mode(config: CorruptionConfig, rng: random.Random) -> str:
    draw = rng.random()
    if draw < config.padding_prob_reflect:
        return "reflect"
    if draw < config.padding_prob_reflect + config.padding_prob_replicate:
        return "replicate"
    return "zero"


def sample_direction(
    magnitude: int,
    rng: random.Random,
    allow_cardinal: bool = True,
    allow_diagonal: bool = True,
) -> Tuple[int, int, str, str]:
    candidates = []
    if allow_cardinal:
        candidates.extend(CARDINAL_DIRECTIONS)
    if allow_diagonal:
        candidates.extend(DIAGONAL_DIRECTIONS)
    dy, dx, name = rng.choice(candidates)
    family = "cardinal" if dy == 0 or dx == 0 else "diagonal"
    return magnitude * dy, magnitude * dx, name, family


def _torch_padding_mode(mode: str) -> str:
    if mode == "zero":
        return "constant"
    if mode in {"reflect", "replicate"}:
        return mode
    raise ValueError(f"Unsupported padding mode {mode}")


def translate_nonwrapping(
    image: torch.Tensor, shift_y: int, shift_x: int, padding_mode: str
) -> torch.Tensor:
    if image.ndim != 2:
        raise RuntimeError("translate_nonwrapping expects [H,W]")
    pad = max(abs(int(shift_y)), abs(int(shift_x)))
    mode = _torch_padding_mode(padding_mode)
    if mode == "constant":
        padded = F.pad(image[None, None], (pad, pad, pad, pad), mode=mode, value=0.0)[0, 0]
    else:
        padded = F.pad(image[None, None], (pad, pad, pad, pad), mode=mode)[0, 0]
    start_y, start_x = pad - int(shift_y), pad - int(shift_x)
    return padded[start_y : start_y + image.shape[0], start_x : start_x + image.shape[1]]


def border_only(image: torch.Tensor, width: int, padding_mode: str) -> torch.Tensor:
    """Replace only the outer band; central anatomy remains exactly aligned."""
    if image.ndim != 2:
        raise RuntimeError("border_only expects [H,W]")
    width = int(width)
    if 2 * width >= min(image.shape):
        raise ValueError("Border width is too large")
    central = image[width:-width, width:-width]
    mode = _torch_padding_mode(padding_mode)
    if mode == "constant":
        output = F.pad(
            central[None, None], (width, width, width, width), mode=mode, value=0.0
        )[0, 0]
    else:
        output = F.pad(
            central[None, None], (width, width, width, width), mode=mode
        )[0, 0]
    if not torch.equal(output[width:-width, width:-width], central):
        raise RuntimeError("Border control changed central anatomy")
    return output


def shift_reliability_target(magnitude: int, config: CorruptionConfig) -> float:
    return {
        2: config.reliability_shift_2,
        4: config.reliability_shift_4,
        8: config.reliability_shift_8,
    }[int(magnitude)]


def border_reliability_target(width: int, config: CorruptionConfig) -> float:
    return {
        2: config.reliability_border_2,
        4: config.reliability_border_4,
        8: config.reliability_border_8,
    }[int(width)]


def wrong_slice_reliability_target(delta_z: float) -> float:
    severity = min(1.0, max(0.0, (float(delta_z) - 0.05) / 0.15))
    return 0.65 + severity * (0.10 - 0.65)


def scale_targets_from_base(
    base_target: float, config: CorruptionConfig, allow_coarse_relief: bool
) -> Tuple[float, ...]:
    base = min(1.0, max(0.0, float(base_target)))
    if not allow_coarse_relief:
        return tuple(base for _ in SCALE_NAMES)
    return tuple(
        min(1.0, base + (1.0 - base) * relief)
        for relief in config.scale_relief
    )


def _ranking_margin(target: torch.Tensor, record: Mapping[str, Any]) -> torch.Tensor:
    condition = str(record.get("condition"))
    target_gap = (1.0 - target).clamp_min(0.0)
    if condition == "shift":
        magnitude = int(record.get("magnitude_linf", 0))
        maximum = 0.04 if magnitude <= 2 else 0.07 if magnitude <= 4 else 0.10
    elif condition == "wrong_slice":
        severity = min(1.0, max(0.0, (float(record.get("delta_z_norm", 0)) - 0.05) / 0.15))
        maximum = 0.025 + 0.075 * severity
    elif condition == "wrong_patient":
        maximum = 0.20
    elif condition == "border":
        return target_gap
    else:
        raise ValueError(f"Condition {condition} does not participate in ranking")
    return torch.minimum(target_gap, torch.full_like(target_gap, maximum))


def paired_discrimination_loss(
    q_clean: torch.Tensor,
    q_second: torch.Tensor,
    second_targets: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
    consistency_tolerance: float = 0.03,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Pair ranking; missing is deliberately excluded because m=0 solves it."""
    losses, gaps = [], []
    for index, record in enumerate(records):
        condition = str(record.get("condition"))
        if condition in {"missing", "clean"}:
            continue
        gap = q_clean[index] - q_second[index]
        margin = _ranking_margin(second_targets[index], record)
        if condition == "border":
            loss = F.relu((gap - margin[None]).abs() - consistency_tolerance).mean()
        else:
            loss = F.relu(margin[None] - gap).mean()
        losses.append(loss)
        gaps.append(float(gap.detach().mean().item()))
    if not losses:
        zero = (q_clean.sum() + q_second.sum()) * 0.0
        return zero, {"q_gap": float("nan"), "num_ranked": 0}
    return torch.stack(losses).mean(), {
        "q_gap": float(sum(gaps) / len(gaps)), "num_ranked": len(losses)
    }


# Conditional probabilities within the second view. Border controls expose the
# same padding families with high reliability, breaking the padding/label
# shortcut. Forward views are 50/50; reconstruction objective weights are 70/30.
CORRUPT_MIXTURE: Mapping[str, float] = {
    "border": 0.10,
    "shift": 0.30,
    "wrong_slice": 0.225,
    "wrong_patient": 0.225,
    "missing": 0.15,
}


def _sample_condition(rng: random.Random) -> str:
    draw, cumulative = rng.random(), 0.0
    for condition, probability in CORRUPT_MIXTURE.items():
        cumulative += probability
        if draw <= cumulative + 1e-12:
            return condition
    return "missing"


def corrupt_batch_prnf(
    pd_aux: torch.Tensor,
    sample_indices: Sequence[int],
    dataset: Any,
    negative_sampler: HardNegativeSampler,
    epoch: int,
    batch_index: int,
    seed: int,
    config: CorruptionConfig,
) -> CorruptedBatch:
    """Create exactly one deterministic corrupt view per clean anatomical pair."""
    config.validate()
    if pd_aux.ndim != 3:
        raise RuntimeError(f"Expected [B,H,W], got {tuple(pd_aux.shape)}")
    if len(sample_indices) != pd_aux.shape[0]:
        raise RuntimeError("sample_indices does not match the anatomical batch")

    rng = random.Random(seed + epoch * 1_000_003 + batch_index * 10_007)
    output = pd_aux.clone()
    availability = torch.ones(
        pd_aux.shape[0], device=pd_aux.device, dtype=pd_aux.dtype
    )
    targets = torch.ones(
        pd_aux.shape[0], len(SCALE_NAMES), device=pd_aux.device, dtype=pd_aux.dtype
    )
    records = []

    for position, source_index in enumerate(sample_indices):
        condition = _sample_condition(rng)
        image = output[position]
        record: Dict[str, Any] = {
            "condition": condition,
            "condition_key": condition,
            "source_index": int(source_index),
            "replacement_index": None,
            "padding_mode": None,
            "dx": 0,
            "dy": 0,
            "magnitude_linf": 0,
            "magnitude_l2": 0.0,
            "direction": None,
            "direction_class": None,
            "delta_z_norm": 0.0,
            "fallback_from": None,
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

        elif condition == "shift":
            magnitude = int(rng.choice(config.shift_magnitudes))
            padding_mode = sample_padding_mode(config, rng)
            dy, dx, direction, direction_class = sample_direction(
                magnitude, rng, allow_cardinal=True, allow_diagonal=True
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
                int(source_index), rng
            )
            if candidate is None:
                # Do not silently relabel an unavailable hard negative as such.
                magnitude = 8
                dy, dx, direction, direction_class = sample_direction(
                    magnitude, rng, True, True
                )
                output[position] = translate_nonwrapping(image, dy, dx, "reflect")
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
                replacement = load_pd_auxiliary(dataset, replacement_index, pd_aux.device)
                if replacement.shape != image.shape:
                    raise RuntimeError("Wrong-slice replacement shape mismatch")
                output[position] = replacement
                base_target = wrong_slice_reliability_target(delta_z)
                record.update(
                    {
                        "replacement_index": int(replacement_index),
                        "replacement_patient_id": negative_sampler.patient_ids[replacement_index],
                        "delta_z_norm": float(delta_z),
                    }
                )

        elif condition == "wrong_patient":
            candidate = negative_sampler.wrong_patient_matched_level(
                int(source_index),
                tuple(int(value) for value in image.shape),
                rng,
                config.wrong_patient_top_k,
            )
            if candidate is None:
                magnitude = 8
                dy, dx, direction, direction_class = sample_direction(
                    magnitude, rng, True, True
                )
                output[position] = translate_nonwrapping(image, dy, dx, "reflect")
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
                replacement = load_pd_auxiliary(dataset, replacement_index, pd_aux.device)
                if replacement.shape != image.shape:
                    raise RuntimeError("Wrong-patient replacement shape mismatch")
                output[position] = replacement
                base_target = config.reliability_wrong_patient
                record.update(
                    {
                        "replacement_index": int(replacement_index),
                        "replacement_patient_id": negative_sampler.patient_ids[replacement_index],
                        "delta_z_norm": float(delta_z),
                    }
                )
        else:
            raise RuntimeError(f"Unsupported condition {condition}")

        allow_coarse_relief = str(record["condition"]) in {"shift", "wrong_slice"}
        targets[position] = torch.tensor(
            scale_targets_from_base(
                float(base_target), config, allow_coarse_relief
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
