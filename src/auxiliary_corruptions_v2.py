from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import h5py
import torch
import torch.nn.functional as F


PADDING_MODES = ("reflect", "replicate", "zero")
SCALE_NAMES = ("H/2", "H/4", "H/8", "H/16")
CARDINAL_DIRECTIONS = (
    (0, 1, "+x"),
    (0, -1, "-x"),
    (1, 0, "+y"),
    (-1, 0, "-y"),
)
DIAGONAL_DIRECTIONS = (
    (1, 1, "+y+x"),
    (1, -1, "+y-x"),
    (-1, 1, "-y+x"),
    (-1, -1, "-y-x"),
)
ALL_DIRECTIONS = CARDINAL_DIRECTIONS + DIAGONAL_DIRECTIONS


@dataclass(frozen=True)
class CorruptionConfig:
    shift_magnitudes: Tuple[int, ...] = (2, 4, 8)
    padding_prob_reflect: float = 0.70
    padding_prob_replicate: float = 0.20
    padding_prob_zero: float = 0.10
    # Border-only examples deliberately oversample black borders so that a
    # black edge is observed frequently with a high reliability label.
    border_padding_prob_reflect: float = 0.25
    border_padding_prob_replicate: float = 0.25
    border_padding_prob_zero: float = 0.50
    # Coarser feature scales tolerate local degradation better.  The first
    # value applies to H/2 and the last to H/16.
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
    blur_sigma_min: float = 0.6
    blur_sigma_max: float = 1.5
    reliability_blur_light: float = 0.80
    reliability_blur_severe: float = 0.30
    noise_std_min: float = 0.01
    noise_std_max: float = 0.08
    reliability_noise_light: float = 0.80
    reliability_noise_severe: float = 0.30
    blur_kernel_size: int = 7

    def validate(self) -> None:
        padding_total = (
            self.padding_prob_reflect
            + self.padding_prob_replicate
            + self.padding_prob_zero
        )
        if not math.isclose(padding_total, 1.0, abs_tol=1e-8):
            raise ValueError(
                "Padding probabilities must sum to one; "
                f"received {padding_total:.8f}."
            )
        if any(value < 0 for value in (
            self.padding_prob_reflect,
            self.padding_prob_replicate,
            self.padding_prob_zero,
        )):
            raise ValueError("Padding probabilities cannot be negative.")
        border_padding_total = (
            self.border_padding_prob_reflect
            + self.border_padding_prob_replicate
            + self.border_padding_prob_zero
        )
        if not math.isclose(border_padding_total, 1.0, abs_tol=1e-8):
            raise ValueError(
                "Border padding probabilities must sum to one; "
                f"received {border_padding_total:.8f}."
            )
        if any(value < 0 for value in (
            self.border_padding_prob_reflect,
            self.border_padding_prob_replicate,
            self.border_padding_prob_zero,
        )):
            raise ValueError("Border padding probabilities cannot be negative.")
        if len(self.scale_relief) != len(SCALE_NAMES):
            raise ValueError(
                f"scale_relief must contain {len(SCALE_NAMES)} values, "
                f"received {len(self.scale_relief)}."
            )
        if any(value < 0.0 or value > 1.0 for value in self.scale_relief):
            raise ValueError("scale_relief values must lie in [0, 1].")
        if tuple(sorted(self.scale_relief)) != self.scale_relief:
            raise ValueError("scale_relief must be non-decreasing across scales.")
        if tuple(sorted(self.shift_magnitudes)) != self.shift_magnitudes:
            raise ValueError("shift_magnitudes must be sorted.")
        if any(value <= 0 for value in self.shift_magnitudes):
            raise ValueError("Shift magnitudes must be positive.")
        if not (
            self.reliability_clean
            > self.reliability_shift_2
            > self.reliability_shift_4
            > self.reliability_shift_8
            >= self.reliability_missing
        ):
            raise ValueError("Shift reliability targets must decrease with severity.")
        if not (
            self.reliability_clean
            >= self.reliability_border_2
            >= self.reliability_border_4
            >= self.reliability_border_8
            > self.reliability_shift_8
        ):
            raise ValueError("Border-only targets are inconsistent.")
        if self.blur_kernel_size < 3 or self.blur_kernel_size % 2 == 0:
            raise ValueError("blur_kernel_size must be odd and at least three.")


@dataclass
class CorruptedBatch:
    image: torch.Tensor
    availability: torch.Tensor
    reliability_target: torch.Tensor
    records: List[Dict[str, Any]]


def curriculum_mixture(epoch: int, curriculum: str) -> Dict[str, float]:
    """Return the pre-registered per-sample condition mixture.

    ``smoke5`` is the compressed five-epoch screening curriculum.
    ``formal15`` is the staged curriculum for a fresh 15-epoch pilot.
    """
    if epoch < 1:
        raise ValueError("Epoch numbering starts at one.")

    if curriculum == "smoke5":
        schedules = {
            1: {"clean": 0.80, "border": 0.20},
            2: {"clean": 0.60, "border": 0.10, "shift_mild": 0.30},
            3: {"clean": 0.50, "border": 0.10, "shift": 0.35, "missing": 0.05},
            4: {
                "clean": 0.45,
                "shift": 0.25,
                "border": 0.10,
                "wrong_slice": 0.10,
                "wrong_patient": 0.05,
                "missing": 0.05,
            },
        }
        mixture = schedules.get(
            epoch,
            {
                "clean": 0.50,
                "shift": 0.20,
                "border": 0.075,
                "wrong_slice": 0.075,
                "wrong_patient": 0.05,
                "missing": 0.05,
                "blur": 0.025,
                "noise": 0.025,
            },
        )
    elif curriculum == "formal15":
        if epoch <= 2:
            mixture = {"clean": 0.80, "border": 0.20}
        elif epoch <= 4:
            mixture = {"clean": 0.60, "border": 0.10, "shift_mild": 0.30}
        elif epoch <= 6:
            mixture = {"clean": 0.50, "border": 0.10, "shift": 0.35, "missing": 0.05}
        elif epoch <= 8:
            mixture = {
                "clean": 0.45,
                "shift": 0.30,
                "border": 0.10,
                "wrong_slice": 0.075,
                "wrong_patient": 0.05,
                "missing": 0.025,
            }
        else:
            mixture = {
                "clean": 0.50,
                "shift": 0.20,
                "border": 0.075,
                "wrong_slice": 0.075,
                "wrong_patient": 0.05,
                "missing": 0.05,
                "blur": 0.025,
                "noise": 0.025,
            }
    else:
        raise ValueError(f"Unsupported curriculum: {curriculum}")

    total = sum(mixture.values())
    if not math.isclose(total, 1.0, abs_tol=1e-8):
        raise RuntimeError(f"Curriculum mixture sums to {total:.8f}: {mixture}")
    return mixture


def sample_from_mixture(mixture: Mapping[str, float], rng: random.Random) -> str:
    draw = rng.random()
    cumulative = 0.0
    for name, probability in mixture.items():
        cumulative += probability
        if draw <= cumulative:
            return name
    return next(reversed(mixture))


def sample_padding_mode(config: CorruptionConfig, rng: random.Random) -> str:
    draw = rng.random()
    if draw < config.padding_prob_reflect:
        return "reflect"
    if draw < config.padding_prob_reflect + config.padding_prob_replicate:
        return "replicate"
    return "zero"


def sample_border_padding_mode(
    config: CorruptionConfig,
    rng: random.Random,
) -> str:
    draw = rng.random()
    if draw < config.border_padding_prob_reflect:
        return "reflect"
    if draw < (
        config.border_padding_prob_reflect
        + config.border_padding_prob_replicate
    ):
        return "replicate"
    return "zero"


def scale_targets_from_base(
    base_target: float,
    config: CorruptionConfig,
    allow_coarse_relief: bool,
) -> Tuple[float, ...]:
    """Convert a sample-level target into four scale-aware targets.

    Wrong-patient and missing conditions remain low at every scale.  Spatial
    shifts, border changes, blur, noise and wrong-slice conditions receive
    progressively more tolerance at coarser scales.
    """
    base_target = min(1.0, max(0.0, float(base_target)))
    if not allow_coarse_relief:
        return tuple(base_target for _ in SCALE_NAMES)
    return tuple(
        min(1.0, base_target + (1.0 - base_target) * float(relief))
        for relief in config.scale_relief
    )


def sample_direction(
    magnitude: int,
    rng: random.Random,
    allow_cardinal: bool = True,
    allow_diagonal: bool = True,
) -> Tuple[int, int, str, str]:
    candidates: List[Tuple[int, int, str]] = []
    if allow_cardinal:
        candidates.extend(CARDINAL_DIRECTIONS)
    if allow_diagonal:
        candidates.extend(DIAGONAL_DIRECTIONS)
    if not candidates:
        raise ValueError("At least one direction family must be enabled.")
    unit_dy, unit_dx, direction = rng.choice(candidates)
    direction_class = "cardinal" if (unit_dy == 0 or unit_dx == 0) else "diagonal"
    return magnitude * unit_dy, magnitude * unit_dx, direction, direction_class


def shift_reliability_target(magnitude: int, config: CorruptionConfig) -> float:
    mapping = {
        2: config.reliability_shift_2,
        4: config.reliability_shift_4,
        8: config.reliability_shift_8,
    }
    if magnitude not in mapping:
        raise ValueError(
            f"No pre-registered shift target for magnitude={magnitude}; "
            f"supported values are {sorted(mapping)}."
        )
    return float(mapping[magnitude])


def border_reliability_target(width: int, config: CorruptionConfig) -> float:
    mapping = {
        2: config.reliability_border_2,
        4: config.reliability_border_4,
        8: config.reliability_border_8,
    }
    if width not in mapping:
        raise ValueError(
            f"No pre-registered border target for width={width}; "
            f"supported values are {sorted(mapping)}."
        )
    return float(mapping[width])


def wrong_slice_reliability_target(delta_z_norm: float) -> float:
    """Map same-patient slice mismatch to a 0.2--0.5 soft target."""
    value = float(abs(delta_z_norm))
    if value <= 0.05:
        return 0.50
    if value >= 0.20:
        return 0.20
    fraction = (value - 0.05) / 0.15
    return float(0.50 + fraction * (0.20 - 0.50))


def _pad_mode_for_torch(padding_mode: str) -> str:
    if padding_mode == "zero":
        return "constant"
    if padding_mode in {"reflect", "replicate"}:
        return padding_mode
    raise ValueError(f"Unsupported padding mode: {padding_mode}")


def translate_nonwrapping(
    image: torch.Tensor,
    shift_y: int,
    shift_x: int,
    padding_mode: str,
) -> torch.Tensor:
    """Translate a 2-D image without circular wrap-around.

    Positive ``shift_x`` moves anatomy to the right and positive ``shift_y``
    moves anatomy down. Empty regions are supplied by the requested boundary
    extension. No interpolation or resizing is performed.
    """
    if image.ndim != 2:
        raise RuntimeError(f"Expected [H,W], got {tuple(image.shape)}")
    height, width = image.shape
    pad = max(abs(int(shift_y)), abs(int(shift_x)))
    if pad == 0:
        return image.clone()
    if pad >= min(height, width) and padding_mode == "reflect":
        raise ValueError(
            f"Reflect padding {pad} is invalid for image shape {(height, width)}."
        )

    mode = _pad_mode_for_torch(padding_mode)
    x = image[None, None]
    if mode == "constant":
        padded = F.pad(x, (pad, pad, pad, pad), mode=mode, value=0.0)
    else:
        padded = F.pad(x, (pad, pad, pad, pad), mode=mode)

    start_y = pad - int(shift_y)
    start_x = pad - int(shift_x)
    output = padded[..., start_y:start_y + height, start_x:start_x + width]
    if output.shape[-2:] != (height, width):
        raise RuntimeError(
            f"Shift produced shape {tuple(output.shape[-2:])}, expected {(height, width)}."
        )
    return output[0, 0]


def border_only(
    image: torch.Tensor,
    width: int,
    padding_mode: str,
) -> torch.Tensor:
    """Alter only the peripheral border while leaving central content fixed."""
    if image.ndim != 2:
        raise RuntimeError(f"Expected [H,W], got {tuple(image.shape)}")
    height, image_width = image.shape
    width = int(width)
    if width <= 0 or 2 * width >= min(height, image_width):
        raise ValueError(
            f"Invalid border width={width} for image shape {(height, image_width)}."
        )

    core = image[width:height - width, width:image_width - width][None, None]
    mode = _pad_mode_for_torch(padding_mode)
    if mode == "constant":
        output = F.pad(core, (width, width, width, width), mode=mode, value=0.0)
    else:
        output = F.pad(core, (width, width, width, width), mode=mode)
    output = output[0, 0]

    # This is a shortcut control: central anatomy must remain exactly unchanged.
    if not torch.equal(
        output[width:height - width, width:image_width - width],
        image[width:height - width, width:image_width - width],
    ):
        raise RuntimeError("Border-only corruption modified central content.")
    return output


def gaussian_kernel_2d(
    kernel_size: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coordinates = torch.arange(kernel_size, device=device, dtype=dtype)
    coordinates = coordinates - (kernel_size - 1) / 2
    kernel_1d = torch.exp(-0.5 * (coordinates / float(sigma)).square())
    kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1e-12)
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d / kernel_2d.sum().clamp_min(1e-12)


def gaussian_blur_single(
    image: torch.Tensor,
    sigma: float,
    kernel_size: int,
) -> torch.Tensor:
    kernel = gaussian_kernel_2d(
        kernel_size,
        sigma,
        image.device,
        image.dtype,
    ).view(1, 1, kernel_size, kernel_size)
    padding = kernel_size // 2
    padded = F.pad(
        image[None, None],
        (padding, padding, padding, padding),
        mode="reflect",
    )
    return F.conv2d(padded, kernel)[0, 0]


def linear_target(
    value: float,
    lower: float,
    upper: float,
    target_lower: float,
    target_upper: float,
) -> float:
    if upper <= lower:
        return float(target_upper)
    fraction = min(1.0, max(0.0, (value - lower) / (upper - lower)))
    return float(target_lower + fraction * (target_upper - target_lower))


def _record_z_norm(record: Mapping[str, Any]) -> float:
    slice_idx = int(record["slice_idx"])
    num_slices = max(int(record.get("num_slices", 1)), 1)
    return float(slice_idx) / float(max(num_slices - 1, 1))


class HardNegativeSampler:
    """Anatomy-level matched hard-negative index for an indexed dataset.

    Wrong-patient candidates are restricted to the exact k-space shape bucket.
    If no different patient exists in that bucket, the caller is told that the
    condition is unavailable; training does not silently use a size shortcut.
    """

    def __init__(self, dataset: Any):
        self.dataset = dataset
        self.records = dataset.records
        self.patient_ids = [str(record["patient_id"]) for record in self.records]
        self.z_norm = [_record_z_norm(record) for record in self.records]

        self.by_patient: Dict[str, List[int]] = defaultdict(list)
        self.by_shape: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        self.shape_by_index: List[Tuple[int, int]] = []
        shape_cache: Dict[str, Tuple[int, int]] = {}
        for index, record in enumerate(self.records):
            self.by_patient[self.patient_ids[index]].append(index)
            path = str(record["pdfs_path"])
            if path not in shape_cache:
                with h5py.File(path, "r") as hf:
                    shape_dataset = (
                        hf["reconstruction_rss"]
                        if "reconstruction_rss" in hf
                        else hf["kspace"]
                    )
                    shape_cache[path] = tuple(
                        int(value) for value in shape_dataset.shape[-2:]
                    )
            shape_key = shape_cache[path]
            self.shape_by_index.append(shape_key)
            self.by_shape[shape_key].append(index)

    def same_patient_wrong_slice(
        self,
        source_index: int,
        rng: random.Random,
    ) -> Optional[Tuple[int, float]]:
        patient_id = self.patient_ids[source_index]
        source_z = self.z_norm[source_index]
        candidates = [
            index
            for index in self.by_patient[patient_id]
            if index != source_index
        ]
        if not candidates:
            return None

        preferred = [
            index
            for index in candidates
            if 0.05 <= abs(self.z_norm[index] - source_z) <= 0.25
        ]
        pool = preferred if preferred else candidates
        # Randomise within hard candidates while retaining reproducibility.
        replacement = rng.choice(pool)
        delta = abs(self.z_norm[replacement] - source_z)
        return replacement, float(delta)

    def wrong_patient_matched_level(
        self,
        source_index: int,
        source_shape: Tuple[int, int],
    ) -> Optional[Tuple[int, float]]:
        source_patient = self.patient_ids[source_index]
        source_z = self.z_norm[source_index]

        candidates = [
            index
            for index in self.by_shape.get(tuple(source_shape), [])
            if self.patient_ids[index] != source_patient
        ]

        if not candidates:
            return None
        replacement = min(
            candidates,
            key=lambda index: (
                abs(self.z_norm[index] - source_z),
                self.patient_ids[index],
                index,
            ),
        )
        delta = abs(self.z_norm[replacement] - source_z)
        return replacement, float(delta)


def load_pd_auxiliary(dataset: Any, index: int, device: torch.device) -> torch.Tensor:
    item = dataset[int(index)]
    image = item["pd_aux_image"]
    if not torch.is_tensor(image):
        image = torch.as_tensor(image)
    image = image.float()
    if image.ndim == 3 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 2:
        raise RuntimeError(
            f"Expected alternative PD [H,W] at index={index}, got {tuple(image.shape)}."
        )
    return image.to(device, non_blocking=True)


def corrupt_batch(
    pd_aux: torch.Tensor,
    sample_indices: Sequence[int],
    dataset: Any,
    negative_sampler: HardNegativeSampler,
    epoch: int,
    batch_index: int,
    seed: int,
    curriculum: str,
    config: CorruptionConfig,
) -> CorruptedBatch:
    """Apply per-sample corruption and return audit-ready supervision metadata."""
    config.validate()
    if pd_aux.ndim != 3:
        raise RuntimeError(f"Expected PD batch [B,H,W], got {tuple(pd_aux.shape)}")
    if len(sample_indices) != pd_aux.shape[0]:
        raise RuntimeError("sample_indices length does not match batch size.")

    mixture = curriculum_mixture(epoch, curriculum)
    rng = random.Random(seed + epoch * 1_000_003 + batch_index * 10_007)
    output = pd_aux.clone()
    availability = torch.ones(
        pd_aux.shape[0], device=pd_aux.device, dtype=pd_aux.dtype
    )
    targets = torch.full(
        (pd_aux.shape[0], len(SCALE_NAMES)),
        float(config.reliability_clean),
        device=pd_aux.device,
        dtype=pd_aux.dtype,
    )
    records: List[Dict[str, Any]] = []

    for sample_position, source_index in enumerate(sample_indices):
        condition = sample_from_mixture(mixture, rng)
        image = output[sample_position]
        metadata: Dict[str, Any] = {
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
        allow_coarse_relief = False

        if condition == "clean":
            target = config.reliability_clean

        elif condition == "missing":
            output[sample_position].zero_()
            availability[sample_position] = 0.0
            target = config.reliability_missing

        elif condition in {"shift", "shift_mild"}:
            allow_coarse_relief = True
            magnitudes = (2, 4) if condition == "shift_mild" else config.shift_magnitudes
            magnitude = int(rng.choice(magnitudes))
            padding_mode = sample_padding_mode(config, rng)
            # Epochs using shift_mild deliberately exclude zero padding while
            # the network first learns content disagreement.
            if condition == "shift_mild" and padding_mode == "zero":
                padding_mode = "reflect" if rng.random() < 0.78 else "replicate"
            dy, dx, direction, direction_class = sample_direction(
                magnitude,
                rng,
                allow_cardinal=True,
                allow_diagonal=True,
            )
            output[sample_position] = translate_nonwrapping(
                image, dy, dx, padding_mode
            )
            target = shift_reliability_target(magnitude, config)
            metadata.update(
                {
                    "condition": "shift",
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

        elif condition == "border":
            allow_coarse_relief = True
            width = int(rng.choice(config.shift_magnitudes))
            padding_mode = sample_border_padding_mode(config, rng)
            output[sample_position] = border_only(image, width, padding_mode)
            target = border_reliability_target(width, config)
            metadata.update(
                {
                    "condition_key": f"border{width}",
                    "padding_mode": padding_mode,
                    "magnitude_linf": width,
                    "magnitude_l2": float(width),
                }
            )

        elif condition == "wrong_slice":
            allow_coarse_relief = True
            candidate = negative_sampler.same_patient_wrong_slice(
                int(source_index), rng
            )
            if candidate is None:
                metadata["fallback_from"] = "wrong_slice"
                metadata["condition"] = "clean"
                metadata["condition_key"] = "clean"
                target = config.reliability_clean
                allow_coarse_relief = False
            else:
                replacement_index, delta_z = candidate
                replacement = load_pd_auxiliary(dataset, replacement_index, pd_aux.device)
                if replacement.shape != image.shape:
                    raise RuntimeError(
                        "Same-patient wrong-slice shape mismatch: "
                        f"{tuple(replacement.shape)} vs {tuple(image.shape)}."
                    )
                output[sample_position] = replacement
                target = wrong_slice_reliability_target(delta_z)
                metadata.update(
                    {
                        "replacement_index": int(replacement_index),
                        "delta_z_norm": float(delta_z),
                    }
                )

        elif condition == "wrong_patient":
            candidate = negative_sampler.wrong_patient_matched_level(
                int(source_index), tuple(int(value) for value in image.shape)
            )
            if candidate is None:
                # Do not introduce a shape shortcut. Fall back to a reflect
                # shift with the same low-reliability training role.
                magnitude = 8
                dy, dx, direction, direction_class = sample_direction(
                    magnitude, rng, True, True
                )
                output[sample_position] = translate_nonwrapping(
                    image, dy, dx, "reflect"
                )
                target = shift_reliability_target(magnitude, config)
                allow_coarse_relief = True
                metadata.update(
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
                    raise RuntimeError(
                        "Wrong-patient matched-level shape mismatch: "
                        f"{tuple(replacement.shape)} vs {tuple(image.shape)}."
                    )
                output[sample_position] = replacement
                target = config.reliability_wrong_patient
                metadata.update(
                    {
                        "replacement_index": int(replacement_index),
                        "delta_z_norm": float(delta_z),
                    }
                )

        elif condition == "blur":
            allow_coarse_relief = True
            sigma = rng.uniform(config.blur_sigma_min, config.blur_sigma_max)
            output[sample_position] = gaussian_blur_single(
                image, sigma, config.blur_kernel_size
            )
            target = linear_target(
                sigma,
                config.blur_sigma_min,
                config.blur_sigma_max,
                config.reliability_blur_light,
                config.reliability_blur_severe,
            )
            metadata.update({"severity": float(sigma)})

        elif condition == "noise":
            allow_coarse_relief = True
            ratio = rng.uniform(config.noise_std_min, config.noise_std_max)
            generator = torch.Generator(device=pd_aux.device)
            generator.manual_seed(
                seed
                + epoch * 1_000_003
                + batch_index * 10_007
                + sample_position * 101
            )
            noise = torch.randn(
                image.shape,
                device=image.device,
                dtype=image.dtype,
                generator=generator,
            )
            output[sample_position] = (
                image + ratio * image.std().clamp_min(1e-8) * noise
            ).clamp_min(0.0)
            target = linear_target(
                ratio,
                config.noise_std_min,
                config.noise_std_max,
                config.reliability_noise_light,
                config.reliability_noise_severe,
            )
            metadata.update({"severity": float(ratio)})

        else:
            raise RuntimeError(f"Unhandled condition: {condition}")

        scale_target_values = scale_targets_from_base(
            float(target),
            config,
            allow_coarse_relief=allow_coarse_relief,
        )
        targets[sample_position] = torch.tensor(
            scale_target_values,
            device=pd_aux.device,
            dtype=pd_aux.dtype,
        )
        metadata["reliability_target"] = float(
            sum(scale_target_values) / len(scale_target_values)
        )
        metadata["reliability_target_base"] = float(target)
        metadata["reliability_target_by_scale"] = {
            name: float(value)
            for name, value in zip(SCALE_NAMES, scale_target_values)
        }
        metadata["availability"] = float(availability[sample_position].item())
        records.append(metadata)

    unique_availability = torch.unique(availability.detach())
    if not bool(torch.logical_or(
        unique_availability == 0,
        unique_availability == 1,
    ).all().item()):
        raise RuntimeError(
            f"Availability must be hard 0/1, got {unique_availability.tolist()}."
        )

    return CorruptedBatch(output, availability, targets, records)
