import hashlib

import numpy as np
import torch


def stable_mask_seed(patient_id, slice_idx, acceleration):
    """
    Create a reproducible seed from patient, slice and acceleration.

    Python's built-in hash() must not be used because it is not guaranteed
    to be stable across processes or sessions.
    """
    seed_string = f"{patient_id}_{slice_idx}_{acceleration}"
    return int(
        hashlib.md5(seed_string.encode("utf-8")).hexdigest()[:8],
        16,
    )


def make_gaussian_vd_mask(
    num_cols,
    acceleration,
    seed,
    center_fraction=None,
):
    """
    Create the fixed 1D Gaussian variable-density Cartesian mask used by
    the formal zero-filled baseline and later model experiments.

    Returns
    -------
    mask : np.ndarray
        Float32 array with shape [num_cols].
    num_samples : int
        Number of sampled phase-encoding lines.
    actual_R : float
        Actual acceleration factor.
    """
    rng = np.random.default_rng(seed)

    if center_fraction is None:
        center_fractions = {
            2: 0.16,
            4: 0.08,
            6: 0.06,
            8: 0.04,
        }

        if acceleration not in center_fractions:
            raise ValueError(
                f"Unsupported acceleration: {acceleration}. "
                f"Expected one of {sorted(center_fractions)}."
            )

        center_fraction = center_fractions[acceleration]

    num_low_freqs = max(
        1,
        int(round(num_cols * center_fraction)),
    )

    mask = np.zeros(num_cols, dtype=np.float32)

    center = num_cols // 2
    half = num_low_freqs // 2
    start = max(0, center - half)
    end = min(num_cols, start + num_low_freqs)

    mask[start:end] = 1.0

    target_samples = int(round(num_cols / acceleration))
    remaining_samples = max(
        0,
        target_samples - int(mask.sum()),
    )

    if remaining_samples > 0:
        x = np.arange(num_cols)
        sigma = num_cols / 6.0

        probabilities = np.exp(
            -0.5 * ((x - center) / sigma) ** 2
        )
        probabilities[mask == 1] = 0.0

        if probabilities.sum() <= 0:
            raise RuntimeError(
                "Gaussian sampling probabilities sum to zero."
            )

        probabilities /= probabilities.sum()

        available = np.where(mask == 0)[0]
        chosen = rng.choice(
            np.arange(num_cols),
            size=min(remaining_samples, len(available)),
            replace=False,
            p=probabilities,
        )
        mask[chosen] = 1.0

    num_samples = int(mask.sum())
    actual_R = num_cols / float(num_samples)

    return mask, num_samples, float(actual_R)


def apply_1d_mask_to_kspace(kspace, mask):
    """
    Apply a 1D phase-encoding mask to multicoil k-space.

    Parameters
    ----------
    kspace : np.ndarray or torch.Tensor
        Shape [coils, height, width].
    mask : np.ndarray or torch.Tensor
        Shape [width].
    """
    if kspace.ndim != 3:
        raise ValueError(
            "Expected multicoil k-space shape "
            f"[coils, height, width], got {tuple(kspace.shape)}."
        )

    if kspace.shape[-1] != len(mask):
        raise ValueError(
            f"Mask length {len(mask)} does not match "
            f"k-space width {kspace.shape[-1]}."
        )

    if torch.is_tensor(kspace):
        mask_tensor = torch.as_tensor(
            mask,
            dtype=kspace.real.dtype,
            device=kspace.device,
        )
        return kspace * mask_tensor.view(1, 1, -1)

    mask_array = np.asarray(mask, dtype=np.float32)
    return kspace * mask_array.reshape(1, 1, -1)


def random_cartesian_mask(
    shape,
    center_fraction=0.08,
    acceleration=4,
    seed=None,
):
    """
    Legacy random Cartesian mask retained for older exploratory scripts.

    This function is not the formal mask used in the controlled multicoil
    experiments.
    """
    height, width = shape
    rng = np.random.default_rng(seed)

    num_low_freqs = int(round(width * center_fraction))
    probability = (
        width / acceleration - num_low_freqs
    ) / (
        width - num_low_freqs
    )
    probability = max(0.0, min(1.0, probability))

    mask_1d = rng.uniform(size=width) < probability

    pad = (width - num_low_freqs + 1) // 2
    mask_1d[pad:pad + num_low_freqs] = True

    mask = np.tile(
        mask_1d[None, :],
        (height, 1),
    ).astype(np.float32)

    return torch.from_numpy(mask)
