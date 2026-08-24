#!/usr/bin/env python3
"""Restore serialized RNG byte states after CUDA map_location loading.

The frozen continuation trainer loads the complete checkpoint with
``map_location=device``.  This correctly moves model and optimizer tensors to
CUDA, but it also moves the serialized CPU RNG ByteTensor.  PyTorch requires
``torch.set_rng_state`` (and CUDA RNG state setters) to receive CPU uint8
tensors.  This runtime-only launcher converts only those RNG state arguments
back to CPU uint8 before delegating to the unchanged AMP-SAFE launcher.

No RNG bytes are generated, discarded, reordered, or modified.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path
from typing import Any, Iterable

import torch


_ORIGINAL_SET_RNG_STATE = torch.set_rng_state
_ORIGINAL_CUDA_SET_RNG_STATE_ALL = torch.cuda.set_rng_state_all


def _cpu_byte_state(value: Any) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(
            "Serialized RNG state must be a torch.Tensor, got "
            f"{type(value).__name__}"
        )
    if value.dtype != torch.uint8:
        raise TypeError(
            "Serialized RNG state must retain uint8 dtype, got "
            f"{value.dtype}"
        )
    state = value.detach().to(device="cpu").contiguous()
    if state.ndim != 1 or state.numel() == 0:
        raise RuntimeError(
            f"Invalid RNG-state shape after restoration: {tuple(state.shape)}"
        )
    return state


def _set_cpu_rng_state_compat(value: Any) -> None:
    _ORIGINAL_SET_RNG_STATE(_cpu_byte_state(value))


def _set_cuda_rng_state_all_compat(values: Iterable[Any]) -> None:
    restored = [_cpu_byte_state(value) for value in values]
    if len(restored) != torch.cuda.device_count():
        raise RuntimeError(
            "CUDA RNG-state count differs from visible CUDA device count: "
            f"states={len(restored)}, devices={torch.cuda.device_count()}"
        )
    _ORIGINAL_CUDA_SET_RNG_STATE_ALL(restored)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("amp_safe_launcher", type=Path)
    launcher_args, delegated_args = parser.parse_known_args()

    launcher = launcher_args.amp_safe_launcher.resolve()
    if not launcher.is_file():
        raise FileNotFoundError(launcher)

    torch.set_rng_state = _set_cpu_rng_state_compat
    torch.cuda.set_rng_state_all = _set_cuda_rng_state_all_compat

    print(
        "[RNG-COMPAT] CPU/CUDA RNG setters will receive unchanged CPU "
        "uint8 byte states after checkpoint map_location.",
        flush=True,
    )
    sys.argv = [str(launcher), *delegated_args]
    runpy.run_path(str(launcher), run_name="__main__")


if __name__ == "__main__":
    main()
