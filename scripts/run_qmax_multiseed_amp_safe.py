#!/usr/bin/env python3
"""Run an existing QMax trainer with recoverable AMP-overflow handling.

The original trainers call GradScaler.unscale_() before clip_grad_norm_().
GradScaler therefore records non-finite gradients before this wrapper replaces
the non-finite norm with a finite logging sentinel.  GradScaler.step() then
skips the optimizer update and GradScaler.update() backs the scale off, which
is PyTorch AMP's intended recovery path.

The original training files are not modified, preserving their bound hashes.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path
from typing import Any

import torch


_ORIGINAL_CLIP_GRAD_NORM = torch.nn.utils.clip_grad_norm_
_consecutive_overflows = 0
_total_overflows = 0
_MAX_CONSECUTIVE_OVERFLOWS = 8


def _amp_safe_clip_grad_norm_(
    parameters: Any,
    max_norm: float,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor:
    """Preserve clipping while allowing GradScaler to recover from overflow."""
    global _consecutive_overflows, _total_overflows

    # The trainer itself performs the policy check.  Do not let the PyTorch
    # helper raise before GradScaler.step()/update() can handle an AMP overflow.
    kwargs["error_if_nonfinite"] = False
    norm = _ORIGINAL_CLIP_GRAD_NORM(parameters, max_norm, *args, **kwargs)
    norm_tensor = norm if torch.is_tensor(norm) else torch.as_tensor(norm)

    if bool(torch.isfinite(norm_tensor).all()):
        _consecutive_overflows = 0
        return norm

    _consecutive_overflows += 1
    _total_overflows += 1
    print(
        "[AMP-SAFE] non-finite unscaled gradient detected; "
        "GradScaler will skip this optimizer step and lower its scale "
        f"(total={_total_overflows}, consecutive={_consecutive_overflows}, "
        f"reported_norm={norm_tensor.detach().cpu().item()!r}).",
        flush=True,
    )

    if _consecutive_overflows >= _MAX_CONSECUTIVE_OVERFLOWS:
        raise RuntimeError(
            "Eight consecutive non-finite gradient norms were observed. "
            "This is no longer treated as an isolated AMP overflow."
        )

    # The trainer checks this return value and logs it. GradScaler already saw
    # the non-finite gradients during unscale_(), so its step remains skipped.
    return torch.zeros((), device=norm_tensor.device, dtype=norm_tensor.dtype)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a QMax training script with AMP overflow recovery."
    )
    parser.add_argument("training_script", type=Path)
    wrapper_args, training_args = parser.parse_known_args()

    training_script = wrapper_args.training_script.resolve()
    if not training_script.is_file():
        raise FileNotFoundError(training_script)

    torch.nn.utils.clip_grad_norm_ = _amp_safe_clip_grad_norm_
    sys.path.insert(0, str(training_script.parent))
    sys.argv = [str(training_script), *training_args]

    print(
        f"[AMP-SAFE] launching unchanged trainer: {training_script}",
        flush=True,
    )
    runpy.run_path(str(training_script), run_name="__main__")


if __name__ == "__main__":
    main()
