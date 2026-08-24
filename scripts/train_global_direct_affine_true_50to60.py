#!/usr/bin/env python3
"""Affine=True paired branch of Global-direct epoch-50 low-LR continuation."""

from __future__ import annotations

import train_global_direct_low_lr_50to60 as base

from affine_ablation_runtime import annotate_final_summary, install_runtime


SOURCE_SHA256 = "9a686dd9eaa9611df17042b5610dac2daf48281a2b1efc17602142a15813c94f"
ARM_NAME = "global_direct_adapter_affine_true"


if __name__ == "__main__":
    output_dir = install_runtime(base, SOURCE_SHA256, ARM_NAME)
    base.main()
    annotate_final_summary(output_dir, ARM_NAME)

