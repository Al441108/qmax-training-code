#!/usr/bin/env python3
"""Affine=True paired branch of Hybrid-gain epoch-50 low-LR continuation."""

from __future__ import annotations

import train_hybrid_gain_low_lr_50to60 as base

from affine_ablation_runtime import annotate_final_summary, install_runtime


SOURCE_SHA256 = "6b035ca46a20e7ba192f14b2cdc42e7525d734414d1720b16736cca101149829"
ARM_NAME = "quality_protected_hybrid_gain_adapter_affine_true"


if __name__ == "__main__":
    output_dir = install_runtime(base, SOURCE_SHA256, ARM_NAME)
    base.main()
    annotate_final_summary(output_dir, ARM_NAME)

