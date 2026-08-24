#!/usr/bin/env python3
from __future__ import annotations

"""Audit StageA-Full invariance to missing-PD tensor contents.

The availability mask is fixed to zero while the auxiliary tensor is replaced
by zero, the real PD tensor, a deterministic transformed tensor, and a
deterministic nonzero pattern.  The held-out test set is never accessed.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_qmax_counterfactuals import (  # noqa: E402
    MANIFEST_PROTOCOL_VERSION,
    ManifestDataset,
)
from scripts.evaluate_stagea_full_epoch60_validation import (  # noqa: E402
    _build_model,
    _load_json,
    _validate_checkpoint,
    _validate_checkpoint_location,
    _validate_input_hashes,
)
from scripts.qmax_common import (  # noqa: E402
    IndexedDataset,
    ShapeBucketBatchSampler,
    autocast_context,
    code_hashes,
    make_dataset,
    prepare_batch,
    set_seed,
    sha256_file,
)


PROTOCOL_VERSION = "StageA-Full-epoch60-missing-tensor-invariance-v1"
OUTPUT_TOLERANCE = 1e-7


def _max_abs(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().abs().max().item())


def _finite_tree(prediction: torch.Tensor, auxiliary: Mapping[str, Any]) -> bool:
    if not bool(torch.isfinite(prediction).all()):
        return False
    return all(
        bool(torch.isfinite(value).all())
        for value in auxiliary.values()
        if torch.is_tensor(value)
    )


def _pattern_like(pd: torch.Tensor) -> torch.Tensor:
    height, width = pd.shape[-2:]
    yy = torch.linspace(-1.0, 1.0, height, device=pd.device)
    xx = torch.linspace(-1.0, 1.0, width, device=pd.device)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    pattern = (
        torch.sin(3.0 * math.pi * grid_x)
        + 0.5 * torch.cos(5.0 * math.pi * grid_y)
        + 0.25 * torch.sin(2.0 * math.pi * (grid_x + grid_y))
    )
    scale = pd.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return pattern.unsqueeze(0).expand_as(pd) * scale


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--full_clean_manifest", required=True)
    parser.add_argument("--robustness_manifest", required=True)
    parser.add_argument("--condition_manifest", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    if args.seed != 42 or not args.amp:
        raise ValueError("Locked safety audit requires seed=42 and AMP")
    if not torch.cuda.is_available():
        raise RuntimeError("Missing-tensor audit requires CUDA")
    set_seed(args.seed)
    device = torch.device("cuda")

    paths: Dict[str, Path] = {}
    for name in (
        "checkpoint",
        "metadata_csv",
        "full_clean_manifest",
        "robustness_manifest",
        "condition_manifest",
    ):
        path = Path(getattr(args, name)).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        paths[name] = path
    _validate_checkpoint_location(paths["checkpoint"])
    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise RuntimeError(f"Refusing to overwrite safety audit: {output_path}")

    checkpoint = torch.load(
        paths["checkpoint"], map_location="cpu", weights_only=False
    )
    checkpoint_audit = _validate_checkpoint(
        paths["checkpoint"], checkpoint, code_hashes(PROJECT_ROOT)
    )
    _validate_input_hashes(checkpoint["config"], paths)
    model = _build_model(checkpoint, device)

    robust_manifest = _load_json(paths["robustness_manifest"])
    if (
        robust_manifest.get("protocol_version") != MANIFEST_PROTOCOL_VERSION
        or robust_manifest.get("cohort") != "robustness"
    ):
        raise RuntimeError("Robustness manifest protocol/cohort mismatch")
    source = IndexedDataset(
        make_dataset(
            str(paths["metadata_csv"]),
            "val",
            acceleration=8,
            pd_aux_acceleration=2,
        )
    )
    dataset = ManifestDataset(source, robust_manifest)
    loader = DataLoader(
        dataset,
        batch_sampler=ShapeBucketBatchSampler(
            dataset, args.batch_size, False, args.seed
        ),
        num_workers=args.num_workers,
        pin_memory=True,
    )

    variants = ("real_pd", "transformed_pd", "nonzero_pattern")
    max_output_delta = {name: 0.0 for name in variants}
    max_q_delta = {name: 0.0 for name in variants}
    max_direct = {name: 0.0 for name in ("zero_pd", *variants)}
    max_correction = {name: 0.0 for name in ("zero_pd", *variants)}
    max_final_auxiliary = {name: 0.0 for name in ("zero_pd", *variants)}
    max_dc_on_minus_zero_output = 0.0
    all_finite = True
    num_slices = 0

    def run(
        kspace: torch.Tensor,
        mask: torch.Tensor,
        pd_value: torch.Tensor,
        available: torch.Tensor,
        *,
        dc_zero: bool = False,
    ):
        with autocast_context(device, args.amp):
            return model(
                kspace,
                mask,
                pd_value,
                available,
                return_aux=True,
                dc_zero=dc_zero,
            )

    for batch in loader:
        kspace, mask, pd, _target, _indices = prepare_batch(batch, device)
        available = torch.zeros(pd.shape[0], device=device)
        zero_pd = torch.zeros_like(pd)
        pattern = _pattern_like(pd)
        transformed = -0.75 * torch.flip(pd, dims=(-2, -1)) + 0.25 * pattern
        tensors = {
            "zero_pd": zero_pd,
            "real_pd": pd,
            "transformed_pd": transformed,
            "nonzero_pattern": pattern,
        }

        baseline_prediction, baseline_aux = run(
            kspace, mask, zero_pd, available
        )
        all_finite = all_finite and _finite_tree(
            baseline_prediction, baseline_aux
        )
        baseline_q = baseline_aux["q_hat"]
        pattern_prediction_cache = None
        for name, pd_value in tensors.items():
            if name == "zero_pd":
                prediction, auxiliary = baseline_prediction, baseline_aux
            else:
                prediction, auxiliary = run(
                    kspace, mask, pd_value, available
                )
                max_output_delta[name] = max(
                    max_output_delta[name],
                    _max_abs(prediction - baseline_prediction),
                )
                max_q_delta[name] = max(
                    max_q_delta[name],
                    _max_abs(auxiliary["q_hat"] - baseline_q),
                )
                if name == "nonzero_pattern":
                    pattern_prediction_cache = prediction
            all_finite = all_finite and _finite_tree(prediction, auxiliary)
            max_direct[name] = max(
                max_direct[name],
                _max_abs(auxiliary["direct_to_target_rms"]),
            )
            max_correction[name] = max(
                max_correction[name],
                _max_abs(auxiliary["correction_to_target_rms"]),
            )
            max_final_auxiliary[name] = max(
                max_final_auxiliary[name],
                _max_abs(auxiliary["final_auxiliary_to_target_rms"]),
            )

        dc_zero_prediction, dc_zero_aux = run(
            kspace, mask, pattern, available, dc_zero=True
        )
        all_finite = all_finite and _finite_tree(
            dc_zero_prediction, dc_zero_aux
        )
        if pattern_prediction_cache is None:
            raise RuntimeError("Nonzero-pattern forward was not executed")
        max_dc_on_minus_zero_output = max(
            max_dc_on_minus_zero_output,
            _max_abs(pattern_prediction_cache - dc_zero_prediction),
        )
        num_slices += int(pd.shape[0])

    output_content_pass = all(
        value <= OUTPUT_TOLERANCE for value in max_output_delta.values()
    )
    direct_zero_pass = all(value == 0.0 for value in max_direct.values())
    correction_zero_pass = all(
        value == 0.0 for value in max_correction.values()
    )
    final_auxiliary_zero_pass = all(
        value == 0.0 for value in max_final_auxiliary.values()
    )
    dc_bypass_pass = max_dc_on_minus_zero_output <= OUTPUT_TOLERANCE
    passed = bool(
        num_slices > 0
        and all_finite
        and output_content_pass
        and direct_zero_pass
        and correction_zero_pass
        and final_auxiliary_zero_pass
        and dc_bypass_pass
    )
    audit = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "passed" if passed else "failed",
        "scope": "locked robustness validation cohort; held-out test not accessed",
        "checkpoint_audit": checkpoint_audit,
        "availability_fixed_to_zero": True,
        "tensor_variants": ["zero_pd", *variants],
        "output_tolerance": OUTPUT_TOLERANCE,
        "max_abs_output_delta_vs_zero_tensor": max_output_delta,
        "max_abs_q_delta_vs_zero_tensor": max_q_delta,
        "q_is_allowed_to_change_because_m_masks_its_effect": True,
        "max_direct_rms": max_direct,
        "max_correction_rms": max_correction,
        "max_final_auxiliary_rms": max_final_auxiliary,
        "max_abs_output_delta_dc_on_vs_dc_zero_when_missing": (
            max_dc_on_minus_zero_output
        ),
        "checks": {
            "output_independent_of_missing_tensor_content": output_content_pass,
            "direct_exact_zero": direct_zero_pass,
            "correction_exact_zero": correction_zero_pass,
            "final_auxiliary_exact_zero": final_auxiliary_zero_pass,
            "dc_evidence_cannot_bypass_availability": dc_bypass_pass,
            "all_forward_outputs_finite": all_finite,
        },
        "num_slices": num_slices,
        "num_patients": len(
            {str(record["patient_id"]) for record in dataset.records}
        ),
        "input_hashes": {
            key: sha256_file(value) for key, value in paths.items()
        },
    }
    output_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
