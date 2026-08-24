#!/usr/bin/env python3
from __future__ import annotations

"""Generate one paired random-init template for CompactSwin Core and Full."""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.qmax_stage_b_versioning import (  # noqa: E402
    AUDIT_VERSION,
    RUNTIME_VERSION,
    STRUCTURE_VERSION,
    manifest_digest,
    sha256_file,
    stage_b_audit_hashes,
    stage_b_runtime_hashes,
    stage_b_structure_hashes,
)
from src.m2_prnf_qmax_compactswin_varnet import (  # noqa: E402
    COMPACTSWIN_LAYER_SCALE_INIT,
    COMPACTSWIN_WINDOW_SIZE,
    QMaxCompactSwinAuxPDVarNet,
)
from src.m2_prnf_qmax_varnet import (  # noqa: E402
    QMAX_VARIANTS,
    initialise_qmax_full_from_core,
    qmax_dc_input_columns,
    qmax_shared_state,
)


PROTOCOL_VERSION = "QMax-StageB-Independent-CoreFull-R8-R2-v1"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def model_kwargs(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "num_cascades": int(args.num_cascades),
        "sens_chans": int(args.sens_chans),
        "sens_pools": int(args.sens_pools),
        "chans": int(args.chans),
        "pools": int(args.pools),
        "controller_chans": int(args.controller_chans),
        "initial_aux_alpha": float(args.initial_aux_alpha),
        "initial_gate_probability": float(args.initial_gate_probability),
    }


def compactswin_kwargs(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "h8_blocks": int(args.h8_blocks),
        "h16_blocks": int(args.h16_blocks),
        "window_size": int(args.window_size),
        "activation_checkpointing": bool(args.activation_checkpointing),
    }


def build_stage_b_from_template(
    template: Dict[str, object],
    variant: str,
) -> QMaxCompactSwinAuxPDVarNet:
    if variant not in QMAX_VARIANTS:
        raise ValueError(variant)
    model = QMaxCompactSwinAuxPDVarNet(
        qmax_variant=variant,
        **dict(template["model_kwargs"]),
        **dict(template["compactswin_kwargs"]),
    )
    state_key = (
        "core_state_dict" if variant == "qmax_core" else "full_state_dict"
    )
    model.load_state_dict(template[state_key], strict=True)
    return model


def maximum_difference(
    first: Dict[str, torch.Tensor],
    second: Dict[str, torch.Tensor],
    keys: Iterable[str],
) -> float:
    differences = []
    for key in keys:
        if key not in first or key not in second:
            raise RuntimeError(f"Missing paired initialisation key: {key}")
        if first[key].shape != second[key].shape:
            raise RuntimeError(f"Paired shape mismatch for {key}")
        differences.append(
            float((first[key] - second[key]).abs().max().item())
        )
    return max(differences, default=0.0)


def _extended_old_columns_max_difference(
    core: QMaxCompactSwinAuxPDVarNet,
    full: QMaxCompactSwinAuxPDVarNet,
) -> float:
    values = []
    core_state = core.state_dict()
    full_state = full.state_dict()
    for key, core_value in core_state.items():
        if not key.endswith(
            ("detail_head.in_proj.weight", "correction_head.in_proj.weight")
        ):
            continue
        full_value = full_state[key]
        if full_value.shape[1] != core_value.shape[1] + 1:
            raise RuntimeError(f"Expected one added DC channel for {key}")
        values.append(
            float(
                (
                    core_value
                    - full_value[:, : core_value.shape[1]]
                )
                .abs()
                .max()
                .item()
            )
        )
    return max(values, default=float("nan"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit_json", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_cascades", type=int, default=12)
    parser.add_argument("--chans", type=int, default=18)
    parser.add_argument("--sens_chans", type=int, default=8)
    parser.add_argument("--pools", type=int, default=4)
    parser.add_argument("--sens_pools", type=int, default=4)
    parser.add_argument("--controller_chans", type=int, default=16)
    parser.add_argument("--initial_aux_alpha", type=float, default=0.1)
    parser.add_argument(
        "--initial_gate_probability", type=float, default=0.95
    )
    parser.add_argument("--h8_blocks", type=int, default=2)
    parser.add_argument("--h16_blocks", type=int, default=2)
    parser.add_argument(
        "--window_size", type=int, default=COMPACTSWIN_WINDOW_SIZE
    )
    parser.add_argument(
        "--activation_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    if args.seed != 42:
        raise ValueError("Frozen Stage B requires seed=42")
    if args.chans != 18 or args.pools != 4:
        raise ValueError("Frozen CompactSwin requires chans=18 and pools=4")
    if args.window_size != 8:
        raise ValueError("Frozen CompactSwin window size is 8")
    if args.h8_blocks != 2 or args.h16_blocks != 2:
        raise ValueError("Frozen CompactSwin requires two H/8 and H/16 blocks")
    if not args.activation_checkpointing:
        raise ValueError("Frozen CompactSwin requires activation checkpointing")

    model_config = model_kwargs(args)
    swin_config = compactswin_kwargs(args)
    set_seed(args.seed)
    core = QMaxCompactSwinAuxPDVarNet(
        qmax_variant="qmax_core", **model_config, **swin_config
    )
    set_seed(args.seed)
    full = QMaxCompactSwinAuxPDVarNet(
        qmax_variant="qmax_full", **model_config, **swin_config
    )
    copy_report = initialise_qmax_full_from_core(core, full)

    core_state = core.state_dict()
    full_state = full.state_dict()
    shared_keys = sorted(qmax_shared_state(core))
    q_keys = [key for key in shared_keys if ".reliability." in key]
    backbone_keys = [
        key
        for key in shared_keys
        if key.startswith(("sens_net.", "pd_encoder.", "cascades."))
        and ".fusions." not in key
    ]
    dc_columns = qmax_dc_input_columns(full)
    layer_scales = [
        parameter.detach()
        for name, parameter in core.named_parameters()
        if "layer_scale_" in name
    ]
    structure_hashes = stage_b_structure_hashes(PROJECT_ROOT)
    runtime_hashes = stage_b_runtime_hashes(PROJECT_ROOT)
    audit_hashes = stage_b_audit_hashes(PROJECT_ROOT)
    expected_layer_scale = float(
        torch.tensor(COMPACTSWIN_LAYER_SCALE_INIT).item()
    )
    audit = {
        "protocol_version": PROTOCOL_VERSION,
        "structure_version": STRUCTURE_VERSION,
        "runtime_version": RUNTIME_VERSION,
        "audit_version": AUDIT_VERSION,
        "seed": args.seed,
        "variants": sorted(QMAX_VARIANTS),
        "model_kwargs": model_config,
        "compactswin_kwargs": swin_config,
        "copy_report": {
            "num_same_shape": copy_report["num_same_shape"],
            "num_extended": copy_report["num_extended"],
            "zero_extended_inputs": copy_report["zero_extended_inputs"],
        },
        "core_full_shared_max_diff": maximum_difference(
            core_state, full_state, shared_keys
        ),
        "core_full_backbone_max_diff": maximum_difference(
            core_state, full_state, backbone_keys
        ),
        "core_full_q_max_diff": maximum_difference(
            core_state, full_state, q_keys
        ),
        "extended_old_columns_max_diff": (
            _extended_old_columns_max_difference(core, full)
        ),
        "full_dc_columns_max_abs": max(
            (
                float(value.abs().max().item())
                for value in dc_columns.values()
            ),
            default=float("nan"),
        ),
        "num_dc_columns": len(dc_columns),
        "num_shared_keys": len(shared_keys),
        "num_shared_q_keys": len(q_keys),
        "num_shared_backbone_keys": len(backbone_keys),
        "layer_scale_min": min(
            float(value.min().item()) for value in layer_scales
        ),
        "layer_scale_max": max(
            float(value.max().item()) for value in layer_scales
        ),
        "structure_hashes": structure_hashes,
        "structure_digest": manifest_digest(structure_hashes),
        "runtime_tool_hashes_at_generation": runtime_hashes,
        "runtime_tool_digest_at_generation": manifest_digest(runtime_hashes),
        "audit_tool_hashes_at_generation": audit_hashes,
        "audit_tool_digest_at_generation": manifest_digest(audit_hashes),
    }
    audit["passed"] = (
        audit["core_full_shared_max_diff"] == 0.0
        and audit["core_full_backbone_max_diff"] == 0.0
        and audit["core_full_q_max_diff"] == 0.0
        and audit["extended_old_columns_max_diff"] == 0.0
        and audit["full_dc_columns_max_abs"] == 0.0
        and audit["num_dc_columns"] == 8
        and audit["num_shared_q_keys"] > 0
        and audit["layer_scale_min"] == expected_layer_scale
        and audit["layer_scale_max"] == expected_layer_scale
    )
    if not audit["passed"]:
        raise RuntimeError(
            "Paired CompactSwin initialisation failed:\n"
            + json.dumps(audit, indent=2)
        )

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output) + ".tmp")
    torch.save(
        {
            "protocol_version": PROTOCOL_VERSION,
            "seed": args.seed,
            "variants": sorted(QMAX_VARIANTS),
            "model_kwargs": model_config,
            "compactswin_kwargs": swin_config,
            "structure_version": STRUCTURE_VERSION,
            "structure_hashes": structure_hashes,
            "structure_digest": audit["structure_digest"],
            "core_state_dict": {
                key: value.detach().cpu().clone()
                for key, value in core_state.items()
            },
            "full_state_dict": {
                key: value.detach().cpu().clone()
                for key, value in full_state.items()
            },
            "initialisation_audit": audit,
        },
        temporary,
    )
    os.replace(temporary, output)
    template_hash = sha256_file(output)
    Path(str(output) + ".sha256").write_text(
        f"{template_hash}  {output.name}\n", encoding="utf-8"
    )
    audit["template"] = str(output)
    audit["template_sha256"] = template_hash
    audit_path = Path(args.audit_json).resolve()
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
