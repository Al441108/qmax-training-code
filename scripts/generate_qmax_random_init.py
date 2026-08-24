#!/usr/bin/env python3
from __future__ import annotations

"""Generate and audit the single step-0 QMax seed-42 template."""

import argparse
import hashlib
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

from src.m2_prnf_fusion_pilot_varnet import (  # noqa: E402
    M2PRNFFusionPilotVarNet,
)
from src.m2_prnf_qmax_varnet import (  # noqa: E402
    QMaxAuxPDVarNet,
    copy_matching_state,
    initialise_qmax_full_from_core,
    qmax_dc_input_columns,
    qmax_shared_state,
)


PROTOCOL_VERSION = "QMax-StageA-R8-R2-v2"


def initialisation_source_hashes(project_root: Path) -> Dict[str, str]:
    paths = (
        "src/m2_prnf_varnet.py",
        "src/m2_prnf_fusion_pilot_varnet.py",
        "src/m2_prnf_qmax_varnet.py",
        "scripts/generate_qmax_random_init.py",
        "QMAX_STAGE_A_PROTOCOL_R8.json",
    )
    return {
        relative: sha256_file(project_root / relative)
        for relative in paths
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def maximum_difference(
    first: Dict[str, torch.Tensor],
    second: Dict[str, torch.Tensor],
    keys: Iterable[str],
) -> float:
    values = []
    for key in keys:
        if key not in first or key not in second:
            raise RuntimeError(f"Missing shared key: {key}")
        if first[key].shape != second[key].shape:
            raise RuntimeError(f"Shared shape differs for {key}")
        values.append(float((first[key] - second[key]).abs().max().item()))
    return max(values, default=0.0)


def model_kwargs(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "num_cascades": int(args.num_cascades),
        "sens_chans": int(args.sens_chans),
        "sens_pools": int(args.sens_pools),
        "chans": int(args.chans),
        "pools": int(args.pools),
        "controller_chans": int(args.controller_chans),
        "initial_aux_alpha": float(args.initial_aux_alpha),
        "initial_gate_probability": float(
            args.initial_gate_probability
        ),
    }


def build_from_template(
    template: Dict[str, object],
    variant: str,
) -> QMaxAuxPDVarNet:
    """Public helper used by preflight and training."""

    kwargs = dict(template["model_kwargs"])
    core = QMaxAuxPDVarNet(qmax_variant="qmax_core", **kwargs)
    core.load_state_dict(template["p1_state_dict"], strict=True)
    if variant == "qmax_core":
        return core
    if variant != "qmax_full":
        raise ValueError(variant)
    full = QMaxAuxPDVarNet(qmax_variant="qmax_full", **kwargs)
    initialise_qmax_full_from_core(core, full)
    return full


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
    args = parser.parse_args()

    if args.seed != 42:
        raise ValueError("Frozen Stage-A template requires seed=42")
    kwargs = model_kwargs(args)

    # P1 is the canonical random initialisation template.
    set_seed(args.seed)
    p1 = QMaxAuxPDVarNet(qmax_variant="qmax_core", **kwargs)

    # P2 receives every common tensor from P1.  Only the new DC input columns
    # differ, and they are strictly zero.
    set_seed(args.seed)
    p2 = QMaxAuxPDVarNet(qmax_variant="qmax_full", **kwargs)
    p2_copy_report = initialise_qmax_full_from_core(p1, p2)

    # P0 is built only for the step-0 audit.  Existing fifth-arm training is
    # not overwritten or rerun.
    set_seed(args.seed)
    p0 = M2PRNFFusionPilotVarNet(
        model_variant="prnf_no_need",
        fusion_design="hybrid_direct_residual",
        need_scope="residual",
        residual_scale=0.1,
        initial_need_probability=0.95,
        need_floor=0.25,
        **kwargs,
    )
    p0_natural_state = {
        key: value.detach().clone()
        for key, value in p0.state_dict().items()
    }
    p1_state = p1.state_dict()
    p2_state = p2.state_dict()
    backbone_prefixes = (
        "sens_net.",
        "pd_encoder.",
        "cascades.",
    )
    # Controller-specific modules are excluded from the backbone audit.
    backbone_keys = [
        key
        for key in p1_state
        if key.startswith(backbone_prefixes)
        and ".fusions." not in key
    ]
    q_keys = [
        key for key in p1_state if ".reliability." in key
    ]
    p0_natural_backbone_max_diff = maximum_difference(
        p0_natural_state, p1_state, backbone_keys
    )
    p0_natural_q_max_diff = maximum_difference(
        p0_natural_state, p1_state, q_keys
    )
    # Enforce the same tensors as well as proving the natural seeded build
    # already agrees. P0 is not saved or retrained by this script.
    p0_copy_report = copy_matching_state(p1, p0)
    p0_state = p0.state_dict()
    p1_p2_shared_keys = sorted(qmax_shared_state(p1))
    dc_columns = qmax_dc_input_columns(p2)
    audit = {
        "protocol_version": PROTOCOL_VERSION,
        "seed": int(args.seed),
        "model_kwargs": kwargs,
        "source_hashes": initialisation_source_hashes(PROJECT_ROOT),
        "p0_copy_report": {
            "num_copied": p0_copy_report["num_copied"]
        },
        "p2_copy_report": {
            "num_same_shape": p2_copy_report["num_same_shape"],
            "num_extended": p2_copy_report["num_extended"],
            "zero_extended_inputs": p2_copy_report[
                "zero_extended_inputs"
            ],
        },
        "p0_p1_shared_backbone_max_diff": maximum_difference(
            p0_state, p1_state, backbone_keys
        ),
        "p0_p1_shared_q_max_diff": maximum_difference(
            p0_state, p1_state, q_keys
        ),
        "p0_p1_natural_seeded_backbone_max_diff": (
            p0_natural_backbone_max_diff
        ),
        "p0_p1_natural_seeded_q_max_diff": p0_natural_q_max_diff,
        "p1_p2_shared_max_diff": maximum_difference(
            p1_state, p2_state, p1_p2_shared_keys
        ),
        "p2_dc_columns_max_abs": max(
            (
                float(value.abs().max().item())
                for value in dc_columns.values()
            ),
            default=float("nan"),
        ),
        "num_backbone_keys": len(backbone_keys),
        "num_q_keys": len(q_keys),
        "num_p1_p2_shared_keys": len(p1_p2_shared_keys),
        "num_dc_columns": len(dc_columns),
    }
    audit["passed"] = (
        audit["p0_p1_shared_backbone_max_diff"] == 0.0
        and audit["p0_p1_shared_q_max_diff"] == 0.0
        and audit["p0_p1_natural_seeded_backbone_max_diff"] == 0.0
        and audit["p0_p1_natural_seeded_q_max_diff"] == 0.0
        and audit["p1_p2_shared_max_diff"] == 0.0
        and audit["p2_dc_columns_max_abs"] == 0.0
        and audit["num_dc_columns"] == 8
    )
    if not audit["passed"]:
        raise RuntimeError(
            "Step-0 initialisation audit failed:\n"
            + json.dumps(audit, indent=2)
        )

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output) + ".tmp")
    torch.save(
        {
            "protocol_version": PROTOCOL_VERSION,
            "seed": int(args.seed),
            "model_kwargs": kwargs,
            "source_hashes": audit["source_hashes"],
            "p1_state_dict": {
                key: value.detach().cpu().clone()
                for key, value in p1_state.items()
            },
            "initialisation_audit": audit,
        },
        temporary,
    )
    os.replace(temporary, output)
    template_hash = sha256_file(output)
    hash_path = Path(str(output) + ".sha256")
    hash_path.write_text(
        f"{template_hash}  {output.name}\n", encoding="utf-8"
    )
    audit.update(
        {
            "template": str(output),
            "template_sha256": template_hash,
        }
    )
    audit_path = Path(args.audit_json).resolve()
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
