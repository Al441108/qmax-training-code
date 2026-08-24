#!/usr/bin/env python3
"""Audited runtime support for the InstanceNorm affine=True paired ablation.

This module deliberately leaves the locked reconstruction source untouched.
It converts only the 48 M2PRNFFeatureFusion adapter InstanceNorm layers from
affine=False to affine=True, initialises gamma=1 and beta=0, loads every shared
model/Adam state exactly, and leaves the new affine parameters with empty Adam
state so that their moments begin at the branch point.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn


AFFINE_PARAMETER_NAMES: set[str] = set()
EXPECTED_SOURCE_SHA256: str | None = None
RESUME_PATH: Path | None = None
ORIGINAL_TORCH_LOAD = torch.load
ORIGINAL_ADAM = torch.optim.Adam


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def enable_affine_adapters(model: nn.Module) -> set[str]:
    """Replace exactly 48 adapter InstanceNorm2d layers with identity affine IN."""
    converted: list[str] = []
    for module_name, module in model.named_modules():
        if module.__class__.__name__ != "M2PRNFFeatureFusion":
            continue
        adapter = getattr(module, "adapter", None)
        if not isinstance(adapter, nn.Sequential) or len(adapter) != 2:
            raise RuntimeError(f"Unexpected adapter at {module_name}: {adapter}")
        old = adapter[1]
        if not isinstance(old, nn.InstanceNorm2d):
            raise RuntimeError(
                f"Expected InstanceNorm2d at {module_name}.adapter.1, got {type(old)}"
            )
        if old.affine:
            raise RuntimeError(
                f"Source model already has affine=True at {module_name}.adapter.1"
            )
        conv = adapter[0]
        reference = next(conv.parameters())
        new = nn.InstanceNorm2d(
            old.num_features,
            eps=old.eps,
            momentum=old.momentum,
            affine=True,
            track_running_stats=old.track_running_stats,
            device=reference.device,
            dtype=reference.dtype,
        )
        with torch.no_grad():
            new.weight.fill_(1.0)
            new.bias.zero_()
        adapter[1] = new
        converted.append(f"{module_name}.adapter.1")

    if len(converted) != 48:
        raise RuntimeError(f"Expected 48 fusion adapters, converted {len(converted)}")
    names = {
        f"{prefix}.{suffix}"
        for prefix in converted
        for suffix in ("weight", "bias")
    }
    model_names = {name for name, _ in model.named_parameters()}
    if not names <= model_names:
        raise RuntimeError(
            f"Affine parameters missing after conversion: {sorted(names - model_names)}"
        )
    global AFFINE_PARAMETER_NAMES
    AFFINE_PARAMETER_NAMES = names
    return names


def load_affine_model_state(
    model: nn.Module, state_dict: dict[str, torch.Tensor]
) -> Any:
    """Allow only the 96 identity affine tensors to be absent at the branch."""
    result = nn.Module.load_state_dict(model, state_dict, strict=False)
    missing = set(result.missing_keys)
    unexpected = set(result.unexpected_keys)
    if missing not in (set(), AFFINE_PARAMETER_NAMES):
        raise RuntimeError(
            "Illegal model-state mismatch. "
            f"Missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    if unexpected:
        raise RuntimeError(f"Unexpected model-state keys: {sorted(unexpected)}")
    if missing == AFFINE_PARAMETER_NAMES:
        parameters = dict(model.named_parameters())
        for name in AFFINE_PARAMETER_NAMES:
            expected = 0.0 if name.endswith(".bias") else 1.0
            observed = parameters[name].detach()
            if not torch.equal(observed, torch.full_like(observed, expected)):
                raise RuntimeError(f"Non-identity affine initialisation at {name}")
    return result


def _flatten_group_parameters(groups: list[dict[str, Any]]) -> list[Any]:
    return [parameter for group in groups for parameter in group["params"]]


class AffineAdam(ORIGINAL_ADAM):
    """Adam that maps the epoch-50 state by parameter name/order audit."""

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        saved_ids = _flatten_group_parameters(state_dict["param_groups"])
        current_parameters = _flatten_group_parameters(self.param_groups)
        affine_ids = {
            id(parameter)
            for group in self.param_groups
            for parameter in group["params"]
            if id(parameter) in _AFFINE_OBJECT_IDS
        }
        shared_parameters = [
            parameter
            for parameter in current_parameters
            if id(parameter) not in affine_ids
        ]

        if len(saved_ids) == len(current_parameters):
            super().load_state_dict(state_dict)
            return
        if len(state_dict["param_groups"]) != 1 or len(self.param_groups) != 1:
            raise RuntimeError("Affine state migration expects exactly one Adam group")
        if len(saved_ids) != len(shared_parameters):
            raise RuntimeError(
                "Adam parameter-count mismatch: "
                f"saved={len(saved_ids)}, shared={len(shared_parameters)}, "
                f"current={len(current_parameters)}"
            )

        current_index = {
            id(parameter): index
            for index, parameter in enumerate(current_parameters)
        }
        remapped_state: dict[int, Any] = {}
        for saved_id, parameter in zip(saved_ids, shared_parameters):
            if saved_id in state_dict["state"]:
                remapped_state[current_index[id(parameter)]] = state_dict["state"][
                    saved_id
                ]
        remapped_group = dict(state_dict["param_groups"][0])
        remapped_group["params"] = list(range(len(current_parameters)))
        super().load_state_dict(
            {"state": remapped_state, "param_groups": [remapped_group]}
        )

        for parameter in current_parameters:
            state = self.state.get(parameter, {})
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state and state[key].shape != parameter.shape:
                    raise RuntimeError(
                        f"Adam {key} shape mismatch: {state[key].shape} "
                        f"versus {parameter.shape}"
                    )
        for parameter in current_parameters:
            if id(parameter) in affine_ids and self.state.get(parameter):
                raise RuntimeError("New affine parameter unexpectedly inherited Adam state")


_AFFINE_OBJECT_IDS: set[int] = set()


def make_affine_model_class(base_class):
    class AffineModel(base_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            names = enable_affine_adapters(self)
            parameters = dict(self.named_parameters())
            global _AFFINE_OBJECT_IDS
            _AFFINE_OBJECT_IDS = {id(parameters[name]) for name in names}

        def load_state_dict(self, state_dict, strict=True, assign=False):
            if assign:
                raise RuntimeError("assign=True is not supported by this audit")
            return load_affine_model_state(self, state_dict)

    AffineModel.__name__ = f"Affine{base_class.__name__}"
    return AffineModel


def _argument_value(flag: str) -> str:
    try:
        return sys.argv[sys.argv.index(flag) + 1]
    except (ValueError, IndexError) as error:
        raise RuntimeError(f"Required argument missing: {flag}") from error


def prepare_output_scaffold(
    resume_path: Path,
    output_dir: Path,
    expected_source_sha256: str,
    arm_name: str,
) -> dict[str, Any]:
    """Create the immutable 50-epoch branch ledger in a new output directory."""
    if not resume_path.is_file():
        raise FileNotFoundError(resume_path)
    observed_sha = sha256_file(resume_path)
    if observed_sha != expected_source_sha256:
        raise RuntimeError(
            f"Wrong source checkpoint for {arm_name}: "
            f"expected {expected_source_sha256}, observed {observed_sha}"
        )
    checkpoint = ORIGINAL_TORCH_LOAD(
        resume_path, map_location="cpu", weights_only=False
    )
    source_is_affine = bool(checkpoint["config"].get("adapter_affine", False))
    if int(checkpoint["epoch"]) == 50 and not source_is_affine:
        output_dir.mkdir(parents=True, exist_ok=True)
        history = list(checkpoint["history"])
        if len(history) != 50:
            raise RuntimeError(f"Expected 50 history rows, got {len(history)}")
        scaffold = {
            "config.json": json.dumps(checkpoint["config"], indent=2) + "\n",
            "train_patient_ids.txt": "\n".join(
                checkpoint["config"]["train_patient_ids"]
            ),
            "val_patient_ids.txt": "\n".join(
                checkpoint["config"]["val_patient_ids"]
            ),
        }
        for filename, expected_text in scaffold.items():
            path = output_dir / filename
            if path.exists():
                if path.read_text(encoding="utf-8") != expected_text:
                    raise RuntimeError(f"Existing scaffold differs: {path}")
            else:
                path.write_text(expected_text, encoding="utf-8")
        log_path = output_dir / "training_log.csv"
        if log_path.exists():
            with log_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if rows != [
                {key: str(value) for key, value in row.items()} for row in history
            ]:
                raise RuntimeError(f"Existing scaffold differs: {log_path}")
        else:
            with log_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(history[0]))
                writer.writeheader()
                writer.writerows(history)
    elif not source_is_affine:
        raise RuntimeError("Initial affine branch must use the locked epoch-50 source")

    manifest = {
        "ablation_protocol": "adapter-instance-norm-affine-paired-v1",
        "arm": arm_name,
        "adapter_normalization": "InstanceNorm2d",
        "adapter_affine": True,
        "identity_initialisation": {"weight": 1.0, "bias": 0.0},
        "source_checkpoint": str(resume_path),
        "source_checkpoint_sha256": observed_sha,
        "source_epoch": int(checkpoint["epoch"]),
        "selection_baseline_reset": (
            "best_val is reset at the non-affine epoch-50 branch so model_best "
            "selects only among affine epochs 51-60"
        ),
        "optimizer_migration": (
            "All shared Adam states are mapped in original parameter order; "
            "the 96 new affine tensors start with empty Adam state."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "affine_ablation_manifest.json"
    if manifest_path.exists():
        installed = json.loads(manifest_path.read_text(encoding="utf-8"))
        if installed != manifest:
            raise RuntimeError("Installed affine ablation manifest has changed")
    else:
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    return checkpoint


def install_runtime(base_module, expected_source_sha256: str, arm_name: str) -> Path:
    """Patch one audited training module, preserving its training loop."""
    resume_path = Path(_argument_value("--resume")).resolve()
    output_dir = Path(_argument_value("--output_dir")).resolve()
    prepare_output_scaffold(
        resume_path, output_dir, expected_source_sha256, arm_name
    )

    base_module.M2PRNFFusionPilotVarNet = make_affine_model_class(
        base_module.M2PRNFFusionPilotVarNet
    )
    base_module.torch.optim.Adam = AffineAdam

    def audited_load(path, *args, **kwargs):
        checkpoint = ORIGINAL_TORCH_LOAD(path, *args, **kwargs)
        if (
            isinstance(path, (str, Path))
            and Path(path).resolve() == resume_path
            and int(checkpoint["epoch"]) == 50
            and not checkpoint["config"].get("adapter_affine", False)
        ):
            checkpoint = dict(checkpoint)
            checkpoint["best_epoch"] = 0
            checkpoint["best_val"] = float("inf")
        return checkpoint

    base_module.torch.load = audited_load
    original_save = base_module.save_checkpoint

    def audited_save(
        path,
        model,
        optimizer,
        epoch,
        best_epoch,
        best_val,
        config,
        history,
        run_corruption_audit,
    ):
        config.update(
            {
                "adapter_normalization": "InstanceNorm2d",
                "adapter_affine": True,
                "adapter_affine_initial_weight": 1.0,
                "adapter_affine_initial_bias": 0.0,
                "adapter_affine_parameter_tensors": 96,
                "ablation_protocol_version": (
                    "adapter-instance-norm-affine-paired-v1"
                ),
                "ablation_arm": arm_name,
                "ablation_runtime_script": str(Path(__file__).resolve()),
                "ablation_runtime_script_sha256": sha256_file(
                    Path(__file__).resolve()
                ),
            }
        )
        return original_save(
            path,
            model,
            optimizer,
            epoch,
            best_epoch,
            best_val,
            config,
            history,
            run_corruption_audit,
        )

    base_module.save_checkpoint = audited_save
    return output_dir


def annotate_final_summary(output_dir: Path, arm_name: str) -> None:
    summary_path = output_dir / "final_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "adapter_normalization": "InstanceNorm2d",
            "adapter_affine": True,
            "adapter_affine_initial_weight": 1.0,
            "adapter_affine_initial_bias": 0.0,
            "ablation_protocol_version": "adapter-instance-norm-affine-paired-v1",
            "ablation_arm": arm_name,
        }
    )
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
