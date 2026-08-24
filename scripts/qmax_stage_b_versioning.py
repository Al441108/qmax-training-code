#!/usr/bin/env python3
from __future__ import annotations

"""Layered hashes for QMax Stage B.

Only ``stage_b_structure_hashes`` determines whether a trained checkpoint is
scientifically compatible. Runtime and audit hashes are provenance records:
changes to them require their own checks, but never invalidate learned weights.
"""

import ast
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Sequence


STRUCTURE_VERSION = "QMax-StageB-independent-structure-v1"
RUNTIME_VERSION = "QMax-StageB-independent-runtime-v1"
AUDIT_VERSION = "QMax-StageB-independent-audit-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_files(project_root: Path, relatives: Iterable[str]) -> None:
    missing = [
        relative
        for relative in relatives
        if not (project_root / relative).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Locked Stage-B dependency missing: " + ", ".join(missing)
        )


def _ast_symbol_hash(path: Path, symbols: Sequence[str]) -> str:
    """Hash selected top-level definitions without binding unrelated tools."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    wanted = set(symbols)
    selected = [
        node
        for node in tree.body
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        )
        and node.name in wanted
    ]
    found = {node.name for node in selected}
    if found != wanted:
        raise RuntimeError(
            f"Cannot hash requested symbols in {path}: "
            f"missing={sorted(wanted - found)}"
        )
    canonical = ast.dump(
        ast.Module(body=selected, type_ignores=[]),
        annotate_fields=True,
        include_attributes=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _file_hashes(
    project_root: Path, relatives: Sequence[str]
) -> Dict[str, str]:
    _require_files(project_root, relatives)
    return {
        f"file:{relative}": sha256_file(project_root / relative)
        for relative in relatives
    }


def stage_b_structure_hashes(project_root: Path) -> Dict[str, str]:
    """Hash model/training semantics only.

    Full-file hashes cover the model, forward, data, corruption and the
    extracted loss/optimizer contract. Selected AST symbols from qmax_common
    bind batching and batch preparation without coupling checkpoint/logging
    utilities from the same legacy module.
    """

    files = (
        "src/m2_prnf_varnet.py",
        "src/m2_prnf_corruptions.py",
        "src/m2_prnf_fusion_pilot_varnet.py",
        "src/m2_prnf_qmax_varnet.py",
        "src/m2_prnf_qmax_compactswin_varnet.py",
        "src/qmax_deterministic_corruptions.py",
        "src/dataset_paired_multicoil_aux_pd_r2.py",
        "src/fft_utils.py",
        "src/masks.py",
        "scripts/generate_qmax_stage_b_init.py",
        "scripts/qmax_stage_b_training_contract.py",
        "QMAX_STAGE_B_STRUCTURE_CONTRACT_R8.json",
    )
    output = _file_hashes(project_root, files)
    common_path = project_root / "scripts/qmax_common.py"
    if not common_path.is_file():
        raise FileNotFoundError(common_path)
    output[
        "ast:scripts/qmax_common.py:"
        "ShapeBucketBatchSampler,make_dataset,select_patient_ids,"
        "prepare_batch,l1_per_sample"
    ] = _ast_symbol_hash(
        common_path,
        (
            "ShapeBucketBatchSampler",
            "make_dataset",
            "select_patient_ids",
            "prepare_batch",
            "l1_per_sample",
        ),
    )
    return output


def stage_b_runtime_hashes(project_root: Path) -> Dict[str, str]:
    """Hash mutable execution/checkpoint/logging tools for provenance."""

    return _file_hashes(
        project_root,
        (
            "scripts/train_qmax_stage_b.py",
            "scripts/qmax_common.py",
            "scripts/generate_qmax_stage_b_init.py",
            "scripts/qmax_stage_b_versioning.py",
            "slurm/submit_qmax_stage_b_preflight.slurm",
            "slurm/submit_qmax_stage_b_core_pilot.slurm",
            "slurm/submit_qmax_stage_b_full_pilot.slurm",
            "slurm/submit_qmax_stage_b_core_epoch1to30.slurm",
            "slurm/submit_qmax_stage_b_full_epoch1to30.slurm",
            "slurm/submit_qmax_stage_b_core_epoch31to60.slurm",
            "slurm/submit_qmax_stage_b_full_epoch31to60.slurm",
        ),
    )


def stage_b_audit_hashes(project_root: Path) -> Dict[str, str]:
    """Hash read-only verification/evaluation tools for provenance."""

    return _file_hashes(
        project_root,
        (
            "scripts/preflight_qmax_stage_b.py",
            "scripts/audit_qmax_stage_b_checkpoint.py",
            "slurm/submit_qmax_stage_b_checkpoint_audit.slurm",
        ),
    )


def manifest_digest(values: Dict[str, str]) -> str:
    canonical = json.dumps(
        values, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
