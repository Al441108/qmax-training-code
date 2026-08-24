#!/usr/bin/env python3
from __future__ import annotations

"""Safely reconcile a QMax training CSV with model_last.pt history.

The checkpoint is treated as authoritative. The script is dry-run by default.
With --apply it preserves a timestamped backup and atomically rewrites only
training_log.csv. It never deletes checkpoints or per-epoch artifacts.
"""

import argparse
import csv
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def canonical_json(value: Any) -> Any:
    return json.loads(
        json.dumps(value, sort_keys=True, allow_nan=False)
    )


def values_match(csv_value: str, checkpoint_value: Any) -> bool:
    if isinstance(checkpoint_value, bool):
        return csv_value == str(checkpoint_value)
    if isinstance(checkpoint_value, int):
        try:
            return int(csv_value) == checkpoint_value
        except ValueError:
            return False
    if isinstance(checkpoint_value, float):
        try:
            observed = float(csv_value)
        except ValueError:
            return False
        if math.isnan(checkpoint_value):
            return math.isnan(observed)
        return math.isclose(
            observed,
            checkpoint_value,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    return csv_value == str(checkpoint_value)


def validate_history(history: Any, completed_epoch: int) -> List[Dict[str, Any]]:
    if not isinstance(history, list):
        raise RuntimeError("Checkpoint history must be a list")
    if len(history) != completed_epoch:
        raise RuntimeError(
            "Checkpoint history length differs from checkpoint epoch: "
            f"{len(history)} != {completed_epoch}"
        )
    rows: List[Dict[str, Any]] = []
    for expected_epoch, value in enumerate(history, start=1):
        if not isinstance(value, Mapping):
            raise RuntimeError(
                f"History row {expected_epoch} is not a mapping"
            )
        row = dict(value)
        if int(row.get("epoch", -1)) != expected_epoch:
            raise RuntimeError(
                f"History epoch sequence breaks at {expected_epoch}"
            )
        rows.append(row)
    return rows


def read_csv(path: Path) -> tuple[List[str], List[Dict[str, str]]]:
    if not path.is_file():
        return [], []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def validate_common_prefix(
    csv_rows: Sequence[Mapping[str, str]],
    history: Sequence[Mapping[str, Any]],
) -> None:
    common = min(len(csv_rows), len(history))
    for index in range(common):
        observed = csv_rows[index]
        expected = history[index]
        if int(observed.get("epoch", -1)) != int(expected["epoch"]):
            raise RuntimeError(
                f"CSV/checkpoint epoch mismatch at row {index + 1}"
            )
        for key, expected_value in expected.items():
            if key not in observed:
                raise RuntimeError(f"CSV is missing field {key!r}")
            if not values_match(observed[key], expected_value):
                raise RuntimeError(
                    "CSV/checkpoint value mismatch at "
                    f"epoch={expected['epoch']} field={key}: "
                    f"{observed[key]!r} != {expected_value!r}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--audit_json", required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Create a backup and atomically repair training_log.csv.",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    audit_path = Path(args.audit_json).resolve()
    expected_checkpoint = output_dir / "model_last.pt"
    if checkpoint_path != expected_checkpoint:
        raise RuntimeError(
            "Only model_last.pt from the exact output_dir may be used: "
            f"{checkpoint_path} != {expected_checkpoint}"
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    required = {
        "epoch",
        "history",
        "config",
        "model_state_dict",
        "optimizer_state_dict",
        "grad_scaler_state_dict",
        "rng_state",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise RuntimeError(f"Checkpoint missing keys: {missing}")
    completed_epoch = int(checkpoint["epoch"])
    history = validate_history(checkpoint["history"], completed_epoch)
    if not history:
        raise RuntimeError("Refusing to repair from an epoch-zero checkpoint")

    config_path = output_dir / "config.json"
    if not config_path.is_file():
        raise RuntimeError("Output directory has no config.json")
    installed_config = json.loads(config_path.read_text(encoding="utf-8"))
    if canonical_json(installed_config) != canonical_json(
        checkpoint["config"]
    ):
        raise RuntimeError("config.json differs from checkpoint config")

    log_path = output_dir / "training_log.csv"
    existing_fields, existing_rows = read_csv(log_path)
    fieldnames = list(history[0].keys())
    for row in history:
        if list(row.keys()) != fieldnames:
            raise RuntimeError("Checkpoint history field order is inconsistent")
    if existing_fields and existing_fields != fieldnames:
        raise RuntimeError(
            "Existing CSV fields differ from checkpoint history fields"
        )
    validate_common_prefix(existing_rows, history)
    if len(existing_rows) > completed_epoch + 1:
        raise RuntimeError(
            "CSV is more than one epoch ahead of model_last.pt; "
            "manual investigation is required"
        )
    if len(existing_rows) == completed_epoch + 1:
        if int(existing_rows[-1].get("epoch", -1)) != completed_epoch + 1:
            raise RuntimeError(
                "The single CSV row ahead of model_last.pt is not the "
                "immediately following epoch"
            )

    repair_needed = (
        existing_fields != fieldnames or len(existing_rows) != completed_epoch
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = output_dir / (
        f"training_log.before_resume_repair.{timestamp}.csv"
    )
    orphan_epoch = completed_epoch + 1
    orphan_artifacts = sorted(
        str(path)
        for path in output_dir.glob(f"epoch_{orphan_epoch:02d}_*")
    )

    applied = False
    if args.apply and repair_needed:
        if log_path.is_file():
            shutil.copy2(log_path, backup_path)
        atomic_write_csv(log_path, history, fieldnames)
        applied = True
        repaired_fields, repaired_rows = read_csv(log_path)
        if (
            repaired_fields != fieldnames
            or len(repaired_rows) != completed_epoch
        ):
            raise RuntimeError("Post-repair CSV verification failed")
        validate_common_prefix(repaired_rows, history)

    result = {
        "status": (
            "repaired"
            if applied
            else "repair_needed"
            if repair_needed
            else "already_consistent"
        ),
        "audit_version": "QMax-resume-log-repair-v1",
        "dry_run": not args.apply,
        "checkpoint": str(checkpoint_path),
        "output_dir": str(output_dir),
        "completed_epoch": completed_epoch,
        "checkpoint_history_rows": len(history),
        "existing_csv_rows": len(existing_rows),
        "repair_needed": repair_needed,
        "repair_applied": applied,
        "backup": (
            str(backup_path)
            if applied and backup_path.is_file()
            else None
        ),
        "orphan_artifacts_not_deleted": orphan_artifacts,
    }
    atomic_write_json(audit_path, result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
