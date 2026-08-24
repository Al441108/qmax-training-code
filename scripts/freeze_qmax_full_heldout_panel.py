#!/usr/bin/env python3
from __future__ import annotations

"""Freeze the QMax-Full three-seed held-out panel without computing metrics."""

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.qmax_multiseed_common import IndexedDataset, make_dataset, sha256_file  # noqa: E402


PROTOCOL_VERSION = "QMax-Full-heldout-panel-v1"
EXPECTED_SEEDS = (42, 123, 2026)
EXPECTED_PATIENTS = 34


def parse_main_lock(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    fields: dict[str, str] = {}
    hashes: dict[str, str] = {}
    section = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line
            continue
        match = re.match(r"^([0-9a-f]{64})\s+(.+)$", line)
        if section == "[checkpoint_sha256]" and match:
            hashes[str(Path(match.group(2)).resolve())] = match.group(1)
        elif "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            fields[key.strip()] = value.strip()
    required = {
        "status": "MAIN_EXPERIMENTS_COMPLETE",
        "formal_model": "QMax-Full",
        "formal_checkpoint_policy": "epoch60/model_last.pt",
        "model_seeds": "42,123,2026",
        "held_out_test_accessed": "false",
        "post_freeze_training_allowed": "false",
        "post_freeze_model_selection_allowed": "false",
    }
    mismatch = {
        key: {"expected": value, "observed": fields.get(key)}
        for key, value in required.items()
        if fields.get(key) != value
    }
    if mismatch:
        raise RuntimeError(
            "MAIN_EXPERIMENTS_COMPLETE.lock is not valid:\n"
            + json.dumps(mismatch, indent=2)
        )
    if len(hashes) != 3:
        raise RuntimeError(f"Expected three checkpoint hashes in lock, got {len(hashes)}")
    return {"fields": fields, "checkpoint_hashes": hashes, "sha256": sha256_file(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main_lock", required=True)
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--model_seed", action="append", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if tuple(args.model_seed) != EXPECTED_SEEDS or len(args.checkpoint) != 3:
        raise ValueError("Seeds/checkpoints must be supplied once in order: 42, 123, 2026")

    main_lock = Path(args.main_lock).resolve()
    metadata = Path(args.metadata_csv).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite frozen held-out manifest: {output}")
    for path in (main_lock, metadata):
        if not path.is_file():
            raise FileNotFoundError(path)

    lock = parse_main_lock(main_lock)
    checkpoints = []
    for seed, value in zip(args.model_seed, args.checkpoint):
        path = Path(value).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.name != "model_last.pt" or path.parent.name != "epoch60":
            raise RuntimeError(f"Seed {seed} is not epoch60/model_last.pt: {path}")
        observed = sha256_file(path)
        expected = lock["checkpoint_hashes"].get(str(path))
        if expected is None or observed != expected:
            raise RuntimeError(
                f"Seed {seed} checkpoint is absent from or differs from main lock"
            )
        checkpoints.append(
            {"model_seed": seed, "path": str(path), "sha256": observed}
        )

    # Dataset construction reads split metadata and file inventories only. No
    # sample is indexed here, so no target image or reconstruction metric is read.
    source = IndexedDataset(
        make_dataset(
            str(metadata), "test", acceleration=8, pd_aux_acceleration=2
        )
    )
    by_patient: dict[str, list[int]] = defaultdict(list)
    identities = set()
    for record in source.records:
        patient = str(record["patient_id"])
        slice_idx = int(record["slice_idx"])
        identity = (patient, slice_idx)
        if identity in identities:
            raise RuntimeError(f"Duplicate held-out identity: {identity}")
        identities.add(identity)
        by_patient[patient].append(slice_idx)
    if len(by_patient) != EXPECTED_PATIENTS:
        raise RuntimeError(
            f"Expected {EXPECTED_PATIENTS} test patients, found {len(by_patient)}"
        )
    if not identities:
        raise RuntimeError("Held-out test split is empty")

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "frozen",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "one-time clean held-out evaluation; no metrics computed during freeze",
        "split": "test",
        "expected_num_patients": EXPECTED_PATIENTS,
        "num_patients": len(by_patient),
        "num_slices": len(identities),
        "pdfs_acceleration": 8,
        "pd_aux_acceleration": 2,
        "evaluation_mode": "actual-q / all-components-on only",
        "checkpoint_selection": "all three prespecified epoch60/model_last checkpoints; no seed selection",
        "aggregation": "patient-equal within seed; seed-equal across seeds",
        "main_lock": str(main_lock),
        "main_lock_sha256": lock["sha256"],
        "metadata_csv": str(metadata),
        "metadata_sha256": sha256_file(metadata),
        "checkpoints": checkpoints,
        "patients": [
            {
                "patient_id": patient,
                "slice_indices": sorted(indices),
                "num_slices": len(indices),
            }
            for patient, indices in sorted(by_patient.items())
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output) + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({**payload, "patients": "[redacted from stdout]"}, indent=2))


if __name__ == "__main__":
    main()
