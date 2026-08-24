#!/usr/bin/env python3
from __future__ import annotations

"""Audit and clone the locked global-direct epoch-15 state for continuation."""

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_m2_prnf_fusion import locked_code_hashes  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pilot_decision", required=True)
    parser.add_argument("--pilot_audit", required=True)
    args = parser.parse_args()

    pilot_dir = Path(args.pilot_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    decision_path = require_file(Path(args.pilot_decision))
    audit_path = require_file(Path(args.pilot_audit))
    if not pilot_dir.is_dir():
        raise FileNotFoundError(pilot_dir)

    required = {
        name: require_file(pilot_dir / name)
        for name in (
            "model_last.pt", "model_best.pt", "config.json", "training_log.csv",
            "train_patient_ids.txt", "val_patient_ids.txt", "final_summary.json",
            "run_corruption_audit.json",
        )
    }
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if decision.get("protocol_version") != "M2-PRNF-R8-v1.4.1-fusion-pilot-audited":
        raise RuntimeError("Unexpected fusion-pilot decision protocol")
    if decision.get("recommended_fusion_design") != "global_direct":
        raise RuntimeError("The audited fusion pilot did not recommend global_direct")
    if not audit.get("cross_checkpoint_fairness", {}).get("passed", False):
        raise RuntimeError("The source four-arm fairness audit did not pass")
    if audit.get("code_hashes") != locked_code_hashes():
        raise RuntimeError("Installed locked v1.4 code differs from the evaluated pilot")

    checkpoint = torch.load(required["model_last.pt"], map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    expected = {
        "variant": "prnf_no_need",
        "fusion_design": "global_direct",
        "need_scope": "residual",
        "run_stage": "pilot",
        "epochs": 15,
        "seed": 42,
        "acceleration": 8,
        "pd_aux_acceleration": 2,
        "batch_size": 4,
        "grad_accum_steps": 1,
    }
    observed = {key: config.get(key) for key in expected}
    if observed != expected:
        raise RuntimeError(
            "Pilot checkpoint identity mismatch:\n"
            + json.dumps({"expected": expected, "observed": observed}, indent=2)
        )
    if int(checkpoint.get("epoch", -1)) != 15:
        raise RuntimeError("model_last.pt is not the completed epoch-15 state")
    if int(checkpoint.get("sampler_next_epoch", -1)) != 15:
        raise RuntimeError("Pilot sampler state is not positioned after epoch 15")
    if len(checkpoint.get("history", [])) != 15:
        raise RuntimeError("Pilot checkpoint does not contain 15 history rows")
    if checkpoint.get("code_hashes") != locked_code_hashes():
        raise RuntimeError("Pilot checkpoint code hashes do not match installed v1.4 code")
    with required["training_log.csv"].open(newline="", encoding="utf-8") as file:
        log_rows = list(csv.DictReader(file))
    if len(log_rows) != 15 or int(log_rows[-1]["epoch"]) != 15:
        raise RuntimeError("Pilot training_log.csv is not complete through epoch 15")

    trainer = require_file(
        PROJECT_ROOT / "scripts" / "train_m2_prnf_global_direct_extend50.py"
    )
    preparer = Path(__file__).resolve()
    provenance = {
        "protocol_version": (
            "M2-PRNF-R8-v1.4.1-global-direct-continuation-15-to-50-audited"
        ),
        "source_pilot_dir": str(pilot_dir),
        "source_model_last_sha256": sha256_file(required["model_last.pt"]),
        "source_model_best_sha256": sha256_file(required["model_best.pt"]),
        "source_epoch": 15,
        "source_best_epoch": int(checkpoint.get("best_epoch", -1)),
        "source_best_val_patient_l1": float(checkpoint.get("best_val")),
        "target_total_epochs": 50,
        "pilot_decision": str(decision_path),
        "pilot_decision_sha256": sha256_file(decision_path),
        "pilot_audit": str(audit_path),
        "pilot_audit_sha256": sha256_file(audit_path),
        "locked_v14_code_hashes": locked_code_hashes(),
        "continuation_trainer_sha256": sha256_file(trainer),
        "preparer_sha256": sha256_file(preparer),
    }

    provenance_path = output_dir / "extension_provenance.json"
    if output_dir.exists():
        if not provenance_path.is_file():
            raise RuntimeError(
                f"Refusing to use non-empty/unproven output directory: {output_dir}"
            )
        installed = json.loads(provenance_path.read_text(encoding="utf-8"))
        if installed != provenance:
            raise RuntimeError("Existing extension provenance differs from the source pilot")
        require_file(output_dir / "model_last.pt")
        print(json.dumps({"status": "ready_to_resume", **provenance}, indent=2))
        return

    staging = output_dir.with_name(output_dir.name + ".initialising")
    if staging.exists():
        raise RuntimeError(f"Stale initialisation directory exists: {staging}")
    staging.mkdir(parents=True)
    for name in (
        "model_last.pt", "model_best.pt", "config.json", "training_log.csv",
        "train_patient_ids.txt", "val_patient_ids.txt", "run_corruption_audit.json",
    ):
        shutil.copy2(required[name], staging / name)
    shutil.copy2(required["final_summary.json"], staging / "pilot15_final_summary.json")
    for pattern in ("epoch_*_clean_audit.json", "epoch_*_corruption_audit.json"):
        for source in sorted(pilot_dir.glob(pattern)):
            shutil.copy2(source, staging / source.name)
    (staging / "extension_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    staging.rename(output_dir)
    print(json.dumps({"status": "initialised_from_epoch_15", **provenance}, indent=2))


if __name__ == "__main__":
    main()
