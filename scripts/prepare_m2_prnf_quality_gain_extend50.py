#!/usr/bin/env python3
from __future__ import annotations

"""Audit and clone the locked quality-gain epoch-15 state for continuation."""

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

from scripts.train_m2_prnf_quality_gain import locked_code_hashes  # noqa: E402


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


def assert_identical_model_states(best_checkpoint, last_checkpoint) -> None:
    best_state = best_checkpoint.get("model_state_dict", {})
    last_state = last_checkpoint.get("model_state_dict", {})
    if set(best_state) != set(last_state):
        raise RuntimeError("Epoch-15 model_best/model_last state keys differ")
    for key in best_state:
        if not torch.equal(best_state[key], last_state[key]):
            raise RuntimeError(
                f"Epoch-15 model_best/model_last tensors differ at {key}"
            )


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
    protocol = "M2-PRNF-R8-v1.5.0-quality-gain-comparison-audited"
    if decision.get("protocol_version") != protocol:
        raise RuntimeError("Unexpected five-arm quality-gain decision protocol")
    arm = decision.get("arm_results", {}).get("quality_protected_hybrid_gain", {})
    if arm.get("qualification") not in {"borderline", "eligible"}:
        raise RuntimeError("Quality-gain candidate was rejected by the five-arm audit")
    if not arm.get("clean_noninferiority_vs_global_direct", {}).get("passed", False):
        raise RuntimeError("Quality-gain candidate failed clean non-inferiority")
    if not arm.get("missing_exact_zero", False):
        raise RuntimeError("Quality-gain candidate failed exact missing-PD safety")
    corrupt_status = arm.get("corrupt_safety_status", {})
    if any(corrupt_status.get(key) != "eligible" for key in (
        "shift8", "wrong_slice", "wrong_patient"
    )):
        raise RuntimeError("Quality-gain candidate failed a corruption-safety gate")
    if not arm.get("numerical_residual_stability", {}).get("passed", False):
        raise RuntimeError("Quality-gain candidate failed residual-stability audit")
    if not audit.get("cross_checkpoint_fairness", {}).get("passed", False):
        raise RuntimeError("Five-arm cross-checkpoint fairness audit did not pass")
    installed_hashes = locked_code_hashes()
    audited_hashes = audit.get("code_hashes", {}).get("quality_gain_candidate")
    if audited_hashes != installed_hashes:
        raise RuntimeError("Installed locked v1.5 code differs from the evaluated pilot")
    audited_checkpoint = audit.get("checkpoint_audit", {}).get(
        "quality_protected_hybrid_gain", {}
    )
    if audited_checkpoint.get("sha256") != sha256_file(required["model_best.pt"]):
        raise RuntimeError("The evaluated quality-gain model_best checkpoint changed")

    last_checkpoint = torch.load(
        required["model_last.pt"], map_location="cpu", weights_only=False
    )
    best_checkpoint = torch.load(
        required["model_best.pt"], map_location="cpu", weights_only=False
    )
    config = last_checkpoint.get("config", {})
    expected = {
        "variant": "prnf_no_need",
        "fusion_design": "hybrid_direct_residual",
        "run_stage": "quality_gain_pilot",
        "epochs": 15,
        "seed": 42,
        "acceleration": 8,
        "pd_aux_acceleration": 2,
        "batch_size": 4,
        "grad_accum_steps": 1,
        "learning_rate": 3e-4,
        "lambda_residual_gain": 0.2,
        "residual_gain_margin_relative": 0.002,
        "residual_gain_ramp_epochs": 5,
    }
    observed = {key: config.get(key) for key in expected}
    if observed != expected:
        raise RuntimeError(
            "Pilot checkpoint identity mismatch:\n"
            + json.dumps({"expected": expected, "observed": observed}, indent=2)
        )
    expected_optimizer = {
        "name": "Adam", "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0
    }
    if config.get("optimizer") != expected_optimizer:
        raise RuntimeError("Pilot optimizer configuration is not the locked Adam setup")
    if config.get("scheduler") not in (None, {}):
        raise RuntimeError("Unexpected learning-rate scheduler in the source pilot")
    for label, checkpoint in (
        ("model_last", last_checkpoint), ("model_best", best_checkpoint)
    ):
        if int(checkpoint.get("epoch", -1)) != 15:
            raise RuntimeError(f"{label}.pt is not the completed epoch-15 state")
        if int(checkpoint.get("best_epoch", -1)) != 15:
            raise RuntimeError(f"{label}.pt does not record epoch 15 as best")
        if checkpoint.get("code_hashes") != installed_hashes:
            raise RuntimeError(f"{label}.pt code hashes do not match installed v1.5 code")
    if int(last_checkpoint.get("sampler_next_epoch", -1)) != 15:
        raise RuntimeError("Pilot sampler state is not positioned after epoch 15")
    if len(last_checkpoint.get("history", [])) != 15:
        raise RuntimeError("Pilot checkpoint does not contain 15 history rows")
    for key in (
        "optimizer_state_dict", "rng_state", "run_corruption_audit"
    ):
        if key not in last_checkpoint:
            raise RuntimeError(f"model_last.pt is missing continuation state: {key}")
    assert_identical_model_states(best_checkpoint, last_checkpoint)

    with required["training_log.csv"].open(newline="", encoding="utf-8") as file:
        log_rows = list(csv.DictReader(file))
    if len(log_rows) != 15 or int(log_rows[-1]["epoch"]) != 15:
        raise RuntimeError("Pilot training_log.csv is not complete through epoch 15")
    final_summary = json.loads(required["final_summary.json"].read_text(encoding="utf-8"))
    if int(final_summary.get("completed_epochs", -1)) != 15:
        raise RuntimeError("Pilot final_summary does not report 15 completed epochs")
    if int(final_summary.get("best_epoch", -1)) != 15:
        raise RuntimeError("Pilot final_summary does not report epoch 15 as best")
    if abs(float(last_checkpoint.get("best_val")) - float(arm.get("clean_l1"))) > 1e-6:
        raise RuntimeError("Pilot checkpoint and audited clean L1 are inconsistent")

    trainer = require_file(
        PROJECT_ROOT / "scripts" / "train_m2_prnf_quality_gain_extend50.py"
    )
    preparer = Path(__file__).resolve()
    provenance = {
        "protocol_version": (
            "M2-PRNF-R8-v1.5.0-quality-gain-continuation-15-to-50-audited"
        ),
        "source_pilot_dir": str(pilot_dir),
        "source_model_last_sha256": sha256_file(required["model_last.pt"]),
        "source_model_best_sha256": sha256_file(required["model_best.pt"]),
        "source_epoch": 15,
        "source_best_epoch": 15,
        "source_best_val_patient_l1": float(last_checkpoint.get("best_val")),
        "target_total_epochs": 50,
        "pilot_decision": str(decision_path),
        "pilot_decision_sha256": sha256_file(decision_path),
        "pilot_audit": str(audit_path),
        "pilot_audit_sha256": sha256_file(audit_path),
        "locked_v15_code_hashes": installed_hashes,
        "continuation_trainer_sha256": sha256_file(trainer),
        "preparer_sha256": sha256_file(preparer),
    }

    provenance_path = output_dir / "extension_provenance.json"
    if output_dir.exists():
        if not provenance_path.is_file():
            raise RuntimeError(
                f"Refusing to use non-provenanced output directory: {output_dir}"
            )
        installed = json.loads(provenance_path.read_text(encoding="utf-8"))
        if installed != provenance:
            raise RuntimeError("Existing extension provenance differs from source pilot")
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
