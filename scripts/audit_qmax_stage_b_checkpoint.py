#!/usr/bin/env python3
from __future__ import annotations

"""Read-only Stage-B checkpoint/recovery audit.

The checkpoint is never modified. A failure affects only this report/job and
does not retroactively change the status of the training job that produced it.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.qmax_common import (  # noqa: E402
    IndexedDataset,
    ShapeBucketBatchSampler,
    autocast_context,
    capture_rng_state,
    make_dataset,
    make_grad_scaler,
    prepare_batch,
    restore_rng_state,
    sha256_file,
)
from scripts.qmax_stage_b_training_contract import (  # noqa: E402
    build_optimizer,
)
from scripts.qmax_stage_b_versioning import (  # noqa: E402
    AUDIT_VERSION,
    RUNTIME_VERSION,
    STRUCTURE_VERSION,
    manifest_digest,
    stage_b_audit_hashes,
    stage_b_runtime_hashes,
    stage_b_structure_hashes,
)
from src.m2_prnf_qmax_compactswin_varnet import (  # noqa: E402
    QMaxCompactSwinAuxPDVarNet,
)
from src.m2_prnf_qmax_varnet import QMaxAuxPDVarNet  # noqa: E402


REQUIRED_KEYS = {
    "epoch",
    "best_epoch",
    "best_val",
    "model_state_dict",
    "optimizer_state_dict",
    "grad_scaler_state_dict",
    "config",
    "history",
    "rng_state",
    "sampler_next_epoch",
    "code_hashes",
    "run_corruption_audit",
}


def _rng_equal(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> bool:
    if set(first) != set(second):
        return False
    if first["python"] != second["python"]:
        return False
    first_numpy = first["numpy"]
    second_numpy = second["numpy"]
    if (
        first_numpy[0] != second_numpy[0]
        or not np.array_equal(first_numpy[1], second_numpy[1])
        or first_numpy[2:] != second_numpy[2:]
    ):
        return False
    if not torch.equal(first["torch_cpu"], second["torch_cpu"]):
        return False
    first_cuda = first["torch_cuda"]
    second_cuda = second["torch_cuda"]
    return len(first_cuda) == len(second_cuda) and all(
        torch.equal(left, right)
        for left, right in zip(first_cuda, second_cuda)
    )


def _build_model(config: Mapping[str, Any]) -> torch.nn.Module:
    kwargs = dict(config["model_kwargs"])
    backbone = kwargs.pop(
        "backbone_variant", config.get("backbone_variant")
    )
    variant = str(config["qmax_variant"])
    if backbone == "convolutional":
        return QMaxAuxPDVarNet(qmax_variant=variant, **kwargs)
    if backbone == "compactswin":
        return QMaxCompactSwinAuxPDVarNet(
            qmax_variant=variant, **kwargs
        )
    raise RuntimeError(f"Unknown Stage-B backbone: {backbone}")


def _artifact_checks(
    checkpoint_path: Path, checkpoint: Mapping[str, Any]
) -> Dict[str, bool]:
    run_dir = checkpoint_path.parent
    if not (run_dir / "config.json").is_file() and (
        run_dir.parent / "config.json"
    ).is_file():
        run_dir = run_dir.parent
    history = list(checkpoint["history"])
    epoch = int(checkpoint["epoch"])
    checks = {
        "history_length_matches_epoch": len(history) == epoch,
        "history_last_epoch_matches": (
            bool(history) and int(history[-1].get("epoch", -1)) == epoch
        ),
        "sampler_next_epoch_matches": (
            int(checkpoint["sampler_next_epoch"]) == epoch
        ),
        "config_json_exists": (run_dir / "config.json").is_file(),
        "training_log_exists": (
            run_dir / "training_log.csv"
        ).is_file(),
        "train_patient_manifest_exists": (
            run_dir / "train_patient_ids.txt"
        ).is_file(),
        "val_patient_manifest_exists": (
            run_dir / "val_patient_ids.txt"
        ).is_file(),
    }
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epoch_seconds", type=float, default=None)
    parser.add_argument("--max_gpu_memory_gb", type=float, default=None)
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    output = Path(args.output_json).resolve()
    checkpoint_candidate = Path(args.checkpoint).resolve()
    training_root = checkpoint_candidate.parent
    if (
        training_root.name.startswith("epoch")
        and training_root.name[5:].isdigit()
    ):
        training_root = training_root.parent
    if output == training_root or training_root in output.parents:
        raise ValueError(
            "Audit report must be outside the immutable training run "
            f"directory: {training_root}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {
        "audit_version": AUDIT_VERSION,
        "status": "failed",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "read_only": True,
    }
    try:
        checkpoint_path = Path(args.checkpoint).resolve()
        metadata_path = Path(args.metadata_csv).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
        if not torch.cuda.is_available():
            raise RuntimeError("Checkpoint audit requires CUDA")
        device = torch.device("cuda")

        structure_hashes = stage_b_structure_hashes(PROJECT_ROOT)
        runtime_hashes = stage_b_runtime_hashes(PROJECT_ROOT)
        audit_hashes = stage_b_audit_hashes(PROJECT_ROOT)
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        missing = sorted(REQUIRED_KEYS - set(checkpoint))
        if missing:
            raise RuntimeError(f"Checkpoint missing keys: {missing}")
        config = dict(checkpoint["config"])
        checks = _artifact_checks(checkpoint_path, checkpoint)
        checks.update(
            {
                "legacy_code_hash_alias_matches_structure": (
                    checkpoint["code_hashes"] == structure_hashes
                ),
                "config_structure_hashes_match": (
                    config.get("structure_hashes")
                    == structure_hashes
                ),
                "config_structure_digest_matches": (
                    config.get("structure_digest")
                    == manifest_digest(structure_hashes)
                ),
            }
        )

        model = _build_model(config).to(device)
        model.load_state_dict(
            checkpoint["model_state_dict"], strict=True
        )
        checks["strict_model_load"] = True
        optimizer = build_optimizer(model)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        checks["optimizer_load"] = True
        scaler = make_grad_scaler(bool(args.amp))
        scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])
        checks["grad_scaler_load"] = True

        restore_rng_state(checkpoint["rng_state"])
        restored = capture_rng_state()
        checks["rng_restore_exact"] = _rng_equal(
            checkpoint["rng_state"], restored
        )

        dataset = IndexedDataset(
            make_dataset(str(metadata_path), "val", 8, 2)
        )
        sampler = ShapeBucketBatchSampler(
            dataset, args.batch_size, False, 42
        )
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        batch = next(iter(loader))
        kspace, mask, pd, _, _ = prepare_batch(batch, device)
        model.eval()
        with torch.no_grad(), autocast_context(device, args.amp):
            prediction = model(
                kspace,
                mask,
                pd,
                torch.ones(pd.shape[0], device=device),
            )
        checks["post_load_forward_finite"] = bool(
            torch.isfinite(prediction).all()
        )

        last_history = dict(checkpoint["history"][-1])
        if args.max_epoch_seconds is not None:
            checks["epoch_time_within_gate"] = (
                float(last_history["epoch_seconds"])
                < args.max_epoch_seconds
            )
        if args.max_gpu_memory_gb is not None:
            checks["training_peak_memory_within_gate"] = (
                float(last_history["peak_gpu_memory_gb"])
                < args.max_gpu_memory_gb
            )

        report.update(
            {
                "status": (
                    "passed" if all(checks.values()) else "failed"
                ),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "epoch": int(checkpoint["epoch"]),
                "qmax_variant": config["qmax_variant"],
                "backbone_variant": config["backbone_variant"],
                "checks": checks,
                "structure_version": STRUCTURE_VERSION,
                "structure_digest": manifest_digest(structure_hashes),
                "runtime_version": RUNTIME_VERSION,
                "checkpoint_runtime_tool_digest": config.get(
                    "runtime_tool_digest_at_launch"
                ),
                "current_runtime_tool_digest": manifest_digest(
                    runtime_hashes
                ),
                "runtime_tool_changed_since_checkpoint": (
                    config.get("runtime_tool_digest_at_launch")
                    != manifest_digest(runtime_hashes)
                ),
                "current_audit_tool_digest": manifest_digest(
                    audit_hashes
                ),
                "runtime_change_invalidates_weights": False,
                "audit_change_invalidates_weights": False,
                "last_epoch_resources": {
                    "epoch_seconds": float(
                        last_history["epoch_seconds"]
                    ),
                    "peak_gpu_memory_gb": float(
                        last_history["peak_gpu_memory_gb"]
                    ),
                },
            }
        )
    except Exception as error:
        report["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if report["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
