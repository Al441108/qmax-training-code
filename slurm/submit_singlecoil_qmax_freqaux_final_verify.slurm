#!/usr/bin/env bash
#SBATCH --job-name=sc_qmax_freqverify
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:29:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/singlecoil_qmax_benchmark/logs/sc_qmax_freqverify_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/singlecoil_qmax_benchmark/logs/sc_qmax_freqverify_%j.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/projects/u6dm/fastmri_project/fastmri_pipeline
BENCH_ROOT=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/singlecoil_qmax_benchmark

FORMAL_DIR="${BENCH_ROOT}/outputs/qmax_freqaux/formal_seed1337_100k"
MODEL_LAST="${FORMAL_DIR}/model_last.pt"
CHECKPOINT_100K="${FORMAL_DIR}/checkpoint_update100000.pt"
TRAINING_SUMMARY="${FORMAL_DIR}/training_summary.json"
TRAIN_METRICS="${FORMAL_DIR}/train_metrics.jsonl"

VERIFY_DIR="${BENCH_ROOT}/outputs/qmax_freqaux/final_verification_seed1337_100k"
VERIFY_JSON="${VERIFY_DIR}/verification.json"
VERIFY_SHA256="${VERIFY_DIR}/verified_files_sha256.txt"

HASH_FILE="${BENCH_ROOT}/protocol/singlecoil_freqaux_formal_code_sha256.txt"
AMENDMENT_FILE="${BENCH_ROOT}/protocol/PROTOCOL_AMENDMENT_001_SINGLECOIL_FREQAUX_PARALLEL.lock"
MAIN_LOCK="${PROJECT_ROOT}/outputs/qmax_stage_a/final_release/MAIN_EXPERIMENTS_COMPLETE.lock"

cd "${PROJECT_ROOT}"

source "${HOME}/miniforge3/bin/activate"
conda activate fastmri311

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/external_benchmarks:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

mkdir -p \
  "${BENCH_ROOT}/logs" \
  "${VERIFY_DIR}"

echo "===== VERIFY REQUIRED FILES ====="

for required in \
  "${MODEL_LAST}" \
  "${CHECKPOINT_100K}" \
  "${TRAINING_SUMMARY}" \
  "${TRAIN_METRICS}" \
  "${HASH_FILE}" \
  "${AMENDMENT_FILE}" \
  "${MAIN_LOCK}"
do
  test -f "${required}" || {
    echo "Missing required file: ${required}"
    exit 1
  }
  echo "found: ${required}"
done

echo "===== VERIFY PROTOCOL LOCKS ====="

grep -Fqx \
  "status=APPROVED_BEFORE_HELD_OUT_TEST" \
  "${AMENDMENT_FILE}"

grep -Fqx \
  "design=prespecified_parallel_models" \
  "${AMENDMENT_FILE}"

grep -Fqx \
  "arm_a=QMaxSinglecoilFull" \
  "${AMENDMENT_FILE}"

grep -Fqx \
  "arm_b=QMaxSinglecoilFullFreqAux" \
  "${AMENDMENT_FILE}"

grep -Fqx \
  "reporting_policy=report_both_parallel_models" \
  "${AMENDMENT_FILE}"

grep -Fqx \
  "held_out_test_accessed=false" \
  "${AMENDMENT_FILE}"

EXPECTED_LOCK_HASH=$(
  awk -F= \
    '$1 == "original_lock_sha256" {print $2}' \
    "${AMENDMENT_FILE}"
)

test -n "${EXPECTED_LOCK_HASH}"

printf '%s  %s\n' \
  "${EXPECTED_LOCK_HASH}" \
  "${MAIN_LOCK}" |
  sha256sum -c -

echo "===== VERIFY FROZEN CODE HASHES ====="

sha256sum -c "${HASH_FILE}"

echo "===== VERIFY FINAL CHECKPOINT ====="

python - \
  "${MODEL_LAST}" \
  "${CHECKPOINT_100K}" \
  "${TRAINING_SUMMARY}" \
  "${TRAIN_METRICS}" \
  "${VERIFY_JSON}" <<'PY'
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch


model_last_path = Path(sys.argv[1])
checkpoint_100k_path = Path(sys.argv[2])
summary_path = Path(sys.argv[3])
metrics_path = Path(sys.argv[4])
verification_path = Path(sys.argv[5])

EXPECTED_MODEL = "QMaxSinglecoilFullFreqAux"
EXPECTED_UPDATE = 100000
EXPECTED_SEED = 1337
EXPECTED_BATCH_SIZE = 4
EXPECTED_PRECISION = "fp32"
EXPECTED_FREQUENCY_CHANNELS = 64
EXPECTED_BATCHES_PER_EPOCH = 2086

EXPECTED_EPOCH = (
    EXPECTED_UPDATE // EXPECTED_BATCHES_PER_EPOCH
)
EXPECTED_BATCH_POSITION = (
    EXPECTED_UPDATE % EXPECTED_BATCHES_PER_EPOCH
)


def torch_load(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def exact_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor):
        return (
            isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left, right)
        )

    if isinstance(left, np.ndarray):
        return (
            isinstance(right, np.ndarray)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and np.array_equal(left, right)
        )

    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and set(left) == set(right)
            and all(
                exact_equal(left[key], right[key])
                for key in left
            )
        )

    if isinstance(left, (tuple, list)):
        return (
            isinstance(right, type(left))
            and len(left) == len(right)
            and all(
                exact_equal(a, b)
                for a, b in zip(left, right)
            )
        )

    return left == right


def audit_tensor_tree(value: Any) -> Tuple[int, int]:
    tensor_count = 0
    element_count = 0

    if isinstance(value, torch.Tensor):
        tensor_count += 1
        element_count += value.numel()

        if value.is_floating_point() or value.is_complex():
            if not bool(torch.isfinite(value).all()):
                raise RuntimeError(
                    "Checkpoint contains non-finite tensor"
                )

    elif isinstance(value, dict):
        for child in value.values():
            child_tensors, child_elements = (
                audit_tensor_tree(child)
            )
            tensor_count += child_tensors
            element_count += child_elements

    elif isinstance(value, (tuple, list)):
        for child in value:
            child_tensors, child_elements = (
                audit_tensor_tree(child)
            )
            tensor_count += child_tensors
            element_count += child_elements

    return tensor_count, element_count


model_last = torch_load(model_last_path)
checkpoint_100k = torch_load(checkpoint_100k_path)

summary = json.loads(
    summary_path.read_text(encoding="utf-8")
)

metric_rows = [
    json.loads(line)
    for line in metrics_path.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]

if not metric_rows:
    raise RuntimeError("train_metrics.jsonl is empty")


def verify_checkpoint(
    checkpoint: Dict[str, Any],
    label: str,
) -> None:
    required_keys = {
        "model_name",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "mask_rng_state",
        "rng_state",
        "global_update",
        "epoch",
        "batch_position",
        "seed",
        "batch_size",
        "precision",
        "frequency_channels",
    }

    missing = required_keys - set(checkpoint)
    if missing:
        raise RuntimeError(
            f"{label} missing keys: {sorted(missing)}"
        )

    assert checkpoint["model_name"] == EXPECTED_MODEL
    assert int(checkpoint["global_update"]) == EXPECTED_UPDATE
    assert int(checkpoint["seed"]) == EXPECTED_SEED
    assert int(checkpoint["batch_size"]) == EXPECTED_BATCH_SIZE
    assert checkpoint["precision"] == EXPECTED_PRECISION
    assert (
        int(checkpoint["frequency_channels"])
        == EXPECTED_FREQUENCY_CHANNELS
    )
    assert int(checkpoint["epoch"]) == EXPECTED_EPOCH
    assert (
        int(checkpoint["batch_position"])
        == EXPECTED_BATCH_POSITION
    )


verify_checkpoint(model_last, "model_last.pt")
verify_checkpoint(
    checkpoint_100k,
    "checkpoint_update100000.pt",
)

summary_expected = {
    "status": "complete",
    "model_name": EXPECTED_MODEL,
    "global_update": EXPECTED_UPDATE,
    "seed": EXPECTED_SEED,
    "batch_size": EXPECTED_BATCH_SIZE,
    "precision": EXPECTED_PRECISION,
    "frequency_channels": EXPECTED_FREQUENCY_CHANNELS,
}

for key, expected in summary_expected.items():
    if summary.get(key) != expected:
        raise RuntimeError(
            f"Summary mismatch for {key}: "
            f"{summary.get(key)!r} != {expected!r}"
        )

if "epoch" in summary:
    assert int(summary["epoch"]) == EXPECTED_EPOCH

if "batch_position" in summary:
    assert (
        int(summary["batch_position"])
        == EXPECTED_BATCH_POSITION
    )

updates = [
    int(row["update"])
    for row in metric_rows
]

if updates[-1] != EXPECTED_UPDATE:
    raise RuntimeError(
        f"Last logged update is {updates[-1]}, "
        f"expected {EXPECTED_UPDATE}"
    )

update_counts = {}
for update in updates:
    update_counts[update] = (
        update_counts.get(update, 0) + 1
    )

unique_updates = sorted(update_counts)

expected_updates = list(
    range(1, EXPECTED_UPDATE + 1)
)

if unique_updates != expected_updates:
    missing_updates = sorted(
        set(expected_updates) - set(unique_updates)
    )
    unexpected_updates = sorted(
        set(unique_updates) - set(expected_updates)
    )

    raise RuntimeError(
        "Training metric coverage is incomplete: "
        f"missing={missing_updates[:20]}, "
        f"unexpected={unexpected_updates[:20]}"
    )

restart_transitions = [
    {
        "previous_update": updates[index - 1],
        "resumed_update": updates[index],
    }
    for index in range(1, len(updates))
    if updates[index] <= updates[index - 1]
]

for transition in restart_transitions:
    resumed_update = transition["resumed_update"]

    # 正式训练每5000 updates保存一次checkpoint。
    if (resumed_update - 1) % 5000 != 0:
        raise RuntimeError(
            "Metric log contains an unexplained "
            f"non-monotonic transition: {transition}"
        )

duplicate_updates = sorted(
    update
    for update, count in update_counts.items()
    if count > 1
)

print(
    "logged metric rows:",
    len(metric_rows),
)
print(
    "unique metric updates:",
    len(unique_updates),
)
print(
    "duplicate replay updates:",
    len(duplicate_updates),
)
print(
    "restart transitions:",
    restart_transitions,
)

for row in metric_rows:
    for key, value in row.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError(
                f"Non-finite training metric: "
                f"update={row.get('update')} key={key}"
            )

model_tensor_count, model_element_count = (
    audit_tensor_tree(model_last["model_state"])
)
optimizer_tensor_count, optimizer_element_count = (
    audit_tensor_tree(model_last["optimizer_state"])
)

if model_tensor_count == 0 or model_element_count == 0:
    raise RuntimeError("Model state is empty")

if optimizer_tensor_count == 0:
    raise RuntimeError("Optimizer state is empty")

model_state_equal = exact_equal(
    model_last["model_state"],
    checkpoint_100k["model_state"],
)
optimizer_state_equal = exact_equal(
    model_last["optimizer_state"],
    checkpoint_100k["optimizer_state"],
)
scheduler_state_equal = exact_equal(
    model_last["scheduler_state"],
    checkpoint_100k["scheduler_state"],
)
mask_rng_equal = exact_equal(
    model_last["mask_rng_state"],
    checkpoint_100k["mask_rng_state"],
)
rng_state_equal = exact_equal(
    model_last["rng_state"],
    checkpoint_100k["rng_state"],
)

if not model_state_equal:
    raise RuntimeError(
        "model_last and checkpoint_update100000 "
        "model states differ"
    )

if not optimizer_state_equal:
    raise RuntimeError(
        "model_last and checkpoint_update100000 "
        "optimizer states differ"
    )

if not scheduler_state_equal:
    raise RuntimeError(
        "Final scheduler states differ"
    )

if not mask_rng_equal:
    raise RuntimeError(
        "Final mask RNG states differ"
    )

if not rng_state_equal:
    raise RuntimeError(
        "Final global RNG states differ"
    )

verification = {
    "status": "passed",
    "held_out_accessed": False,
    "model_name": EXPECTED_MODEL,
    "global_update": EXPECTED_UPDATE,
    "epoch": EXPECTED_EPOCH,
    "batch_position": EXPECTED_BATCH_POSITION,
    "batches_per_epoch": EXPECTED_BATCHES_PER_EPOCH,
    "seed": EXPECTED_SEED,
    "batch_size": EXPECTED_BATCH_SIZE,
    "precision": EXPECTED_PRECISION,
    "frequency_channels": EXPECTED_FREQUENCY_CHANNELS,
    "model_last_path": str(model_last_path),
    "checkpoint_100k_path": str(
        checkpoint_100k_path
    ),
    "model_last_sha256": sha256_file(
        model_last_path
    ),
    "checkpoint_100k_sha256": sha256_file(
        checkpoint_100k_path
    ),
    "training_summary_sha256": sha256_file(
        summary_path
    ),
    "train_metrics_sha256": sha256_file(
        metrics_path
    ),
    "logged_metric_rows": len(metric_rows),
    "last_logged_update": updates[-1],
    "model_tensor_count": model_tensor_count,
    "model_element_count": model_element_count,
    "optimizer_tensor_count": optimizer_tensor_count,
    "optimizer_element_count": optimizer_element_count,
    "model_state_equal": model_state_equal,
    "optimizer_state_equal": optimizer_state_equal,
    "scheduler_state_equal": scheduler_state_equal,
    "mask_rng_equal": mask_rng_equal,
    "rng_state_equal": rng_state_equal,
}

temporary_path = verification_path.with_suffix(
    ".json.tmp"
)
temporary_path.write_text(
    json.dumps(
        verification,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
os.replace(temporary_path, verification_path)

print(json.dumps(
    verification,
    indent=2,
    sort_keys=True,
))
print("FREQAUX FINAL CHECKPOINT VERIFIED")
PY

echo "===== RECORD VERIFIED FILE HASHES ====="

sha256sum \
  "${MODEL_LAST}" \
  "${CHECKPOINT_100K}" \
  "${TRAINING_SUMMARY}" \
  "${TRAIN_METRICS}" \
  "${HASH_FILE}" \
  "${AMENDMENT_FILE}" \
  "${MAIN_LOCK}" |
  tee "${VERIFY_SHA256}"

echo "===== VERIFY OUTPUT ====="

python -c "
import json
from pathlib import Path

path = Path('${VERIFY_JSON}')
result = json.loads(path.read_text())

assert result['status'] == 'passed'
assert result['held_out_accessed'] is False
assert result['model_name'] == 'QMaxSinglecoilFullFreqAux'
assert result['global_update'] == 100000
assert result['seed'] == 1337
assert result['batch_size'] == 4
assert result['precision'] == 'fp32'
assert result['frequency_channels'] == 64
assert result['model_state_equal'] is True
assert result['optimizer_state_equal'] is True
assert result['scheduler_state_equal'] is True
assert result['mask_rng_equal'] is True
assert result['rng_state_equal'] is True

print(json.dumps(result, indent=2))
"

echo "QMAX FREQUENCY-AUX FINAL VERIFICATION PASSED"
echo "verification=${VERIFY_JSON}"
echo "held_out_accessed=false"