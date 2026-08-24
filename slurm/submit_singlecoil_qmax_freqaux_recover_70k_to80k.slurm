#!/usr/bin/env bash
#SBATCH --job-name=sc_qmax_freq70to80
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/singlecoil_qmax_benchmark/logs/sc_qmax_freq70to80_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/singlecoil_qmax_benchmark/logs/sc_qmax_freq70to80_%j.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/projects/u6dm/fastmri_project/fastmri_pipeline
BENCH_ROOT=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/singlecoil_qmax_benchmark
FSMNET="${BENCH_ROOT}/repo/FSMNet"

FORMAL_DIR="${BENCH_ROOT}/outputs/qmax_freqaux/formal_seed1337_100k"
SOURCE_CHECKPOINT="${FORMAL_DIR}/model_last.pt"
FROZEN_70K="${FORMAL_DIR}/checkpoint_update070000.pt"
TARGET_CHECKPOINT="${FORMAL_DIR}/checkpoint_update080000.pt"

HASH_FILE="${BENCH_ROOT}/protocol/singlecoil_freqaux_formal_code_sha256.txt"
AMENDMENT_FILE="${BENCH_ROOT}/protocol/PROTOCOL_AMENDMENT_001_SINGLECOIL_FREQAUX_PARALLEL.lock"
MAIN_LOCK="${PROJECT_ROOT}/outputs/qmax_stage_a/final_release/MAIN_EXPERIMENTS_COMPLETE.lock"

RESUME_PREFLIGHT_JOB_ID=5952661
RESUME_OUT="${BENCH_ROOT}/logs/sc_qmax_freqresume_${RESUME_PREFLIGHT_JOB_ID}.out"
RESUME_ERR="${BENCH_ROOT}/logs/sc_qmax_freqresume_${RESUME_PREFLIGHT_JOB_ID}.err"

SOURCE_UPDATE=70000
TARGET_UPDATE=80000

cd "${PROJECT_ROOT}"

source "${HOME}/miniforge3/bin/activate"
conda activate fastmri311

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/external_benchmarks:${PYTHONPATH:-}"

echo "===== PROTOCOL GATES ====="

test -f "${MAIN_LOCK}"
test -f "${AMENDMENT_FILE}"
test -f "${HASH_FILE}"
test -f "${RESUME_OUT}"
test -f "${RESUME_ERR}"
test ! -s "${RESUME_ERR}"

grep -Fqx \
  "status=APPROVED_BEFORE_HELD_OUT_TEST" \
  "${AMENDMENT_FILE}"

grep -Fqx \
  "design=prespecified_parallel_models" \
  "${AMENDMENT_FILE}"

grep -Fqx \
  "arm_b=QMaxSinglecoilFullFreqAux" \
  "${AMENDMENT_FILE}"

grep -Fqx \
  "held_out_test_accessed=false" \
  "${AMENDMENT_FILE}"

grep -Fq \
  "QMAX FREQUENCY-AUX CHECKPOINT/RESUME PREFLIGHT PASSED" \
  "${RESUME_OUT}"

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

sha256sum -c "${HASH_FILE}"

echo "===== SOURCE CHECKPOINT GATES ====="

test -f "${SOURCE_CHECKPOINT}"
test -f "${FROZEN_70K}"

if [ -e "${TARGET_CHECKPOINT}" ]; then
  echo "Refusing to overwrite existing 80k checkpoint:"
  echo "${TARGET_CHECKPOINT}"
  exit 1
fi

python - \
  "${SOURCE_CHECKPOINT}" \
  "${FROZEN_70K}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


model_last_path = Path(sys.argv[1])
frozen_path = Path(sys.argv[2])


def load(path):
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


model_last = load(model_last_path)
frozen = load(frozen_path)

required = {
    "model_name",
    "global_update",
    "epoch",
    "batch_position",
    "seed",
    "batch_size",
    "precision",
    "frequency_channels",
    "model_state",
    "optimizer_state",
    "scheduler_state",
    "mask_rng_state",
    "rng_state",
}

for label, checkpoint in (
    ("model_last", model_last),
    ("checkpoint_update070000", frozen),
):
    missing = required - set(checkpoint)
    assert not missing, (label, sorted(missing))

    assert checkpoint["model_name"] == (
        "QMaxSinglecoilFullFreqAux"
    )
    assert checkpoint["global_update"] == 70000
    assert checkpoint["epoch"] == 33
    assert checkpoint["batch_position"] == 1162
    assert checkpoint["seed"] == 1337
    assert checkpoint["batch_size"] == 4
    assert checkpoint["precision"] == "fp32"
    assert checkpoint["frequency_channels"] == 64

for key in (
    "model_state",
    "optimizer_state",
    "scheduler_state",
    "mask_rng_state",
    "rng_state",
):
    if not exact_equal(model_last[key], frozen[key]):
        raise RuntimeError(
            f"model_last and frozen 70k differ: {key}"
        )

print("70K CHECKPOINT PAIR EXACTLY VERIFIED")
print("source update:", model_last["global_update"])
print("epoch:", model_last["epoch"])
print("batch position:", model_last["batch_position"])
PY

echo "===== SOURCE CHECKPOINT HASHES ====="

sha256sum \
  "${SOURCE_CHECKPOINT}" \
  "${FROZEN_70K}"

echo "===== RECOVER 70K TO 80K ====="

python -m py_compile \
  src/m2_prnf_qmax_singlecoil.py \
  src/m2_prnf_qmax_singlecoil_freqaux.py \
  external_benchmarks/singlecoil_paired_dataset_raw.py \
  external_benchmarks/raw_shape_batch_sampler.py \
  external_benchmarks/train_singlecoil_qmax_freqaux.py

python -c "
import torch
assert torch.cuda.is_available(), 'GPU unavailable'
print('GPU:', torch.cuda.get_device_name(0))
"

python external_benchmarks/train_singlecoil_qmax_freqaux.py \
  --bench-root "${BENCH_ROOT}" \
  --fsmnet-root "${FSMNET}" \
  --output-dir "${FORMAL_DIR}" \
  --resume "${SOURCE_CHECKPOINT}" \
  --seed 1337 \
  --batch-size 4 \
  --frequency-channels 64 \
  --max-updates 100000 \
  --stop-at-update "${TARGET_UPDATE}" \
  --save-every 5000 \
  --log-every 100 \
  --learning-rate 1e-4 \
  --weight-decay 1e-4 \
  --lr-step-size 20000 \
  --lr-gamma 0.5 \
  --grad-clip 0.01

echo "===== VERIFY RECOVERED 80K CHECKPOINT ====="

test -f "${TARGET_CHECKPOINT}"
test -f "${FORMAL_DIR}/model_last.pt"
test -f "${FORMAL_DIR}/training_summary.json"
test -f "${FORMAL_DIR}/train_metrics.jsonl"

python - \
  "${FORMAL_DIR}/model_last.pt" \
  "${TARGET_CHECKPOINT}" \
  "${FORMAL_DIR}/training_summary.json" <<'PY'
import json
import sys
from pathlib import Path

import torch


def load(path):
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


model_last_path = Path(sys.argv[1])
checkpoint_path = Path(sys.argv[2])
summary_path = Path(sys.argv[3])

model_last = load(model_last_path)
checkpoint = load(checkpoint_path)
summary = json.loads(
    summary_path.read_text(encoding="utf-8")
)

for label, value in (
    ("model_last", model_last),
    ("checkpoint_update080000", checkpoint),
):
    assert value["model_name"] == (
        "QMaxSinglecoilFullFreqAux"
    )
    assert value["global_update"] == 80000
    assert value["epoch"] == 38
    assert value["batch_position"] == 732
    assert value["seed"] == 1337
    assert value["batch_size"] == 4
    assert value["precision"] == "fp32"
    assert value["frequency_channels"] == 64

assert summary["status"] == "complete"
assert summary["model_name"] == (
    "QMaxSinglecoilFullFreqAux"
)
assert summary["global_update"] == 80000
assert summary["seed"] == 1337
assert summary["batch_size"] == 4
assert summary["precision"] == "fp32"
assert summary["frequency_channels"] == 64

print("RECOVERED 80K CHECKPOINT VERIFIED")
PY

sha256sum \
  "${FORMAL_DIR}/model_last.pt" \
  "${TARGET_CHECKPOINT}" \
  "${FORMAL_DIR}/training_summary.json"

echo "QMAX FREQUENCY-AUX 70K TO 80K RECOVERY PASSED"
echo "held_out_accessed=false"