#!/usr/bin/env bash
#SBATCH --job-name=sc_qmax_freqresume
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/singlecoil_qmax_benchmark/logs/sc_qmax_freqresume_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/singlecoil_qmax_benchmark/logs/sc_qmax_freqresume_%j.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/projects/u6dm/fastmri_project/fastmri_pipeline
BENCH_ROOT=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/singlecoil_qmax_benchmark
FSMNET="${BENCH_ROOT}/repo/FSMNet"

PILOT_DIR="${BENCH_ROOT}/outputs/qmax_freqaux/pilot_seed1337"
PILOT_CHECKPOINT="${PILOT_DIR}/model_last.pt"
PILOT_SUMMARY="${PILOT_DIR}/training_summary.json"
PILOT_LOG="${BENCH_ROOT}/logs/sc_qmax_freqpilot_5940333.out"
PILOT_ERR="${BENCH_ROOT}/logs/sc_qmax_freqpilot_5940333.err"

RESUME_ROOT="${BENCH_ROOT}/outputs/qmax_freqaux/resume_preflight_${SLURM_JOB_ID}"
BRANCH_A="${RESUME_ROOT}/branch_a"
BRANCH_B="${RESUME_ROOT}/branch_b"

source "${HOME}/miniforge3/bin/activate"
conda activate fastmri311

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/external_benchmarks:${PYTHONPATH:-}"

test -f "${PILOT_CHECKPOINT}"
test -f "${PILOT_SUMMARY}"
test -f "${PILOT_LOG}"
test -f "${PILOT_ERR}"
test ! -s "${PILOT_ERR}"

grep -Fq \
  "ONE-EPOCH QMAX FREQUENCY-AUX PILOT PASSED" \
  "${PILOT_LOG}"

python -m py_compile \
  src/m2_prnf_qmax_singlecoil_freqaux.py \
  external_benchmarks/train_singlecoil_qmax_freqaux.py

python -c "
import json
from pathlib import Path

summary = json.loads(
    Path('${PILOT_SUMMARY}').read_text()
)

assert summary['status'] == 'complete'
assert summary['model_name'] == 'QMaxSinglecoilFullFreqAux'
assert summary['global_update'] == 2086
assert summary['epoch'] == 1
assert summary['batch_position'] == 0
assert summary['precision'] == 'fp32'
assert summary['seed'] == 1337
assert summary['batch_size'] == 4
assert summary['frequency_channels'] == 64

print('PILOT CHECKPOINT VERIFIED')
"

test ! -e "${RESUME_ROOT}"
mkdir -p "${BRANCH_A}" "${BRANCH_B}"

sha256sum "${PILOT_CHECKPOINT}" |
  tee "${RESUME_ROOT}/pilot_checkpoint_sha256.txt"

run_branch() {
    local output_dir="$1"

    python external_benchmarks/train_singlecoil_qmax_freqaux.py \
      --bench-root "${BENCH_ROOT}" \
      --fsmnet-root "${FSMNET}" \
      --output-dir "${output_dir}" \
      --resume "${PILOT_CHECKPOINT}" \
      --seed 1337 \
      --batch-size 4 \
      --frequency-channels 64 \
      --max-updates 100000 \
      --stop-at-update 2088 \
      --save-every 1 \
      --log-every 1 \
      --learning-rate 1e-4 \
      --weight-decay 1e-4 \
      --lr-step-size 20000 \
      --lr-gamma 0.5 \
      --grad-clip 0.01
}

echo "===== RUN BRANCH A ====="
run_branch "${BRANCH_A}"

echo "===== RUN BRANCH B ====="
run_branch "${BRANCH_B}"

export BRANCH_A BRANCH_B

python - <<'PY'
import json
import os
from pathlib import Path

import numpy as np
import torch


branch_a = Path(os.environ["BRANCH_A"])
branch_b = Path(os.environ["BRANCH_B"])


def torch_load(path):
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


def exact_equal(left, right):
    if isinstance(left, torch.Tensor):
        return (
            isinstance(right, torch.Tensor)
            and torch.equal(left, right)
        )

    if isinstance(left, np.ndarray):
        return (
            isinstance(right, np.ndarray)
            and np.array_equal(left, right)
        )

    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
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


def maximum_tensor_difference(left, right):
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor):
            raise AssertionError("Tensor type mismatch")

        if left.shape != right.shape:
            raise AssertionError("Tensor shape mismatch")

        if left.is_floating_point() or left.is_complex():
            difference = (
                left - right
            ).abs()

            if not torch.isfinite(difference).all():
                raise AssertionError(
                    "Non-finite checkpoint difference"
                )

            if difference.numel() == 0:
                return 0.0

            return float(difference.max())

        if not torch.equal(left, right):
            raise AssertionError(
                "Non-floating tensor mismatch"
            )

        return 0.0

    if isinstance(left, dict):
        if not isinstance(right, dict):
            raise AssertionError("Dictionary type mismatch")

        if left.keys() != right.keys():
            raise AssertionError(
                "Dictionary key mismatch"
            )

        maximum = 0.0
        for key in left:
            maximum = max(
                maximum,
                maximum_tensor_difference(
                    left[key],
                    right[key],
                ),
            )
        return maximum

    if isinstance(left, (tuple, list)):
        if (
            not isinstance(right, type(left))
            or len(left) != len(right)
        ):
            raise AssertionError(
                "Sequence mismatch"
            )

        maximum = 0.0
        for a, b in zip(left, right):
            maximum = max(
                maximum,
                maximum_tensor_difference(a, b),
            )
        return maximum

    if left != right:
        raise AssertionError(
            f"Scalar mismatch: {left!r} != {right!r}"
        )

    return 0.0


checkpoint_a = torch_load(
    branch_a / "model_last.pt"
)
checkpoint_b = torch_load(
    branch_b / "model_last.pt"
)

for checkpoint in (checkpoint_a, checkpoint_b):
    assert checkpoint["model_name"] == (
        "QMaxSinglecoilFullFreqAux"
    )
    assert checkpoint["global_update"] == 2088
    assert checkpoint["epoch"] == 1
    assert checkpoint["batch_position"] == 2
    assert checkpoint["seed"] == 1337
    assert checkpoint["batch_size"] == 4
    assert checkpoint["precision"] == "fp32"
    assert checkpoint["frequency_channels"] == 64

model_difference = maximum_tensor_difference(
    checkpoint_a["model_state"],
    checkpoint_b["model_state"],
)

optimizer_difference = maximum_tensor_difference(
    checkpoint_a["optimizer_state"],
    checkpoint_b["optimizer_state"],
)

scheduler_equal = exact_equal(
    checkpoint_a["scheduler_state"],
    checkpoint_b["scheduler_state"],
)

mask_rng_equal = exact_equal(
    checkpoint_a["mask_rng_state"],
    checkpoint_b["mask_rng_state"],
)

rng_equal = exact_equal(
    checkpoint_a["rng_state"],
    checkpoint_b["rng_state"],
)

rows_a = [
    json.loads(line)
    for line in (
        branch_a / "train_metrics.jsonl"
    ).read_text().splitlines()
    if line.strip()
]

rows_b = [
    json.loads(line)
    for line in (
        branch_b / "train_metrics.jsonl"
    ).read_text().splitlines()
    if line.strip()
]

assert [row["update"] for row in rows_a] == [
    2087,
    2088,
]
assert [row["update"] for row in rows_b] == [
    2087,
    2088,
]

loss_difference = max(
    abs(a["loss"] - b["loss"])
    for a, b in zip(rows_a, rows_b)
)

print(
    "model maximum absolute difference:",
    model_difference,
)
print(
    "optimizer maximum absolute difference:",
    optimizer_difference,
)
print("scheduler equal:", scheduler_equal)
print("mask RNG equal:", mask_rng_equal)
print("all RNG equal:", rng_equal)
print(
    "loss maximum absolute difference:",
    loss_difference,
)

assert model_difference <= 1e-4
assert optimizer_difference <= 1e-4
assert scheduler_equal
assert mask_rng_equal
assert rng_equal
assert loss_difference <= 2e-4

summary = {
    "status": "passed",
    "source_update": 2086,
    "target_update": 2088,
    "model_max_abs_difference": model_difference,
    "optimizer_max_abs_difference": (
        optimizer_difference
    ),
    "scheduler_equal": scheduler_equal,
    "mask_rng_equal": mask_rng_equal,
    "rng_equal": rng_equal,
    "loss_max_abs_difference": loss_difference,
    "tolerance": 1e-4,
}

(
    Path(os.environ["BRANCH_A"]).parent
    / "resume_preflight_summary.json"
).write_text(
    json.dumps(summary, indent=2) + "\n"
)
PY

test -f \
  "${RESUME_ROOT}/resume_preflight_summary.json"

cat \
  "${RESUME_ROOT}/resume_preflight_summary.json"

echo \
  "QMAX FREQUENCY-AUX CHECKPOINT/RESUME PREFLIGHT PASSED"