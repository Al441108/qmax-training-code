#!/usr/bin/env bash
#SBATCH --job-name=sc_qmax_resume
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/singlecoil_qmax_benchmark/logs/sc_qmax_resume_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/singlecoil_qmax_benchmark/logs/sc_qmax_resume_%j.err
#SBATCH --export=ALL

set -euo pipefail

PIPELINE=/projects/u6dm/fastmri_project/fastmri_pipeline
BENCH_ROOT=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/singlecoil_qmax_benchmark
FSMNET="${BENCH_ROOT}/repo/FSMNet"

RUN_ROOT="${BENCH_ROOT}/outputs/resume_preflight_${SLURM_JOB_ID}"
CONTINUOUS="${RUN_ROOT}/continuous_4"
SPLIT="${RUN_ROOT}/split_2_plus_2"

cd "${PIPELINE}"

source "$HOME/miniforge3/bin/activate"
conda activate fastmri311

mkdir -p \
  "${BENCH_ROOT}/logs" \
  "${CONTINUOUS}" \
  "${SPLIT}"

export PYTHONPATH="${PIPELINE}/external_benchmarks:${PIPELINE}:${PYTHONPATH:-}"

python -m py_compile \
  external_benchmarks/train_singlecoil_qmax.py

echo "===== CONTINUOUS 0 -> 4 ====="

python external_benchmarks/train_singlecoil_qmax.py \
  --bench-root "${BENCH_ROOT}" \
  --fsmnet-root "${FSMNET}" \
  --output-dir "${CONTINUOUS}" \
  --batch-size 4 \
  --seed 1337 \
  --max-updates 4 \
  --stop-at-update 4 \
  --save-every 2 \
  --log-every 1

echo "===== SPLIT 0 -> 2 ====="

python external_benchmarks/train_singlecoil_qmax.py \
  --bench-root "${BENCH_ROOT}" \
  --fsmnet-root "${FSMNET}" \
  --output-dir "${SPLIT}" \
  --batch-size 4 \
  --seed 1337 \
  --max-updates 4 \
  --stop-at-update 2 \
  --save-every 2 \
  --log-every 1

echo "===== RESUME 2 -> 4 ====="

python external_benchmarks/train_singlecoil_qmax.py \
  --bench-root "${BENCH_ROOT}" \
  --fsmnet-root "${FSMNET}" \
  --output-dir "${SPLIT}" \
  --resume "${SPLIT}/model_last.pt" \
  --batch-size 4 \
  --seed 1337 \
  --max-updates 4 \
  --stop-at-update 4 \
  --save-every 2 \
  --log-every 1

echo "===== COMPARE FINAL CHECKPOINTS ====="

export CONTINUOUS_CHECKPOINT="${CONTINUOUS}/model_last.pt"
export SPLIT_CHECKPOINT="${SPLIT}/model_last.pt"

python - <<'PY'
import os
import torch

a = torch.load(
    os.environ["CONTINUOUS_CHECKPOINT"],
    map_location="cpu",
    weights_only=False,
)
b = torch.load(
    os.environ["SPLIT_CHECKPOINT"],
    map_location="cpu",
    weights_only=False,
)

assert a["global_update"] == b["global_update"] == 4
assert a["epoch"] == b["epoch"]
assert a["batch_position"] == b["batch_position"]
assert a["scheduler_state"] == b["scheduler_state"]

maximum_difference = 0.0
different_tensors = 0

for name, first in a["model_state"].items():
    second = b["model_state"][name]

    if first.dtype.is_floating_point:
        difference = float(
            (first - second).abs().max().item()
        )
    else:
        difference = 0.0 if torch.equal(
            first,
            second,
        ) else float("inf")

    maximum_difference = max(
        maximum_difference,
        difference,
    )

    if difference != 0.0:
        different_tensors += 1

print("global_update:", a["global_update"])
print("epoch:", a["epoch"])
print("batch_position:", a["batch_position"])
print("different model tensors:", different_tensors)
print("model maximum absolute difference:",
      maximum_difference)

# FP32 CUDA kernels are numerically, but not bitwise, reproducible.
assert maximum_difference <= 1e-4

print("NUMERICAL CHECKPOINT/RESUME PREFLIGHT PASSED")
PY

echo "RUN_ROOT=${RUN_ROOT}"