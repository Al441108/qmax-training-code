#!/usr/bin/env bash
#SBATCH --job-name=sc_qmax_branch
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=00:20:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/singlecoil_qmax_benchmark/logs/sc_qmax_branch_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/singlecoil_qmax_benchmark/logs/sc_qmax_branch_%j.err
#SBATCH --export=ALL

set -euo pipefail

PIPELINE=/projects/u6dm/fastmri_project/fastmri_pipeline
BENCH_ROOT=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/singlecoil_qmax_benchmark
FSMNET="${BENCH_ROOT}/repo/FSMNet"

SOURCE_JOB_ID="${SOURCE_JOB_ID:-5903215}"
SOURCE="${BENCH_ROOT}/outputs/resume_preflight_${SOURCE_JOB_ID}/continuous_4"
BRANCH="${BENCH_ROOT}/outputs/resume_branch_${SLURM_JOB_ID}"

SOURCE_UPDATE2="${SOURCE}/checkpoint_update000002.pt"
SOURCE_UPDATE4="${SOURCE}/checkpoint_update000004.pt"

cd "${PIPELINE}"

source "$HOME/miniforge3/bin/activate"
conda activate fastmri311

mkdir -p "${BRANCH}"

test -f "${SOURCE_UPDATE2}"
test -f "${SOURCE_UPDATE4}"

export PYTHONPATH="${PIPELINE}/external_benchmarks:${PIPELINE}:${PYTHONPATH:-}"

echo "===== RESUME SHARED UPDATE-2 CHECKPOINT ====="

python external_benchmarks/train_singlecoil_qmax.py \
  --bench-root "${BENCH_ROOT}" \
  --fsmnet-root "${FSMNET}" \
  --output-dir "${BRANCH}" \
  --resume "${SOURCE_UPDATE2}" \
  --batch-size 4 \
  --seed 1337 \
  --max-updates 4 \
  --stop-at-update 4 \
  --save-every 2 \
  --log-every 1

export SOURCE_UPDATE4
export BRANCH_UPDATE4="${BRANCH}/checkpoint_update000004.pt"

echo "===== COMPARE SHARED-CHECKPOINT BRANCHES ====="

python - <<'PY'
import os
import numpy as np
import torch

reference = torch.load(
    os.environ["SOURCE_UPDATE4"],
    map_location="cpu",
    weights_only=False,
)
resumed = torch.load(
    os.environ["BRANCH_UPDATE4"],
    map_location="cpu",
    weights_only=False,
)

assert reference["global_update"] == resumed["global_update"] == 4
assert reference["epoch"] == resumed["epoch"]
assert reference["batch_position"] == resumed["batch_position"]
assert reference["scheduler_state"] == resumed["scheduler_state"]

maximum_difference = 0.0
different_tensors = 0

for name, first in reference["model_state"].items():
    second = resumed["model_state"][name]

    if first.dtype.is_floating_point:
        difference = float(
            (first - second).abs().max().item()
        )
    else:
        difference = (
            0.0 if torch.equal(first, second)
            else float("inf")
        )

    maximum_difference = max(maximum_difference, difference)

    if difference != 0:
        different_tensors += 1

reference_mask_rng = reference["mask_rng_state"]
resumed_mask_rng = resumed["mask_rng_state"]

mask_rng_equal = (
    reference_mask_rng[0] == resumed_mask_rng[0]
    and np.array_equal(
        reference_mask_rng[1],
        resumed_mask_rng[1],
    )
    and reference_mask_rng[2:] == resumed_mask_rng[2:]
)

torch_rng_equal = torch.equal(
    reference["rng_state"]["torch_cpu"],
    resumed["rng_state"]["torch_cpu"],
)

cuda_rng_equal = all(
    torch.equal(first, second)
    for first, second in zip(
        reference["rng_state"]["torch_cuda"],
        resumed["rng_state"]["torch_cuda"],
    )
)

print("different model tensors:", different_tensors)
print("model maximum absolute difference:", maximum_difference)
print("mask RNG equal:", mask_rng_equal)
print("torch CPU RNG equal:", torch_rng_equal)
print("torch CUDA RNG equal:", cuda_rng_equal)

assert maximum_difference <= 1e-7
assert mask_rng_equal
assert torch_rng_equal
assert cuda_rng_equal

print("SHARED-CHECKPOINT RESUME PREFLIGHT PASSED")
PY

echo "BRANCH=${BRANCH}"