#!/usr/bin/env bash
#SBATCH --job-name=sc_qmax_pilot
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/singlecoil_qmax_benchmark/logs/sc_qmax_pilot_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/singlecoil_qmax_benchmark/logs/sc_qmax_pilot_%j.err
#SBATCH --export=ALL

set -euo pipefail

PIPELINE=/projects/u6dm/fastmri_project/fastmri_pipeline
BENCH_ROOT=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/singlecoil_qmax_benchmark
FSMNET="${BENCH_ROOT}/repo/FSMNet"

MAIN_LOCK="${PIPELINE}/outputs/qmax_stage_a/final_release/MAIN_EXPERIMENTS_COMPLETE.lock"
OUTPUT_DIR="${BENCH_ROOT}/outputs/pilot_seed1337"
export OUTPUT_DIR
LOG_DIR="${BENCH_ROOT}/logs"

cd "${PIPELINE}"

source "$HOME/miniforge3/bin/activate"
conda activate fastmri311

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

test -f "${MAIN_LOCK}" || {
    echo "Main experiments are not frozen:"
    echo "${MAIN_LOCK}"
    exit 1
}

test -f "${BENCH_ROOT}/manifests/train.csv"
test -f "${BENCH_ROOT}/manifests/test_locked.csv"
test -f "${FSMNET}/dataloaders/subsample.py"

test -f src/m2_prnf_qmax_singlecoil.py
test -f external_benchmarks/singlecoil_paired_dataset.py
test -f external_benchmarks/singlecoil_paired_dataset_raw.py
test -f external_benchmarks/raw_shape_batch_sampler.py
test -f external_benchmarks/train_singlecoil_qmax.py

export PYTHONPATH="${PIPELINE}/external_benchmarks:${PIPELINE}:${PYTHONPATH:-}"

python -m py_compile \
  src/m2_prnf_qmax_singlecoil.py \
  external_benchmarks/singlecoil_paired_dataset.py \
  external_benchmarks/singlecoil_paired_dataset_raw.py \
  external_benchmarks/raw_shape_batch_sampler.py \
  external_benchmarks/preflight_singlecoil_qmax_raw.py \
  external_benchmarks/train_singlecoil_qmax.py

python -c \
  "import torch; assert torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0))"

python external_benchmarks/train_singlecoil_qmax.py \
  --bench-root "${BENCH_ROOT}" \
  --fsmnet-root "${FSMNET}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size 4 \
  --seed 1337 \
  --max-updates 2086 \
  --stop-at-update 2086 \
  --save-every 500 \
  --log-every 100 \
  --learning-rate 1e-4 \
  --weight-decay 1e-4 \
  --lr-step-size 20000 \
  --lr-gamma 0.5 \
  --grad-clip 0.01

test -f "${OUTPUT_DIR}/model_last.pt"
test -f "${OUTPUT_DIR}/training_summary.json"

python - <<'PY'
import json
import os
import torch

output_dir = os.environ["OUTPUT_DIR"]

checkpoint = torch.load(
    os.path.join(output_dir, "model_last.pt"),
    map_location="cpu",
    weights_only=False,
)

with open(
    os.path.join(output_dir, "training_summary.json"),
    "r",
    encoding="utf-8",
) as handle:
    summary = json.load(handle)

assert checkpoint["global_update"] == 2086
assert checkpoint["epoch"] == 1
assert checkpoint["batch_position"] == 0
assert summary["global_update"] == 2086
assert summary["status"] == "complete"

print(json.dumps(summary, indent=2))
print("ONE-EPOCH PILOT PASSED")
PY