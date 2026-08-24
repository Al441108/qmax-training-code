#!/usr/bin/env bash
#SBATCH --job-name=sc_qmax_preflight
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/singlecoil_qmax_benchmark/logs/sc_qmax_preflight_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/singlecoil_qmax_benchmark/logs/sc_qmax_preflight_%j.err
#SBATCH --export=ALL

set -euo pipefail

PIPELINE=/projects/u6dm/fastmri_project/fastmri_pipeline
BENCH_ROOT=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/singlecoil_qmax_benchmark
FSMNET="${BENCH_ROOT}/repo/FSMNet"
BATCH_SIZE="${PREFLIGHT_BATCH_SIZE:-1}"

cd "${PIPELINE}"

source "$HOME/miniforge3/bin/activate"
conda activate fastmri311

mkdir -p "${BENCH_ROOT}/logs"

test -f external_benchmarks/preflight_singlecoil_qmax_raw.py
test -f external_benchmarks/singlecoil_paired_dataset.py
test -f external_benchmarks/singlecoil_paired_dataset_raw.py
test -f external_benchmarks/raw_shape_batch_sampler.py
test -f src/m2_prnf_qmax_singlecoil.py
test -f "${BENCH_ROOT}/manifests/train.csv"
test -f "${FSMNET}/dataloaders/subsample.py"

python -m py_compile \
  src/m2_prnf_qmax_singlecoil.py \
  external_benchmarks/singlecoil_paired_dataset.py \
  external_benchmarks/singlecoil_paired_dataset_raw.py \
  external_benchmarks/raw_shape_batch_sampler.py \
  external_benchmarks/preflight_singlecoil_qmax_raw.py

export BENCH_ROOT
export FSMNET
export PYTHONPATH="${PIPELINE}/external_benchmarks:${PIPELINE}:${PYTHONPATH:-}"

python -c \
  "import torch; assert torch.cuda.is_available(), 'GPU is not visible'; print('GPU:', torch.cuda.get_device_name(0))"

python external_benchmarks/preflight_singlecoil_qmax_raw.py \
  --bench-root "${BENCH_ROOT}" \
  --fsmnet-root "${FSMNET}" \
  --batch-size "${BATCH_SIZE}" \
  --raw-height 640 \
  --raw-width 368 \
  --seed 1337