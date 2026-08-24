#!/usr/bin/env bash
#SBATCH --job-name=sc_qmax_freqaux
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/singlecoil_qmax_benchmark/logs/sc_qmax_freqaux_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/singlecoil_qmax_benchmark/logs/sc_qmax_freqaux_%j.err
#SBATCH --export=ALL

set -euo pipefail

PIPELINE=/projects/u6dm/fastmri_project/fastmri_pipeline
BENCH_ROOT=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/singlecoil_qmax_benchmark
FSMNET="${BENCH_ROOT}/repo/FSMNet"
BATCH_SIZE="${BATCH_SIZE:-1}"

cd "${PIPELINE}"

source "$HOME/miniforge3/bin/activate"
conda activate fastmri311

mkdir -p "${BENCH_ROOT}/logs"

export BENCH_ROOT
export FSMNET
export PYTHONPATH="${PIPELINE}/external_benchmarks:${PIPELINE}:${PYTHONPATH:-}"

test -f src/m2_prnf_qmax_singlecoil_freqaux.py
test -f external_benchmarks/preflight_singlecoil_qmax_freqaux.py

python -m py_compile \
  src/m2_prnf_qmax_singlecoil_freqaux.py \
  external_benchmarks/preflight_singlecoil_qmax_freqaux.py

python -c \
  "import torch; assert torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0))"

python external_benchmarks/preflight_singlecoil_qmax_freqaux.py \
  --bench-root "${BENCH_ROOT}" \
  --fsmnet-root "${FSMNET}" \
  --batch-size "${BATCH_SIZE}" \
  --raw-height 640 \
  --raw-width 368 \
  --frequency-channels 64 \
  --seed 1337