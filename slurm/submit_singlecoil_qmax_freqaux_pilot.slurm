#!/usr/bin/env bash
#SBATCH --job-name=sc_qmax_freqpilot
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/singlecoil_qmax_benchmark/logs/sc_qmax_freqpilot_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/singlecoil_qmax_benchmark/logs/sc_qmax_freqpilot_%j.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/projects/u6dm/fastmri_project/fastmri_pipeline
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/external_benchmarks:${PYTHONPATH:-}"
BENCH_ROOT=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/singlecoil_qmax_benchmark
FSMNET="${BENCH_ROOT}/repo/FSMNet"
OUTPUT_DIR="${BENCH_ROOT}/outputs/qmax_freqaux/pilot_seed1337"

: "${PREFLIGHT_JOB_ID:?PREFLIGHT_JOB_ID must contain the passed batch-4 preflight Job ID}"

PREFLIGHT_OUT="${BENCH_ROOT}/logs/sc_qmax_freqaux_${PREFLIGHT_JOB_ID}.out"
PREFLIGHT_ERR="${BENCH_ROOT}/logs/sc_qmax_freqaux_${PREFLIGHT_JOB_ID}.err"

cd "${PROJECT_ROOT}"

source "${HOME}/miniforge3/bin/activate"
conda activate fastmri311

test -f external_benchmarks/train_singlecoil_qmax_freqaux.py
test -f external_benchmarks/preflight_singlecoil_qmax_freqaux.py
test -f src/m2_prnf_qmax_singlecoil_freqaux.py
test -f "${PREFLIGHT_OUT}"
test -f "${PREFLIGHT_ERR}"

grep -Fq \
  "batch size: 4" \
  "${PREFLIGHT_OUT}"

grep -Fq \
  "QMAX FREQUENCY-AUX FORWARD/BACKWARD PREFLIGHT PASSED" \
  "${PREFLIGHT_OUT}"

test ! -s "${PREFLIGHT_ERR}"

python -m py_compile \
  external_benchmarks/train_singlecoil_qmax_freqaux.py

python -c \
  "import torch; assert torch.cuda.is_available(), 'GPU unavailable'; print('GPU:', torch.cuda.get_device_name(0))"

mkdir -p "${OUTPUT_DIR}"

python external_benchmarks/train_singlecoil_qmax_freqaux.py \
  --bench-root "${BENCH_ROOT}" \
  --fsmnet-root "${FSMNET}" \
  --output-dir "${OUTPUT_DIR}" \
  --seed 1337 \
  --batch-size 4 \
  --frequency-channels 64 \
  --max-updates 100000 \
  --stop-at-update 2086 \
  --save-every 500 \
  --log-every 100 \
  --learning-rate 1e-4 \
  --weight-decay 1e-4 \
  --lr-step-size 20000 \
  --lr-gamma 0.5 \
  --grad-clip 0.01

python -c \
  "import json, pathlib; p=pathlib.Path('${OUTPUT_DIR}/training_summary.json'); x=json.loads(p.read_text()); assert x['status']=='complete'; assert x['global_update']==2086; assert x['batch_size']==4; assert x['precision']=='fp32'; print(json.dumps(x, indent=2))"

test -f "${OUTPUT_DIR}/model_last.pt"
test -f "${OUTPUT_DIR}/checkpoint_update002086.pt"
test -f "${OUTPUT_DIR}/train_metrics.jsonl"

echo "ONE-EPOCH QMAX FREQUENCY-AUX PILOT PASSED"