#!/usr/bin/env bash
#SBATCH --job-name=sc_qmax_heldout3
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=01:59:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/singlecoil_qmax_benchmark/logs/sc_qmax_heldout3_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/singlecoil_qmax_benchmark/logs/sc_qmax_heldout3_%j.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/projects/u6dm/fastmri_project/fastmri_pipeline
BENCH_ROOT=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/singlecoil_qmax_benchmark
FSMNET="${BENCH_ROOT}/repo/FSMNet"

QMAX_CHECKPOINT="${BENCH_ROOT}/outputs/qmax_full_seed1337/formal_100k/model_last.pt"
FREQ_CHECKPOINT="${BENCH_ROOT}/outputs/qmax_freqaux/formal_seed1337_100k/model_last.pt"
FREQ_VERIFICATION="${BENCH_ROOT}/outputs/qmax_freqaux/final_verification_seed1337_100k/verification.json"
QMAX_MONITOR="${BENCH_ROOT}/outputs/qmax_full_seed1337/formal_monitor_evaluation_update100000/summary.json"
ZF_MONITOR="${BENCH_ROOT}/outputs/baselines/zero_filled_train_monitor_seed1337/summary.json"
OUTPUT_DIR="${BENCH_ROOT}/outputs/formal_heldout/three_arm_seed1337"
CODE_HASH_FILE="${BENCH_ROOT}/protocol/singlecoil_three_arm_heldout_code_sha256.txt"
GATE_AMENDMENT="${BENCH_ROOT}/protocol/PROTOCOL_AMENDMENT_003_HELDOUT_MONITOR_TOLERANCE.lock"

cd "${PROJECT_ROOT}"
source "${HOME}/miniforge3/bin/activate"
conda activate fastmri311
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/external_benchmarks:${PYTHONPATH:-}"

mkdir -p "${BENCH_ROOT}/logs" "${OUTPUT_DIR}"

for required in \
  external_benchmarks/evaluate_singlecoil_three_arm_heldout.py \
  external_benchmarks/singlecoil_paired_dataset_raw.py \
  src/m2_prnf_qmax_singlecoil.py \
  src/m2_prnf_qmax_singlecoil_freqaux.py \
  "${QMAX_CHECKPOINT}" \
  "${FREQ_CHECKPOINT}" \
  "${FREQ_VERIFICATION}" \
  "${QMAX_MONITOR}" \
  "${ZF_MONITOR}" \
  "${BENCH_ROOT}/manifests/train_monitor.csv" \
  "${BENCH_ROOT}/manifests/test_locked.csv" \
  "${FSMNET}/metric.py" \
  "${CODE_HASH_FILE}" \
  "${GATE_AMENDMENT}"
do
  test -f "${required}" || { echo "Missing: ${required}"; exit 1; }
done

grep -Fqx \
  "status=APPROVED_BEFORE_HELD_OUT_ACCESS" \
  "${GATE_AMENDMENT}"

grep -Fqx \
  "held_out_accessed=false" \
  "${GATE_AMENDMENT}"

grep -Fqx \
  "change=uniform_monitor_reproduction_tolerances_without_changing_heldout_metrics" \
  "${GATE_AMENDMENT}"

sha256sum -c "${CODE_HASH_FILE}"

python -m py_compile \
  external_benchmarks/evaluate_singlecoil_three_arm_heldout.py

python -c "import torch; assert torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0))"

python external_benchmarks/evaluate_singlecoil_three_arm_heldout.py \
  --bench-root "${BENCH_ROOT}" \
  --fsmnet-root "${FSMNET}" \
  --qmax-checkpoint "${QMAX_CHECKPOINT}" \
  --freq-checkpoint "${FREQ_CHECKPOINT}" \
  --freq-verification "${FREQ_VERIFICATION}" \
  --qmax-monitor-summary "${QMAX_MONITOR}" \
  --zf-monitor-summary "${ZF_MONITOR}" \
  --output-dir "${OUTPUT_DIR}" \
  --mask-seed 1337 \
  --bootstrap-resamples 10000 \
  --bootstrap-seed 1337

python -c "
import json
from pathlib import Path
p=Path('${OUTPUT_DIR}/summary.json')
x=json.loads(p.read_text())
assert x['status']=='complete'
assert x['held_out_accessed'] is True
assert x['complete_heldout_evaluation'] is True
assert x['heldout_slices']==1665
assert x['heldout_volumes']==45
print(json.dumps(x['overfitting'], indent=2))
"

echo "THREE-ARM FORMAL HELD-OUT EVALUATION VERIFIED"
echo "output_dir=${OUTPUT_DIR}"
