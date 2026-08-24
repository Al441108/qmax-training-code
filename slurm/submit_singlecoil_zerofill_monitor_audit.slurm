#!/usr/bin/env bash
#SBATCH --job-name=sc_qmax_zfaudit
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:29:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/singlecoil_qmax_benchmark/logs/sc_qmax_zfaudit_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/singlecoil_qmax_benchmark/logs/sc_qmax_zfaudit_%j.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/projects/u6dm/fastmri_project/fastmri_pipeline
BENCH_ROOT=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/singlecoil_qmax_benchmark
FSMNET="${BENCH_ROOT}/repo/FSMNet"

QMAX_EVAL="${BENCH_ROOT}/outputs/qmax_full_seed1337/formal_monitor_evaluation_update100000"
OUTPUT_DIR="${BENCH_ROOT}/outputs/baselines/zero_filled_train_monitor_seed1337"

cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/external_benchmarks:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

source "${HOME}/miniforge3/bin/activate"
conda activate fastmri311

mkdir -p \
  "${BENCH_ROOT}/logs" \
  "${OUTPUT_DIR}"

test -f external_benchmarks/evaluate_singlecoil_zerofill_monitor.py
test -f external_benchmarks/singlecoil_paired_dataset_raw.py
test -f "${FSMNET}/metric.py"
test -f "${BENCH_ROOT}/manifests/train_monitor.csv"
test -f "${QMAX_EVAL}/summary.json"
test -f "${QMAX_EVAL}/slice_metrics.csv"
test -f "${QMAX_EVAL}/volume_metrics.csv"

python -m py_compile \
  external_benchmarks/evaluate_singlecoil_zerofill_monitor.py

python external_benchmarks/evaluate_singlecoil_zerofill_monitor.py \
  --bench-root "${BENCH_ROOT}" \
  --fsmnet-root "${FSMNET}" \
  --qmax-eval-dir "${QMAX_EVAL}" \
  --output-dir "${OUTPUT_DIR}" \
  --mask-rng-seed 1337

python -c "
import json
from pathlib import Path

output = Path('${OUTPUT_DIR}')
summary = json.loads((output / 'summary.json').read_text())

assert summary['status'] == 'complete'
assert summary['model_type'] == 'zero_filled'
assert summary['held_out_accessed'] is False
assert summary['complete_manifest_evaluation'] is True
assert summary['evaluated_slices'] == 571
assert summary['evaluated_volumes'] == 16
assert summary['mask_rng_seed'] == 1337

for name in (
    'slice_metrics.csv',
    'volume_metrics.csv',
    'paired_qmax_vs_zerofill.csv',
):
    assert (output / name).is_file(), name

print(json.dumps(summary, indent=2))
"

echo "ZERO-FILLED TRAIN-MONITOR BASELINE AUDIT PASSED"
echo "output_dir=${OUTPUT_DIR}"