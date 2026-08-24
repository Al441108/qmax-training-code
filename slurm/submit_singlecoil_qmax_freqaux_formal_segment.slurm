#!/usr/bin/env bash
#SBATCH --job-name=sc_qmax_freqformal
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/singlecoil_qmax_benchmark/logs/sc_qmax_freqformal_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/singlecoil_qmax_benchmark/logs/sc_qmax_freqformal_%j.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/projects/u6dm/fastmri_project/fastmri_pipeline
BENCH_ROOT=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/singlecoil_qmax_benchmark
FSMNET="${BENCH_ROOT}/repo/FSMNet"

OUTPUT_DIR="${BENCH_ROOT}/outputs/qmax_freqaux/formal_seed1337_100k"
HASH_FILE="${BENCH_ROOT}/protocol/singlecoil_freqaux_formal_code_sha256.txt"
AMENDMENT_FILE="${BENCH_ROOT}/protocol/PROTOCOL_AMENDMENT_001_SINGLECOIL_FREQAUX_PARALLEL.lock"
MAIN_LOCK="${PROJECT_ROOT}/outputs/qmax_stage_a/final_release/MAIN_EXPERIMENTS_COMPLETE.lock"

PILOT_SUMMARY="${BENCH_ROOT}/outputs/qmax_freqaux/pilot_seed1337/training_summary.json"
PILOT_OUT="${BENCH_ROOT}/logs/sc_qmax_freqpilot_5940333.out"
PILOT_ERR="${BENCH_ROOT}/logs/sc_qmax_freqpilot_5940333.err"

: "${TARGET_UPDATE:?TARGET_UPDATE is required}"
: "${RESUME_MODE:?RESUME_MODE is required}"

case "${TARGET_UPDATE}" in
    20000|40000|60000|80000|100000)
        ;;
    *)
        echo "Invalid TARGET_UPDATE=${TARGET_UPDATE}"
        exit 1
        ;;
esac

EXPECTED_MODE=fresh
if [ "${TARGET_UPDATE}" -gt 20000 ]; then
    EXPECTED_MODE=resume
fi

if [ "${RESUME_MODE}" != "${EXPECTED_MODE}" ]; then
    echo \
      "TARGET_UPDATE=${TARGET_UPDATE} requires " \
      "RESUME_MODE=${EXPECTED_MODE}, got ${RESUME_MODE}"
    exit 1
fi

source "${HOME}/miniforge3/bin/activate"
conda activate fastmri311

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/external_benchmarks:${PYTHONPATH:-}"

test -f "${MAIN_LOCK}"
test -f "${AMENDMENT_FILE}"
test -f "${HASH_FILE}"

grep -Fqx \
  "status=APPROVED_BEFORE_HELD_OUT_TEST" \
  "${AMENDMENT_FILE}"

grep -Fqx \
  "design=prespecified_parallel_models" \
  "${AMENDMENT_FILE}"

grep -Fqx \
  "arm_a=QMaxSinglecoilFull" \
  "${AMENDMENT_FILE}"

grep -Fqx \
  "arm_b=QMaxSinglecoilFullFreqAux" \
  "${AMENDMENT_FILE}"

grep -Fqx \
  "reporting_policy=report_both_parallel_models" \
  "${AMENDMENT_FILE}"

grep -Fqx \
  "held_out_test_accessed=false" \
  "${AMENDMENT_FILE}"

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

test -f "${PILOT_SUMMARY}"
test -f "${PILOT_OUT}"
test -f "${PILOT_ERR}"
test ! -s "${PILOT_ERR}"

grep -Fq \
  "ONE-EPOCH QMAX FREQUENCY-AUX PILOT PASSED" \
  "${PILOT_OUT}"

python -m py_compile \
  src/m2_prnf_qmax_singlecoil.py \
  src/m2_prnf_qmax_singlecoil_freqaux.py \
  external_benchmarks/singlecoil_paired_dataset.py \
  external_benchmarks/singlecoil_paired_dataset_raw.py \
  external_benchmarks/raw_shape_batch_sampler.py \
  external_benchmarks/preflight_singlecoil_qmax_raw.py \
  external_benchmarks/train_singlecoil_qmax_freqaux.py

python -c "
import torch
assert torch.cuda.is_available(), 'GPU unavailable'
print('GPU:', torch.cuda.get_device_name(0))
"

mkdir -p "${OUTPUT_DIR}"

RESUME_ARGUMENTS=()

if [ "${RESUME_MODE}" = "fresh" ]; then
    if (
      [ -e "${OUTPUT_DIR}/model_last.pt" ] ||
      [ -s "${OUTPUT_DIR}/train_metrics.jsonl" ]
    ); then
        echo "Refusing to overwrite existing formal run"
        exit 1
    fi
else
    PREVIOUS_UPDATE=$((TARGET_UPDATE - 20000))
    CHECKPOINT="${OUTPUT_DIR}/model_last.pt"

    PREVIOUS_CHECKPOINT="$(
      printf \
        '%s/checkpoint_update%06d.pt' \
        "${OUTPUT_DIR}" \
        "${PREVIOUS_UPDATE}"
    )"

    test -f "${CHECKPOINT}"
    test -f "${PREVIOUS_CHECKPOINT}"

    python - \
      "${CHECKPOINT}" \
      "${PREVIOUS_UPDATE}" <<'PY'
import sys
from pathlib import Path

import torch


checkpoint_path = Path(sys.argv[1])
expected_update = int(sys.argv[2])

try:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
except TypeError:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

assert checkpoint["model_name"] == (
    "QMaxSinglecoilFullFreqAux"
)
assert checkpoint["global_update"] == expected_update
assert checkpoint["seed"] == 1337
assert checkpoint["batch_size"] == 4
assert checkpoint["precision"] == "fp32"
assert checkpoint["frequency_channels"] == 64

print(
    "verified resume checkpoint:",
    checkpoint_path,
    expected_update,
)
PY

    RESUME_ARGUMENTS=(
      --resume "${CHECKPOINT}"
    )
fi

python external_benchmarks/train_singlecoil_qmax_freqaux.py \
  --bench-root "${BENCH_ROOT}" \
  --fsmnet-root "${FSMNET}" \
  --output-dir "${OUTPUT_DIR}" \
  "${RESUME_ARGUMENTS[@]}" \
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

TARGET_CHECKPOINT="$(
  printf \
    '%s/checkpoint_update%06d.pt' \
    "${OUTPUT_DIR}" \
    "${TARGET_UPDATE}"
)"

test -f "${TARGET_CHECKPOINT}"
test -f "${OUTPUT_DIR}/model_last.pt"
test -f "${OUTPUT_DIR}/training_summary.json"
test -f "${OUTPUT_DIR}/train_metrics.jsonl"

python - \
  "${OUTPUT_DIR}/model_last.pt" \
  "${OUTPUT_DIR}/training_summary.json" \
  "${TARGET_UPDATE}" <<'PY'
import json
import sys
from pathlib import Path

import torch


checkpoint_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
expected_update = int(sys.argv[3])

try:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
except TypeError:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

summary = json.loads(
    summary_path.read_text()
)

assert checkpoint["model_name"] == (
    "QMaxSinglecoilFullFreqAux"
)
assert checkpoint["global_update"] == expected_update
assert checkpoint["seed"] == 1337
assert checkpoint["batch_size"] == 4
assert checkpoint["precision"] == "fp32"
assert checkpoint["frequency_channels"] == 64

assert summary["status"] == "complete"
assert summary["model_name"] == (
    "QMaxSinglecoilFullFreqAux"
)
assert summary["global_update"] == expected_update
assert summary["seed"] == 1337
assert summary["batch_size"] == 4
assert summary["precision"] == "fp32"
assert summary["frequency_channels"] == 64

print(
    "verified completed formal segment:",
    expected_update,
)
PY

echo \
  "QMAX FREQUENCY-AUX FORMAL SEGMENT COMPLETE " \
  "TARGET_UPDATE=${TARGET_UPDATE}"