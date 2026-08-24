#!/usr/bin/env bash
#SBATCH --job-name=sc_qmax_formal
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/singlecoil_qmax_benchmark/logs/sc_qmax_formal_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/singlecoil_qmax_benchmark/logs/sc_qmax_formal_%j.err
#SBATCH --export=ALL

set -euo pipefail

PIPELINE=/projects/u6dm/fastmri_project/fastmri_pipeline
BENCH_ROOT=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/singlecoil_qmax_benchmark
FSMNET="${BENCH_ROOT}/repo/FSMNet"

MAIN_LOCK="${PIPELINE}/outputs/qmax_stage_a/final_release/MAIN_EXPERIMENTS_COMPLETE.lock"
PILOT_DIR="${BENCH_ROOT}/outputs/pilot_seed1337"
OUTPUT_DIR="${BENCH_ROOT}/outputs/qmax_full_seed1337/formal_100k"
HASH_FILE="${BENCH_ROOT}/protocol/singlecoil_formal_training_code_sha256.txt"

TARGET_UPDATE="${TARGET_UPDATE:?TARGET_UPDATE is required}"
RESUME_MODE="${RESUME_MODE:?RESUME_MODE is required}"

case "${TARGET_UPDATE}" in
    20000|40000|60000|80000|100000)
        ;;
    *)
        echo "Invalid TARGET_UPDATE=${TARGET_UPDATE}"
        exit 1
        ;;
esac

case "${RESUME_MODE}" in
    fresh|resume)
        ;;
    *)
        echo "Invalid RESUME_MODE=${RESUME_MODE}"
        exit 1
        ;;
esac

if [[ "${TARGET_UPDATE}" == "20000" && "${RESUME_MODE}" != "fresh" ]]; then
    echo "The first segment must use RESUME_MODE=fresh"
    exit 1
fi

if [[ "${TARGET_UPDATE}" != "20000" && "${RESUME_MODE}" != "resume" ]]; then
    echo "Later segments must use RESUME_MODE=resume"
    exit 1
fi

cd "${PIPELINE}"

source "$HOME/miniforge3/bin/activate"
conda activate fastmri311

mkdir -p \
    "${BENCH_ROOT}/logs" \
    "${OUTPUT_DIR}"

test -f "${MAIN_LOCK}"
test -f "${PILOT_DIR}/model_last.pt"
test -f "${PILOT_DIR}/training_summary.json"
test -f "${HASH_FILE}"

# Refuse to run if any frozen training file has changed.
sha256sum -c "${HASH_FILE}"

export PYTHONPATH="${PIPELINE}/external_benchmarks:${PIPELINE}:${PYTHONPATH:-}"

python -m py_compile \
    src/m2_prnf_qmax_singlecoil.py \
    external_benchmarks/singlecoil_paired_dataset.py \
    external_benchmarks/singlecoil_paired_dataset_raw.py \
    external_benchmarks/raw_shape_batch_sampler.py \
    external_benchmarks/preflight_singlecoil_qmax_raw.py \
    external_benchmarks/train_singlecoil_qmax.py

# Confirm the one-epoch pilot genuinely passed.
python - "${PILOT_DIR}" <<'PY'
import json
import os
import sys
import torch

pilot_dir = sys.argv[1]

with open(
    os.path.join(pilot_dir, "training_summary.json"),
    "r",
    encoding="utf-8",
) as handle:
    summary = json.load(handle)

checkpoint = torch.load(
    os.path.join(pilot_dir, "model_last.pt"),
    map_location="cpu",
    weights_only=False,
)

assert summary["status"] == "complete"
assert summary["global_update"] == 2086
assert checkpoint["global_update"] == 2086

print("PILOT GATE PASSED")
PY

RESUME_ARGUMENTS=()

if [[ "${RESUME_MODE}" == "fresh" ]]; then
    if [[ -e "${OUTPUT_DIR}/model_last.pt" ]]; then
        echo "Refusing to overwrite an existing formal run:"
        echo "${OUTPUT_DIR}/model_last.pt"
        exit 1
    fi
else
    CHECKPOINT="${OUTPUT_DIR}/model_last.pt"
    EXPECTED_START=$((TARGET_UPDATE - 20000))

    test -f "${CHECKPOINT}"

    python - "${CHECKPOINT}" "${EXPECTED_START}" <<'PY'
import sys
import torch

checkpoint_path = sys.argv[1]
expected_update = int(sys.argv[2])

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=False,
)

actual_update = int(checkpoint["global_update"])

assert actual_update == expected_update, (
    f"Expected update {expected_update}, got {actual_update}"
)

print(
    f"RESUME GATE PASSED: update={actual_update}"
)
PY

    RESUME_ARGUMENTS=(
        --resume "${CHECKPOINT}"
    )
fi

echo "===== FORMAL SEGMENT ====="
echo "target_update=${TARGET_UPDATE}"
echo "resume_mode=${RESUME_MODE}"
echo "output_dir=${OUTPUT_DIR}"

python external_benchmarks/train_singlecoil_qmax.py \
    --bench-root "${BENCH_ROOT}" \
    --fsmnet-root "${FSMNET}" \
    --output-dir "${OUTPUT_DIR}" \
    "${RESUME_ARGUMENTS[@]}" \
    --batch-size 4 \
    --seed 1337 \
    --max-updates 100000 \
    --stop-at-update "${TARGET_UPDATE}" \
    --save-every 20000 \
    --log-every 100 \
    --learning-rate 1e-4 \
    --weight-decay 1e-4 \
    --lr-step-size 20000 \
    --lr-gamma 0.5 \
    --grad-clip 0.01

python - "${OUTPUT_DIR}/model_last.pt" "${TARGET_UPDATE}" <<'PY'
import sys
import torch

checkpoint_path = sys.argv[1]
expected_update = int(sys.argv[2])

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=False,
)

assert checkpoint["global_update"] == expected_update
assert checkpoint["precision"] == "fp32"
assert checkpoint["seed"] == 1337

print(
    f"FORMAL SEGMENT PASSED: update={expected_update}"
)
PY