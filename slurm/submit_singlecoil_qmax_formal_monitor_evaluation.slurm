#!/usr/bin/env bash
#SBATCH --job-name=sc_qmax_moneval
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/singlecoil_qmax_benchmark/logs/sc_qmax_moneval_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/singlecoil_qmax_benchmark/logs/sc_qmax_moneval_%j.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/projects/u6dm/fastmri_project/fastmri_pipeline
BENCH_ROOT=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/singlecoil_qmax_benchmark
FSMNET="${BENCH_ROOT}/repo/FSMNet"

CHECKPOINT="${BENCH_ROOT}/outputs/qmax_full_seed1337/formal_100k/model_last.pt"
TRAINING_SUMMARY="${BENCH_ROOT}/outputs/qmax_full_seed1337/formal_100k/training_summary.json"
MONITOR_MANIFEST="${BENCH_ROOT}/manifests/train_monitor.csv"

OUTPUT_DIR=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/outputs/singlecoil_qmax/qmax_full_seed1337/formal_monitor_evaluation_update100000

source "${HOME}/miniforge3/bin/activate"
conda activate fastmri311

cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/external_benchmarks:${PYTHONPATH:-}"

test -f "${CHECKPOINT}"
test -f "${TRAINING_SUMMARY}"
test -f "${MONITOR_MANIFEST}"
test -f "${FSMNET}/metric.py"
test -f external_benchmarks/evaluate_singlecoil_qmax_shared.py

python -m py_compile \
  external_benchmarks/evaluate_singlecoil_qmax_shared.py

python -c "
import torch
assert torch.cuda.is_available(), 'GPU unavailable'
print('GPU:', torch.cuda.get_device_name(0))
"

python - \
  "${CHECKPOINT}" \
  "${TRAINING_SUMMARY}" <<'PY'
import json
import sys
from pathlib import Path

import torch


checkpoint_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])

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
    "QMaxSinglecoilFull"
)
assert checkpoint["qmax_variant"] == "qmax_full"
assert checkpoint["global_update"] == 100000
assert checkpoint["seed"] == 1337
assert checkpoint["precision"] == "fp32"

assert summary["status"] == "complete"
assert summary["global_update"] == 100000
assert summary["seed"] == 1337
assert summary["precision"] == "fp32"

print("FINAL QMAX TRAINING RESULT VERIFIED")
PY

mkdir -p "$(dirname "${OUTPUT_DIR}")"

python external_benchmarks/evaluate_singlecoil_qmax_shared.py \
  --model-type qmax_full \
  --checkpoint "${CHECKPOINT}" \
  --manifest "${MONITOR_MANIFEST}" \
  --fsmnet-root "${FSMNET}" \
  --output-dir "${OUTPUT_DIR}" \
  --dataset-mode train \
  --seed 1337 \
  --require-update 100000

python - "${OUTPUT_DIR}/summary.json" <<'PY'
import json
import math
import sys
from pathlib import Path


summary = json.loads(
    Path(sys.argv[1]).read_text()
)

assert summary["status"] == "complete"
assert summary["model_type"] == "qmax_full"
assert summary["checkpoint_model_name"] == (
    "QMaxSinglecoilFull"
)
assert summary["checkpoint_update"] == 100000
assert summary["checkpoint_seed"] == 1337
assert summary["precision"] == "fp32"

assert summary["held_out_accessed"] is False
assert summary["manifest_name"] == (
    "train_monitor.csv"
)
assert summary[
    "complete_manifest_evaluation"
] is True

assert summary["available_slices"] == 571
assert summary["evaluated_slices"] == 571
assert summary["evaluated_volumes"] == 16

for key in (
    "volume_nmse_mean",
    "volume_psnr_mean",
    "volume_ssim_mean",
    "slice_nmse_mean",
    "slice_psnr_mean",
    "slice_ssim_mean",
    "mean_inference_seconds_per_slice",
    "peak_gpu_memory_gib",
):
    value = summary[key]
    assert isinstance(value, (int, float))
    assert math.isfinite(value), (
        key,
        value,
    )

print(json.dumps(summary, indent=2))
PY

test -s "${OUTPUT_DIR}/slice_metrics.csv"
test -s "${OUTPUT_DIR}/volume_metrics.csv"
test -s "${OUTPUT_DIR}/summary.json"
test -s "${OUTPUT_DIR}/evaluation_manifest.json"

echo "output_dir=${OUTPUT_DIR}"
echo "QMAX FULL FORMAL TRAIN-MONITOR EVALUATION PASSED"