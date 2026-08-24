#!/usr/bin/env bash
#SBATCH --job-name=qmax_fmri_roi_report
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/qmax_fmri_roi_report_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/qmax_fmri_roi_report_%j.err
#SBATCH --export=ALL

set -euo pipefail
PROJECT_ROOT=/projects/u6dm/fastmri_project/fastmri_pipeline
SCRATCH_ROOT=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline
SCRIPT=${PROJECT_ROOT}/scripts/qmax_fastmriplus_roi_report.py
INFERENCE=${SCRATCH_ROOT}/outputs/qmax_stage_a/fastmriplus_roi/qmax_full_epoch60_inference_cache/inference_manifest.json
OUTPUT=${SCRATCH_ROOT}/outputs/qmax_stage_a/fastmriplus_roi/qmax_full_epoch60_report

cd "${PROJECT_ROOT}"
mkdir -p "${SCRATCH_ROOT}/logs" "${OUTPUT}"
for path in "${SCRIPT}" "${INFERENCE}"; do test -f "${path}"; done
test ! -e "${OUTPUT}/roi_qeff_audit.json"

python -u "${SCRIPT}" \
  --inference_manifest "${INFERENCE}" \
  --output_dir "${OUTPUT}" \
  --bootstrap_iterations 10000 \
  --seed 42

python - "${OUTPUT}/roi_qeff_audit.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
assert report["status"] == "passed", report
assert "held-out test not accessed" in report["scope"], report
assert report["figure_contract"]["backend"] == "Python/matplotlib only", report
PY

echo "report_output=${OUTPUT}"
