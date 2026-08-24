#!/usr/bin/env bash
#SBATCH --job-name=qmax_3seed_summary
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/qmax_3seed_summary_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/qmax_3seed_summary_%j.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/projects/u6dm/fastmri_project/fastmri_pipeline
SCRATCH_ROOT=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline
SEED42_VALIDATION=${SCRATCH_ROOT}/outputs/qmax_stage_a/final_validation/qmax_full_seed42_epoch60
SEED42_COMPONENTS=${SCRATCH_ROOT}/outputs/qmax_stage_a/component_counterfactuals/qmax_full_seed42_epoch60
SEED123=${SCRATCH_ROOT}/outputs/qmax_stage_a_multiseed/final_validation/qmax_full_seed123_epoch60
SEED2026=${SCRATCH_ROOT}/outputs/qmax_stage_a_multiseed/final_validation/qmax_full_seed2026_epoch60
OUTPUT_DIR=${SCRATCH_ROOT}/outputs/qmax_stage_a_multiseed/three_seed_summary
SUMMARIZER=${PROJECT_ROOT}/scripts/summarize_qmax_full_three_seed.py

cd "${PROJECT_ROOT}"
mkdir -p "${SCRATCH_ROOT}/logs" "$(dirname "${OUTPUT_DIR}")"

for path in "${SEED42_VALIDATION}" "${SEED42_COMPONENTS}" "${SEED123}" "${SEED2026}"; do
  if [[ ! -d "${path}" ]]; then
    echo "missing_required_directory=${path}" >&2
    exit 1
  fi
done
test -f "${SUMMARIZER}"
test ! -e "${OUTPUT_DIR}/three_seed_summary_audit.json"

python -u "${SUMMARIZER}" \
  --seed42_validation_dir "${SEED42_VALIDATION}" \
  --seed42_component_dir "${SEED42_COMPONENTS}" \
  --seed123_dir "${SEED123}" \
  --seed2026_dir "${SEED2026}" \
  --output_dir "${OUTPUT_DIR}" \
  --bootstrap_resamples 10000

python -c "import json; p=json.load(open('${OUTPUT_DIR}/three_seed_summary_audit.json')); assert p['status']=='passed', p"
echo "summary_output=${OUTPUT_DIR}"
