#!/usr/bin/env bash
#SBATCH --job-name=qmax_e30_compare
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/qmax_epoch30_valid_compare_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/qmax_epoch30_valid_compare_%j.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/projects/u6dm/fastmri_project/fastmri_pipeline
SCRATCH_ROOT=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline
EVAL_ROOT=${SCRATCH_ROOT}/outputs/qmax_epoch30_four_arm/evaluations
CF_ROOT=${SCRATCH_ROOT}/outputs/qmax_epoch30_valid_three_arm/stagea_counterfactuals
OUTPUT_DIR=${SCRATCH_ROOT}/outputs/qmax_epoch30_valid_three_arm/comparison
STAGEA_ROOT=${SCRATCH_ROOT}/outputs/qmax_stage_a/formal
STAGEB_ROOT=${SCRATCH_ROOT}/outputs/qmax_stage_b_independent/formal

cd "${PROJECT_ROOT}"
mkdir -p "${SCRATCH_ROOT}/logs" "$(dirname "${OUTPUT_DIR}")"
test ! -e "${OUTPUT_DIR}/epoch30_valid_three_arm_report.json"

python scripts/compare_qmax_epoch30_valid_arms.py \
  --stagea_core_eval_dir "${EVAL_ROOT}/stagea_core" \
  --stagea_full_eval_dir "${EVAL_ROOT}/stagea_full" \
  --stageb_full_eval_dir "${EVAL_ROOT}/stageb_full" \
  --stagea_core_counterfactual_dir "${CF_ROOT}/stagea_core" \
  --stagea_full_counterfactual_dir "${CF_ROOT}/stagea_full" \
  --stagea_core_training_log "${STAGEA_ROOT}/qmax_core_seed42_epoch30/training_log.csv" \
  --stagea_full_training_log "${STAGEA_ROOT}/qmax_full_seed42_epoch30/training_log.csv" \
  --stageb_core_training_log "${STAGEB_ROOT}/qmax_compactswin_qmax_core_seed42_epoch60/training_log.csv" \
  --stageb_full_training_log "${STAGEB_ROOT}/qmax_compactswin_qmax_full_seed42_epoch60/training_log.csv" \
  --arm_status_json "${PROJECT_ROOT}/QMAX_EPOCH30_ARM_STATUS.json" \
  --output_dir "${OUTPUT_DIR}" \
  --bootstrap_resamples 10000 \
  --seed 42

echo "report=${OUTPUT_DIR}/epoch30_valid_three_arm_report.json"
