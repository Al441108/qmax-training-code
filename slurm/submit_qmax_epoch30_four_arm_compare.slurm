#!/usr/bin/env bash
#SBATCH --job-name=qmax_e30_compare
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/qmax_epoch30_compare_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/qmax_epoch30_compare_%j.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/projects/u6dm/fastmri_project/fastmri_pipeline
SCRATCH_ROOT=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline
EVAL_ROOT=${SCRATCH_ROOT}/outputs/qmax_epoch30_four_arm/evaluations
OUTPUT_DIR=${SCRATCH_ROOT}/outputs/qmax_epoch30_four_arm/comparison

cd "${PROJECT_ROOT}"
mkdir -p "${SCRATCH_ROOT}/logs" "$(dirname "${OUTPUT_DIR}")"

for ARM in stagea_core stagea_full stageb_core stageb_full; do
  test -f "${EVAL_ROOT}/${ARM}/epoch30_evaluation_audit.json"
  test -f "${EVAL_ROOT}/${ARM}/epoch30_slice_metrics.csv"
  test -f "${EVAL_ROOT}/${ARM}/epoch30_patient_metrics.csv"
done
test ! -e "${OUTPUT_DIR}/epoch30_four_arm_paired_l1_comparisons.json"

python scripts/compare_qmax_epoch30_four_arm.py \
  --stagea_core_dir "${EVAL_ROOT}/stagea_core" \
  --stagea_full_dir "${EVAL_ROOT}/stagea_full" \
  --stageb_core_dir "${EVAL_ROOT}/stageb_core" \
  --stageb_full_dir "${EVAL_ROOT}/stageb_full" \
  --output_dir "${OUTPUT_DIR}" \
  --bootstrap_resamples 10000 \
  --seed 42

echo "comparison=${OUTPUT_DIR}/epoch30_four_arm_paired_l1_comparisons.json"
