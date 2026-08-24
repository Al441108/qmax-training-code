#!/usr/bin/env bash
#SBATCH --job-name=qmax_a_e60_components
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/qmax_a_e60_components_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/qmax_a_e60_components_%j.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/projects/u6dm/fastmri_project/fastmri_pipeline
SCRATCH_ROOT=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline
METADATA=/lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv
FULL_CLEAN=${SCRATCH_ROOT}/outputs/m2_prnf/preflight_R8_bs4/full_clean_manifest.json
ROBUSTNESS=${SCRATCH_ROOT}/outputs/m2_prnf/preflight_R8_bs4/robustness_manifest.json
CONDITIONS=${SCRATCH_ROOT}/outputs/m2_prnf_fusion/preflight_R8_bs4/condition_manifest.json
CHECKPOINT=${SCRATCH_ROOT}/outputs/qmax_stage_a/formal/qmax_full_seed42_epoch30/epoch60/model_last.pt
OUTPUT_DIR=${SCRATCH_ROOT}/outputs/qmax_stage_a/component_counterfactuals/qmax_full_seed42_epoch60
EVALUATOR=${PROJECT_ROOT}/scripts/evaluate_stagea_full_epoch60_components.py

cd "${PROJECT_ROOT}"
mkdir -p "${SCRATCH_ROOT}/logs" "$(dirname "${OUTPUT_DIR}")"

test -f "${CHECKPOINT}"
test -f "${METADATA}"
test -f "${FULL_CLEAN}"
test -f "${ROBUSTNESS}"
test -f "${CONDITIONS}"
test -f "${EVALUATOR}"
test -f "${PROJECT_ROOT}/scripts/evaluate_stagea_full_epoch60_validation.py"
test -f "${PROJECT_ROOT}/scripts/evaluate_qmax_counterfactuals.py"
test ! -e "${OUTPUT_DIR}/stagea_full_epoch60_component_audit.json"

export PYTHONHASHSEED=42
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "checkpoint=${CHECKPOINT}"
sha256sum "${CHECKPOINT}" "${EVALUATOR}"

python -u "${EVALUATOR}" \
  --checkpoint "${CHECKPOINT}" \
  --metadata_csv "${METADATA}" \
  --full_clean_manifest "${FULL_CLEAN}" \
  --robustness_manifest "${ROBUSTNESS}" \
  --condition_manifest "${CONDITIONS}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size 2 \
  --num_workers "${SLURM_CPUS_PER_TASK:-4}" \
  --bootstrap_resamples 10000 \
  --seed 42 \
  --amp

test -f "${OUTPUT_DIR}/stagea_full_epoch60_component_audit.json"
grep -E '"status"|"missing_direct_and_correction_exact_zero_all_modes"|"max_abs_metric_delta"' \
  "${OUTPUT_DIR}/stagea_full_epoch60_component_audit.json"

echo "component_output=${OUTPUT_DIR}"
