#!/usr/bin/env bash
#SBATCH --job-name=qmax_a_missing_safe
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/qmax_a_missing_safe_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/qmax_a_missing_safe_%j.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/projects/u6dm/fastmri_project/fastmri_pipeline
SCRATCH_ROOT=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline
METADATA=/lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv
FULL_CLEAN=${SCRATCH_ROOT}/outputs/m2_prnf/preflight_R8_bs4/full_clean_manifest.json
ROBUSTNESS=${SCRATCH_ROOT}/outputs/m2_prnf/preflight_R8_bs4/robustness_manifest.json
CONDITIONS=${SCRATCH_ROOT}/outputs/m2_prnf_fusion/preflight_R8_bs4/condition_manifest.json
CHECKPOINT=${SCRATCH_ROOT}/outputs/qmax_stage_a/formal/qmax_full_seed42_epoch30/epoch60/model_last.pt
OUTPUT_JSON=${SCRATCH_ROOT}/outputs/qmax_stage_a/final_audits/missing_tensor/stagea_full_epoch60_missing_tensor_audit.json
AUDITOR=${PROJECT_ROOT}/scripts/audit_stagea_full_missing_tensor.py

cd "${PROJECT_ROOT}"
mkdir -p "${SCRATCH_ROOT}/logs" "$(dirname "${OUTPUT_JSON}")"

test -f "${CHECKPOINT}"
test -f "${METADATA}"
test -f "${FULL_CLEAN}"
test -f "${ROBUSTNESS}"
test -f "${CONDITIONS}"
test -f "${AUDITOR}"
test ! -e "${OUTPUT_JSON}"

export PYTHONHASHSEED=42
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

python -u "${AUDITOR}" \
  --checkpoint "${CHECKPOINT}" \
  --metadata_csv "${METADATA}" \
  --full_clean_manifest "${FULL_CLEAN}" \
  --robustness_manifest "${ROBUSTNESS}" \
  --condition_manifest "${CONDITIONS}" \
  --output_json "${OUTPUT_JSON}" \
  --batch_size 2 \
  --num_workers "${SLURM_CPUS_PER_TASK:-4}" \
  --seed 42 \
  --amp

test -f "${OUTPUT_JSON}"
grep -E '"status"|"output_independent_of_missing_tensor_content"|"dc_evidence_cannot_bypass_availability"|"max_abs_output_delta_vs_zero_tensor"' \
  "${OUTPUT_JSON}"

echo "missing_tensor_audit=${OUTPUT_JSON}"
