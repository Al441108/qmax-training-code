#!/usr/bin/env bash
#SBATCH --job-name=qmax_cf_eval
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/qmax_cf_eval_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/qmax_cf_eval_%j.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/projects/u6dm/fastmri_project/fastmri_pipeline
SCRATCH_ROOT=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline
METADATA=/lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv
FULL_CLEAN=${SCRATCH_ROOT}/outputs/m2_prnf/preflight_R8_bs4/full_clean_manifest.json
ROBUSTNESS=${SCRATCH_ROOT}/outputs/m2_prnf/preflight_R8_bs4/robustness_manifest.json
CONDITIONS=${SCRATCH_ROOT}/outputs/m2_prnf_fusion/preflight_R8_bs4/condition_manifest.json

: "${QMAX_CHECKPOINT:?Set QMAX_CHECKPOINT to epoch30/model_last.pt}"
: "${QMAX_EVAL_OUTPUT:?Set QMAX_EVAL_OUTPUT to a new output directory}"

cd "${PROJECT_ROOT}"
mkdir -p "${SCRATCH_ROOT}/logs/qmax_stage_a"
test -f "${QMAX_CHECKPOINT}"

python scripts/evaluate_qmax_counterfactuals.py \
  --checkpoint "${QMAX_CHECKPOINT}" \
  --metadata_csv "${METADATA}" \
  --full_clean_manifest "${FULL_CLEAN}" \
  --robustness_manifest "${ROBUSTNESS}" \
  --condition_manifest "${CONDITIONS}" \
  --output_dir "${QMAX_EVAL_OUTPUT}" \
  --batch_size 2 \
  --num_workers "${SLURM_CPUS_PER_TASK:-4}" \
  --bootstrap_resamples 10000 \
  --seed 42 \
  --amp

echo "audit=${QMAX_EVAL_OUTPUT}/qmax_counterfactual_audit.json"
