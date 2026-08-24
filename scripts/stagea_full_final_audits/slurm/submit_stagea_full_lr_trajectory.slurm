#!/usr/bin/env bash
#SBATCH --job-name=qmax_a_lr_traj
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/qmax_a_lr_traj_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/qmax_a_lr_traj_%j.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/projects/u6dm/fastmri_project/fastmri_pipeline
SCRATCH_ROOT=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline
RUN_DIR=${SCRATCH_ROOT}/outputs/qmax_stage_a/formal/qmax_full_seed42_epoch30
METADATA=/lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv
FULL_CLEAN=${SCRATCH_ROOT}/outputs/m2_prnf/preflight_R8_bs4/full_clean_manifest.json
ROBUSTNESS=${SCRATCH_ROOT}/outputs/m2_prnf/preflight_R8_bs4/robustness_manifest.json
CONDITIONS=${SCRATCH_ROOT}/outputs/m2_prnf_fusion/preflight_R8_bs4/condition_manifest.json
OUTPUT_DIR=${SCRATCH_ROOT}/outputs/qmax_stage_a/final_audits/lr_trajectory_30_40_50_60
EVALUATOR=${PROJECT_ROOT}/scripts/evaluate_stagea_full_lr_trajectory.py

cd "${PROJECT_ROOT}"
mkdir -p "${SCRATCH_ROOT}/logs" "$(dirname "${OUTPUT_DIR}")"

for epoch in 30 40 50 60; do
  test -f "${RUN_DIR}/epoch${epoch}/model_last.pt"
done
test -f "${METADATA}"
test -f "${FULL_CLEAN}"
test -f "${ROBUSTNESS}"
test -f "${CONDITIONS}"
test -f "${EVALUATOR}"
test ! -e "${OUTPUT_DIR}/stagea_full_lr_trajectory_audit.json"

export PYTHONHASHSEED=42
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

python -u "${EVALUATOR}" \
  --checkpoint30 "${RUN_DIR}/epoch30/model_last.pt" \
  --checkpoint40 "${RUN_DIR}/epoch40/model_last.pt" \
  --checkpoint50 "${RUN_DIR}/epoch50/model_last.pt" \
  --checkpoint60 "${RUN_DIR}/epoch60/model_last.pt" \
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

test -f "${OUTPUT_DIR}/stagea_full_lr_trajectory_audit.json"
grep -E '"status"|"formal_selection_checkpoint"|"shared_init_template_sha256"' \
  "${OUTPUT_DIR}/stagea_full_lr_trajectory_audit.json"

echo "trajectory_output=${OUTPUT_DIR}"
