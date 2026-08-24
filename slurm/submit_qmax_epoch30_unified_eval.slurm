#!/usr/bin/env bash
#SBATCH --job-name=qmax_e30_eval
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/qmax_epoch30_eval_%x_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/qmax_epoch30_eval_%x_%j.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/projects/u6dm/fastmri_project/fastmri_pipeline
SCRATCH_ROOT=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline
METADATA=/lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv
FULL_CLEAN=${SCRATCH_ROOT}/outputs/m2_prnf/preflight_R8_bs4/full_clean_manifest.json
ROBUSTNESS=${SCRATCH_ROOT}/outputs/m2_prnf/preflight_R8_bs4/robustness_manifest.json
CONDITIONS=${SCRATCH_ROOT}/outputs/m2_prnf_fusion/preflight_R8_bs4/condition_manifest.json
OUTPUT_ROOT=${SCRATCH_ROOT}/outputs/qmax_epoch30_four_arm/evaluations

: "${QMAX_EPOCH30_ARM:?Set stagea_core, stagea_full, stageb_core, or stageb_full}"

case "${QMAX_EPOCH30_ARM}" in
  stagea_core)
    CHECKPOINT=${SCRATCH_ROOT}/outputs/qmax_stage_a/formal/qmax_core_seed42_epoch30/epoch30/model_last.pt
    ;;
  stagea_full)
    CHECKPOINT=${SCRATCH_ROOT}/outputs/qmax_stage_a/formal/qmax_full_seed42_epoch30/epoch30/model_last.pt
    ;;
  stageb_core)
    CHECKPOINT=${SCRATCH_ROOT}/outputs/qmax_stage_b_independent/formal/qmax_compactswin_qmax_core_seed42_epoch60/epoch30/model_last.pt
    ;;
  stageb_full)
    CHECKPOINT=${SCRATCH_ROOT}/outputs/qmax_stage_b_independent/formal/qmax_compactswin_qmax_full_seed42_epoch60/epoch30/model_last.pt
    ;;
  *)
    echo "Invalid QMAX_EPOCH30_ARM=${QMAX_EPOCH30_ARM}" >&2
    exit 2
    ;;
esac

OUTPUT_DIR=${OUTPUT_ROOT}/${QMAX_EPOCH30_ARM}

cd "${PROJECT_ROOT}"
mkdir -p "${SCRATCH_ROOT}/logs" "${OUTPUT_ROOT}"
test -f "${CHECKPOINT}"
test -f "${METADATA}"
test -f "${FULL_CLEAN}"
test -f "${ROBUSTNESS}"
test -f "${CONDITIONS}"
test ! -e "${OUTPUT_DIR}/epoch30_evaluation_audit.json"

python scripts/evaluate_qmax_epoch30_unified.py \
  --arm "${QMAX_EPOCH30_ARM}" \
  --checkpoint "${CHECKPOINT}" \
  --metadata_csv "${METADATA}" \
  --full_clean_manifest "${FULL_CLEAN}" \
  --robustness_manifest "${ROBUSTNESS}" \
  --condition_manifest "${CONDITIONS}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size 2 \
  --num_workers "${SLURM_CPUS_PER_TASK:-4}" \
  --seed 42 \
  --amp

echo "audit=${OUTPUT_DIR}/epoch30_evaluation_audit.json"
