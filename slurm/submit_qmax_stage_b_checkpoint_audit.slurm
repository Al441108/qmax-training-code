#!/usr/bin/env bash
#SBATCH --job-name=qmax_b_ckpt_audit
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/qmax_stage_b_checkpoint_audit_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/qmax_stage_b_checkpoint_audit_%j.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/projects/u6dm/fastmri_project/fastmri_pipeline
SCRATCH_ROOT=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline
METADATA=/lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv

: "${QMAX_STAGE_B_CHECKPOINT:?Set a Stage-B model_last.pt checkpoint}"
: "${QMAX_STAGE_B_AUDIT_OUTPUT:?Set a separate audit JSON path}"

RESOURCE_ARGS=()
if [[ -n "${QMAX_STAGE_B_MAX_EPOCH_SECONDS:-}" ]]; then
  RESOURCE_ARGS+=(
    --max_epoch_seconds "${QMAX_STAGE_B_MAX_EPOCH_SECONDS}"
  )
fi
if [[ -n "${QMAX_STAGE_B_MAX_GPU_MEMORY_GB:-}" ]]; then
  RESOURCE_ARGS+=(
    --max_gpu_memory_gb "${QMAX_STAGE_B_MAX_GPU_MEMORY_GB}"
  )
fi

cd "${PROJECT_ROOT}"
mkdir -p "${SCRATCH_ROOT}/logs" "$(dirname "${QMAX_STAGE_B_AUDIT_OUTPUT}")"
test -f "${QMAX_STAGE_B_CHECKPOINT}"
test ! -e "${QMAX_STAGE_B_AUDIT_OUTPUT}"

python scripts/audit_qmax_stage_b_checkpoint.py \
  --checkpoint "${QMAX_STAGE_B_CHECKPOINT}" \
  --output_json "${QMAX_STAGE_B_AUDIT_OUTPUT}" \
  --metadata_csv "${METADATA}" \
  --batch_size 2 \
  --num_workers "${SLURM_CPUS_PER_TASK:-4}" \
  --amp \
  "${RESOURCE_ARGS[@]}"

echo "audit=${QMAX_STAGE_B_AUDIT_OUTPUT}"

