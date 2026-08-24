#!/usr/bin/env bash
#SBATCH --job-name=qmax_b_preflight
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/qmax_stage_b_independent_preflight_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/qmax_stage_b_independent_preflight_%j.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/projects/u6dm/fastmri_project/fastmri_pipeline
SCRATCH_ROOT=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline
METADATA=/lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv
OUTPUT_ROOT=${SCRATCH_ROOT}/outputs/qmax_stage_b_independent
INIT_DIR=${OUTPUT_ROOT}/random_init
INIT_TEMPLATE=${INIT_DIR}/compactswin_core_full_random_init_seed42.pt

cd "${PROJECT_ROOT}"
mkdir -p "${SCRATCH_ROOT}/logs" "${INIT_DIR}" \
  "${OUTPUT_ROOT}/preflight_qmax_core" \
  "${OUTPUT_ROOT}/preflight_qmax_full"

if [[ ! -f "${INIT_TEMPLATE}" ]]; then
  python scripts/generate_qmax_stage_b_init.py \
    --output "${INIT_TEMPLATE}" \
    --audit_json "${INIT_DIR}/compactswin_core_full_random_init_seed42_audit.json" \
    --seed 42 \
    --num_cascades 12 \
    --chans 18 \
    --sens_chans 8 \
    --pools 4 \
    --sens_pools 4 \
    --controller_chans 16 \
    --initial_aux_alpha 0.1 \
    --initial_gate_probability 0.95 \
    --h8_blocks 2 \
    --h16_blocks 2 \
    --window_size 8 \
    --activation_checkpointing
fi

for VARIANT in qmax_core qmax_full; do
  python scripts/preflight_qmax_stage_b.py \
    --metadata_csv "${METADATA}" \
    --stage_b_init_template "${INIT_TEMPLATE}" \
    --qmax_variant "${VARIANT}" \
    --output_json "${OUTPUT_ROOT}/preflight_${VARIANT}/qmax_stage_b_preflight.json" \
    --batch_size 4 \
    --num_workers "${SLURM_CPUS_PER_TASK:-4}" \
    --seed 42 \
    --max_gpu_memory_gb 110 \
    --amp
done

echo "core_preflight=${OUTPUT_ROOT}/preflight_qmax_core/qmax_stage_b_preflight.json"
echo "full_preflight=${OUTPUT_ROOT}/preflight_qmax_full/qmax_stage_b_preflight.json"
