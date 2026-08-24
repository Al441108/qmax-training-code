#!/usr/bin/env bash
#SBATCH --job-name=qmax_b_full_pilot
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/qmax_stage_b_full_pilot_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/qmax_stage_b_full_pilot_%j.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/projects/u6dm/fastmri_project/fastmri_pipeline
SCRATCH_ROOT=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline
METADATA=/lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv
FULL_CLEAN=${SCRATCH_ROOT}/outputs/m2_prnf/preflight_R8_bs4/full_clean_manifest.json
ROBUSTNESS=${SCRATCH_ROOT}/outputs/m2_prnf/preflight_R8_bs4/robustness_manifest.json
CONDITIONS=${SCRATCH_ROOT}/outputs/m2_prnf_fusion/preflight_R8_bs4/condition_manifest.json
OUTPUT_ROOT=${SCRATCH_ROOT}/outputs/qmax_stage_b_independent
INIT_TEMPLATE=${OUTPUT_ROOT}/random_init/compactswin_core_full_random_init_seed42.pt
PREFLIGHT=${OUTPUT_ROOT}/preflight_qmax_full/qmax_stage_b_preflight.json
RUN_OUTPUT=${OUTPUT_ROOT}/pilot/qmax_compactswin_qmax_full_seed42_epoch1

cd "${PROJECT_ROOT}"
mkdir -p "${SCRATCH_ROOT}/logs" "$(dirname "${RUN_OUTPUT}")"
test -f "${INIT_TEMPLATE}"
test -f "${PREFLIGHT}"
test ! -e "${RUN_OUTPUT}/config.json"

python scripts/train_qmax_stage_b.py \
  --metadata_csv "${METADATA}" \
  --output_dir "${RUN_OUTPUT}" \
  --init_template "${INIT_TEMPLATE}" \
  --full_clean_manifest "${FULL_CLEAN}" \
  --robustness_manifest "${ROBUSTNESS}" \
  --condition_manifest "${CONDITIONS}" \
  --preflight_json "${PREFLIGHT}" \
  --qmax_variant qmax_full \
  --backbone_variant compactswin \
  --run_mode pilot \
  --epochs 1 \
  --learning_rate 3e-4 \
  --batch_size 4 \
  --grad_accum_steps 1 \
  --num_workers "${SLURM_CPUS_PER_TASK:-4}" \
  --lambda_rel 0.05 \
  --lambda_rank 0.02 \
  --lambda_residual_gain 0.2 \
  --residual_gain_margin_relative 0.002 \
  --aux_loss_ramp_epochs 5 \
  --residual_gain_ramp_epochs 5 \
  --gradient_clip_norm 10 \
  --seed 42 \
  --amp
