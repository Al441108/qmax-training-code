#!/usr/bin/env bash
#SBATCH --job-name=qmax_a_full_31to60
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=23:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/qmax_stage_a_full_31to60_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/qmax_stage_a_full_31to60_%j.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/projects/u6dm/fastmri_project/fastmri_pipeline
SCRATCH_ROOT=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline
METADATA=/lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv
FULL_CLEAN=${SCRATCH_ROOT}/outputs/m2_prnf/preflight_R8_bs4/full_clean_manifest.json
ROBUSTNESS=${SCRATCH_ROOT}/outputs/m2_prnf/preflight_R8_bs4/robustness_manifest.json
CONDITIONS=${SCRATCH_ROOT}/outputs/m2_prnf_fusion/preflight_R8_bs4/condition_manifest.json
HISTORICAL_P0=${SCRATCH_ROOT}/outputs/m2_prnf/formal_R8_bs4/quality_protected_hybrid_gain_seed42_ep50/model_last.pt
INIT_TEMPLATE=${SCRATCH_ROOT}/outputs/qmax_stage_a/random_init/random_init_seed42.pt
PREFLIGHT=${SCRATCH_ROOT}/outputs/qmax_stage_a/preflight_seed42/qmax_preflight.json
AMP_SAFE_WRAPPER=${PROJECT_ROOT}/scripts/run_qmax_amp_safe.py
CONTINUATION_TRAINER=${PROJECT_ROOT}/scripts/train_qmax_stage_a_continue_31to60.py

QMAX_STAGE_A_VARIANT=qmax_full
OUTPUT_DIR=${SCRATCH_ROOT}/outputs/qmax_stage_a/formal/qmax_full_seed42_epoch30

RESUME=${OUTPUT_DIR}/model_last.pt

cd "${PROJECT_ROOT}"
mkdir -p "${SCRATCH_ROOT}/logs"
test -f "${INIT_TEMPLATE}"
test -f "${PREFLIGHT}"
test -f "${HISTORICAL_P0}"
test -f "${OUTPUT_DIR}/config.json"
test -f "${RESUME}"
test -f "${AMP_SAFE_WRAPPER}"
test -f "${CONTINUATION_TRAINER}"

python -c 'import sys,torch; x=torch.load(sys.argv[1],map_location="cpu",weights_only=False); epoch=int(x["epoch"]); variant=x["config"]["qmax_variant"]; assert 30 <= epoch < 60, "expected epoch 30..59, got %s" % epoch; assert variant == sys.argv[2], "checkpoint variant %s != requested %s" % (variant,sys.argv[2])' \
  "${RESUME}" "${QMAX_STAGE_A_VARIANT}"

echo "AMP-SAFE wrapper and continuation trainer hashes:"
sha256sum "${AMP_SAFE_WRAPPER}" "${CONTINUATION_TRAINER}"

python "${AMP_SAFE_WRAPPER}" "${CONTINUATION_TRAINER}" \
  --metadata_csv "${METADATA}" \
  --output_dir "${OUTPUT_DIR}" \
  --init_template "${INIT_TEMPLATE}" \
  --full_clean_manifest "${FULL_CLEAN}" \
  --robustness_manifest "${ROBUSTNESS}" \
  --condition_manifest "${CONDITIONS}" \
  --preflight_json "${PREFLIGHT}" \
  --historical_p0_checkpoint "${HISTORICAL_P0}" \
  --qmax_variant "${QMAX_STAGE_A_VARIANT}" \
  --run_mode formal \
  --epochs 60 \
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
  --amp \
  --resume "${RESUME}"

test -f "${OUTPUT_DIR}/epoch60/model_last.pt"
echo "epoch60_checkpoint=${OUTPUT_DIR}/epoch60/model_last.pt"
