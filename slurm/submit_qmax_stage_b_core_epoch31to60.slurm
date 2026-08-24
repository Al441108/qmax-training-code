#!/usr/bin/env bash
#SBATCH --job-name=qmax_b_core_31to60
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/qmax_stage_b_core_31to60_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/qmax_stage_b_core_31to60_%j.err
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
PREFLIGHT=${OUTPUT_ROOT}/preflight_qmax_core/qmax_stage_b_preflight.json
RUN_OUTPUT=${OUTPUT_ROOT}/formal/qmax_compactswin_qmax_core_seed42_epoch60
RESUME=${RUN_OUTPUT}/model_last.pt

cd "${PROJECT_ROOT}"
mkdir -p "${SCRATCH_ROOT}/logs"
test -f "${INIT_TEMPLATE}"
test -f "${PREFLIGHT}"
test -f "${RESUME}"
python -c 'import sys,torch; x=torch.load(sys.argv[1],map_location="cpu",weights_only=False); epoch=int(x["epoch"]); assert 30 <= epoch < 60, "expected epoch 30..59, got %s" % epoch' "${RESUME}"

python scripts/train_qmax_stage_b.py \
  --metadata_csv "${METADATA}" \
  --output_dir "${RUN_OUTPUT}" \
  --init_template "${INIT_TEMPLATE}" \
  --full_clean_manifest "${FULL_CLEAN}" \
  --robustness_manifest "${ROBUSTNESS}" \
  --condition_manifest "${CONDITIONS}" \
  --preflight_json "${PREFLIGHT}" \
  --qmax_variant qmax_core \
  --backbone_variant compactswin \
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

test -f "${RUN_OUTPUT}/epoch60/model_last.pt"
echo "epoch60_checkpoint=${RUN_OUTPUT}/epoch60/model_last.pt"
