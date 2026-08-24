#!/usr/bin/env bash
#SBATCH --job-name=affine_hyb_s42
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/affine_hyb_s42_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/affine_hyb_s42_%j.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/projects/u6dm/fastmri_project/fastmri_pipeline
METADATA=/lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv
FULL_CLEAN=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/outputs/m2_prnf/preflight_R8_bs4/full_clean_manifest.json
ROBUSTNESS=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/outputs/m2_prnf/preflight_R8_bs4/robustness_manifest.json
CONDITIONS=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/outputs/m2_prnf_fusion/preflight_R8_bs4/condition_manifest.json
RESUME=${RESUME_CHECKPOINT:-/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/outputs/m2_prnf/formal_R8_bs4/quality_protected_hybrid_gain_seed42_ep50/model_last.pt}
OUTPUT_DIR=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/outputs/adapter_affine_ablation/hybrid_gain_affine_true_seed42_50to60

cd "${PROJECT_ROOT}"
mkdir -p logs
test -f "${RESUME}"
test -f scripts/affine_ablation_runtime.py
test -f scripts/train_hybrid_gain_affine_true_50to60.py
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

python scripts/train_hybrid_gain_affine_true_50to60.py \
  --metadata_csv "${METADATA}" \
  --output_dir "${OUTPUT_DIR}" \
  --full_clean_manifest "${FULL_CLEAN}" \
  --robustness_manifest "${ROBUSTNESS}" \
  --condition_manifest "${CONDITIONS}" \
  --variant prnf_no_need \
  --fusion_design hybrid_direct_residual \
  --need_scope residual \
  --residual_scale 0.1 \
  --run_stage posthoc_low_lr_50_to_60 \
  --acceleration 8 \
  --pd_aux_acceleration 2 \
  --epochs 60 \
  --learning_rate 3e-5 \
  --batch_size 4 \
  --grad_accum_steps 1 \
  --num_workers "${SLURM_CPUS_PER_TASK:-4}" \
  --num_cascades 12 \
  --chans 18 \
  --sens_chans 8 \
  --pools 4 \
  --sens_pools 4 \
  --controller_chans 16 \
  --initial_aux_alpha 0.1 \
  --initial_gate_probability 0.95 \
  --initial_need_probability 0.95 \
  --need_floor 0.25 \
  --lambda_rel 0.05 \
  --lambda_rank 0.02 \
  --aux_loss_ramp_epochs 5 \
  --lambda_residual_gain 0.2 \
  --residual_gain_margin_relative 0.002 \
  --residual_gain_ramp_epochs 5 \
  --seed 42 \
  --resume "${RESUME}"

echo "summary=${OUTPUT_DIR}/final_summary.json"
