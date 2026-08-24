#!/bin/bash
#SBATCH --job-name=lr30k_hyb_50to60
#SBATCH --output=logs/m2_prnf_low_lr/hybrid_gain_%j.out
#SBATCH --error=logs/m2_prnf_low_lr/hybrid_gain_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G

set -euo pipefail

PROJECT_ROOT="/projects/u6dm/fastmri_project/fastmri_pipeline"
METADATA="/lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv"
OUTPUT_DIR="outputs/m2_prnf/posthoc_low_lr_50to60/hybrid_gain_seed42"
MANIFEST_ROOT="outputs/m2_prnf/preflight_R8_bs4"
CONDITION_MANIFEST="outputs/m2_prnf_fusion/preflight_R8_bs4/condition_manifest.json"

cd "$PROJECT_ROOT"
source "$HOME/miniforge3/bin/activate"
conda activate fastmri311
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONDONTWRITEBYTECODE=1
mkdir -p logs/m2_prnf_low_lr

test -f "$OUTPUT_DIR/branch_provenance.json"
test -f "$OUTPUT_DIR/model_last.pt"

python scripts/train_hybrid_gain_low_lr_50to60.py \
  --metadata_csv "$METADATA" \
  --output_dir "$OUTPUT_DIR" \
  --full_clean_manifest "$MANIFEST_ROOT/full_clean_manifest.json" \
  --robustness_manifest "$MANIFEST_ROOT/robustness_manifest.json" \
  --condition_manifest "$CONDITION_MANIFEST" \
  --variant prnf_no_need \
  --fusion_design hybrid_direct_residual \
  --run_stage posthoc_low_lr_50_to_60 \
  --need_scope residual \
  --residual_scale 0.1 \
  --lambda_residual_gain 0.2 \
  --residual_gain_margin_relative 0.002 \
  --residual_gain_ramp_epochs 5 \
  --acceleration 8 \
  --pd_aux_acceleration 2 \
  --epochs 60 \
  --learning_rate 3e-5 \
  --batch_size 4 \
  --grad_accum_steps 1 \
  --num_workers 4 \
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
  --seed 42 \
  --resume "$OUTPUT_DIR/model_last.pt"

cat "$OUTPUT_DIR/final_summary.json"
