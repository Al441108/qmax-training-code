#!/bin/bash
#SBATCH --job-name=prnf_gdir_15to50
#SBATCH --output=logs/m2_prnf/global_direct_15to50_%j.out
#SBATCH --error=logs/m2_prnf/global_direct_15to50_%j.err
#SBATCH --time=22:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G

set -euo pipefail

PROJECT_ROOT="/projects/u6dm/fastmri_project/fastmri_pipeline"
METADATA="/lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv"
PILOT_DIR="outputs/m2_prnf_fusion/pilot15_R8_bs4/global_direct_seed42"
OUTPUT_DIR="outputs/m2_prnf/formal_R8_bs4/global_direct_prnf_no_need_seed42_ep50"
MANIFEST_ROOT="outputs/m2_prnf/preflight_R8_bs4"
CONDITION_MANIFEST="outputs/m2_prnf_fusion/preflight_R8_bs4/condition_manifest.json"
PILOT_EVAL="outputs/evaluation/m2_prnf_fusion_pilot15_R8_bs4_seed42_evalfix1"

cd "$PROJECT_ROOT"
source "$HOME/miniforge3/bin/activate"
conda activate fastmri311
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs/m2_prnf

python -m py_compile \
  scripts/prepare_m2_prnf_global_direct_extend50.py \
  scripts/train_m2_prnf_global_direct_extend50.py

python scripts/prepare_m2_prnf_global_direct_extend50.py \
  --pilot_dir "$PILOT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --pilot_decision "$PILOT_EVAL/fusion_pilot_decision.json" \
  --pilot_audit "$PILOT_EVAL/fusion_pilot_evaluation_audit.json"

python scripts/train_m2_prnf_global_direct_extend50.py \
  --metadata_csv "$METADATA" \
  --output_dir "$OUTPUT_DIR" \
  --full_clean_manifest "$MANIFEST_ROOT/full_clean_manifest.json" \
  --robustness_manifest "$MANIFEST_ROOT/robustness_manifest.json" \
  --condition_manifest "$CONDITION_MANIFEST" \
  --variant prnf_no_need \
  --fusion_design global_direct \
  --need_scope residual \
  --residual_scale 0.1 \
  --run_stage pilot_extension_15_to_50 \
  --acceleration 8 \
  --pd_aux_acceleration 2 \
  --epochs 50 \
  --learning_rate 3e-4 \
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
