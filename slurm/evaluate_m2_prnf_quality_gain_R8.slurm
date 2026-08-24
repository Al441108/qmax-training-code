#!/bin/bash
#SBATCH --job-name=prnf_qgain_eval
#SBATCH --output=logs/m2_prnf_quality_gain/eval_%j.out
#SBATCH --error=logs/m2_prnf_quality_gain/eval_%j.err
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G

set -euo pipefail
PROJECT_ROOT="/projects/u6dm/fastmri_project/fastmri_pipeline"
METADATA="/lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv"
OLD_ROOT="outputs/m2_prnf_fusion/pilot15_R8_bs4"
GAIN_ROOT="outputs/m2_prnf_quality_gain/pilot15_R8_bs4"
MANIFEST_ROOT="outputs/m2_prnf/preflight_R8_bs4"
CONDITION_MANIFEST="outputs/m2_prnf_fusion/preflight_R8_bs4/condition_manifest.json"
OUTPUT_DIR="outputs/evaluation/m2_prnf_quality_gain_five_arm_R8_bs4_seed42"

cd "$PROJECT_ROOT"
source "$HOME/miniforge3/bin/activate"
conda activate fastmri311
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs/m2_prnf_quality_gain "$OUTPUT_DIR"

python -m py_compile scripts/evaluate_m2_prnf_quality_gain_R8.py

python scripts/evaluate_m2_prnf_quality_gain_R8.py \
  --metadata_csv "$METADATA" \
  --full_clean_manifest "$MANIFEST_ROOT/full_clean_manifest.json" \
  --robustness_manifest "$MANIFEST_ROOT/robustness_manifest.json" \
  --condition_manifest "$CONDITION_MANIFEST" \
  --output_dir "$OUTPUT_DIR" \
  --legacy_local_direct_checkpoint "$OLD_ROOT/legacy_local_direct_seed42/model_best.pt" \
  --global_direct_checkpoint "$OLD_ROOT/global_direct_seed42/model_best.pt" \
  --residual_only_checkpoint "$OLD_ROOT/residual_only_seed42/model_best.pt" \
  --hybrid_direct_residual_checkpoint "$OLD_ROOT/hybrid_direct_residual_seed42/model_best.pt" \
  --quality_protected_hybrid_gain_checkpoint "$GAIN_ROOT/quality_protected_hybrid_gain_seed42/model_best.pt" \
  --batch_size 2 \
  --num_workers 4 \
  --bootstrap_resamples 10000 \
  --seed 42

cat "$OUTPUT_DIR/fusion_pilot_decision.json"
