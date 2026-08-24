#!/bin/bash
#SBATCH --job-name=prnf_fusion_eval
#SBATCH --output=logs/m2_prnf_fusion/eval_%j.out
#SBATCH --error=logs/m2_prnf_fusion/eval_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G

set -euo pipefail
PROJECT_ROOT="/projects/u6dm/fastmri_project/fastmri_pipeline"
METADATA="/lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv"
ROOT="outputs/m2_prnf_fusion/pilot15_R8_bs4"
MANIFEST_ROOT="outputs/m2_prnf/preflight_R8_bs4"
CONDITION_MANIFEST="outputs/m2_prnf_fusion/preflight_R8_bs4/condition_manifest.json"
OUTPUT_DIR="outputs/evaluation/m2_prnf_fusion_pilot15_R8_bs4_seed42"

cd "$PROJECT_ROOT"
source "$HOME/miniforge3/bin/activate"
conda activate fastmri311
mkdir -p logs/m2_prnf_fusion "$OUTPUT_DIR"
test -f "$CONDITION_MANIFEST" || { echo "Missing condition manifest"; exit 1; }

python scripts/evaluate_m2_prnf_fusion_R8.py \
  --metadata_csv "$METADATA" \
  --full_clean_manifest "$MANIFEST_ROOT/full_clean_manifest.json" \
  --robustness_manifest "$MANIFEST_ROOT/robustness_manifest.json" \
  --condition_manifest "$CONDITION_MANIFEST" \
  --output_dir "$OUTPUT_DIR" \
  --legacy_local_direct_checkpoint "$ROOT/legacy_local_direct_seed42/model_best.pt" \
  --global_direct_checkpoint "$ROOT/global_direct_seed42/model_best.pt" \
  --residual_only_checkpoint "$ROOT/residual_only_seed42/model_best.pt" \
  --hybrid_direct_residual_checkpoint "$ROOT/hybrid_direct_residual_seed42/model_best.pt" \
  --batch_size 2 \
  --num_workers 4 \
  --bootstrap_resamples 10000 \
  --seed 42

cat "$OUTPUT_DIR/fusion_pilot_decision.json"
