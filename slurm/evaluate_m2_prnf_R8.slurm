#!/bin/bash
#SBATCH --job-name=prnf_eval_R8
#SBATCH --output=logs/m2_prnf/prnf_eval_R8_%j.out
#SBATCH --error=logs/m2_prnf/prnf_eval_R8_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G

set -euo pipefail
PROJECT_ROOT="/projects/u6dm/fastmri_project/fastmri_pipeline"
METADATA="/lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv"
FULL_CLEAN_MANIFEST="outputs/m2_prnf/preflight_R8_bs4/full_clean_manifest.json"
ROBUSTNESS_MANIFEST="outputs/m2_prnf/preflight_R8_bs4/robustness_manifest.json"
ROOT="outputs/m2_prnf/formal_R8_bs4"
OUTPUT_DIR="outputs/evaluation/m2_prnf_R8_bs4_seed42"

cd "$PROJECT_ROOT"
source "$HOME/miniforge3/bin/activate"
conda activate fastmri311
mkdir -p logs/m2_prnf "$OUTPUT_DIR"
test -f "$FULL_CLEAN_MANIFEST" || { echo "Missing manifest: $FULL_CLEAN_MANIFEST"; exit 1; }
test -f "$ROBUSTNESS_MANIFEST" || { echo "Missing manifest: $ROBUSTNESS_MANIFEST"; exit 1; }

python scripts/evaluate_m2_prnf_R8.py \
  --metadata_csv "$METADATA" \
  --full_clean_manifest "$FULL_CLEAN_MANIFEST" \
  --robustness_manifest "$ROBUSTNESS_MANIFEST" \
  --output_dir "$OUTPUT_DIR" \
  --m2u_clean_checkpoint "$ROOT/m2u_clean_seed42_ep50/model_best.pt" \
  --m2u_augmented_checkpoint "$ROOT/m2u_augmented_seed42_ep50/model_best.pt" \
  --m2u_augcap_mask_checkpoint "$ROOT/m2u_augcap_mask_seed42_ep50/model_best.pt" \
  --prnf_no_rel_checkpoint "$ROOT/prnf_no_rel_seed42_ep50/model_best.pt" \
  --prnf_no_need_checkpoint "$ROOT/prnf_no_need_seed42_ep50/model_best.pt" \
  --prnf_full_checkpoint "$ROOT/prnf_full_seed42_ep50/model_best.pt" \
  --batch_size 2 \
  --num_workers 4 \
  --bootstrap_resamples 10000 \
  --seed 42

cat "$OUTPUT_DIR/m2_prnf_decision.json"
