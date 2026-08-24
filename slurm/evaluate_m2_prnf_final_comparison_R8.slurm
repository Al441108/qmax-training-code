#!/bin/bash
#SBATCH --job-name=prnf_final_R8_eval
#SBATCH --output=logs/m2_prnf/final_comparison_R8_%j.out
#SBATCH --error=logs/m2_prnf/final_comparison_R8_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G

set -euo pipefail

PROJECT_ROOT="/projects/u6dm/fastmri_project/fastmri_pipeline"
METADATA="/lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv"
MANIFEST_ROOT="outputs/m2_prnf/preflight_R8_bs4"
CONDITION_MANIFEST="outputs/m2_prnf_fusion/preflight_R8_bs4/condition_manifest.json"
OUTPUT_DIR="outputs/evaluation/m2_prnf_final_comparison_R8_bs4_seed42"

M2U_CLEAN="outputs/m2_prnf/formal_R8_bs4/m2u_clean_seed42_ep50/model_best.pt"
M2U_AUGMENTED="outputs/m2_prnf/formal_R8_bs4/m2u_augmented_seed42_ep50/model_best.pt"
M2U_AUGCAP_MASK="outputs/m2_prnf/formal_R8_bs4/m2u_augcap_mask_seed42_ep50/model_best.pt"
GLOBAL_DIRECT="outputs/m2_prnf/formal_R8_bs4/global_direct_prnf_no_need_seed42_ep50/model_best.pt"
QUALITY_GAIN="outputs/m2_prnf/formal_R8_bs4/quality_protected_hybrid_gain_seed42_ep50/model_best.pt"

cd "$PROJECT_ROOT"
source "$HOME/miniforge3/bin/activate"
conda activate fastmri311
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR:-/tmp}/m2_prnf_final_eval_pycache"
mkdir -p logs/m2_prnf "$OUTPUT_DIR" "$PYTHONPYCACHEPREFIX"

for file in \
  "$METADATA" \
  "$MANIFEST_ROOT/full_clean_manifest.json" \
  "$MANIFEST_ROOT/robustness_manifest.json" \
  "$CONDITION_MANIFEST" \
  "$M2U_CLEAN" \
  "$M2U_AUGMENTED" \
  "$M2U_AUGCAP_MASK" \
  "$GLOBAL_DIRECT" \
  "$QUALITY_GAIN"; do
  [[ -f "$file" ]] || { echo "Missing required file: $file"; exit 1; }
done

python -B -m py_compile scripts/evaluate_m2_prnf_final_comparison_R8.py

python -B scripts/evaluate_m2_prnf_final_comparison_R8.py \
  --metadata_csv "$METADATA" \
  --full_clean_manifest "$MANIFEST_ROOT/full_clean_manifest.json" \
  --robustness_manifest "$MANIFEST_ROOT/robustness_manifest.json" \
  --condition_manifest "$CONDITION_MANIFEST" \
  --m2u_clean_checkpoint "$M2U_CLEAN" \
  --m2u_augmented_checkpoint "$M2U_AUGMENTED" \
  --m2u_augcap_mask_checkpoint "$M2U_AUGCAP_MASK" \
  --global_direct_checkpoint "$GLOBAL_DIRECT" \
  --quality_gain_checkpoint "$QUALITY_GAIN" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size 2 \
  --num_workers 4 \
  --bootstrap_resamples 10000 \
  --seed 42

printf 'Final R=8 comparison completed: %s\n' "$(date)"
cat "$OUTPUT_DIR/final_comparison_decision.json"
