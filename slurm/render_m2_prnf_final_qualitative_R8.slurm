#!/bin/bash
#SBATCH --job-name=prnf_final_R8_fig
#SBATCH --output=logs/m2_prnf/final_qualitative_R8_%j.out
#SBATCH --error=logs/m2_prnf/final_qualitative_R8_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

set -euo pipefail

PROJECT_ROOT="/projects/u6dm/fastmri_project/fastmri_pipeline"
METADATA="/lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv"
MANIFEST_ROOT="outputs/m2_prnf/preflight_R8_bs4"
CONDITION_MANIFEST="outputs/m2_prnf_fusion/preflight_R8_bs4/condition_manifest.json"
COMPARISON_DIR="outputs/evaluation/m2_prnf_final_comparison_R8_bs4_seed42"
OUTPUT_DIR="outputs/figures/m2_prnf_final_comparison_R8_bs4_seed42"

M2U_CLEAN="outputs/m2_prnf/formal_R8_bs4/m2u_clean_seed42_ep50/model_best.pt"
M2U_AUGMENTED="outputs/m2_prnf/formal_R8_bs4/m2u_augmented_seed42_ep50/model_best.pt"
M2U_AUGCAP_MASK="outputs/m2_prnf/formal_R8_bs4/m2u_augcap_mask_seed42_ep50/model_best.pt"
GLOBAL_DIRECT="outputs/m2_prnf/formal_R8_bs4/global_direct_prnf_no_need_seed42_ep50/model_best.pt"
QUALITY_GAIN="outputs/m2_prnf/formal_R8_bs4/quality_protected_hybrid_gain_seed42_ep50/model_best.pt"

cd "$PROJECT_ROOT"
source "$HOME/miniforge3/bin/activate"
conda activate fastmri311
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR:-/tmp}/m2_prnf_fig_pycache"
mkdir -p logs/m2_prnf "$OUTPUT_DIR" "$PYTHONPYCACHEPREFIX"

python -B scripts/render_m2_prnf_final_qualitative_R8.py \
  --metadata_csv "$METADATA" \
  --full_clean_manifest "$MANIFEST_ROOT/full_clean_manifest.json" \
  --robustness_manifest "$MANIFEST_ROOT/robustness_manifest.json" \
  --condition_manifest "$CONDITION_MANIFEST" \
  --comparison_per_slice "$COMPARISON_DIR/final_comparison_per_slice.csv" \
  --m2u_clean_checkpoint "$M2U_CLEAN" \
  --m2u_augmented_checkpoint "$M2U_AUGMENTED" \
  --m2u_augcap_mask_checkpoint "$M2U_AUGCAP_MASK" \
  --global_direct_checkpoint "$GLOBAL_DIRECT" \
  --quality_gain_checkpoint "$QUALITY_GAIN" \
  --output_dir "$OUTPUT_DIR" \
  --seed 42

cat "$OUTPUT_DIR/representative_case_selection.json"
