#!/bin/bash
#SBATCH --job-name=prnf_step2_scales
#SBATCH --output=logs/m2_prnf_step2/scales_%j.out
#SBATCH --error=logs/m2_prnf_step2/scales_%j.err
#SBATCH --time=02:00:00
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
GAIN_CHECKPOINT="outputs/m2_prnf_quality_gain/pilot15_R8_bs4/quality_protected_hybrid_gain_seed42/model_best.pt"
M2U_EPOCH15="outputs/m2_prnf/formal_R8_bs4/m2u_augmented_seed42_ep50/epoch_15_clean_audit.json"
OUTPUT_DIR="outputs/evaluation/m2_prnf_step2_diagnostics_R8_bs4_seed42/scales"

cd "$PROJECT_ROOT"
source "$HOME/miniforge3/bin/activate"
conda activate fastmri311
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs/m2_prnf_step2 "$OUTPUT_DIR"

for path in \
  "$MANIFEST_ROOT/full_clean_manifest.json" \
  "$MANIFEST_ROOT/robustness_manifest.json" \
  "$CONDITION_MANIFEST" \
  "$GAIN_CHECKPOINT" \
  "$M2U_EPOCH15"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path"; exit 1; }
done

python -m py_compile scripts/evaluate_m2_prnf_step2_diagnostics_R8.py

python scripts/evaluate_m2_prnf_step2_diagnostics_R8.py \
  --phase scales \
  --metadata_csv "$METADATA" \
  --full_clean_manifest "$MANIFEST_ROOT/full_clean_manifest.json" \
  --robustness_manifest "$MANIFEST_ROOT/robustness_manifest.json" \
  --condition_manifest "$CONDITION_MANIFEST" \
  --quality_gain_checkpoint "$GAIN_CHECKPOINT" \
  --m2u_augmented_epoch15_audit "$M2U_EPOCH15" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size 2 \
  --num_workers 4 \
  --bootstrap_resamples 10000 \
  --seed 42

cat "$OUTPUT_DIR/step2_scales_decision.json"
