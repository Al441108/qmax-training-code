#!/bin/bash
#SBATCH --job-name=prnf_qgain_preflight
#SBATCH --output=logs/m2_prnf_quality_gain/preflight_%j.out
#SBATCH --error=logs/m2_prnf_quality_gain/preflight_%j.err
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
REFERENCE_CONFIG="outputs/m2_prnf/formal_R8_bs4/prnf_full_seed42_ep50/config.json"
CONDITION_MANIFEST="outputs/m2_prnf_fusion/preflight_R8_bs4/condition_manifest.json"
OUTPUT_JSON="outputs/m2_prnf_quality_gain/preflight_R8_bs4/preflight_results.json"

cd "$PROJECT_ROOT"
source "$HOME/miniforge3/bin/activate"
conda activate fastmri311
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs/m2_prnf_quality_gain "$(dirname "$OUTPUT_JSON")"

for path in \
  "$MANIFEST_ROOT/full_clean_manifest.json" \
  "$MANIFEST_ROOT/robustness_manifest.json" \
  "$CONDITION_MANIFEST" \
  "$REFERENCE_CONFIG"; do
  test -f "$path" || { echo "Missing required input: $path"; exit 1; }
done

python -m py_compile \
  scripts/train_m2_prnf_quality_gain.py \
  scripts/preflight_m2_prnf_quality_gain.py \
  scripts/evaluate_m2_prnf_quality_gain_R8.py

python scripts/preflight_m2_prnf_quality_gain.py \
  --metadata_csv "$METADATA" \
  --output_json "$OUTPUT_JSON" \
  --full_clean_manifest "$MANIFEST_ROOT/full_clean_manifest.json" \
  --robustness_manifest "$MANIFEST_ROOT/robustness_manifest.json" \
  --condition_manifest "$CONDITION_MANIFEST" \
  --reference_config "$REFERENCE_CONFIG" \
  --formal_batch_size 4 \
  --formal_max_gpu_memory_gb 94 \
  --acceleration 8 \
  --pd_aux_acceleration 2 \
  --residual_scale 0.1 \
  --lambda_residual_gain 0.2 \
  --residual_gain_margin_relative 0.002 \
  --seed 42

cat "$OUTPUT_JSON"
