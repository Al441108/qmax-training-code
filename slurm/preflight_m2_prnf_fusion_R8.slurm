#!/bin/bash
#SBATCH --job-name=prnf_fusion_preflight
#SBATCH --output=logs/m2_prnf_fusion/preflight_%j.out
#SBATCH --error=logs/m2_prnf_fusion/preflight_%j.err
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G

set -euo pipefail
PROJECT_ROOT="/projects/u6dm/fastmri_project/fastmri_pipeline"
METADATA="/lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv"
V13_PREFLIGHT="outputs/m2_prnf/preflight_R8_bs4"
REFERENCE_CONFIG="outputs/m2_prnf/formal_R8_bs4/prnf_full_seed42_ep50/config.json"
OUTPUT_JSON="outputs/m2_prnf_fusion/preflight_R8_bs4/preflight_results.json"
CONDITION_MANIFEST="outputs/m2_prnf_fusion/preflight_R8_bs4/condition_manifest.json"

cd "$PROJECT_ROOT"
source "$HOME/miniforge3/bin/activate"
conda activate fastmri311
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs/m2_prnf_fusion "$(dirname "$OUTPUT_JSON")"

test -f "$V13_PREFLIGHT/full_clean_manifest.json" || { echo "Missing v1.3 clean manifest"; exit 1; }
test -f "$V13_PREFLIGHT/robustness_manifest.json" || { echo "Missing v1.3 robustness manifest"; exit 1; }
test -f "$REFERENCE_CONFIG" || { echo "Missing completed prnf_full config: $REFERENCE_CONFIG"; exit 1; }

python -m py_compile \
  src/m2_prnf_varnet.py \
  src/m2_prnf_corruptions.py \
  src/m2_prnf_fusion_pilot_varnet.py \
  scripts/train_m2_prnf_fusion.py \
  scripts/preflight_m2_prnf_fusion.py \
  scripts/evaluate_m2_prnf_fusion_R8.py

python scripts/preflight_m2_prnf_fusion.py \
  --metadata_csv "$METADATA" \
  --output_json "$OUTPUT_JSON" \
  --full_clean_manifest "$V13_PREFLIGHT/full_clean_manifest.json" \
  --robustness_manifest "$V13_PREFLIGHT/robustness_manifest.json" \
  --condition_manifest "$CONDITION_MANIFEST" \
  --reference_config "$REFERENCE_CONFIG" \
  --formal_batch_size 4 \
  --formal_max_gpu_memory_gb 94 \
  --acceleration 8 \
  --pd_aux_acceleration 2 \
  --residual_scale 0.1 \
  --seed 42

cat "$OUTPUT_JSON"
