#!/bin/bash
#SBATCH --job-name=prnf_preflight_R8_bs4
#SBATCH --output=logs/m2_prnf/prnf_preflight_R8_bs4_%j.out
#SBATCH --error=logs/m2_prnf/prnf_preflight_R8_bs4_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G

set -euo pipefail
PROJECT_ROOT="/projects/u6dm/fastmri_project/fastmri_pipeline"
METADATA="/lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv"
OUTPUT_JSON="outputs/m2_prnf/preflight_R8_bs4/preflight_results.json"
FULL_CLEAN_MANIFEST="outputs/m2_prnf/preflight_R8_bs4/full_clean_manifest.json"
ROBUSTNESS_MANIFEST="outputs/m2_prnf/preflight_R8_bs4/robustness_manifest.json"

cd "$PROJECT_ROOT"
source "$HOME/miniforge3/bin/activate"
conda activate fastmri311
mkdir -p logs/m2_prnf "$(dirname "$OUTPUT_JSON")"

python -m py_compile \
  src/m2_prnf_varnet.py \
  src/m2_prnf_corruptions.py \
  scripts/train_m2_prnf.py \
  scripts/preflight_m2_prnf.py \
  scripts/evaluate_m2_prnf_R8.py

python scripts/preflight_m2_prnf.py \
  --metadata_csv "$METADATA" \
  --output_json "$OUTPUT_JSON" \
  --full_clean_manifest "$FULL_CLEAN_MANIFEST" \
  --robustness_manifest "$ROBUSTNESS_MANIFEST" \
  --robustness_slices_per_patient 12 \
  --formal_batch_size 4 \
  --formal_max_gpu_memory_gb 110.0 \
  --acceleration 8 \
  --pd_aux_acceleration 2 \
  --seed 42

cat "$OUTPUT_JSON"
