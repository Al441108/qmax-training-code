#!/bin/bash
#SBATCH --job-name=prnf_clean_diag_R8
#SBATCH --output=logs/m2_prnf/prnf_clean_diag_R8_%j.out
#SBATCH --error=logs/m2_prnf/prnf_clean_diag_R8_%j.err
#SBATCH --time=16:00:00
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
M2U_CLEAN="$ROOT/m2u_clean_seed42_ep50/model_best.pt"
PRNF_FULL="$ROOT/prnf_full_seed42_ep50/model_best.pt"
OUTPUT_DIR="outputs/evaluation/m2_prnf_clean_diagnostics_R8_bs4_seed42"

cd "$PROJECT_ROOT"
source "$HOME/miniforge3/bin/activate"
conda activate fastmri311
mkdir -p logs/m2_prnf "$OUTPUT_DIR"

for required in \
  "$FULL_CLEAN_MANIFEST" \
  "$ROBUSTNESS_MANIFEST" \
  "$M2U_CLEAN" \
  "$PRNF_FULL"
do
  test -f "$required" || { echo "Missing required file: $required"; exit 1; }
done

python -m py_compile scripts/evaluate_m2_prnf_clean_diagnostics_R8.py

python scripts/evaluate_m2_prnf_clean_diagnostics_R8.py \
  --metadata_csv "$METADATA" \
  --full_clean_manifest "$FULL_CLEAN_MANIFEST" \
  --robustness_manifest "$ROBUSTNESS_MANIFEST" \
  --m2u_clean_checkpoint "$M2U_CLEAN" \
  --prnf_full_checkpoint "$PRNF_FULL" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size 2 \
  --num_workers 4 \
  --bootstrap_resamples 10000 \
  --seed 42

cat "$OUTPUT_DIR/clean_diagnostic_summary.csv"
cat "$OUTPUT_DIR/clean_diagnostic_paired_bootstrap.csv"
cat "$OUTPUT_DIR/clean_diagnostic_decision.json"
