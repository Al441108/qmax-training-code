#!/bin/bash
#SBATCH --job-name=m2gdv2_preflight
#SBATCH --output=logs/m2gd_v2/m2gdv2_preflight_%j.out
#SBATCH --error=logs/m2gd_v2/m2gdv2_preflight_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G

set -euo pipefail

PROJECT_ROOT="/projects/u6dm/fastmri_project/fastmri_pipeline"
METADATA="/lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv"
M2U_CHECKPOINT="outputs/m2u/m2u_pdR2_to_pdfs_R8_pilot5/model_best.pt"
OUTPUT_JSON="outputs/m2gd_v2/preflight_R8_fixed/preflight_results.json"

cd "$PROJECT_ROOT"
source "$HOME/miniforge3/bin/activate"
conda activate fastmri311
mkdir -p "$(dirname "$OUTPUT_JSON")"

printf '%s\n' "============================================================"
printf 'M2-GD v2 preflight | job=%s | host=%s | start=%s\n' \
  "${SLURM_JOB_ID:-unknown}" "$(hostname)" "$(date)"
printf 'M2-U checkpoint: %s\n' "$M2U_CHECKPOINT"
printf '%s\n' "============================================================"

[[ -f "$M2U_CHECKPOINT" ]] || { echo "Missing checkpoint: $M2U_CHECKPOINT"; exit 1; }

python -m py_compile \
  src/auxiliary_corruptions_v2.py \
  src/m2gd_v2_auxiliary_varnet.py \
  scripts/preflight_m2gd_v2.py \
  scripts/train_varnet_m2gd_v2.py \
  scripts/evaluate_m2gd_v2_smoke_audit_R8.py

python scripts/preflight_m2gd_v2.py \
  --metadata_csv "$METADATA" \
  --m2u_checkpoint "$M2U_CHECKPOINT" \
  --output_json "$OUTPUT_JSON" \
  --initial_gate_probability 0.99 \
  --max_initial_relative_difference 0.02 \
  --acceleration 8 \
  --pd_aux_acceleration 2

printf 'Preflight completed: %s\n' "$(date)"
cat "$OUTPUT_JSON"
