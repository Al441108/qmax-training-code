#!/bin/bash
#SBATCH --job-name=m2gdv2_R8_audit
#SBATCH --output=logs/m2gd_v2/m2gdv2_R8_audit_%j.out
#SBATCH --error=logs/m2gd_v2/m2gdv2_R8_audit_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G

set -euo pipefail

PROJECT_ROOT="/projects/u6dm/fastmri_project/fastmri_pipeline"
METADATA="/lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv"
M2U_CHECKPOINT="outputs/m2u/m2u_pdR2_to_pdfs_R8_pilot5/model_best.pt"
M2GD_V2_CHECKPOINT="outputs/m2gd_v2/m2gd_v2_R8_smoke5_fixed/model_best.pt"
OUTPUT_DIR="outputs/evaluation/m2gd_v2_fixed_smoke_audit_R8"

cd "$PROJECT_ROOT"
source "$HOME/miniforge3/bin/activate"
conda activate fastmri311
mkdir -p "$OUTPUT_DIR"

[[ -f "$M2U_CHECKPOINT" ]] || { echo "Missing M2-U checkpoint: $M2U_CHECKPOINT"; exit 1; }
[[ -f "$M2GD_V2_CHECKPOINT" ]] || { echo "Missing v2 checkpoint: $M2GD_V2_CHECKPOINT"; exit 1; }

printf '%s\n' "============================================================"
printf 'M2-GD v2 reduced audit | job=%s | host=%s | start=%s\n' \
  "${SLURM_JOB_ID:-unknown}" "$(hostname)" "$(date)"
printf 'v2 checkpoint: %s\n' "$M2GD_V2_CHECKPOINT"
printf '%s\n' "============================================================"

python scripts/evaluate_m2gd_v2_smoke_audit_R8.py \
  --metadata_csv "$METADATA" \
  --m2u_checkpoint "$M2U_CHECKPOINT" \
  --m2gd_v2_checkpoint "$M2GD_V2_CHECKPOINT" \
  --output_dir "$OUTPUT_DIR" \
  --num_audit_patients 6 \
  --slices_per_patient 12 \
  --batch_size 2 \
  --num_workers 4 \
  --seed 42 \
  --expected_m2u_epoch 50 \
  --expected_m2gd_v2_epoch 5

printf 'Reduced audit completed: %s\n' "$(date)"
cat "$OUTPUT_DIR/m2gd_v2_smoke_audit_decision.json"
