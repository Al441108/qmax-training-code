#!/bin/bash
#SBATCH --job-name=m2gdv21_fullclean_R8
#SBATCH --output=logs/m2gd_v21/m2gdv21_fullclean_R8_%j.out
#SBATCH --error=logs/m2gd_v21/m2gdv21_fullclean_R8_%j.err
#SBATCH --time=08:00:00
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
STAGEA_CHECKPOINT="outputs/m2gd_v21/m2gd_v21_R8_paired3_r2/model_best.pt"
STAGEB_CHECKPOINT="outputs/m2gd_v21/m2gd_v21_R8_FINAL/model_final_fusioncal3_r3.pt"
OUTPUT_DIR="outputs/evaluation/m2gd_v21_full_clean_R8_final"

M2U_SHA256="f8c0dfa87ec9271760a659d57076d758195e725400eaacb1f2fe66681aa74bfd"
M2GD_V2_SHA256="bf6eead003e6bb52db631bc95641f18143b26dc616b87dcdd8d97fc2184ad992"
STAGEA_SHA256="e58f6148f1103460391c77d42c144fc925a9f22c2ebef5ba8e88d794870fce5f"
STAGEB_SHA256="a917421b98a3c8482c7cd019bba12eaf5568c72b4398ec142c47007fb9213837"

cd "$PROJECT_ROOT"
source "$HOME/miniforge3/bin/activate"
conda activate fastmri311
mkdir -p logs/m2gd_v21 "$OUTPUT_DIR"

for checkpoint in \
  "$M2U_CHECKPOINT" \
  "$M2GD_V2_CHECKPOINT" \
  "$STAGEA_CHECKPOINT" \
  "$STAGEB_CHECKPOINT"; do
  [[ -f "$checkpoint" ]] || {
    echo "Missing checkpoint: $checkpoint" >&2
    exit 1
  }
done

if [[ -e "$OUTPUT_DIR/full_clean_completed.json" ]]; then
  echo "Completed final output already exists: $OUTPUT_DIR/full_clean_completed.json" >&2
  echo "Nothing was overwritten." >&2
  exit 1
fi

printf '%s\n' "============================================================"
printf 'M2-GD v2.1 full clean R=8 evaluation | job=%s | host=%s | start=%s\n' \
  "${SLURM_JOB_ID:-unknown}" "$(hostname)" "$(date)"
printf 'Stage-B final checkpoint: %s\n' "$STAGEB_CHECKPOINT"
printf '%s\n' "============================================================"

python -m py_compile \
  scripts/evaluate_m2gd_v21_smoke_audit_R8.py \
  scripts/evaluate_m2gd_v21_full_clean_R8.py

python scripts/evaluate_m2gd_v21_full_clean_R8.py \
  --metadata_csv "$METADATA" \
  --m2u_checkpoint "$M2U_CHECKPOINT" \
  --m2gd_v2_checkpoint "$M2GD_V2_CHECKPOINT" \
  --stagea_checkpoint "$STAGEA_CHECKPOINT" \
  --stageb_checkpoint "$STAGEB_CHECKPOINT" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size 2 \
  --num_workers 4 \
  --seed 42 \
  --bootstrap_resamples 10000 \
  --expected_num_patients 25 \
  --expected_num_slices 878 \
  --expected_m2u_epoch 50 \
  --expected_m2gd_v2_epoch 5 \
  --expected_stagea_epoch 3 \
  --expected_stageb_epoch 3 \
  --expected_m2u_sha256 "$M2U_SHA256" \
  --expected_m2gd_v2_sha256 "$M2GD_V2_SHA256" \
  --expected_stagea_sha256 "$STAGEA_SHA256" \
  --expected_stageb_sha256 "$STAGEB_SHA256"

printf 'Full clean evaluation completed: %s\n' "$(date)"
cat "$OUTPUT_DIR/full_clean_completed.json"
