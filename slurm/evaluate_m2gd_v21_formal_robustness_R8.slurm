#!/bin/bash
#SBATCH --job-name=m2gdv21_R8_formal_robust
#SBATCH --output=logs/m2gd_v21/m2gdv21_R8_formal_robust_%j.out
#SBATCH --error=logs/m2gd_v21/m2gdv21_R8_formal_robust_%j.err
#SBATCH --time=12:00:00
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
OUTPUT_DIR="outputs/evaluation/m2gd_v21_formal_robustness_R8_r1"
EXPECTED_STAGEB_SHA256="a917421b98a3c8482c7cd019bba12eaf5568c72b4398ec142c47007fb9213837"
CONSTANT_Q="0.9214919610725095"

cd "$PROJECT_ROOT"
source "$HOME/miniforge3/bin/activate"
conda activate fastmri311
mkdir -p "$OUTPUT_DIR" logs/m2gd_v21

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

ACTUAL_STAGEB_SHA256=$(sha256sum "$STAGEB_CHECKPOINT" | awk '{print $1}')
[[ "$ACTUAL_STAGEB_SHA256" == "$EXPECTED_STAGEB_SHA256" ]] || {
  echo "Stage-B SHA-256 mismatch" >&2
  echo "Expected: $EXPECTED_STAGEB_SHA256" >&2
  echo "Actual:   $ACTUAL_STAGEB_SHA256" >&2
  exit 1
}

python -m py_compile \
  scripts/evaluate_m2gd_v21_smoke_audit_R8.py \
  scripts/evaluate_m2gd_v21_formal_robustness_R8.py

printf '%s\n' "============================================================"
printf 'Formal R8 robustness | job=%s | host=%s | start=%s\n' \
  "${SLURM_JOB_ID:-unknown}" "$(hostname)" "$(date)"
printf 'Selected Stage-B checkpoint: %s\n' "$STAGEB_CHECKPOINT"
printf 'Selected Stage-B SHA-256:   %s\n' "$ACTUAL_STAGEB_SHA256"
printf 'Output: %s\n' "$OUTPUT_DIR"
printf '%s\n' "============================================================"

python scripts/evaluate_m2gd_v21_formal_robustness_R8.py \
  --metadata_csv "$METADATA" \
  --m2u_checkpoint "$M2U_CHECKPOINT" \
  --m2gd_v2_checkpoint "$M2GD_V2_CHECKPOINT" \
  --stagea_checkpoint "$STAGEA_CHECKPOINT" \
  --stageb_checkpoint "$STAGEB_CHECKPOINT" \
  --output_dir "$OUTPUT_DIR" \
  --num_patients 25 \
  --slices_per_patient 12 \
  --batch_size 2 \
  --num_workers 4 \
  --bootstrap_iterations 10000 \
  --constant_q "$CONSTANT_Q" \
  --seed 42 \
  --expected_m2u_epoch 50 \
  --expected_m2gd_v2_epoch 5 \
  --expected_stagea_epoch 3 \
  --expected_stageb_epoch 3 \
  --expected_stageb_sha256 "$EXPECTED_STAGEB_SHA256"

printf 'Formal robustness evaluation completed: %s\n' "$(date)"
cat "$OUTPUT_DIR/formal_robustness_decision.json"
