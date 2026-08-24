#!/bin/bash
#SBATCH --job-name=m2gdv21_fullclean
#SBATCH --output=logs/m2gd_v21/%x_%j.out
#SBATCH --error=logs/m2gd_v21/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G

set -euo pipefail

: "${R:?Submit with --export=ALL,R=4, R=6, or R=8}"
case "$R" in
  4|6)
    DECISION="outputs/evaluation/m2gd_v21_fusioncal_audit_R${R}_r1/m2gd_v21_smoke_audit_decision.json"
    ;;
  8)
    DECISION="outputs/evaluation/m2gd_v21_fusioncal_audit_R8_r3/m2gd_v21_smoke_audit_decision.json"
    ;;
  *)
    echo "Unsupported R=$R; expected 4, 6, or 8." >&2
    exit 1
    ;;
esac

PROJECT_ROOT="/projects/u6dm/fastmri_project/fastmri_pipeline"
METADATA="/lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv"
OUTPUT_DIR="outputs/evaluation/m2gd_v21_full_clean_R${R}_final_r1"

cd "$PROJECT_ROOT"
source "$HOME/miniforge3/bin/activate"
conda activate fastmri311
mkdir -p logs/m2gd_v21 "$OUTPUT_DIR"

[[ -f "$DECISION" ]] || { echo "Missing final audit decision: $DECISION" >&2; exit 1; }
jq -e '
  .final_model_eligible == true and
  .selected_model == "stage_b_fusion_only_calibration" and
  .fallback_to_stagea == false and
  ([.criteria[]] | all)
' "$DECISION" >/dev/null || {
  echo "Final audit did not select an eligible Stage-B model: $DECISION" >&2
  exit 1
}

M2U_CHECKPOINT=$(jq -er '.checkpoint_identity.m2u.path' "$DECISION")
M2GD_V2_CHECKPOINT=$(jq -er '.checkpoint_identity.m2gd_v2.path' "$DECISION")
STAGEA_CHECKPOINT=$(jq -er '.checkpoint_identity.m2gd_v21_stagea.path' "$DECISION")
STAGEB_CHECKPOINT=$(jq -er '.selected_checkpoint' "$DECISION")
AUDITED_STAGEB_CHECKPOINT=$(jq -er '.checkpoint_identity.m2gd_v21.path' "$DECISION")
[[ "$STAGEB_CHECKPOINT" == "$AUDITED_STAGEB_CHECKPOINT" ]] || {
  echo "Selected Stage-B path differs from audited identity." >&2
  exit 1
}

M2U_SHA256=$(jq -er '.checkpoint_identity.m2u.sha256' "$DECISION")
M2GD_V2_SHA256=$(jq -er '.checkpoint_identity.m2gd_v2.sha256' "$DECISION")
STAGEA_SHA256=$(jq -er '.checkpoint_identity.m2gd_v21_stagea.sha256' "$DECISION")
STAGEB_SHA256=$(jq -er '.checkpoint_identity.m2gd_v21.sha256' "$DECISION")
M2U_EPOCH=$(jq -er '.checkpoint_identity.m2u.epoch' "$DECISION")
M2GD_V2_EPOCH=$(jq -er '.checkpoint_identity.m2gd_v2.epoch' "$DECISION")
STAGEA_EPOCH=$(jq -er '.checkpoint_identity.m2gd_v21_stagea.epoch' "$DECISION")
STAGEB_EPOCH=$(jq -er '.checkpoint_identity.m2gd_v21.epoch' "$DECISION")

for checkpoint in "$M2U_CHECKPOINT" "$M2GD_V2_CHECKPOINT" "$STAGEA_CHECKPOINT" "$STAGEB_CHECKPOINT"; do
  [[ -f "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; exit 1; }
done
[[ ! -e "$OUTPUT_DIR/full_clean_completed.json" ]] || {
  echo "Completed output already exists; nothing was overwritten: $OUTPUT_DIR" >&2
  exit 1
}

python -m py_compile \
  scripts/evaluate_m2gd_v21_smoke_audit.py \
  scripts/evaluate_m2gd_v21_full_clean.py

python scripts/evaluate_m2gd_v21_full_clean.py \
  --metadata_csv "$METADATA" \
  --m2u_checkpoint "$M2U_CHECKPOINT" \
  --m2gd_v2_checkpoint "$M2GD_V2_CHECKPOINT" \
  --stagea_checkpoint "$STAGEA_CHECKPOINT" \
  --stageb_checkpoint "$STAGEB_CHECKPOINT" \
  --output_dir "$OUTPUT_DIR" \
  --acceleration "$R" \
  --pd_aux_acceleration 2 \
  --batch_size 2 \
  --num_workers 4 \
  --seed 42 \
  --bootstrap_resamples 10000 \
  --expected_num_patients 25 \
  --expected_num_slices 878 \
  --expected_m2u_epoch "$M2U_EPOCH" \
  --expected_m2gd_v2_epoch "$M2GD_V2_EPOCH" \
  --expected_stagea_epoch "$STAGEA_EPOCH" \
  --expected_stageb_epoch "$STAGEB_EPOCH" \
  --expected_m2u_sha256 "$M2U_SHA256" \
  --expected_m2gd_v2_sha256 "$M2GD_V2_SHA256" \
  --expected_stagea_sha256 "$STAGEA_SHA256" \
  --expected_stageb_sha256 "$STAGEB_SHA256"

cat "$OUTPUT_DIR/full_clean_completed.json"
