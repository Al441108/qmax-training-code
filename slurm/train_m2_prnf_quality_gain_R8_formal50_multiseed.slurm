#!/bin/bash
#SBATCH --job-name=qgain_ms50
#SBATCH --output=logs/m2_prnf_multiseed/qgain_seed%j.out
#SBATCH --error=logs/m2_prnf_multiseed/qgain_seed%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G

set -euo pipefail

PROJECT_ROOT="/projects/u6dm/fastmri_project/fastmri_pipeline"
METADATA="/lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv"
SEED="${SEED:?Submit with --export=ALL,SEED=123 or SEED=2026}"
MANIFEST_ROOT="outputs/m2_prnf/preflight_R8_bs4"
CONDITION_MANIFEST="outputs/m2_prnf_fusion/preflight_R8_bs4/condition_manifest.json"
OLD_PREFLIGHT="outputs/m2_prnf_quality_gain/preflight_R8_bs4/preflight_results.json"
OUTPUT_DIR="outputs/m2_prnf_quality_gain/formal50_multiseed_R8_bs4/quality_protected_hybrid_gain_seed${SEED}_ep50"

case "$SEED" in
  123|2026) ;;
  *)
    echo "Confirmation seed must be 123 or 2026; observed $SEED"
    exit 2
    ;;
esac

cd "$PROJECT_ROOT"
source "$HOME/miniforge3/bin/activate"
conda activate fastmri311
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs/m2_prnf_multiseed "$OUTPUT_DIR"

for required in \
  "$METADATA" \
  "$MANIFEST_ROOT/full_clean_manifest.json" \
  "$MANIFEST_ROOT/robustness_manifest.json" \
  "$CONDITION_MANIFEST" \
  "$OLD_PREFLIGHT" \
  "MULTISEED_CONFIRMATION_PROTOCOL_R8.json" \
  "scripts/train_m2_prnf_quality_gain_formal50.py"
do
  test -f "$required" || {
    echo "Missing required file: $required"
    exit 1
  }
done

python - "$OLD_PREFLIGHT" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], encoding="utf-8"))
if result.get("status") != "passed":
    raise SystemExit(f"Existing fifth-arm shape preflight failed: {result}")
print("Existing architecture/shape preflight: PASSED")
PY

python -m py_compile \
  scripts/train_m2_prnf_quality_gain_formal50.py

RESUME_ARGS=()
if [[ -f "$OUTPUT_DIR/model_last.pt" ]]; then
  RESUME_ARGS=(--resume "$OUTPUT_DIR/model_last.pt")
  echo "Resuming interrupted seed-$SEED run"
else
  if find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type f | grep -q .; then
    echo "Output directory contains files but no model_last.pt: $OUTPUT_DIR"
    echo "Refusing to mix a new run with a partial directory."
    exit 1
  fi
  echo "Starting independent seed-$SEED run from random initialisation"
fi

python scripts/train_m2_prnf_quality_gain_formal50.py \
  --metadata_csv "$METADATA" \
  --output_dir "$OUTPUT_DIR" \
  --full_clean_manifest "$MANIFEST_ROOT/full_clean_manifest.json" \
  --robustness_manifest "$MANIFEST_ROOT/robustness_manifest.json" \
  --condition_manifest "$CONDITION_MANIFEST" \
  --variant prnf_no_need \
  --fusion_design hybrid_direct_residual \
  --run_stage quality_gain_formal50 \
  --need_scope residual \
  --residual_scale 0.1 \
  --lambda_residual_gain 0.2 \
  --residual_gain_margin_relative 0.002 \
  --residual_gain_ramp_epochs 5 \
  --acceleration 8 \
  --pd_aux_acceleration 2 \
  --epochs 50 \
  --learning_rate 3e-4 \
  --batch_size 4 \
  --grad_accum_steps 1 \
  --num_workers 4 \
  --num_cascades 12 \
  --chans 18 \
  --sens_chans 8 \
  --pools 4 \
  --sens_pools 4 \
  --controller_chans 16 \
  --initial_aux_alpha 0.1 \
  --initial_gate_probability 0.95 \
  --initial_need_probability 0.95 \
  --need_floor 0.25 \
  --lambda_rel 0.05 \
  --lambda_rank 0.02 \
  --aux_loss_ramp_epochs 5 \
  --seed "$SEED" \
  "${RESUME_ARGS[@]}"

jq '{
  variant,
  fusion_design,
  completed_epochs,
  best_epoch,
  best_val_patient_l1,
  parameter_count
}' "$OUTPUT_DIR/final_summary.json"

sha256sum \
  "$OUTPUT_DIR/model_best.pt" \
  "$OUTPUT_DIR/model_last.pt" \
  | tee "$OUTPUT_DIR/checkpoint_sha256.txt"
