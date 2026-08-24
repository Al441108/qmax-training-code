#!/bin/bash
#SBATCH --job-name=gdirect_ms50
#SBATCH --output=logs/m2_prnf_multiseed/gdirect_%j.out
#SBATCH --error=logs/m2_prnf_multiseed/gdirect_%j.err
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
OUTPUT_DIR="outputs/m2_prnf_fusion/formal50_R8_bs4/global_direct/prnf_no_need_seed${SEED}"

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

PILOT_EVAL=""
for candidate in \
  "outputs/evaluation/m2_prnf_fusion_pilot15_R8_bs4_seed42_evalfix1" \
  "outputs/evaluation/m2_prnf_fusion_pilot15_R8_bs4_seed42"
do
  if [[ -f "$candidate/fusion_pilot_decision.json" ]] && \
     [[ -f "$candidate/fusion_pilot_evaluation_audit.json" ]]; then
    PILOT_EVAL="$candidate"
    break
  fi
done

test -n "$PILOT_EVAL" || {
  echo "No audited fusion-pilot evaluation directory was found"
  exit 1
}

for required in \
  "$METADATA" \
  "$MANIFEST_ROOT/full_clean_manifest.json" \
  "$MANIFEST_ROOT/robustness_manifest.json" \
  "$CONDITION_MANIFEST" \
  "$PILOT_EVAL/fusion_pilot_decision.json" \
  "$PILOT_EVAL/fusion_pilot_evaluation_audit.json" \
  "scripts/train_m2_prnf_fusion.py"
do
  test -f "$required" || {
    echo "Missing required file: $required"
    exit 1
  }
done

python - "$PILOT_EVAL/fusion_pilot_decision.json" <<'PY'
import json
import sys

decision = json.load(open(sys.argv[1], encoding="utf-8"))
selected = decision.get("recommended_fusion_design")
if selected != "global_direct":
    raise SystemExit(
        "Locked pilot recommendation is not global_direct: "
        f"{selected!r}"
    )
print("Fusion selection gate: PASSED")
PY

python - \
  "$PILOT_EVAL/fusion_pilot_evaluation_audit.json" \
  "$CONDITION_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

from scripts.train_m2_prnf_fusion import (
    locked_code_hashes,
    sha256_file,
)

audit = json.load(open(sys.argv[1], encoding="utf-8"))
if audit.get("code_hashes") != locked_code_hashes():
    raise SystemExit(
        "Installed fusion code differs from the evaluated pilot code"
    )
observed = sha256_file(Path(sys.argv[2]))
expected = audit.get("condition_manifest", {}).get("sha256")
if expected != observed:
    raise SystemExit(
        "Condition manifest differs from the evaluated pilot manifest"
    )
print("Fusion provenance gate: PASSED")
PY

python -m py_compile scripts/train_m2_prnf_fusion.py

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

python scripts/train_m2_prnf_fusion.py \
  --metadata_csv "$METADATA" \
  --output_dir "$OUTPUT_DIR" \
  --full_clean_manifest "$MANIFEST_ROOT/full_clean_manifest.json" \
  --robustness_manifest "$MANIFEST_ROOT/robustness_manifest.json" \
  --condition_manifest "$CONDITION_MANIFEST" \
  --variant prnf_no_need \
  --fusion_design global_direct \
  --need_scope all_auxiliary \
  --residual_scale 0.1 \
  --run_stage formal \
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
