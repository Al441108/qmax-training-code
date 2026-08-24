#!/bin/bash
#SBATCH --job-name=prnf_fusion_ep50
#SBATCH --output=logs/m2_prnf_fusion/formal50_%x_%j.out
#SBATCH --error=logs/m2_prnf_fusion/formal50_%x_%j.err
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G

set -euo pipefail
PROJECT_ROOT="/projects/u6dm/fastmri_project/fastmri_pipeline"
METADATA="/lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv"
FUSION_DESIGN="${FUSION_DESIGN:?Set the pilot-selected FUSION_DESIGN}"
VARIANT="${VARIANT:?Set VARIANT=prnf_no_rel|prnf_no_need|prnf_full}"
SEED="${SEED:-42}"
MANIFEST_ROOT="outputs/m2_prnf/preflight_R8_bs4"
CONDITION_MANIFEST="outputs/m2_prnf_fusion/preflight_R8_bs4/condition_manifest.json"
PILOT_DECISION="outputs/evaluation/m2_prnf_fusion_pilot15_R8_bs4_seed42/fusion_pilot_decision.json"
PILOT_AUDIT="outputs/evaluation/m2_prnf_fusion_pilot15_R8_bs4_seed42/fusion_pilot_evaluation_audit.json"
OUTPUT_DIR="outputs/m2_prnf_fusion/formal50_R8_bs4/${FUSION_DESIGN}/${VARIANT}_seed${SEED}"

case "$FUSION_DESIGN" in
  residual_only|hybrid_direct_residual) NEED_SCOPE=residual ;;
  legacy_local_direct|global_direct) NEED_SCOPE=all_auxiliary ;;
  *) echo "Unknown FUSION_DESIGN=$FUSION_DESIGN"; exit 2 ;;
esac
case "$VARIANT" in
  prnf_no_rel|prnf_no_need|prnf_full) ;;
  *) echo "Unknown VARIANT=$VARIANT"; exit 2 ;;
esac

cd "$PROJECT_ROOT"
source "$HOME/miniforge3/bin/activate"
conda activate fastmri311
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs/m2_prnf_fusion "$OUTPUT_DIR"
test -f "$PILOT_DECISION" || { echo "Missing pilot decision: $PILOT_DECISION"; exit 1; }
test -f "$PILOT_AUDIT" || { echo "Missing pilot audit: $PILOT_AUDIT"; exit 1; }
test -f "$CONDITION_MANIFEST" || { echo "Missing condition manifest"; exit 1; }
python - "$PILOT_DECISION" "$FUSION_DESIGN" "${ALLOW_MANUAL_SELECTION:-0}" <<'PY'
import json, sys
decision = json.load(open(sys.argv[1], encoding="utf-8"))
selected = decision.get("recommended_fusion_design")
if selected != sys.argv[2] and sys.argv[3] != "1":
    raise SystemExit(
        f"Requested fusion {sys.argv[2]} differs from pilot recommendation {selected}. "
        "Set ALLOW_MANUAL_SELECTION=1 only after documenting the reason."
    )
PY
python - "$PILOT_AUDIT" "$CONDITION_MANIFEST" <<'PY'
import json, sys
from pathlib import Path
from scripts.train_m2_prnf_fusion import locked_code_hashes, sha256_file
audit = json.load(open(sys.argv[1], encoding="utf-8"))
if audit.get("code_hashes") != locked_code_hashes():
    raise SystemExit("Installed code differs from the evaluated pilot code")
if audit.get("condition_manifest", {}).get("sha256") != sha256_file(Path(sys.argv[2])):
    raise SystemExit("Condition manifest differs from the evaluated pilot")
PY

RESUME_ARGS=()
if [[ -f "$OUTPUT_DIR/model_last.pt" ]]; then
  RESUME_ARGS=(--resume "$OUTPUT_DIR/model_last.pt")
fi

python scripts/train_m2_prnf_fusion.py \
  --metadata_csv "$METADATA" \
  --output_dir "$OUTPUT_DIR" \
  --full_clean_manifest "$MANIFEST_ROOT/full_clean_manifest.json" \
  --robustness_manifest "$MANIFEST_ROOT/robustness_manifest.json" \
  --condition_manifest "$CONDITION_MANIFEST" \
  --variant "$VARIANT" \
  --fusion_design "$FUSION_DESIGN" \
  --need_scope "$NEED_SCOPE" \
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

cat "$OUTPUT_DIR/final_summary.json"
