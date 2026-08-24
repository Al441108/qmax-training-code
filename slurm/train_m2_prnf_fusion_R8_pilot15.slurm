#!/bin/bash
#SBATCH --job-name=prnf_fusion_p15
#SBATCH --output=logs/m2_prnf_fusion/pilot15_%x_%j.out
#SBATCH --error=logs/m2_prnf_fusion/pilot15_%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G

set -euo pipefail
PROJECT_ROOT="/projects/u6dm/fastmri_project/fastmri_pipeline"
METADATA="/lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv"
FUSION_DESIGN="${FUSION_DESIGN:?Use --export=ALL,FUSION_DESIGN=<arm>,SEED=42}"
SEED="${SEED:-42}"
PREFLIGHT="outputs/m2_prnf_fusion/preflight_R8_bs4/preflight_results.json"
MANIFEST_ROOT="outputs/m2_prnf/preflight_R8_bs4"
CONDITION_MANIFEST="outputs/m2_prnf_fusion/preflight_R8_bs4/condition_manifest.json"
OUTPUT_DIR="outputs/m2_prnf_fusion/pilot15_R8_bs4/${FUSION_DESIGN}_seed${SEED}"

case "$FUSION_DESIGN" in
  legacy_local_direct|global_direct|residual_only|hybrid_direct_residual) ;;
  *) echo "Unknown FUSION_DESIGN=$FUSION_DESIGN"; exit 2 ;;
esac

cd "$PROJECT_ROOT"
source "$HOME/miniforge3/bin/activate"
conda activate fastmri311
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs/m2_prnf_fusion "$OUTPUT_DIR"
test -f "$PREFLIGHT" || { echo "Missing v1.4 preflight"; exit 1; }
test -f "$CONDITION_MANIFEST" || { echo "Missing condition manifest"; exit 1; }
python - "$PREFLIGHT" "$CONDITION_MANIFEST" <<'PY'
import json, sys
from pathlib import Path
from scripts.train_m2_prnf_fusion import locked_code_hashes, sha256_file
result = json.load(open(sys.argv[1], encoding="utf-8"))
if result.get("status") != "passed":
    raise SystemExit(f"v1.4 preflight failed: {result}")
if result.get("code_hashes") != locked_code_hashes():
    raise SystemExit("Installed v1.4 code changed after preflight; rerun preflight")
if result.get("manifest_sha256", {}).get("conditions") != sha256_file(Path(sys.argv[2])):
    raise SystemExit("Condition manifest changed after preflight")
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
  --variant prnf_no_need \
  --fusion_design "$FUSION_DESIGN" \
  --need_scope residual \
  --residual_scale 0.1 \
  --run_stage pilot \
  --acceleration 8 \
  --pd_aux_acceleration 2 \
  --epochs 15 \
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
