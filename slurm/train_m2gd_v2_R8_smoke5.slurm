#!/bin/bash
#SBATCH --job-name=m2gdv2_R8_smoke5
#SBATCH --output=logs/m2gd_v2/m2gdv2_R8_smoke5_%j.out
#SBATCH --error=logs/m2gd_v2/m2gdv2_R8_smoke5_%j.err
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
OUTPUT_DIR="outputs/m2gd_v2/m2gd_v2_R8_smoke5_fixed"
PREFLIGHT_JSON="outputs/m2gd_v2/preflight_R8_fixed/preflight_results.json"

cd "$PROJECT_ROOT"
source "$HOME/miniforge3/bin/activate"
conda activate fastmri311
mkdir -p "$OUTPUT_DIR"

[[ -f "$M2U_CHECKPOINT" ]] || { echo "Missing checkpoint: $M2U_CHECKPOINT"; exit 1; }
[[ -f "$PREFLIGHT_JSON" ]] || { echo "Run preflight first: $PREFLIGHT_JSON"; exit 1; }
python - "$PREFLIGHT_JSON" <<'PY'
import json, sys
path = sys.argv[1]
result = json.load(open(path, "r", encoding="utf-8"))
if result.get("status") != "passed":
    raise SystemExit(f"Preflight did not pass: {result}")
print("Preflight gate: PASSED")
PY

printf '%s\n' "============================================================"
printf 'M2-GD v2 R8 smoke5 | job=%s | host=%s | start=%s\n' \
  "${SLURM_JOB_ID:-unknown}" "$(hostname)" "$(date)"
printf 'Output: %s\n' "$OUTPUT_DIR"
printf '%s\n' "============================================================"

python scripts/train_varnet_m2gd_v2.py \
  --metadata_csv "$METADATA" \
  --m2u_checkpoint "$M2U_CHECKPOINT" \
  --output_dir "$OUTPUT_DIR" \
  --acceleration 8 \
  --pd_aux_acceleration 2 \
  --epochs 5 \
  --curriculum smoke5 \
  --batch_size 4 \
  --grad_accum_steps 2 \
  --num_workers 4 \
  --num_cascades 12 \
  --chans 18 \
  --sens_chans 8 \
  --pools 4 \
  --sens_pools 4 \
  --initial_aux_alpha 0.1 \
  --initial_gate_probability 0.99 \
  --pretrained_lr 1e-5 \
  --new_module_lr 1e-4 \
  --lambda_rel 0.05 \
  --freeze_epochs 2 \
  --padding_prob_reflect 0.70 \
  --padding_prob_replicate 0.20 \
  --padding_prob_zero 0.10 \
  --seed 42

printf 'Smoke training completed: %s\n' "$(date)"
cat "$OUTPUT_DIR/final_summary.json"
