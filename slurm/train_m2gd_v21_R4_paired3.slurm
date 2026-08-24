#!/bin/bash
#SBATCH --job-name=m2gdv21_R4_pair3
#SBATCH --output=logs/m2gd_v21/m2gdv21_R4_pair3_%j.out
#SBATCH --error=logs/m2gd_v21/m2gdv21_R4_pair3_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G

set -euo pipefail
PROJECT_ROOT="/projects/u6dm/fastmri_project/fastmri_pipeline"
METADATA="/lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv"
M2GD_V2_CHECKPOINT="outputs/m2gd_v2/m2gd_v2_R4_smoke5_fixed/model_best.pt"
PREFLIGHT_JSON="outputs/m2gd_v21/preflight_R4_r1/preflight_results.json"
OUTPUT_DIR="outputs/m2gd_v21/m2gd_v21_R4_paired3_r1"

cd "$PROJECT_ROOT"
source "$HOME/miniforge3/bin/activate"
conda activate fastmri311
mkdir -p "$OUTPUT_DIR" logs/m2gd_v21
[[ -f "$M2GD_V2_CHECKPOINT" ]] || { echo "Missing R4 V2 checkpoint: $M2GD_V2_CHECKPOINT"; exit 1; }
[[ -f "$PREFLIGHT_JSON" ]] || { echo "Run R4 V2.1 preflight first: $PREFLIGHT_JSON"; exit 1; }
python - "$PREFLIGHT_JSON" <<'PY'
import json, sys
result = json.load(open(sys.argv[1], "r", encoding="utf-8"))
if result.get("status") != "passed":
    raise SystemExit(f"R4 V2.1 preflight did not pass: {result}")
print("R4 V2.1 preflight gate: PASSED")
PY

python scripts/train_varnet_m2gd_v21.py \
  --metadata_csv "$METADATA" \
  --m2gd_v2_checkpoint "$M2GD_V2_CHECKPOINT" \
  --output_dir "$OUTPUT_DIR" \
  --acceleration 4 \
  --pd_aux_acceleration 2 \
  --epochs 3 \
  --curriculum paired3 \
  --batch_size 2 \
  --grad_accum_steps 2 \
  --num_workers 4 \
  --num_cascades 12 \
  --chans 18 \
  --sens_chans 8 \
  --pools 4 \
  --sens_pools 4 \
  --initial_aux_alpha 0.1 \
  --initial_gate_probability 0.7436 \
  --reliability_lr 1e-4 \
  --lambda_rel 0.05 \
  --rank_weight 1.0 \
  --padding_weight 0.25 \
  --padding_prob_reflect 0.70 \
  --padding_prob_replicate 0.20 \
  --padding_prob_zero 0.10 \
  --border_padding_prob_reflect 0.25 \
  --border_padding_prob_replicate 0.25 \
  --border_padding_prob_zero 0.50 \
  --seed 42

cat "$OUTPUT_DIR/final_summary.json"
