#!/bin/bash
#SBATCH --job-name=smk40_qgain_e16
#SBATCH --output=logs/smoke40/qgain_e16_%j.out
#SBATCH --error=logs/smoke40/qgain_e16_%j.err
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G

set -euo pipefail

PROJECT_ROOT="/projects/u6dm/fastmri_project/fastmri_pipeline"
METADATA="/lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv"
PILOT_DIR="outputs/m2_prnf_quality_gain/pilot15_R8_bs4/quality_protected_hybrid_gain_seed42"
OUTPUT_DIR="outputs/smoke40/${SLURM_JOB_ID}/quality_gain_epoch16"
MANIFEST_ROOT="outputs/m2_prnf/preflight_R8_bs4"
CONDITION_MANIFEST="outputs/m2_prnf_fusion/preflight_R8_bs4/condition_manifest.json"
PILOT_EVAL="outputs/evaluation/m2_prnf_quality_gain_five_arm_R8_bs4_seed42"
TARGET_EPOCH=16
TARGET_AUDIT="$OUTPUT_DIR/epoch_16_clean_audit.json"

cd "$PROJECT_ROOT"
source "$HOME/miniforge3/bin/activate"
conda activate fastmri311
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs/smoke40

python -m py_compile \
  scripts/prepare_m2_prnf_quality_gain_extend50.py \
  scripts/train_m2_prnf_quality_gain_extend50.py

python scripts/prepare_m2_prnf_quality_gain_extend50.py \
  --pilot_dir "$PILOT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --pilot_decision "$PILOT_EVAL/fusion_pilot_decision.json" \
  --pilot_audit "$PILOT_EVAL/fusion_pilot_evaluation_audit.json"

python scripts/train_m2_prnf_quality_gain_extend50.py \
  --metadata_csv "$METADATA" \
  --output_dir "$OUTPUT_DIR" \
  --full_clean_manifest "$MANIFEST_ROOT/full_clean_manifest.json" \
  --robustness_manifest "$MANIFEST_ROOT/robustness_manifest.json" \
  --condition_manifest "$CONDITION_MANIFEST" \
  --variant prnf_no_need \
  --fusion_design hybrid_direct_residual \
  --run_stage quality_gain_extension_15_to_50 \
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
  --seed 42 \
  --resume "$OUTPUT_DIR/model_last.pt" &
TRAIN_PID=$!

DEADLINE=$((SECONDS + 2220))
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  sleep 15
  if [[ -f "$TARGET_AUDIT" && -f "$OUTPUT_DIR/model_last.pt" ]]; then
    OBSERVED_EPOCH=$(python -c \
      "import torch; print(int(torch.load('$OUTPUT_DIR/model_last.pt', map_location='cpu', weights_only=False)['epoch']))")
  else
    OBSERVED_EPOCH=0
  fi
  if (( OBSERVED_EPOCH >= TARGET_EPOCH )); then
    echo "Smoke target reached: epoch=$OBSERVED_EPOCH"
    kill -TERM "$TRAIN_PID" 2>/dev/null || true
    wait "$TRAIN_PID" || true
    python - "$OUTPUT_DIR" "$TARGET_EPOCH" <<'PY'
import csv
import json
import sys
from pathlib import Path
import torch

root = Path(sys.argv[1])
target = int(sys.argv[2])
checkpoint = torch.load(root / "model_last.pt", map_location="cpu", weights_only=False)
rows = list(csv.DictReader(open(root / "training_log.csv", encoding="utf-8")))
result = {
    "status": "passed",
    "target_epoch": target,
    "checkpoint_epoch": int(checkpoint["epoch"]),
    "training_log_rows": len(rows),
    "output_dir": str(root),
}
if result["checkpoint_epoch"] < target or len(rows) < target:
    raise SystemExit(f"Smoke verification failed: {result}")
(root / "smoke40_result.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
PY
    exit 0
  fi
  if (( SECONDS >= DEADLINE )); then
    echo "Smoke target was not reached before the 37-minute safety deadline."
    kill -TERM "$TRAIN_PID" 2>/dev/null || true
    wait "$TRAIN_PID" || true
    exit 1
  fi
done

wait "$TRAIN_PID"
echo "Trainer exited before epoch 16 was observed."
exit 1
