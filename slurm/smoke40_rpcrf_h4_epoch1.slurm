#!/bin/bash
#SBATCH --job-name=smk40_rpcrf_e1
#SBATCH --output=logs/smoke40/rpcrf_e1_%j.out
#SBATCH --error=logs/smoke40/rpcrf_e1_%j.err
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G

set -euo pipefail

PROJECT_ROOT="/projects/u6dm/fastmri_project/fastmri_pipeline"
METADATA="/lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv"
MANIFEST_ROOT="outputs/m2_prnf/preflight_R8_bs4"
CONDITION_MANIFEST="outputs/m2_prnf_fusion/preflight_R8_bs4/condition_manifest.json"
OUTPUT_DIR="outputs/smoke40/${SLURM_JOB_ID}/rpcrf_h4_epoch1"
TARGET_EPOCH=1
TARGET_AUDIT="$OUTPUT_DIR/epoch_01_clean_audit.json"

cd "$PROJECT_ROOT"
source "$HOME/miniforge3/bin/activate"
conda activate fastmri311
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs/smoke40 "$OUTPUT_DIR"

python -m py_compile \
  src/m2_prnf_rpcrf_varnet.py \
  scripts/train_m2_prnf_rpcrf.py

python scripts/train_m2_prnf_rpcrf.py \
  --metadata_csv "$METADATA" \
  --metadata_identity_csv "$METADATA" \
  --output_dir "$OUTPUT_DIR" \
  --full_clean_manifest "$MANIFEST_ROOT/full_clean_manifest.json" \
  --robustness_manifest "$MANIFEST_ROOT/robustness_manifest.json" \
  --condition_manifest "$CONDITION_MANIFEST" \
  --variant prnf_no_need \
  --fusion_design hybrid_direct_residual \
  --run_stage rpcrf_h4_formal50 \
  --need_scope residual \
  --residual_scale 0.1 \
  --residual_reliability_power 1.5 \
  --lambda_correction 0.05 \
  --lambda_overshoot 0.01 \
  --overshoot_kappa 1.25 \
  --correction_weight_clip 4.0 \
  --lambda_residual_gain 0.10 \
  --residual_gain_margin_relative 0.002 \
  --residual_gain_ramp_epochs 5 \
  --acceleration 8 \
  --pd_aux_acceleration 2 \
  --epochs 50 \
  --learning_rate 3e-4 \
  --batch_size 2 \
  --grad_accum_steps 2 \
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
  --seed 42 &
TRAIN_PID=$!

DEADLINE=$((SECONDS + 2220))
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  sleep 15
  if [[ -f "$TARGET_AUDIT" && -f "$OUTPUT_DIR/model_last.pt" ]]; then
    OBSERVED_EPOCH=$(python -c \
      "import torch; print(int(torch.load('$OUTPUT_DIR/model_last.pt', map_location='cpu', weights_only=False)['epoch']))")
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
  fi
  if (( SECONDS >= DEADLINE )); then
    echo "Smoke target was not reached before the 37-minute safety deadline."
    kill -TERM "$TRAIN_PID" 2>/dev/null || true
    wait "$TRAIN_PID" || true
    exit 1
  fi
done

wait "$TRAIN_PID"
echo "Trainer exited before the smoke target checkpoint was observed."
exit 1
