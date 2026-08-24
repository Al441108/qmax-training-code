#!/bin/bash
#SBATCH --job-name=singlePDFS_R8_e50
#SBATCH --output=logs/varnet_single/singlePDFS_R8_e50_%j.out
#SBATCH --error=logs/varnet_single/singlePDFS_R8_e50_%j.err
#SBATCH --partition=workq
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G

set -euo pipefail

PROJECT_ROOT="/projects/u6dm/fastmri_project/fastmri_pipeline"

METADATA="/lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv"

OUTPUT_DIR="$PROJECT_ROOT/outputs/varnet_single/pdfs_R8_c12_ch18_ep30_bs8_nw4"

RESUME_CHECKPOINT="$OUTPUT_DIR/model_last.pt"

cd "$PROJECT_ROOT"

source "$HOME/miniforge3/bin/activate"
conda activate fastmri311

mkdir -p logs/varnet_single

test -f "$METADATA" || {
    echo "ERROR: metadata not found: $METADATA"
    exit 1
}

test -f "$RESUME_CHECKPOINT" || {
    echo "ERROR: resume checkpoint not found: $RESUME_CHECKPOINT"
    exit 1
}

test -f "$OUTPUT_DIR/config.json" || {
    echo "ERROR: config not found: $OUTPUT_DIR/config.json"
    exit 1
}

echo "============================================================"
echo "Single PD-FS VarNet R=8: epoch 30 -> 50"
echo "Job: ${SLURM_JOB_ID:-unknown}"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Checkpoint: $RESUME_CHECKPOINT"
echo "Output: $OUTPUT_DIR"
echo "============================================================"

python - "$RESUME_CHECKPOINT" <<'PY'
import sys
import torch

checkpoint = torch.load(
    sys.argv[1],
    map_location="cpu",
    weights_only=False,
)

required = [
    "epoch",
    "model_state_dict",
    "optimizer_state_dict",
    "config",
]

missing = [
    key for key in required
    if key not in checkpoint
]

if missing:
    raise SystemExit(
        f"Checkpoint missing required keys: {missing}"
    )

epoch = int(checkpoint["epoch"])

if epoch != 30:
    raise SystemExit(
        f"Expected epoch 30 checkpoint, observed epoch {epoch}"
    )

print("Resume checkpoint gate: PASSED")
print("Checkpoint epoch:", epoch)
PY

python scripts/train_varnet_single.py \
  --metadata_csv "$METADATA" \
  --contrast pdfs \
  --acceleration 8 \
  --epochs 50 \
  --learning_rate 0.0003 \
  --num_cascades 12 \
  --chans 18 \
  --sens_chans 8 \
  --pools 4 \
  --sens_pools 4 \
  --batch_size 8 \
  --num_workers 4 \
  --seed 42 \
  --output_dir "$OUTPUT_DIR" \
  --resume "$RESUME_CHECKPOINT"

echo "============================================================"
echo "Training completed: $(date)"
echo "============================================================"

jq '{
  status,
  contrast,
  acceleration,
  completed_epochs,
  best_epoch,
  best_validation_results,
  checkpoint_reload_matches,
  train_patients,
  validation_patients,
  parameter_count
}' "$OUTPUT_DIR/final_summary.json"

tail -n 5 "$OUTPUT_DIR/training_log.csv"

sha256sum \
  "$OUTPUT_DIR/model_best.pt" \
  "$OUTPUT_DIR/model_last.pt" \
  | tee "$OUTPUT_DIR/checkpoint_sha256_epoch50.txt"