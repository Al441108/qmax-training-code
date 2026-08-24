#!/bin/bash
#SBATCH --job-name=eval_single_varnet_val
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/u6dm/albert44.u6dm/fastmri_project/logs/%x_%j.out
#SBATCH --error=/scratch/u6dm/albert44.u6dm/fastmri_project/logs/%x_%j.err

set -euo pipefail

PROJECT_DIR=/projects/u6dm/fastmri_project/fastmri_pipeline
DATA_ROOT=/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split
SCRATCH_ROOT=/scratch/u6dm/albert44.u6dm/fastmri_project

cd "$PROJECT_DIR"

source ~/miniforge3/bin/activate
conda activate fastmri311

echo "===== SINGLE VARNET EVALUATION: VAL SPLIT ====="
echo "Start time: $(date)"
echo "Host: $(hostname)"
echo "Project dir: $PROJECT_DIR"
echo "Scratch root: $SCRATCH_ROOT"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

METADATA="$DATA_ROOT/metadata/reorganised_dataset_split_isambard.csv"
EVAL_ROOT="$SCRATCH_ROOT/outputs/evaluation/single_varnet_val"

mkdir -p "$EVAL_ROOT"

COMMON_ARGS="\
  --metadata_csv $METADATA \
  --split val \
  --batch_size 1 \
  --num_workers 0 \
  --num_cascades 12 \
  --sens_chans 8 \
  --sens_pools 4 \
  --chans 18 \
  --pools 4 \
  --save_png_every 100 \
  --max_png 12"

echo "===== Evaluating PD R=4 ====="
python scripts/evaluate_varnet_single.py \
  $COMMON_ARGS \
  --contrast pd \
  --acceleration 4 \
  --checkpoint "$SCRATCH_ROOT/outputs/varnet_single/pd_R4_c12_ch18_ep30_bs8/model_best.pt" \
  --output_dir "$EVAL_ROOT/pd_R4"

echo "===== Evaluating PD R=6 ====="
python scripts/evaluate_varnet_single.py \
  $COMMON_ARGS \
  --contrast pd \
  --acceleration 6 \
  --checkpoint "$SCRATCH_ROOT/outputs/varnet_single/pd_R6_c12_ch18_ep30_bs8_nw4/model_best.pt" \
  --output_dir "$EVAL_ROOT/pd_R6"

echo "===== Evaluating PD R=8 ====="
python scripts/evaluate_varnet_single.py \
  $COMMON_ARGS \
  --contrast pd \
  --acceleration 8 \
  --checkpoint "$SCRATCH_ROOT/outputs/varnet_single/pd_R8_c12_ch18_ep30_bs8_nw4/model_best.pt" \
  --output_dir "$EVAL_ROOT/pd_R8"

echo "===== Evaluating PD-FS R=4 ====="
python scripts/evaluate_varnet_single.py \
  $COMMON_ARGS \
  --contrast pdfs \
  --acceleration 4 \
  --checkpoint "$SCRATCH_ROOT/outputs/varnet_single/pdfs_R4_c12_ch18_ep30_bs8/model_best.pt" \
  --output_dir "$EVAL_ROOT/pdfs_R4"

echo "===== Evaluating PD-FS R=6 ====="
python scripts/evaluate_varnet_single.py \
  $COMMON_ARGS \
  --contrast pdfs \
  --acceleration 6 \
  --checkpoint "$SCRATCH_ROOT/outputs/varnet_single/pdfs_R6_c12_ch18_ep30_bs8_nw4/model_best.pt" \
  --output_dir "$EVAL_ROOT/pdfs_R6"

echo "===== Evaluating PD-FS R=8 ====="
python scripts/evaluate_varnet_single.py \
  $COMMON_ARGS \
  --contrast pdfs \
  --acceleration 8 \
  --checkpoint "$SCRATCH_ROOT/outputs/varnet_single/pdfs_R8_c12_ch18_ep30_bs8_nw4/model_best.pt" \
  --output_dir "$EVAL_ROOT/pdfs_R8"

echo "===== ALL EVALUATIONS COMPLETED ====="
echo "End time: $(date)"
