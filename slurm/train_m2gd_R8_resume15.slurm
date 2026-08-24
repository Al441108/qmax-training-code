#!/bin/bash
#SBATCH --job-name=m2gd_R8_ep15
#SBATCH --output=logs/m2gd/m2gd_R8_resume15_%j.out
#SBATCH --error=logs/m2gd/m2gd_R8_resume15_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G

set -euo pipefail

cd /projects/u6dm/fastmri_project/fastmri_pipeline

source ~/miniforge3/bin/activate
conda activate fastmri311

CHECKPOINT="outputs/m2gd/m2gd_R8_pilot5/model_last.pt"
OUTPUT_DIR="outputs/m2gd/m2gd_R8_pilot5"

echo "============================================================"
echo "M2-GD R=8 continuation: epoch 5 -> epoch 15"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Output directory: ${OUTPUT_DIR}"
echo "============================================================"

if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: checkpoint not found: ${CHECKPOINT}"
    exit 1
fi

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError("No CUDA GPU visible")

print("GPU:", torch.cuda.get_device_name(0))
PY

python -m py_compile \
  src/m2gd_auxiliary_varnet.py \
  scripts/train_varnet_m2gd.py

python scripts/train_varnet_m2gd.py \
  --metadata_csv /lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv \
  --acceleration 8 \
  --pd_aux_acceleration 2 \
  --epochs 15 \
  --learning_rate 1e-4 \
  --num_cascades 12 \
  --chans 18 \
  --sens_chans 8 \
  --pools 4 \
  --sens_pools 4 \
  --batch_size 8 \
  --num_workers 4 \
  --gate_diagnostic_batches 3 \
  --lambda_rel 0.05 \
  --lambda_budget 0 \
  --resume "${CHECKPOINT}" \
  --output_dir "${OUTPUT_DIR}"

echo "============================================================"
echo "End: $(date)"
echo "M2-GD continuation to epoch 15 completed"
echo "============================================================"
