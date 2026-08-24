#!/bin/bash
#SBATCH --job-name=m2gd_R8_smoke
#SBATCH --output=logs/m2gd/m2gd_R8_smoke_%j.out
#SBATCH --error=logs/m2gd/m2gd_R8_smoke_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G

set -euo pipefail

cd /projects/u6dm/fastmri_project/fastmri_pipeline

source ~/miniforge3/bin/activate
conda activate fastmri311

mkdir -p logs/m2gd
mkdir -p outputs/m2gd/m2gd_R8_engineering_smoke

echo "============================================================"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "============================================================"

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError("No CUDA GPU is visible on this compute node")

print("GPU:", torch.cuda.get_device_name(0))
print(
    "GPU memory:",
    round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
    "GB",
)
PY

python -m py_compile \
  src/m2gd_auxiliary_varnet.py \
  scripts/train_varnet_m2gd.py

python scripts/train_varnet_m2gd.py \
  --metadata_csv /lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv \
  --acceleration 8 \
  --pd_aux_acceleration 2 \
  --num_train_patients 2 \
  --num_val_patients 2 \
  --epochs 1 \
  --learning_rate 1e-4 \
  --num_cascades 12 \
  --chans 18 \
  --sens_chans 8 \
  --pools 4 \
  --sens_pools 4 \
  --batch_size 1 \
  --num_workers 0 \
  --max_train_batches 2 \
  --max_val_batches 1 \
  --gate_diagnostic_batches 1 \
  --lambda_rel 0.05 \
  --lambda_budget 0 \
  --output_dir outputs/m2gd/m2gd_R8_engineering_smoke

echo "============================================================"
echo "End: $(date)"
echo "M2-GD engineering smoke completed"
echo "============================================================"
