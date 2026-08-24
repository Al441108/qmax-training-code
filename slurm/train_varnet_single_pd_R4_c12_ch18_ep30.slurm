#!/bin/bash
#SBATCH --job-name=varnet_pd_R4_c12_ch18
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/u6dm/albert44.u6dm/fastmri_project/logs/%x_%j.out
#SBATCH --error=/scratch/u6dm/albert44.u6dm/fastmri_project/logs/%x_%j.err

set -euo pipefail

PROJECT_DIR=/projects/u6dm/fastmri_project/fastmri_pipeline
DATA_ROOT=/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split
SCRATCH_ROOT=/scratch/u6dm/albert44.u6dm/fastmri_project

mkdir -p "$SCRATCH_ROOT/logs"
mkdir -p "$SCRATCH_ROOT/outputs/varnet_single"

cd "$PROJECT_DIR"

echo "===== JOB INFO ====="
echo "Job ID: ${SLURM_JOB_ID:-not_set}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "DATA_ROOT=$DATA_ROOT"
echo "SCRATCH_ROOT=$SCRATCH_ROOT"

source ~/miniforge3/bin/activate
conda activate fastmri311

echo "===== PYTHON CHECK ====="
which python
python --version

echo "===== GPU CHECK ====="
nvidia-smi || true

echo "===== PACKAGE CHECK ====="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
python -c "import numpy, pandas, h5py, scipy, skimage, matplotlib, tqdm, yaml, requests; print('basic packages OK')"
python -c "import fastmri; print('fastmri OK')"

echo "===== START TRAIN: PD R=4 c12 ch18 ep30 ====="

python scripts/train_varnet_single.py \
  --metadata_csv "$DATA_ROOT/metadata/reorganised_dataset_split_isambard.csv" \
  --contrast pd \
  --acceleration 4 \
  --epochs 30 \
  --learning_rate 0.0003 \
  --num_cascades 12 \
  --chans 18 \
  --sens_chans 8 \
  --pools 4 \
  --sens_pools 4 \
  --batch_size 8 \
  --num_workers 4 \
  --seed 42 \
  --output_dir "$SCRATCH_ROOT/outputs/varnet_single/pd_R4_c12_ch18_ep30_bs8"

echo "End time: $(date)"
echo "===== DONE ====="
