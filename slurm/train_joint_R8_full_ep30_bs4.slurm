#!/bin/bash
#SBATCH --job-name=joint_R8_full_bs4
#SBATCH --output=/projects/u6dm/fastmri_project/fastmri_pipeline/logs/joint_varnet/joint_R8_full_bs4_%j.out
#SBATCH --error=/projects/u6dm/fastmri_project/fastmri_pipeline/logs/joint_varnet/joint_R8_full_bs4_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G

set -euo pipefail

cd /projects/u6dm/fastmri_project/fastmri_pipeline

source ~/miniforge3/bin/activate
conda activate fastmri311

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Hostname:"
hostname

echo "Python:"
which python

echo "CUDA check:"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo "Start full joint R=8 training bs4"
date

python scripts/train_varnet_joint.py \
  --metadata_csv /projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv \
  --acceleration 8 \
  --epochs 30 \
  --num_cascades 12 \
  --chans 18 \
  --sens_chans 8 \
  --pools 4 \
  --sens_pools 4 \
  --batch_size 4 \
  --num_workers 4 \
  --learning_rate 0.0003 \
  --cross_fusion concat \
  --output_dir /projects/u6dm/fastmri_project/fastmri_pipeline/outputs/varnet_joint/joint_R8_c12_ch18_ep30_bs4

echo "Finished full joint R=8 training bs4"
date
