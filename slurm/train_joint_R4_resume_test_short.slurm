#!/bin/bash
#SBATCH --job-name=joint_R4_resume_short
#SBATCH --output=/projects/u6dm/fastmri_project/fastmri_pipeline/logs/joint_varnet/joint_R4_resume_short_%j.out
#SBATCH --error=/projects/u6dm/fastmri_project/fastmri_pipeline/logs/joint_varnet/joint_R4_resume_short_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G

set -euo pipefail

cd /projects/u6dm/fastmri_project/fastmri_pipeline

source ~/miniforge3/bin/activate
conda activate fastmri311

echo "Hostname:"
hostname

echo "CUDA check:"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo "Start short joint R4 resume test"
date

python scripts/train_varnet_joint.py \
  --metadata_csv /projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv \
  --acceleration 4 \
  --epochs 2 \
  --num_cascades 12 \
  --chans 18 \
  --sens_chans 8 \
  --pools 4 \
  --sens_pools 4 \
  --batch_size 1 \
  --num_workers 2 \
  --learning_rate 0.0003 \
  --cross_fusion concat \
  --max_train_batches 5 \
  --max_val_batches 5 \
  --output_dir /projects/u6dm/fastmri_project/fastmri_pipeline/outputs/varnet_joint/joint_R4_smoke_c12_ch18_bs1 \
  --resume /projects/u6dm/fastmri_project/fastmri_pipeline/outputs/varnet_joint/joint_R4_smoke_c12_ch18_bs1/model_last.pt

echo "Finished short joint R4 resume test"
date
