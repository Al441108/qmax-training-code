#!/bin/bash
#SBATCH --job-name=eval_joint_val
#SBATCH --output=/projects/u6dm/fastmri_project/fastmri_pipeline/logs/joint_varnet/eval_joint_val_%j.out
#SBATCH --error=/projects/u6dm/fastmri_project/fastmri_pipeline/logs/joint_varnet/eval_joint_val_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G

set -euo pipefail

cd /projects/u6dm/fastmri_project/fastmri_pipeline

source ~/miniforge3/bin/activate
conda activate fastmri311

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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

echo "Start joint VarNet formal validation evaluation"
date

python scripts/evaluate_varnet_joint.py \
  --metadata_csv /projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv \
  --batch_size 4 \
  --num_workers 4 \
  --output_dir /projects/u6dm/fastmri_project/fastmri_pipeline/outputs/evaluation/joint_varnet_val \
  --R4_checkpoint /projects/u6dm/fastmri_project/fastmri_pipeline/outputs/varnet_joint/joint_R4_c12_ch18_ep30_bs4/model_best.pt \
  --R6_checkpoint /projects/u6dm/fastmri_project/fastmri_pipeline/outputs/varnet_joint/joint_R6_c12_ch18_ep30_bs4/model_best.pt \
  --R8_checkpoint /projects/u6dm/fastmri_project/fastmri_pipeline/outputs/varnet_joint/joint_R8_c12_ch18_ep30_bs4/model_best.pt

echo "Finished joint VarNet formal validation evaluation"
date
