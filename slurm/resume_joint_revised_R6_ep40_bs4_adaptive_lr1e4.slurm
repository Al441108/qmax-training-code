#!/bin/bash
#SBATCH --job-name=joint_rev_R6_ep40
#SBATCH --output=/projects/u6dm/fastmri_project/fastmri_pipeline/logs/joint_varnet/joint_revised_R6_ep40_adaptive_lr1e4_%j.out
#SBATCH --error=/projects/u6dm/fastmri_project/fastmri_pipeline/logs/joint_varnet/joint_revised_R6_ep40_adaptive_lr1e4_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G

set -euo pipefail

cd /projects/u6dm/fastmri_project/fastmri_pipeline

source ~/miniforge3/bin/activate
conda activate fastmri311

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Resume M1-revised adaptive-norm R=6 from 30 to 40 epochs"
date

python scripts/train_varnet_joint_revised.py \
  --metadata_csv /projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv \
  --acceleration 6 \
  --epochs 40 \
  --num_cascades 12 \
  --chans 18 \
  --sens_chans 8 \
  --pools 4 \
  --sens_pools 4 \
  --batch_size 4 \
  --num_workers 4 \
  --learning_rate 0.0001 \
  --cross_fusion concat \
  --output_dir /projects/u6dm/fastmri_project/fastmri_pipeline/outputs/varnet_joint_revised/joint_R6_jvn_adaptive_lr1e4_ep30_bs4 \
  --resume /projects/u6dm/fastmri_project/fastmri_pipeline/outputs/varnet_joint_revised/joint_R6_jvn_adaptive_lr1e4_ep30_bs4/model_last.pt

echo "Finished M1-revised adaptive-norm R=6 extended training to 40 epochs"
date
