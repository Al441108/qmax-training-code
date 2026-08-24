#!/bin/bash
#SBATCH --job-name=debug_joint_R4_central
#SBATCH --output=/projects/u6dm/fastmri_project/fastmri_pipeline/logs/joint_varnet/debug_joint_R4_central_%j.out
#SBATCH --error=/projects/u6dm/fastmri_project/fastmri_pipeline/logs/joint_varnet/debug_joint_R4_central_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G

set -euo pipefail

cd /projects/u6dm/fastmri_project/fastmri_pipeline

source ~/miniforge3/bin/activate
conda activate fastmri311

python scripts/debug_joint_eval_one_batch.py \
  --metadata_csv /projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv \
  --acceleration 4 \
  --checkpoint /projects/u6dm/fastmri_project/fastmri_pipeline/outputs/varnet_joint/joint_R4_c12_ch18_ep30_bs4/model_best.pt \
  --out_dir /projects/u6dm/fastmri_project/fastmri_pipeline/outputs/debug/joint_eval_R4_central \
  --dataset_index 15 \
  --batch_size 1 \
  --num_workers 0
