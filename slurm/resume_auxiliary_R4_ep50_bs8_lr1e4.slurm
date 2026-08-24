#!/bin/bash
#SBATCH --job-name=aux_R4_ep50
#SBATCH --output=/projects/u6dm/fastmri_project/fastmri_pipeline/logs/joint_varnet/auxiliary_R4_ep50_%j.out
#SBATCH --error=/projects/u6dm/fastmri_project/fastmri_pipeline/logs/joint_varnet/auxiliary_R4_ep50_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G

set -euo pipefail

cd /projects/u6dm/fastmri_project/fastmri_pipeline
source ~/miniforge3/bin/activate
conda activate fastmri311

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/train_varnet_auxiliary.py \
  --metadata_csv /projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv \
  --acceleration 4 \
  --epochs 50 \
  --num_cascades 12 \
  --chans 18 \
  --sens_chans 8 \
  --pools 4 \
  --sens_pools 4 \
  --batch_size 8 \
  --num_workers 4 \
  --learning_rate 0.0001 \
  --output_dir outputs/varnet_auxiliary/aux_pdfs_R4_ep30_bs8_lr1e4 \
  --resume outputs/varnet_auxiliary/aux_pdfs_R4_ep30_bs8_lr1e4/model_last.pt
