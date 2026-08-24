#!/bin/bash
#SBATCH --job-name=aux_pdR2_R4_ep50
#SBATCH --output=/projects/u6dm/fastmri_project/fastmri_pipeline/logs/joint_varnet/aux_pdR2_R4_ep50_%j.out
#SBATCH --error=/projects/u6dm/fastmri_project/fastmri_pipeline/logs/joint_varnet/aux_pdR2_R4_ep50_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G

set -euo pipefail

cd /projects/u6dm/fastmri_project/fastmri_pipeline

source ~/miniforge3/bin/activate
conda activate fastmri311

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Start AuxPDVarNet PD R=2 -> PD-FS R=4 full training"
date

python scripts/train_varnet_auxiliary_pd_r2.py \
  --metadata_csv /projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv \
  --acceleration 4 \
  --pd_aux_acceleration 2 \
  --epochs 50 \
  --num_cascades 12 \
  --chans 18 \
  --sens_chans 8 \
  --pools 4 \
  --sens_pools 4 \
  --batch_size 8 \
  --num_workers 4 \
  --learning_rate 0.0001 \
  --output_dir outputs/varnet_auxiliary/aux_pdR2_to_pdfs_R4_ep50_bs8_lr1e4

echo "Finished AuxPDVarNet PD R=2 -> PD-FS R=4 full training"
date
