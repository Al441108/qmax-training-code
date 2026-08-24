#!/bin/bash
#SBATCH --job-name=aux_pdR2_R8_smoke
#SBATCH --output=/projects/u6dm/fastmri_project/fastmri_pipeline/logs/joint_varnet/aux_pdR2_R8_smoke_%j.out
#SBATCH --error=/projects/u6dm/fastmri_project/fastmri_pipeline/logs/joint_varnet/aux_pdR2_R8_smoke_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G

set -euo pipefail

cd /projects/u6dm/fastmri_project/fastmri_pipeline

source ~/miniforge3/bin/activate
conda activate fastmri311

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Start AuxPDVarNet PD R=2 -> PD-FS R=8 smoke test"
date

python scripts/train_varnet_auxiliary_pd_r2.py \
  --metadata_csv /projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv \
  --acceleration 8 \
  --pd_aux_acceleration 2 \
  --epochs 1 \
  --num_cascades 12 \
  --chans 18 \
  --sens_chans 8 \
  --pools 4 \
  --sens_pools 4 \
  --batch_size 8 \
  --num_workers 4 \
  --learning_rate 0.0001 \
  --max_train_batches 30 \
  --max_val_batches 10 \
  --output_dir outputs/varnet_auxiliary/aux_pdR2_to_pdfs_R8_smoke_bs8_lr1e4

echo "Finished AuxPDVarNet PD R=2 -> PD-FS R=8 smoke test"
date
