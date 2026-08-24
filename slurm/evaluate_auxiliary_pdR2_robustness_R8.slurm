#!/bin/bash
#SBATCH --job-name=robust_aux_R8
#SBATCH --output=/projects/u6dm/fastmri_project/fastmri_pipeline/logs/joint_varnet/robust_aux_R8_%j.out
#SBATCH --error=/projects/u6dm/fastmri_project/fastmri_pipeline/logs/joint_varnet/robust_aux_R8_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G

set -euo pipefail

cd /projects/u6dm/fastmri_project/fastmri_pipeline

source ~/miniforge3/bin/activate
conda activate fastmri311

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Start R=8 robustness evaluation for PD R=2 auxiliary model"
date

python scripts/evaluate_auxiliary_pdR2_robustness_R8.py \
  --metadata_csv /projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv \
  --aux_checkpoint outputs/varnet_auxiliary/aux_pdR2_to_pdfs_R8_ep50_bs8_lr1e4/model_best.pt \
  --single_checkpoint outputs/varnet_single/pdfs_R8_c12_ch18_ep30_bs8_nw4/model_best.pt \
  --output_dir outputs/evaluation/auxiliary_pdR2_to_pdfs_val/robustness_R8 \
  --batch_size 8 \
  --num_workers 4 \
  --shift_pixels 2 4 8

echo "Finished R=8 robustness evaluation"
date
