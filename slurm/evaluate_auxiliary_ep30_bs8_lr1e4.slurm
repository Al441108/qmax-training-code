#!/bin/bash
#SBATCH --job-name=eval_aux_b1
#SBATCH --output=/projects/u6dm/fastmri_project/fastmri_pipeline/logs/joint_varnet/evaluate_auxiliary_ep30_bs8_%j.out
#SBATCH --error=/projects/u6dm/fastmri_project/fastmri_pipeline/logs/joint_varnet/evaluate_auxiliary_ep30_bs8_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G

set -euo pipefail

cd /projects/u6dm/fastmri_project/fastmri_pipeline

source ~/miniforge3/bin/activate
conda activate fastmri311

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Start B1 AuxPDVarNet formal validation evaluation"
date

python scripts/evaluate_varnet_auxiliary.py \
  --metadata_csv /projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv \
  --output_dir outputs/evaluation/auxiliary_varnet_val/ep30_bs8_lr1e4 \
  --batch_size 8 \
  --num_workers 4 \
  --R4_checkpoint outputs/varnet_auxiliary/aux_pdfs_R4_ep30_bs8_lr1e4/model_best.pt \
  --R6_checkpoint outputs/varnet_auxiliary/aux_pdfs_R6_ep30_bs8_lr1e4/model_best.pt \
  --R8_checkpoint outputs/varnet_auxiliary/aux_pdfs_R8_ep30_bs8_lr1e4/model_best.pt

echo "Finished B1 AuxPDVarNet formal validation evaluation"
date
