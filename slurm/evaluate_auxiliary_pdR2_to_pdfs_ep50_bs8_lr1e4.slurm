#!/bin/bash
#SBATCH --job-name=eval_aux_pdR2
#SBATCH --output=/projects/u6dm/fastmri_project/fastmri_pipeline/logs/joint_varnet/evaluate_aux_pdR2_ep50_%j.out
#SBATCH --error=/projects/u6dm/fastmri_project/fastmri_pipeline/logs/joint_varnet/evaluate_aux_pdR2_ep50_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G

set -euo pipefail

cd /projects/u6dm/fastmri_project/fastmri_pipeline

source ~/miniforge3/bin/activate
conda activate fastmri311

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Start AuxPDVarNet PD R=2 -> PD-FS ep50 formal evaluation"
date

python scripts/evaluate_varnet_auxiliary_pd_r2.py \
  --metadata_csv /projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv \
  --output_dir outputs/evaluation/auxiliary_pdR2_to_pdfs_val/ep50_bs8_lr1e4 \
  --batch_size 8 \
  --num_workers 4 \
  --pd_aux_acceleration 2 \
  --R4_checkpoint outputs/varnet_auxiliary/aux_pdR2_to_pdfs_R4_ep50_bs8_lr1e4/model_best.pt \
  --R6_checkpoint outputs/varnet_auxiliary/aux_pdR2_to_pdfs_R6_ep50_bs8_lr1e4/model_best.pt \
  --R8_checkpoint outputs/varnet_auxiliary/aux_pdR2_to_pdfs_R8_ep50_bs8_lr1e4/model_best.pt

echo "Finished AuxPDVarNet PD R=2 -> PD-FS ep50 formal evaluation"
date
