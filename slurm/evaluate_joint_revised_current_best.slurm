#!/bin/bash
#SBATCH --job-name=eval_joint_rev
#SBATCH --output=/projects/u6dm/fastmri_project/fastmri_pipeline/logs/joint_varnet/evaluate_joint_revised_current_best_%j.out
#SBATCH --error=/projects/u6dm/fastmri_project/fastmri_pipeline/logs/joint_varnet/evaluate_joint_revised_current_best_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G

set -euo pipefail

cd /projects/u6dm/fastmri_project/fastmri_pipeline

source ~/miniforge3/bin/activate
conda activate fastmri311

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Start revised JointVarNet current-best formal evaluation"
date

python scripts/evaluate_varnet_joint_revised.py \
  --metadata_csv /projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv \
  --output_dir outputs/evaluation/joint_revised_varnet_val/current_best \
  --batch_size 4 \
  --num_workers 4 \
  --R4_checkpoint outputs/varnet_joint_revised/joint_R4_jvn_adaptive_lr1e4_pilot_ep5_bs4/model_best.pt \
  --R6_checkpoint outputs/varnet_joint_revised/joint_R6_jvn_adaptive_lr1e4_ep30_bs4/model_best.pt \
  --R8_checkpoint outputs/varnet_joint_revised/joint_R8_jvn_adaptive_lr1e4_ep30_bs4/model_best.pt

echo "Finished revised JointVarNet current-best formal evaluation"
date
