#!/bin/bash
#SBATCH --job-name=alpha_sweep_R8
#SBATCH --output=/projects/u6dm/fastmri_project/fastmri_pipeline/logs/joint_varnet/alpha_sweep_R8_%j.out
#SBATCH --error=/projects/u6dm/fastmri_project/fastmri_pipeline/logs/joint_varnet/alpha_sweep_R8_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G

set -euo pipefail

cd /projects/u6dm/fastmri_project/fastmri_pipeline

source ~/miniforge3/bin/activate
conda activate fastmri311

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs/joint_varnet
mkdir -p outputs/evaluation/alpha_sweep_R8

echo "Start R=8 prediction-level alpha sweep"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not_set}"
date

nvidia-smi

python scripts/evaluate_alpha_sweep_R8.py \
  --metadata_csv /projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv \
  --aux_checkpoint outputs/varnet_auxiliary/aux_pdR2_to_pdfs_R8_ep50_bs8_lr1e4/model_best.pt \
  --single_checkpoint outputs/varnet_single/pdfs_R8_c12_ch18_ep30_bs8_nw4/model_best.pt \
  --output_dir outputs/evaluation/alpha_sweep_R8 \
  --batch_size 8 \
  --num_workers 4 \
  --shift_pixels 2 4 8

echo "Finished R=8 prediction-level alpha sweep"
date