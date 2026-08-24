#!/bin/bash
#SBATCH --job-name=eval_m1_m2u_robust_R8
#SBATCH --output=logs/evaluation/eval_m1_m2u_robust_R8_%j.out
#SBATCH --error=logs/evaluation/eval_m1_m2u_robust_R8_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G

set -euo pipefail

cd /projects/u6dm/fastmri_project/fastmri_pipeline

source ~/miniforge3/bin/activate
conda activate fastmri311

mkdir -p logs/evaluation
mkdir -p outputs/evaluation/m1_m2u_robustness_R8

python scripts/evaluate_m1_m2u_robustness_R8.py \
  --metadata_csv /lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv \
  --single_checkpoint outputs/varnet_single/pdfs_R8_c12_ch18_ep30_bs8_nw4/model_best.pt \
  --m1_checkpoint outputs/varnet_auxiliary/aux_pdR2_to_pdfs_R8_ep50_bs8_lr1e4/model_best.pt \
  --m2u_checkpoint outputs/m2u/m2u_pdR2_to_pdfs_R8_pilot5/model_best.pt \
  --output_dir outputs/evaluation/m1_m2u_robustness_R8 \
  --batch_size 2 \
  --num_workers 4 \
  --shift_pixels 2 4 8
