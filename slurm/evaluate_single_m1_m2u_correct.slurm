#!/bin/bash
#SBATCH --job-name=eval_m2u_correct
#SBATCH --output=logs/evaluation/eval_m2u_correct_%j.out
#SBATCH --error=logs/evaluation/eval_m2u_correct_%j.err
#SBATCH --time=08:00:00
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
mkdir -p outputs/evaluation/m2u_correct_comparison

python scripts/evaluate_single_m1_m2u_correct.py \
  --metadata_csv /lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv \
  --output_dir outputs/evaluation/m2u_correct_comparison \
  --batch_size 2 \
  --num_workers 4 \
  --single_R4 outputs/varnet_single/pdfs_R4_c12_ch18_ep30_bs8/model_best.pt \
  --m1_R4 outputs/varnet_auxiliary/aux_pdR2_to_pdfs_R4_ep50_bs8_lr1e4/model_best.pt \
  --m2u_R4 outputs/m2u/m2u_pdR2_to_pdfs_R4_ep50/model_best.pt \
  --single_R6 outputs/varnet_single/pdfs_R6_c12_ch18_ep30_bs8_nw4/model_best.pt \
  --m1_R6 outputs/varnet_auxiliary/aux_pdR2_to_pdfs_R6_ep50_bs8_lr1e4/model_best.pt \
  --m2u_R6 outputs/m2u/m2u_pdR2_to_pdfs_R6_ep50/model_best.pt \
  --single_R8 outputs/varnet_single/pdfs_R8_c12_ch18_ep30_bs8_nw4/model_best.pt \
  --m1_R8 outputs/varnet_auxiliary/aux_pdR2_to_pdfs_R8_ep50_bs8_lr1e4/model_best.pt \
  --m2u_R8 outputs/m2u/m2u_pdR2_to_pdfs_R8_pilot5/model_best.pt
