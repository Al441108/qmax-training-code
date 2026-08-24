#!/bin/bash
#SBATCH --job-name=m2u_R6_ep50
#SBATCH --output=logs/m2u/m2u_R6_ep50_%j.out
#SBATCH --error=logs/m2u/m2u_R6_ep50_%j.err
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

mkdir -p outputs/m2u/m2u_pdR2_to_pdfs_R6_ep50

echo "============================================================"
echo "Start M2-U R=6 formal 50-epoch training"
echo "Host: $(hostname)"
date
nvidia-smi
echo "============================================================"

python scripts/train_varnet_m2u_pd_r2_optimized.py \
  --metadata_csv /lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv \
  --acceleration 6 \
  --pd_aux_acceleration 2 \
  --epochs 50 \
  --learning_rate 1e-4 \
  --num_cascades 12 \
  --chans 18 \
  --sens_chans 8 \
  --pools 4 \
  --sens_pools 4 \
  --initial_aux_alpha 0.1 \
  --batch_size 8 \
  --num_workers 4 \
  --output_dir outputs/m2u/m2u_pdR2_to_pdfs_R6_ep50

echo "============================================================"
echo "Finished M2-U R=6"
date
echo "============================================================"
