#!/bin/bash
#SBATCH --job-name=m2u_smoke50
#SBATCH --output=logs/m2u/m2u_smoke50_R8_%j.out
#SBATCH --error=logs/m2u/m2u_smoke50_R8_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

set -euo pipefail

cd /projects/u6dm/fastmri_project/fastmri_pipeline

source ~/miniforge3/bin/activate
conda activate fastmri311

mkdir -p logs/m2u
mkdir -p outputs/m2u/smoke50_R8_optimized

echo "Start M2-U 50-batch GPU smoke test"
hostname
date
nvidia-smi

python scripts/train_varnet_m2u_pd_r2_optimized.py \
  --metadata_csv /lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv \
  --acceleration 8 \
  --pd_aux_acceleration 2 \
  --num_train_patients 10 \
  --num_val_patients 5 \
  --epochs 1 \
  --learning_rate 1e-4 \
  --num_cascades 12 \
  --chans 18 \
  --sens_chans 8 \
  --pools 4 \
  --sens_pools 4 \
  --initial_aux_alpha 0.1 \
  --batch_size 2 \
  --num_workers 4 \
  --max_train_batches 50 \
  --max_val_batches 10 \
  --output_dir outputs/m2u/smoke50_R8_optimized

echo "Finished M2-U 50-batch GPU smoke test"
date