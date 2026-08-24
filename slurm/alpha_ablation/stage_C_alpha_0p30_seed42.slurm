#!/bin/bash
#SBATCH --job-name=m2u_a030_ep50
#SBATCH --output=logs/alpha_ablation/stage_C_a030_%j.out
#SBATCH --error=logs/alpha_ablation/stage_C_a030_%j.err
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

OUTPUT_DIR="outputs/m2u_alpha_ablation/stage_C/alpha_0p30_seed42_ep50"

mkdir -p "$OUTPUT_DIR"
mkdir -p logs/alpha_ablation

echo "============================================================"
echo "Stage C full-budget alpha confirmation"
echo "initial alpha=0.30"
echo "seed=42"
echo "epochs=50"
echo "output=${OUTPUT_DIR}"
echo "host=$(hostname)"
date
nvidia-smi
echo "============================================================"

python scripts/train_varnet_m2u_pd_r2_optimized.py \
  --metadata_csv /lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv \
  --acceleration 8 \
  --pd_aux_acceleration 2 \
  --epochs 50 \
  --learning_rate 1e-4 \
  --num_cascades 12 \
  --chans 18 \
  --sens_chans 8 \
  --pools 4 \
  --sens_pools 4 \
  --initial_aux_alpha 0.30 \
  --batch_size 8 \
  --num_workers 4 \
  --seed 42 \
  --output_dir "$OUTPUT_DIR"

echo "Finished Stage C alpha=0.30"
date
