#!/bin/bash
#SBATCH --job-name=m2u_alpha_A
#SBATCH --output=logs/alpha_ablation/stage_A_%A_%a.out
#SBATCH --error=logs/alpha_ablation/stage_A_%A_%a.err
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --array=0-4%3

set -euo pipefail

cd /projects/u6dm/fastmri_project/fastmri_pipeline

source ~/miniforge3/bin/activate
conda activate fastmri311

ALPHAS=("0.01" "0.03" "0.10" "0.20" "0.30")
ALPHA="${ALPHAS[$SLURM_ARRAY_TASK_ID]}"
ALPHA_TAG="${ALPHA/./p}"

OUTPUT_DIR="outputs/m2u_alpha_ablation/stage_A/alpha_${ALPHA_TAG}_seed42_ep5"

mkdir -p "$OUTPUT_DIR"
mkdir -p logs/alpha_ablation

echo "============================================================"
echo "Stage A alpha stability screen"
echo "alpha=${ALPHA}"
echo "seed=42"
echo "epochs=5"
echo "output=${OUTPUT_DIR}"
echo "host=$(hostname)"
date
nvidia-smi
echo "============================================================"

python scripts/train_varnet_m2u_pd_r2_optimized.py \
  --metadata_csv /lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv \
  --acceleration 8 \
  --pd_aux_acceleration 2 \
  --epochs 5 \
  --learning_rate 1e-4 \
  --num_cascades 12 \
  --chans 18 \
  --sens_chans 8 \
  --pools 4 \
  --sens_pools 4 \
  --initial_aux_alpha "$ALPHA" \
  --batch_size 8 \
  --num_workers 4 \
  --seed 42 \
  --output_dir "$OUTPUT_DIR"

echo "Finished Stage A alpha=${ALPHA}"
date
