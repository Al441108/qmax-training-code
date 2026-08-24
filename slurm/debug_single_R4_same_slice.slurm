#!/bin/bash
#SBATCH --job-name=debug_single_R4
#SBATCH --output=/projects/u6dm/fastmri_project/fastmri_pipeline/logs/joint_varnet/debug_single_R4_%j.out
#SBATCH --error=/projects/u6dm/fastmri_project/fastmri_pipeline/logs/joint_varnet/debug_single_R4_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G

set -euo pipefail

cd /projects/u6dm/fastmri_project/fastmri_pipeline

source ~/miniforge3/bin/activate
conda activate fastmri311

METADATA=/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv
OUTDIR=/projects/u6dm/fastmri_project/fastmri_pipeline/outputs/debug/single_eval_R4_same_slice

python scripts/debug_single_eval_one_slice.py \
  --metadata_csv $METADATA \
  --acceleration 4 \
  --contrast PD \
  --checkpoint /projects/u6dm/fastmri_project/fastmri_pipeline/outputs/varnet_single/pd_R4_c12_ch18_ep30_bs8/model_best.pt \
  --dataset_index 15 \
  --out_dir $OUTDIR

python scripts/debug_single_eval_one_slice.py \
  --metadata_csv $METADATA \
  --acceleration 4 \
  --contrast PD-FS \
  --checkpoint /projects/u6dm/fastmri_project/fastmri_pipeline/outputs/varnet_single/pdfs_R4_c12_ch18_ep30_bs8/model_best.pt \
  --dataset_index 15 \
  --out_dir $OUTDIR
