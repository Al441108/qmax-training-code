#!/bin/bash
#SBATCH --job-name=plot_best_aux
#SBATCH --output=/projects/u6dm/fastmri_project/fastmri_pipeline/logs/joint_varnet/plot_best_aux_%j.out
#SBATCH --error=/projects/u6dm/fastmri_project/fastmri_pipeline/logs/joint_varnet/plot_best_aux_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -euo pipefail

cd /projects/u6dm/fastmri_project/fastmri_pipeline

source ~/miniforge3/bin/activate
conda activate fastmri311

python scripts/plot_best_auxiliary_examples_R4_R6_R8.py
