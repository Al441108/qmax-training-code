#!/bin/bash
#SBATCH --job-name=jComp_R6
#SBATCH --output=logs/jComp_R6_%j.out
#SBATCH --error=logs/jComp_R6_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G

cd /projects/u6dm/fastmri_project/fastmri_pipeline
source ~/fastmri311/bin/activate 2>/dev/null || source venv311/bin/activate

python scripts/train_varnet_joint_complementary.py \
  --metadata_csv /projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv \
  --acceleration 6 \
  --pd_mask_type equispaced \
  --pdfs_mask_type gaussian_vd \
  --epochs 30 \
  --batch_size 4 \
  --num_cascades 12 \
  --chans 18 \
  --sens_chans 8 \
  --learning_rate 1e-4 \
  --num_workers 4 \
  --output_dir outputs/varnet_joint_complementary/pdEq_pdfsG_R6_c12_ch18_ep30_bs4_lr1e4
