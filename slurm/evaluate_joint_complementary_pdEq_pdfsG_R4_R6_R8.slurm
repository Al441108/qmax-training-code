#!/bin/bash
#SBATCH --job-name=evalJComp
#SBATCH --output=logs/evalJComp_%j.out
#SBATCH --error=logs/evalJComp_%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G

cd /projects/u6dm/fastmri_project/fastmri_pipeline

source ~/fastmri311/bin/activate 2>/dev/null || source venv311/bin/activate

mkdir -p logs
mkdir -p outputs/varnet_joint_complementary/eval_pdEq_pdfsG_R4_R6_R8

python scripts/evaluate_varnet_joint_complementary.py \
  --metadata_csv /projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv \
  --output_dir outputs/varnet_joint_complementary/eval_pdEq_pdfsG_R4_R6_R8 \
  --batch_size 4 \
  --num_workers 4 \
  --R4_checkpoint outputs/varnet_joint_complementary/pdEq_pdfsG_R4_c12_ch18_ep30_bs4_lr1e4/model_best.pt \
  --R6_checkpoint outputs/varnet_joint_complementary/pdEq_pdfsG_R6_c12_ch18_ep30_bs4_lr1e4/model_best.pt \
  --R8_checkpoint outputs/varnet_joint_complementary/pdEq_pdfsG_R8_c12_ch18_ep30_bs4_lr1e4/model_best.pt \
  --pd_mask_type equispaced \
  --pdfs_mask_type gaussian_vd
