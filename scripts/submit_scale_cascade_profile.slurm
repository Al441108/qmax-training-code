#!/usr/bin/env bash
#SBATCH --job-name=pd_cascade_s42
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/pd_mechanism/pd_cascade_s42_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/pd_mechanism/pd_cascade_s42_%j.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/projects/u6dm/fastmri_project/fastmri_pipeline
METADATA_CSV=/lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv
CHECKPOINT=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/outputs/m2_prnf/formal_R8_bs4/global_direct_prnf_no_need_seed42_ep50/model_best.pt
OUTPUT_DIR=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/outputs/pd_mechanism/scale_cascade_profile_seed42

cd "${PROJECT_ROOT}"
test -f "${CHECKPOINT}"
test -f scripts/profile_pd_scale_cascade.py
test -f scripts/evaluate_pd_oracle_stage2a.py
python -c "import torch; assert torch.cuda.is_available(), 'GPU is not visible'; print(torch.cuda.get_device_name(0))"
mkdir -p "${OUTPUT_DIR}"

python scripts/profile_pd_scale_cascade.py \
  --project-root "${PROJECT_ROOT}" \
  --metadata-csv "${METADATA_CSV}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --split val \
  --num-workers "${SLURM_CPUS_PER_TASK:-4}"

echo "summary=${OUTPUT_DIR}/profile_manifest.json"
