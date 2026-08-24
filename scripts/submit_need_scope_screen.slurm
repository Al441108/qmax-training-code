#!/usr/bin/env bash
#SBATCH --job-name=need_scope_screen
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/pd_mechanism/need_scope_screen_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/pd_mechanism/need_scope_screen_%j.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/projects/u6dm/fastmri_project/fastmri_pipeline
METADATA_CSV=/lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv
: "${HYBRID_CHECKPOINT:?Export HYBRID_CHECKPOINT as the locked prnf_full model_best.pt before sbatch}"
OUTPUT_DIR=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/outputs/pd_mechanism/need_scope_inference_screen

cd "${PROJECT_ROOT}"
test -f "${HYBRID_CHECKPOINT}"
test -f scripts/screen_need_scope_inference.py
test -f scripts/analyze_pd_spatial_helpfulness.py
test -f scripts/evaluate_pd_oracle_stage2a.py
python -c "import torch; assert torch.cuda.is_available(), 'GPU is not visible'; print(torch.cuda.get_device_name(0))"
mkdir -p "${OUTPUT_DIR}"

python scripts/screen_need_scope_inference.py \
  --project-root "${PROJECT_ROOT}" \
  --metadata-csv "${METADATA_CSV}" \
  --checkpoint "${HYBRID_CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --split val \
  --num-workers "${SLURM_CPUS_PER_TASK:-4}" \
  --bootstrap-replicates 10000

echo "summary=${OUTPUT_DIR}/need_scope_screen_summary.json"
