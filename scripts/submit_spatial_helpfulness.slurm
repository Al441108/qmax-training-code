#!/usr/bin/env bash
#SBATCH --job-name=pd_spatialmask_s42
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/pd_mechanism/pd_spatialmask_s42_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/pd_mechanism/pd_spatialmask_s42_%j.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/projects/u6dm/fastmri_project/fastmri_pipeline
METADATA_CSV=/lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv
CHECKPOINT=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/outputs/m2_prnf/formal_R8_bs4/global_direct_prnf_no_need_seed42_ep50/model_best.pt
OUTPUT_DIR=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/outputs/pd_mechanism/spatial_mask_sensitivity_seed42

cd "${PROJECT_ROOT}"

echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "host=$(hostname)"
echo "python=$(command -v python)"
echo "conda_env=${CONDA_DEFAULT_ENV:-unset}"
echo "checkpoint=${CHECKPOINT}"
echo "output_dir=${OUTPUT_DIR}"

test -f "${METADATA_CSV}"
test -f "${CHECKPOINT}"
test -f "${PROJECT_ROOT}/scripts/evaluate_pd_oracle_stage2a.py"
test -f "${PROJECT_ROOT}/scripts/analyze_pd_spatial_helpfulness.py"

python -c "import torch; print('torch=', torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); assert torch.cuda.is_available(), 'GPU is not visible'"

mkdir -p "${OUTPUT_DIR}"

python scripts/analyze_pd_spatial_helpfulness.py \
  --project-root "${PROJECT_ROOT}" \
  --metadata-csv "${METADATA_CSV}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --split val \
  --num-workers "${SLURM_CPUS_PER_TASK:-4}" \
  --foreground-fractions 0.01,0.05,0.10 \
  --primary-foreground-fraction 0.05 \
  --active-error-quantile 0.50 \
  --high-error-quantile 0.75 \
  --edge-quantile 0.75

echo "completed_at=$(date --iso-8601=seconds)"
echo "summary=${OUTPUT_DIR}/spatial_summary.json"
