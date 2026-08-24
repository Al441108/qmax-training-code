#!/usr/bin/env bash
#SBATCH --job-name=pd_dropone_s42
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/pd_mechanism/pd_dropone_s42_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/pd_mechanism/pd_dropone_s42_%j.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/projects/u6dm/fastmri_project/fastmri_pipeline
METADATA_CSV=/lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv
CHECKPOINT=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/outputs/m2_prnf/formal_R8_bs4/global_direct_prnf_no_need_seed42_ep50/model_best.pt
OUTPUT_DIR=${DROPONE_OUTPUT_DIR:-/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/outputs/pd_mechanism/dropone_global_direct_seed42}
INTERVENTIONS=${INTERVENTIONS:-cell,cascade,scale}
EXTRA_ARGS=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  EXTRA_ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

cd "${PROJECT_ROOT}"
mkdir -p logs/pd_mechanism "${OUTPUT_DIR}"

echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "host=$(hostname)"
echo "checkpoint=${CHECKPOINT}"
echo "output_dir=${OUTPUT_DIR}"

test -f "${METADATA_CSV}"
test -f "${CHECKPOINT}"
test -f scripts/evaluate_pd_oracle_stage2a.py
test -f scripts/evaluate_pd_dropone.py
python -c "import torch; assert torch.cuda.is_available(), 'GPU is not visible'; print(torch.cuda.get_device_name(0))"

python scripts/evaluate_pd_dropone.py \
  --project-root "${PROJECT_ROOT}" \
  --metadata-csv "${METADATA_CSV}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --split val \
  --num-workers "${SLURM_CPUS_PER_TASK:-4}" \
  --bootstrap-replicates 10000 \
  --interventions "${INTERVENTIONS}" \
  "${EXTRA_ARGS[@]}"

echo "completed_at=$(date --iso-8601=seconds)"
echo "summary=${OUTPUT_DIR}/dropone_manifest.json"
