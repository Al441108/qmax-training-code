#!/usr/bin/env bash
#SBATCH --job-name=qmax_preflight_s42
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/qmax_preflight_s42_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/qmax_preflight_s42_%j.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/projects/u6dm/fastmri_project/fastmri_pipeline
SCRATCH_ROOT=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline
METADATA=/lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv
FULL_CLEAN=${SCRATCH_ROOT}/outputs/m2_prnf/preflight_R8_bs4/full_clean_manifest.json
ROBUSTNESS=${SCRATCH_ROOT}/outputs/m2_prnf/preflight_R8_bs4/robustness_manifest.json
CONDITIONS=${SCRATCH_ROOT}/outputs/m2_prnf_fusion/preflight_R8_bs4/condition_manifest.json
HISTORICAL_P0=${SCRATCH_ROOT}/outputs/m2_prnf/formal_R8_bs4/quality_protected_hybrid_gain_seed42_ep50/model_last.pt
OUTPUT_DIR=${SCRATCH_ROOT}/outputs/qmax_stage_a/preflight_seed42
INIT_DIR=${SCRATCH_ROOT}/outputs/qmax_stage_a/random_init
INIT_TEMPLATE=${INIT_DIR}/random_init_seed42.pt

cd "${PROJECT_ROOT}"
mkdir -p "${SCRATCH_ROOT}/logs/qmax_stage_a" "${OUTPUT_DIR}" "${INIT_DIR}"

test -f "${METADATA}"
test -f "${FULL_CLEAN}"
test -f "${ROBUSTNESS}"
test -f "${CONDITIONS}"
test -f "${HISTORICAL_P0}"
test -f scripts/generate_qmax_random_init.py
test -f scripts/preflight_qmax_stage_a.py
test -f src/m2_prnf_qmax_varnet.py
test -f QMAX_STAGE_A_PROTOCOL_R8.json

python -c "import torch, fastmri; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"

if [[ ! -f "${INIT_TEMPLATE}" ]]; then
  python scripts/generate_qmax_random_init.py \
    --output "${INIT_TEMPLATE}" \
    --audit_json "${INIT_DIR}/random_init_seed42_audit.json" \
    --seed 42
fi

python scripts/preflight_qmax_stage_a.py \
  --metadata_csv "${METADATA}" \
  --init_template "${INIT_TEMPLATE}" \
  --full_clean_manifest "${FULL_CLEAN}" \
  --robustness_manifest "${ROBUSTNESS}" \
  --condition_manifest "${CONDITIONS}" \
  --historical_p0_checkpoint "${HISTORICAL_P0}" \
  --output_json "${OUTPUT_DIR}/qmax_preflight.json" \
  --corruption_manifest_json "${OUTPUT_DIR}/qmax_preflight_corruption_manifest.json" \
  --formal_batch_size 4 \
  --formal_max_gpu_memory_gb 110 \
  --num_workers "${SLURM_CPUS_PER_TASK:-4}" \
  --seed 42 \
  --amp

python -c "import json; p=json.load(open('${OUTPUT_DIR}/qmax_preflight.json')); assert p['status']=='passed', p"
echo "preflight=${OUTPUT_DIR}/qmax_preflight.json"
