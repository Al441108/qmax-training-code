#!/usr/bin/env bash
#SBATCH --job-name=qmax_ms_s123_pf
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/qmax_ms_s123_pf_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/qmax_ms_s123_pf_%j.err
#SBATCH --export=ALL

set -euo pipefail

SEED=123
PROJECT_ROOT=/projects/u6dm/fastmri_project/fastmri_pipeline
SCRATCH_ROOT=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline
METADATA=/lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv
FULL_CLEAN=${SCRATCH_ROOT}/outputs/m2_prnf/preflight_R8_bs4/full_clean_manifest.json
ROBUSTNESS=${SCRATCH_ROOT}/outputs/m2_prnf/preflight_R8_bs4/robustness_manifest.json
CONDITIONS=${SCRATCH_ROOT}/outputs/m2_prnf_fusion/preflight_R8_bs4/condition_manifest.json
HISTORICAL_P0=${SCRATCH_ROOT}/outputs/m2_prnf/formal_R8_bs4/quality_protected_hybrid_gain_seed42_ep50/model_last.pt
INIT_DIR=${SCRATCH_ROOT}/outputs/qmax_stage_a_multiseed/random_init
INIT_TEMPLATE=${INIT_DIR}/random_init_seed${SEED}.pt
OUTPUT_DIR=${SCRATCH_ROOT}/outputs/qmax_stage_a_multiseed/preflight_seed${SEED}

cd "${PROJECT_ROOT}"
mkdir -p "${INIT_DIR}" "${OUTPUT_DIR}"

for path in "${METADATA}" "${FULL_CLEAN}" "${ROBUSTNESS}" "${CONDITIONS}" "${HISTORICAL_P0}"; do
  test -f "${path}"
done
test -f scripts/generate_qmax_multiseed_init.py
test -f scripts/preflight_qmax_multiseed.py
test -f scripts/qmax_multiseed_common.py
test -f scripts/train_qmax_full_multiseed_epoch1to30.py
test -f scripts/train_qmax_full_multiseed_continue31to60.py
test -f scripts/run_qmax_multiseed_amp_safe.py
test -f QMAX_FULL_MULTISEED_PROTOCOL_R8.json

if [[ ! -f "${INIT_TEMPLATE}" ]]; then
  python scripts/generate_qmax_multiseed_init.py \
    --output "${INIT_TEMPLATE}" \
    --audit_json "${INIT_DIR}/random_init_seed${SEED}_audit.json" \
    --seed "${SEED}"
fi

python scripts/preflight_qmax_multiseed.py \
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
  --seed "${SEED}" \
  --amp

python -c "import json; p=json.load(open('${OUTPUT_DIR}/qmax_preflight.json')); assert p['status']=='passed', p"
echo "preflight=${OUTPUT_DIR}/qmax_preflight.json"
