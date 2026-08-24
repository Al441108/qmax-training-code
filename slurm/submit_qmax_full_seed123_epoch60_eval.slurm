#!/usr/bin/env bash
#SBATCH --job-name=qmax_s123_e60_eval
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/qmax_s123_e60_eval_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/qmax_s123_e60_eval_%j.err
#SBATCH --export=ALL

set -euo pipefail

MODEL_SEED=123
PROJECT_ROOT=/projects/u6dm/fastmri_project/fastmri_pipeline
SCRATCH_ROOT=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline
METADATA=/lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv
FULL_CLEAN=${SCRATCH_ROOT}/outputs/m2_prnf/preflight_R8_bs4/full_clean_manifest.json
ROBUSTNESS=${SCRATCH_ROOT}/outputs/m2_prnf/preflight_R8_bs4/robustness_manifest.json
CONDITIONS=${SCRATCH_ROOT}/outputs/m2_prnf_fusion/preflight_R8_bs4/condition_manifest.json
CHECKPOINT=${SCRATCH_ROOT}/outputs/qmax_stage_a_multiseed/formal/qmax_full_seed${MODEL_SEED}_epoch30/epoch60/model_last.pt
OUTPUT_DIR=${SCRATCH_ROOT}/outputs/qmax_stage_a_multiseed/final_validation/qmax_full_seed${MODEL_SEED}_epoch60
EVALUATOR=${PROJECT_ROOT}/scripts/evaluate_qmax_full_replication_epoch60.py

cd "${PROJECT_ROOT}"
mkdir -p "${SCRATCH_ROOT}/logs" "$(dirname "${OUTPUT_DIR}")"

for path in "${CHECKPOINT}" "${METADATA}" "${FULL_CLEAN}" "${ROBUSTNESS}" "${CONDITIONS}" "${EVALUATOR}"; do
  if [[ ! -f "${path}" ]]; then
    echo "missing_required_file=${path}" >&2
    exit 1
  fi
done
test -f scripts/evaluate_qmax_counterfactuals.py
test -f scripts/compare_qmax_stage_a.py
test -f scripts/qmax_multiseed_common.py
test ! -e "${OUTPUT_DIR}/qmax_full_seed${MODEL_SEED}_epoch60_evaluation_audit.json"

export PYTHONHASHSEED=42
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

python -u "${EVALUATOR}" \
  --checkpoint "${CHECKPOINT}" \
  --metadata_csv "${METADATA}" \
  --full_clean_manifest "${FULL_CLEAN}" \
  --robustness_manifest "${ROBUSTNESS}" \
  --condition_manifest "${CONDITIONS}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_seed "${MODEL_SEED}" \
  --batch_size 2 \
  --num_workers "${SLURM_CPUS_PER_TASK:-4}" \
  --bootstrap_resamples 10000 \
  --amp

python -c "import json; p=json.load(open('${OUTPUT_DIR}/qmax_full_seed${MODEL_SEED}_epoch60_evaluation_audit.json')); assert p['status']=='passed', p"
echo "evaluation_output=${OUTPUT_DIR}"
