#!/usr/bin/env bash
#SBATCH --job-name=qmax_p2_safe
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/qmax_p2_safe_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/qmax_p2_safe_%j.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/projects/u6dm/fastmri_project/fastmri_pipeline
SCRATCH_ROOT=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline
METADATA=/lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv
FULL_CLEAN=${SCRATCH_ROOT}/outputs/m2_prnf/preflight_R8_bs4/full_clean_manifest.json
ROBUSTNESS=${SCRATCH_ROOT}/outputs/m2_prnf/preflight_R8_bs4/robustness_manifest.json
CONDITIONS=${SCRATCH_ROOT}/outputs/m2_prnf_fusion/preflight_R8_bs4/condition_manifest.json
HISTORICAL_P0=${SCRATCH_ROOT}/outputs/m2_prnf/formal_R8_bs4/quality_protected_hybrid_gain_seed42_ep50/model_last.pt
INIT_TEMPLATE=${SCRATCH_ROOT}/outputs/qmax_stage_a/random_init/random_init_seed42.pt
PREFLIGHT=${SCRATCH_ROOT}/outputs/qmax_stage_a/preflight_seed42/qmax_preflight.json
SMOKE=${SCRATCH_ROOT}/outputs/qmax_stage_a/smoke/qmax_full_seed42_epoch1/smoke_resume_audit.json
OUTPUT_DIR=${SCRATCH_ROOT}/outputs/qmax_stage_a/formal/qmax_full_seed42_epoch30

cd "${PROJECT_ROOT}"
mkdir -p "${SCRATCH_ROOT}/logs/qmax_stage_a"
test -f "${INIT_TEMPLATE}"
test -f "${PREFLIGHT}"
test -f "${HISTORICAL_P0}"
test -f "${SMOKE}"
python -c "import json; p=json.load(open('${SMOKE}')); assert p['passed'], p"

RESUME_ARGS=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  RESUME_ARGS=(--resume "${RESUME_CHECKPOINT}")
fi

python scripts/run_qmax_amp_safe.py scripts/train_qmax_stage_a.py \
  --metadata_csv "${METADATA}" \
  --output_dir "${OUTPUT_DIR}" \
  --init_template "${INIT_TEMPLATE}" \
  --full_clean_manifest "${FULL_CLEAN}" \
  --robustness_manifest "${ROBUSTNESS}" \
  --condition_manifest "${CONDITIONS}" \
  --preflight_json "${PREFLIGHT}" \
  --historical_p0_checkpoint "${HISTORICAL_P0}" \
  --qmax_variant qmax_full \
  --run_mode formal \
  --epochs 30 \
  --learning_rate 3e-4 \
  --batch_size 4 \
  --grad_accum_steps 1 \
  --num_workers "${SLURM_CPUS_PER_TASK:-4}" \
  --seed 42 \
  --amp \
  "${RESUME_ARGS[@]}"

test -f "${OUTPUT_DIR}/epoch30/model_last.pt"
echo "selection_checkpoint=${OUTPUT_DIR}/epoch30/model_last.pt"
