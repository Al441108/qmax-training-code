#!/usr/bin/env bash
#SBATCH --job-name=qmax_p1_smoke
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/qmax_p1_smoke_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/qmax_p1_smoke_%j.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/projects/u6dm/fastmri_project/fastmri_pipeline
SCRATCH_ROOT=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline
METADATA=/lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv
FULL_CLEAN=${SCRATCH_ROOT}/outputs/m2_prnf/preflight_R8_bs4/full_clean_manifest.json
ROBUSTNESS=${SCRATCH_ROOT}/outputs/m2_prnf/preflight_R8_bs4/robustness_manifest.json
CONDITIONS=${SCRATCH_ROOT}/outputs/m2_prnf_fusion/preflight_R8_bs4/condition_manifest.json
HISTORICAL_P0=${SCRATCH_ROOT}/outputs/m2_prnf/formal_R8_bs4/quality_protected_hybrid_gain_seed42_ep50/model_last.pt
PREFLIGHT=${SCRATCH_ROOT}/outputs/qmax_stage_a/preflight_seed42/qmax_preflight.json
INIT_TEMPLATE=${SCRATCH_ROOT}/outputs/qmax_stage_a/random_init/random_init_seed42.pt
OUTPUT_DIR=${SCRATCH_ROOT}/outputs/qmax_stage_a/smoke/qmax_core_seed42_epoch1

cd "${PROJECT_ROOT}"
mkdir -p "${SCRATCH_ROOT}/logs/qmax_stage_a"
test -f "${INIT_TEMPLATE}"
test -f "${PREFLIGHT}"
test -f "${HISTORICAL_P0}"
python -c "import json; p=json.load(open('${PREFLIGHT}')); assert p['status']=='passed', p"

python scripts/train_qmax_stage_a.py \
  --metadata_csv "${METADATA}" \
  --output_dir "${OUTPUT_DIR}" \
  --init_template "${INIT_TEMPLATE}" \
  --full_clean_manifest "${FULL_CLEAN}" \
  --robustness_manifest "${ROBUSTNESS}" \
  --condition_manifest "${CONDITIONS}" \
  --preflight_json "${PREFLIGHT}" \
  --historical_p0_checkpoint "${HISTORICAL_P0}" \
  --qmax_variant qmax_core \
  --run_mode smoke \
  --epochs 1 \
  --learning_rate 3e-4 \
  --batch_size 4 \
  --grad_accum_steps 1 \
  --num_workers "${SLURM_CPUS_PER_TASK:-4}" \
  --seed 42 \
  --amp

python -c "import json; p=json.load(open('${OUTPUT_DIR}/smoke_resume_audit.json')); assert p['passed'], p"
echo "smoke=${OUTPUT_DIR}/smoke_resume_audit.json"
