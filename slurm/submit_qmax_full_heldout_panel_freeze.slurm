#!/usr/bin/env bash
#SBATCH --job-name=qmax_test_freeze
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/qmax_test_freeze_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/qmax_test_freeze_%j.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/projects/u6dm/fastmri_project/fastmri_pipeline
SCRATCH=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline
METADATA=/lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv
LOCK=${PROJECT_ROOT}/MAIN_EXPERIMENTS_COMPLETE.lock
OUTPUT=${SCRATCH}/outputs/qmax_stage_a_multiseed/heldout_test/heldout_test_panel_manifest.json
SCRIPT=${PROJECT_ROOT}/scripts/freeze_qmax_full_heldout_panel.py

CKPT42=${SCRATCH}/outputs/qmax_stage_a/formal/qmax_full_seed42_epoch30/epoch60/model_last.pt
CKPT123=${SCRATCH}/outputs/qmax_stage_a_multiseed/formal/qmax_full_seed123_epoch30/epoch60/model_last.pt
CKPT2026=${SCRATCH}/outputs/qmax_stage_a_multiseed/formal/qmax_full_seed2026_epoch30/epoch60/model_last.pt

cd "${PROJECT_ROOT}"
mkdir -p "${SCRATCH}/logs" "$(dirname "${OUTPUT}")"
test -f "${LOCK}"
test -f "${METADATA}"
test -f "${SCRIPT}"
test ! -e "${OUTPUT}"

python -u "${SCRIPT}" \
  --main_lock "${LOCK}" \
  --metadata_csv "${METADATA}" \
  --model_seed 42 --checkpoint "${CKPT42}" \
  --model_seed 123 --checkpoint "${CKPT123}" \
  --model_seed 2026 --checkpoint "${CKPT2026}" \
  --output "${OUTPUT}"

test -f "${OUTPUT}"
python -c "import json; p=json.load(open('${OUTPUT}')); assert p['status']=='frozen'; assert p['num_patients']==34; print('held-out panel frozen:', p['num_patients'], 'patients,', p['num_slices'], 'slices')"

