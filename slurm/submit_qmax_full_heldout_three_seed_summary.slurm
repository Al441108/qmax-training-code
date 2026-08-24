#!/usr/bin/env bash
#SBATCH --job-name=qmax_test_summary
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/qmax_test_summary_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/qmax_test_summary_%j.err
#SBATCH --export=ALL

set -euo pipefail
PROJECT_ROOT=/projects/u6dm/fastmri_project/fastmri_pipeline
SCRATCH=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline
ROOT=${SCRATCH}/outputs/qmax_stage_a_multiseed/heldout_test
OUTPUT=${ROOT}/three_seed_summary
SCRIPT=${PROJECT_ROOT}/scripts/summarize_qmax_full_heldout_three_seed.py
cd "${PROJECT_ROOT}"
mkdir -p "${SCRATCH}/logs" "${OUTPUT}"
test ! -e "${OUTPUT}/heldout_three_seed_summary_audit.json"
python -u "${SCRIPT}" \
  --seed42_patient "${ROOT}/qmax_full_seed42_epoch60/qmax_full_seed42_epoch60_heldout_patient_metrics.csv" \
  --seed42_audit "${ROOT}/qmax_full_seed42_epoch60/qmax_full_seed42_epoch60_heldout_audit.json" \
  --seed123_patient "${ROOT}/qmax_full_seed123_epoch60/qmax_full_seed123_epoch60_heldout_patient_metrics.csv" \
  --seed123_audit "${ROOT}/qmax_full_seed123_epoch60/qmax_full_seed123_epoch60_heldout_audit.json" \
  --seed2026_patient "${ROOT}/qmax_full_seed2026_epoch60/qmax_full_seed2026_epoch60_heldout_patient_metrics.csv" \
  --seed2026_audit "${ROOT}/qmax_full_seed2026_epoch60/qmax_full_seed2026_epoch60_heldout_audit.json" \
  --output_dir "${OUTPUT}" --bootstrap_resamples 10000
test -f "${OUTPUT}/heldout_three_seed_summary_audit.json"

