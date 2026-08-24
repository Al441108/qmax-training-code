#!/usr/bin/env bash
#SBATCH --job-name=qmax_s123_test
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/qmax_s123_test_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/qmax_s123_test_%j.err
#SBATCH --export=ALL

set -euo pipefail
PROJECT_ROOT=/projects/u6dm/fastmri_project/fastmri_pipeline
SCRATCH=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline
METADATA=/lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv
MANIFEST=${SCRATCH}/outputs/qmax_stage_a_multiseed/heldout_test/heldout_test_panel_manifest.json
CHECKPOINT=${SCRATCH}/outputs/qmax_stage_a_multiseed/formal/qmax_full_seed123_epoch30/epoch60/model_last.pt
OUTPUT=${SCRATCH}/outputs/qmax_stage_a_multiseed/heldout_test/qmax_full_seed123_epoch60
SCRIPT=${PROJECT_ROOT}/scripts/evaluate_qmax_full_heldout_seed.py
cd "${PROJECT_ROOT}"
mkdir -p "${SCRATCH}/logs" "$(dirname "${OUTPUT}")"
test -f "${MANIFEST}"; test -f "${CHECKPOINT}"; test -f "${SCRIPT}"
test ! -e "${OUTPUT}/qmax_full_seed123_epoch60_heldout_audit.json"
export PYTHONHASHSEED=42 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
python -u "${SCRIPT}" --checkpoint "${CHECKPOINT}" --metadata_csv "${METADATA}" --test_manifest "${MANIFEST}" --output_dir "${OUTPUT}" --model_seed 123 --batch_size 2 --num_workers "${SLURM_CPUS_PER_TASK:-4}" --bootstrap_resamples 10000 --amp
test -f "${OUTPUT}/qmax_full_seed123_epoch60_heldout_audit.json"

