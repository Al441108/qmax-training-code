#!/usr/bin/env bash
#SBATCH --job-name=six_slice_qual
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/six_slice_qual_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/six_slice_qual_%j.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/projects/u6dm/fastmri_project/fastmri_pipeline
SCRATCH_ROOT=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline
METADATA=/lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv
FULL_CLEAN=${SCRATCH_ROOT}/outputs/m2_prnf/preflight_R8_bs4/full_clean_manifest.json
ROBUSTNESS=${SCRATCH_ROOT}/outputs/m2_prnf/preflight_R8_bs4/robustness_manifest.json
CONDITIONS=${SCRATCH_ROOT}/outputs/m2_prnf_fusion/preflight_R8_bs4/condition_manifest.json

M2U_AUG=${SCRATCH_ROOT}/outputs/m2_prnf/formal_R8_bs4/m2u_augmented_seed42_ep50/model_best.pt
FIFTH=${SCRATCH_ROOT}/outputs/m2_prnf/formal_R8_bs4/quality_protected_hybrid_gain_seed42_ep50/model_best.pt
QMAX_FULL=${SCRATCH_ROOT}/outputs/qmax_stage_a/formal/qmax_full_seed42_epoch30/epoch60/model_last.pt
SINGLE_R8=${SCRATCH_ROOT}/outputs/varnet_single/pdfs_R8_c12_ch18_ep30_bs8_nw4/model_last.pt

SCRIPT=${PROJECT_ROOT}/scripts/render_six_slice_qualitative.py
OUTPUT_DIR=${SCRATCH_ROOT}/outputs/qmax_stage_a/six_slice_qualitative/qmax_full_epoch60

cd "${PROJECT_ROOT}"
mkdir -p "${SCRATCH_ROOT}/logs" "$(dirname "${OUTPUT_DIR}")"

for path in \
  "${METADATA}" "${FULL_CLEAN}" "${ROBUSTNESS}" "${CONDITIONS}" \
  "${M2U_AUG}" "${FIFTH}" "${QMAX_FULL}" "${SINGLE_R8}" "${SCRIPT}"; do
  test -f "${path}"
done

# Validation-only guard: never reuse the held-out test for figure selection.
case "${FULL_CLEAN}:${ROBUSTNESS}:${CONDITIONS}" in
  *heldout*|*held_out*|*test_manifest*)
    echo "Refusing a held-out-test input in the qualitative validation task." >&2
    exit 2
    ;;
esac

test ! -e "${OUTPUT_DIR}/six_slice_qualitative_audit.json"

export PYTHONHASHSEED=42
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

sha256sum "${SCRIPT}" "${M2U_AUG}" "${FIFTH}" "${QMAX_FULL}" "${SINGLE_R8}"

python -u "${SCRIPT}" \
  --metadata_csv "${METADATA}" \
  --full_clean_manifest "${FULL_CLEAN}" \
  --robustness_manifest "${ROBUSTNESS}" \
  --condition_manifest "${CONDITIONS}" \
  --m2u_augmented_checkpoint "${M2U_AUG}" \
  --quality_gain_checkpoint "${FIFTH}" \
  --qmax_full_checkpoint "${QMAX_FULL}" \
  --single_checkpoint "${SINGLE_R8}" \
  --single_num_cascades 12 \
  --single_chans 18 \
  --single_pools 4 \
  --single_sens_chans 8 \
  --single_sens_pools 4 \
  --output_dir "${OUTPUT_DIR}" \
  --seed 42

test -f "${OUTPUT_DIR}/six_slice_qualitative_audit.json"
test -f "${OUTPUT_DIR}/six_slice_panel_metrics.csv"
test -f "${OUTPUT_DIR}/group_contact_sheets/Group_A_three_slice_reconstruction_plate.png"
test -f "${OUTPUT_DIR}/group_contact_sheets/Group_A_three_slice_absolute_error_plate.png"
test -f "${OUTPUT_DIR}/group_contact_sheets/Group_B_three_slice_reconstruction_plate.png"
test -f "${OUTPUT_DIR}/group_contact_sheets/Group_B_three_slice_absolute_error_plate.png"

find "${OUTPUT_DIR}/slice_level_figures" -maxdepth 1 -name '*.png' | sort
PNG_COUNT=$(find "${OUTPUT_DIR}/slice_level_figures" -maxdepth 1 -name '*.png' | wc -l)
test "${PNG_COUNT}" -eq 12
python - "${OUTPUT_DIR}/six_slice_qualitative_audit.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
assert report["status"] == "passed", report
assert report["scope"] == "locked validation only; held-out test not accessed", report
assert report["num_distinct_slices"] == 6, report
PY
echo "slice_level_png_count=${PNG_COUNT}"
echo "output_dir=${OUTPUT_DIR}"
