#!/usr/bin/env bash
#SBATCH --job-name=qmax_val_failfig
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/qmax_val_failfig_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/qmax_val_failfig_%j.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/projects/u6dm/fastmri_project/fastmri_pipeline
SCRATCH_ROOT=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline
METADATA=/lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv
FULL_CLEAN=${SCRATCH_ROOT}/outputs/m2_prnf/preflight_R8_bs4/full_clean_manifest.json
ROBUSTNESS=${SCRATCH_ROOT}/outputs/m2_prnf/preflight_R8_bs4/robustness_manifest.json
CONDITIONS=${SCRATCH_ROOT}/outputs/m2_prnf_fusion/preflight_R8_bs4/condition_manifest.json

VALIDATION_DIR=${SCRATCH_ROOT}/outputs/qmax_stage_a/final_validation/qmax_full_seed42_epoch60
VALIDATION_SLICE=${VALIDATION_DIR}/stagea_full_epoch60_slice_metrics.csv
VALIDATION_AUDIT=${VALIDATION_DIR}/stagea_full_epoch60_validation_audit.json

M2U_AUG=${SCRATCH_ROOT}/outputs/m2_prnf/formal_R8_bs4/m2u_augmented_seed42_ep50/model_best.pt
FIFTH=${SCRATCH_ROOT}/outputs/m2_prnf/formal_R8_bs4/quality_protected_hybrid_gain_seed42_ep50/model_best.pt
QMAX_FULL=${SCRATCH_ROOT}/outputs/qmax_stage_a/formal/qmax_full_seed42_epoch30/epoch60/model_last.pt

QUAL_SCRIPT=${PROJECT_ROOT}/scripts/render_six_slice_qualitative.py
FAILURE_SCRIPT=${PROJECT_ROOT}/scripts/render_qmax_validation_failure_cases.py
OUTPUT_DIR=${SCRATCH_ROOT}/outputs/qmax_stage_a/validation_failure_cases/qmax_full_epoch60

cd "${PROJECT_ROOT}"
mkdir -p "${SCRATCH_ROOT}/logs" "$(dirname "${OUTPUT_DIR}")"

for path in \
  "${METADATA}" "${FULL_CLEAN}" "${ROBUSTNESS}" "${CONDITIONS}" \
  "${VALIDATION_SLICE}" "${VALIDATION_AUDIT}" \
  "${M2U_AUG}" "${FIFTH}" "${QMAX_FULL}" \
  "${QUAL_SCRIPT}" "${FAILURE_SCRIPT}"; do
  test -f "${path}"
done

case "${VALIDATION_SLICE}:${VALIDATION_AUDIT}:${FULL_CLEAN}:${ROBUSTNESS}:${CONDITIONS}" in
  *heldout*|*held_out*|*test_manifest*|*held-out*)
    echo "Refusing held-out-test input in validation failure-case task." >&2
    exit 2
    ;;
esac

test ! -e "${OUTPUT_DIR}/qmax_validation_failure_case_audit.json"

export PYTHONHASHSEED=42
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

python -u "${FAILURE_SCRIPT}" \
  --metadata_csv "${METADATA}" \
  --full_clean_manifest "${FULL_CLEAN}" \
  --robustness_manifest "${ROBUSTNESS}" \
  --condition_manifest "${CONDITIONS}" \
  --validation_slice_metrics "${VALIDATION_SLICE}" \
  --validation_audit "${VALIDATION_AUDIT}" \
  --m2u_augmented_checkpoint "${M2U_AUG}" \
  --quality_gain_checkpoint "${FIFTH}" \
  --qmax_full_checkpoint "${QMAX_FULL}" \
  --output_dir "${OUTPUT_DIR}" \
  --seed 42

AUDIT=${OUTPUT_DIR}/qmax_validation_failure_case_audit.json
test -f "${AUDIT}"
test -f "${OUTPUT_DIR}/selected_failure_cases.json"
test -f "${OUTPUT_DIR}/failure_case_panel_metrics.csv"
test -f "${OUTPUT_DIR}/failure_case_source_arrays.npz"

python - "${AUDIT}" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
assert report["status"] == "passed", report
assert report["scope"] == "locked validation only; held-out test not accessed", report
assert report["num_distinct_patients"] == 6, report
assert report["num_distinct_slices"] == 6, report
assert report["checkpoint_sha256"]["qmax_full"] == "1285dd76f7900859d7ca57e68fa4f54509bed540865a7a638397e20d5012b5aa"
PY

PNG_COUNT=$(find "${OUTPUT_DIR}/slice_level_figures" -maxdepth 1 -name '*.png' | wc -l)
test "${PNG_COUNT}" -eq 12
echo "slice_level_png_count=${PNG_COUNT}"
echo "output_dir=${OUTPUT_DIR}"
