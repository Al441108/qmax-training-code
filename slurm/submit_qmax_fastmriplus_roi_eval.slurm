#!/usr/bin/env bash
#SBATCH --job-name=qmax_fmri_roi_eval
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/qmax_fmri_roi_eval_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/qmax_fmri_roi_eval_%j.err
#SBATCH --export=ALL

set -euo pipefail
PROJECT_ROOT=/projects/u6dm/fastmri_project/fastmri_pipeline
SCRATCH_ROOT=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline
METADATA=/lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv
MANIFEST=${SCRATCH_ROOT}/outputs/m2_prnf/preflight_R8_bs4/full_clean_manifest.json
ANNOTATIONS=${PROJECT_ROOT}/external/fastmri_plus/knee.csv
SCRIPT=${PROJECT_ROOT}/scripts/qmax_fastmriplus_roi.py
PREFLIGHT=${SCRATCH_ROOT}/outputs/qmax_stage_a/fastmriplus_roi/preflight
APPROVAL=${PREFLIGHT}/FASTMRIPLUS_ROI_COORDINATES_APPROVED.json
CHECKPOINT=${SCRATCH_ROOT}/outputs/qmax_stage_a/formal/qmax_full_seed42_epoch30/epoch60/model_last.pt
OUTPUT=${SCRATCH_ROOT}/outputs/qmax_stage_a/fastmriplus_roi/qmax_full_epoch60

cd "${PROJECT_ROOT}"
mkdir -p "${SCRATCH_ROOT}/logs" "${OUTPUT}"
for path in "${METADATA}" "${MANIFEST}" "${ANNOTATIONS}" "${SCRIPT}" \
  "${PREFLIGHT}/mapped_locked_validation_boxes.csv" "${APPROVAL}" "${CHECKPOINT}"; do test -f "${path}"; done
test ! -e "${OUTPUT}/roi_audit.json"
echo "1285dd76f7900859d7ca57e68fa4f54509bed540865a7a638397e20d5012b5aa  ${CHECKPOINT}" | sha256sum -c -

python -u "${SCRIPT}" evaluate \
  --metadata_csv "${METADATA}" \
  --full_clean_manifest "${MANIFEST}" \
  --annotations_csv "${ANNOTATIONS}" \
  --mapped_boxes_csv "${PREFLIGHT}/mapped_locked_validation_boxes.csv" \
  --approval_json "${APPROVAL}" \
  --qmax_checkpoint "${CHECKPOINT}" \
  --output_dir "${OUTPUT}" \
  --ring_pixels 8 \
  --bootstrap_iterations 10000 \
  --seed 42

python - "${OUTPUT}/roi_audit.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
assert report["status"] == "passed", report
assert report["comparison"] == ["zero_filled", "qmax_full_stagea_epoch60"], report
assert "held-out test not accessed" in report["scope"], report
PY
echo "output_dir=${OUTPUT}"
