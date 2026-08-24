#!/usr/bin/env bash
#SBATCH --job-name=qmax_fmri_roi_pf2
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/qmax_fmri_roi_pf_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/qmax_fmri_roi_pf_%j.err
#SBATCH --export=ALL

set -euo pipefail
PROJECT_ROOT=/projects/u6dm/fastmri_project/fastmri_pipeline
SCRATCH_ROOT=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline
METADATA=/lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv
MANIFEST=${SCRATCH_ROOT}/outputs/m2_prnf/preflight_R8_bs4/full_clean_manifest.json
ANNOTATIONS=${PROJECT_ROOT}/external/fastmri_plus/knee.csv
SCRIPT=${PROJECT_ROOT}/scripts/qmax_fastmriplus_roi.py
OUTPUT=${SCRATCH_ROOT}/outputs/qmax_stage_a/fastmriplus_roi/preflight

cd "${PROJECT_ROOT}"
mkdir -p "${SCRATCH_ROOT}/logs" "${OUTPUT}"
for path in "${METADATA}" "${MANIFEST}" "${ANNOTATIONS}" "${SCRIPT}"; do test -f "${path}"; done
test ! -e "${OUTPUT}/preflight_audit.json"

python -u "${SCRIPT}" preflight \
  --metadata_csv "${METADATA}" \
  --full_clean_manifest "${MANIFEST}" \
  --annotations_csv "${ANNOTATIONS}" \
  --output_dir "${OUTPUT}" \
  --slice_offset -1 \
  --flip_up_down \
  --annotation_height 320 \
  --annotation_width 320 \
  --num_overlays 12

test -f "${OUTPUT}/preflight_audit.json"
test -f "${OUTPUT}/fastmriplus_coordinate_overlay.png"
test -f "${OUTPUT}/mapped_locked_validation_boxes.csv"
echo "Inspect before approval: ${OUTPUT}/fastmriplus_coordinate_overlay.png"
