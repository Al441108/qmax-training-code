#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/projects/u6dm/fastmri_project/fastmri_pipeline}"
: "${METADATA_CSV:?Set METADATA_CSV to the metadata CSV used by training}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/artifacts/pd_mechanism/stage1}"

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"

python scripts/audit_pd_aux_stage1.py \
  --project-root "${PROJECT_ROOT}" \
  --metadata-csv "${METADATA_CSV}" \
  --output-dir "${OUTPUT_DIR}" \
  --split val \
  --target-acceleration 8 \
  --pd-acceleration 2 \
  --num-samples 6

printf 'Stage-1 audit complete: %s\n' "${OUTPUT_DIR}"
