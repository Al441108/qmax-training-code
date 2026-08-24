#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/projects/u6dm/fastmri_project/fastmri_pipeline}"
: "${METADATA_CSV:?Set METADATA_CSV to the metadata CSV used by training}"
: "${CHECKPOINT:?Set CHECKPOINT to the locked Global-direct model_best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/artifacts/pd_mechanism/spatial_helpfulness}"

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"

python scripts/analyze_pd_spatial_helpfulness.py \
  --project-root "${PROJECT_ROOT}" \
  --metadata-csv "${METADATA_CSV}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --split val \
  --num-workers 4 \
  --foreground-fraction 0.01 \
  --high-error-quantile 0.75 \
  --edge-quantile 0.75

printf 'Spatial helpfulness analysis complete: %s\n' "${OUTPUT_DIR}"
