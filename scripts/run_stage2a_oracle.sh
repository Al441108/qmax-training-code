#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/projects/u6dm/fastmri_project/fastmri_pipeline}"
: "${METADATA_CSV:?Set METADATA_CSV to the metadata CSV used by training}"
: "${CHECKPOINT:?Set CHECKPOINT to the pre-selected Global-direct model_best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/artifacts/pd_mechanism/stage2a_oracle}"

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"

python scripts/evaluate_pd_oracle_stage2a.py \
  --project-root "${PROJECT_ROOT}" \
  --metadata-csv "${METADATA_CSV}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --split val \
  --batch-size 1 \
  --num-workers 4 \
  --bootstrap-replicates 10000

printf 'Stage-2A oracle evaluation complete: %s\n' "${OUTPUT_DIR}"
