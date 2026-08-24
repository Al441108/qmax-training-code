#!/bin/bash
#SBATCH --job-name=m2gdv2_R6_smoke5
#SBATCH --output=logs/m2gd_v2/m2gdv2_R6_smoke5_%j.out
#SBATCH --error=logs/m2gd_v2/m2gdv2_R6_smoke5_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G

set -euo pipefail

PROJECT_ROOT="/projects/u6dm/fastmri_project/fastmri_pipeline"
METADATA="/lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv"
M2U_CHECKPOINT="outputs/m2u/m2u_pdR2_to_pdfs_R6_ep50/model_best.pt"
OUTPUT_DIR="outputs/m2gd_v2/m2gd_v2_R6_smoke5_fixed"

cd "$PROJECT_ROOT"
source "$HOME/miniforge3/bin/activate"
conda activate fastmri311
mkdir -p "$OUTPUT_DIR" logs/m2gd_v2

[[ -f "$M2U_CHECKPOINT" ]] || {
  echo "Missing R6 M2-U checkpoint: $M2U_CHECKPOINT"
  exit 1
}

python -m py_compile \
  src/auxiliary_corruptions_v2.py \
  src/m2gd_v2_auxiliary_varnet.py \
  scripts/train_varnet_m2gd_v2.py \
  scripts/evaluate_m2gd_v2_smoke_audit.py

# Fail before training if the source is not the registered R6, 50-epoch M2-U.
python - "$M2U_CHECKPOINT" <<'PY'
import json
import sys
from pathlib import Path

import torch

path = Path(sys.argv[1])
try:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
except TypeError:
    checkpoint = torch.load(path, map_location="cpu")

if not isinstance(checkpoint, dict):
    raise SystemExit("M2-U checkpoint is not a dictionary.")
config = checkpoint.get("config", {})
if not isinstance(config, dict):
    raise SystemExit("M2-U checkpoint has no valid config dictionary.")

observed = {
    "epoch": checkpoint.get("epoch"),
    "acceleration": config.get("acceleration", config.get("pdfs_acceleration")),
    "pd_aux_acceleration": config.get("pd_aux_acceleration"),
    "num_cascades": config.get("num_cascades"),
    "chans": config.get("chans"),
    "pools": config.get("pools"),
}
expected = {
    # This M2-U trainer stores zero-based epoch indices: 49 means 50 completed epochs.
    "epoch": 49,
    "acceleration": 6,
    "pd_aux_acceleration": 2,
    "num_cascades": 12,
    "chans": 18,
    "pools": 4,
}
mismatch = {
    key: {"expected": value, "observed": observed.get(key)}
    for key, value in expected.items()
    if observed.get(key) is not None and observed.get(key) != value
}
if observed["epoch"] != 49:
    mismatch["epoch"] = {"expected": 49, "observed": observed["epoch"]}
if observed["acceleration"] != 6:
    mismatch["acceleration"] = {"expected": 6, "observed": observed["acceleration"]}
if observed["pd_aux_acceleration"] not in (None, 2):
    mismatch["pd_aux_acceleration"] = {
        "expected": 2,
        "observed": observed["pd_aux_acceleration"],
    }
if mismatch:
    raise SystemExit(f"R6 M2-U provenance mismatch: {mismatch}")
print("R6 M2-U source preflight: PASSED")
print(json.dumps(observed, indent=2))
PY

printf '%s\n' "============================================================"
printf 'M2-GD v2 R6 smoke5 | job=%s | host=%s | start=%s\n' \
  "${SLURM_JOB_ID:-unknown}" "$(hostname)" "$(date)"
printf 'M2-U source: %s\n' "$M2U_CHECKPOINT"
printf 'Output: %s\n' "$OUTPUT_DIR"
printf '%s\n' "============================================================"

python scripts/train_varnet_m2gd_v2.py \
  --metadata_csv "$METADATA" \
  --m2u_checkpoint "$M2U_CHECKPOINT" \
  --output_dir "$OUTPUT_DIR" \
  --acceleration 6 \
  --pd_aux_acceleration 2 \
  --epochs 5 \
  --curriculum smoke5 \
  --batch_size 4 \
  --grad_accum_steps 2 \
  --num_workers 4 \
  --num_cascades 12 \
  --chans 18 \
  --sens_chans 8 \
  --pools 4 \
  --sens_pools 4 \
  --initial_aux_alpha 0.1 \
  --initial_gate_probability 0.99 \
  --pretrained_lr 1e-5 \
  --new_module_lr 1e-4 \
  --lambda_rel 0.05 \
  --freeze_epochs 2 \
  --padding_prob_reflect 0.70 \
  --padding_prob_replicate 0.20 \
  --padding_prob_zero 0.10 \
  --seed 42

printf 'R6 V2 smoke training completed: %s\n' "$(date)"
cat "$OUTPUT_DIR/final_summary.json"
