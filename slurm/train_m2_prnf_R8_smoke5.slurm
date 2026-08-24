#!/bin/bash
#SBATCH --job-name=prnf_R8_smoke5
#SBATCH --output=logs/m2_prnf/prnf_R8_smoke5_%j.out
#SBATCH --error=logs/m2_prnf/prnf_R8_smoke5_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G

set -euo pipefail
PROJECT_ROOT="/projects/u6dm/fastmri_project/fastmri_pipeline"
METADATA="/lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv"
VARIANT="${VARIANT:-prnf_full}"
SEED="${SEED:-42}"
OUTPUT_DIR="outputs/m2_prnf/smoke5_R8_bs4/${VARIANT}_seed${SEED}"
PREFLIGHT="outputs/m2_prnf/preflight_R8_bs4/preflight_results.json"
FULL_CLEAN_MANIFEST="outputs/m2_prnf/preflight_R8_bs4/full_clean_manifest.json"
ROBUSTNESS_MANIFEST="outputs/m2_prnf/preflight_R8_bs4/robustness_manifest.json"

cd "$PROJECT_ROOT"
source "$HOME/miniforge3/bin/activate"
conda activate fastmri311
mkdir -p logs/m2_prnf "$OUTPUT_DIR"

test -f "$PREFLIGHT" || { echo "Missing preflight: $PREFLIGHT"; exit 1; }
test -f "$FULL_CLEAN_MANIFEST" || { echo "Missing manifest: $FULL_CLEAN_MANIFEST"; exit 1; }
test -f "$ROBUSTNESS_MANIFEST" || { echo "Missing manifest: $ROBUSTNESS_MANIFEST"; exit 1; }
python - "$PREFLIGHT" <<'PY'
import json, sys
result = json.load(open(sys.argv[1], encoding="utf-8"))
if result.get("status") != "passed":
    raise SystemExit(f"Preflight did not pass: {result}")
PY

python scripts/train_m2_prnf.py \
  --metadata_csv "$METADATA" \
  --output_dir "$OUTPUT_DIR" \
  --full_clean_manifest "$FULL_CLEAN_MANIFEST" \
  --robustness_manifest "$ROBUSTNESS_MANIFEST" \
  --variant "$VARIANT" \
  --acceleration 8 \
  --pd_aux_acceleration 2 \
  --epochs 5 \
  --learning_rate 3e-4 \
  --batch_size 4 \
  --grad_accum_steps 1 \
  --num_workers 4 \
  --num_cascades 12 \
  --chans 18 \
  --sens_chans 8 \
  --pools 4 \
  --sens_pools 4 \
  --controller_chans 16 \
  --initial_aux_alpha 0.1 \
  --initial_gate_probability 0.95 \
  --initial_need_probability 0.95 \
  --need_floor 0.25 \
  --lambda_rel 0.05 \
  --lambda_rank 0.02 \
  --aux_loss_ramp_epochs 5 \
  --seed "$SEED"

python - "$OUTPUT_DIR/training_log.csv" <<'PY'
import csv, math, sys
rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
if len(rows) != 5:
    raise SystemExit(f"Smoke expected 5 logged epochs, observed {len(rows)}")
last = {key: float(value) for key, value in rows[-1].items() if key != "epoch"}
if not all(math.isfinite(value) for value in last.values()):
    raise SystemExit("Smoke contains a non-finite final diagnostic")
need = last["train_need_clean"]
spread = last["train_need_p95_clean"] - last["train_need_p05_clean"]
if not 0.05 < need < 0.995:
    raise SystemExit(f"Need mean failed smoke range: {need}")
if spread < 0.001:
    raise SystemExit(f"Need spatial spread collapsed: {spread}")
print(f"Smoke diagnostic gate passed: need={need:.6f}, p95-p05={spread:.6f}")
PY

cat "$OUTPUT_DIR/final_summary.json"
