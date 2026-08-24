#!/bin/bash
#SBATCH --job-name=m2gdv21_R6_preflight
#SBATCH --output=logs/m2gd_v21/m2gdv21_R6_preflight_%j.out
#SBATCH --error=logs/m2gd_v21/m2gdv21_R6_preflight_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G

set -euo pipefail
PROJECT_ROOT="/projects/u6dm/fastmri_project/fastmri_pipeline"
METADATA="/lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv"
M2GD_V2_CHECKPOINT="outputs/m2gd_v2/m2gd_v2_R6_smoke5_fixed/model_best.pt"
OUTPUT_JSON="outputs/m2gd_v21/preflight_R6_r1/preflight_results.json"

cd "$PROJECT_ROOT"
source "$HOME/miniforge3/bin/activate"
conda activate fastmri311
mkdir -p "$(dirname "$OUTPUT_JSON")" logs/m2gd_v21
[[ -f "$M2GD_V2_CHECKPOINT" ]] || { echo "Missing R6 V2 checkpoint: $M2GD_V2_CHECKPOINT"; exit 1; }

python -m py_compile \
  src/auxiliary_corruptions_v21.py \
  src/m2gd_v21_auxiliary_varnet.py \
  scripts/preflight_m2gd_v21.py \
  scripts/train_varnet_m2gd_v21.py \
  scripts/train_varnet_m2gd_v21_stageb.py \
  scripts/evaluate_m2gd_v21_smoke_audit.py

python scripts/preflight_m2gd_v21.py \
  --metadata_csv "$METADATA" \
  --m2gd_v2_checkpoint "$M2GD_V2_CHECKPOINT" \
  --output_json "$OUTPUT_JSON" \
  --initial_gate_probability 0.7436 \
  --max_initial_relative_difference 0.02 \
  --acceleration 6 \
  --pd_aux_acceleration 2

cat "$OUTPUT_JSON"
