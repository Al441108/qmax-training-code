#!/bin/bash
#SBATCH --job-name=m2gdv21_R8_qual
#SBATCH --output=logs/m2gd_v21/m2gdv21_R8_qual_%j.out
#SBATCH --error=logs/m2gd_v21/m2gdv21_R8_qual_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G

set -euo pipefail

PROJECT_ROOT="/projects/u6dm/fastmri_project/fastmri_pipeline"
METADATA="/lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv"
M2U_CHECKPOINT="outputs/m2u/m2u_pdR2_to_pdfs_R8_pilot5/model_best.pt"
SINGLE_CHECKPOINT="outputs/varnet_single/pdfs_R8_c12_ch18_ep30_bs8_nw4/model_best.pt"
STAGEB_CHECKPOINT="outputs/m2gd_v21/m2gd_v21_R8_FINAL/model_final_fusioncal3_r3.pt"
FORMAL_DIR="outputs/evaluation/m2gd_v21_formal_robustness_R8_r1"
OUTPUT_DIR="outputs/figures/m2gd_v21_R8_final_evidence_r2"

cd "$PROJECT_ROOT"
source "$HOME/miniforge3/bin/activate"
conda activate fastmri311
mkdir -p "$OUTPUT_DIR" logs/m2gd_v21

for required in \
  "$SINGLE_CHECKPOINT" \
  "$M2U_CHECKPOINT" \
  "$STAGEB_CHECKPOINT" \
  "$FORMAL_DIR/formal_robustness_per_slice.csv" \
  "$FORMAL_DIR/formal_robustness_decision.json"; do
  [[ -f "$required" ]] || { echo "Missing required file: $required"; exit 1; }
done

python -m py_compile scripts/export_m2gd_v21_qualitative_R8.py

python scripts/export_m2gd_v21_qualitative_R8.py \
  --metadata_csv "$METADATA" \
  --single_checkpoint "$SINGLE_CHECKPOINT" \
  --m2u_checkpoint "$M2U_CHECKPOINT" \
  --stageb_checkpoint "$STAGEB_CHECKPOINT" \
  --formal_results_dir "$FORMAL_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --seed 42

printf 'Qualitative evidence export completed: %s\n' "$(date)"
ls -lh "$OUTPUT_DIR"
