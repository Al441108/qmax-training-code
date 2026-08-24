#!/bin/bash
#SBATCH --job-name=m2gdv21_R4_fusion3
#SBATCH --output=logs/m2gd_v21/m2gdv21_R4_fusion3_%j.out
#SBATCH --error=logs/m2gd_v21/m2gdv21_R4_fusion3_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G

set -euo pipefail
PROJECT_ROOT="/projects/u6dm/fastmri_project/fastmri_pipeline"
METADATA="/lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv"
STAGEA_CHECKPOINT="outputs/m2gd_v21/m2gd_v21_R4_paired3_r1/model_best.pt"
STAGEA_AUDIT="outputs/evaluation/m2gd_v21_stagea_audit_R4_r1/m2gd_v21_smoke_audit_decision.json"
OUTPUT_DIR="outputs/m2gd_v21/m2gd_v21_R4_fusioncal3_r1"

cd "$PROJECT_ROOT"
source "$HOME/miniforge3/bin/activate"
conda activate fastmri311
mkdir -p "$OUTPUT_DIR" logs/m2gd_v21
[[ -f "$STAGEA_CHECKPOINT" ]] || { echo "Missing Stage-A checkpoint: $STAGEA_CHECKPOINT"; exit 1; }
[[ -f "$STAGEA_AUDIT" ]] || { echo "Missing Stage-A audit: $STAGEA_AUDIT"; exit 1; }
python - "$STAGEA_AUDIT" <<'PY'
import json, sys
decision = json.load(open(sys.argv[1], "r", encoding="utf-8"))
gate = decision.get("go_to_fusion_only_calibration")
if gate is None:
    gate = decision.get("go_to_stage_b_joint_finetuning")
if gate is not True:
    failed = [key for key, value in decision.get("criteria", {}).items() if not value]
    raise SystemExit(f"R4 fusion calibration blocked: {failed}")
print("R4 Stage-A gate: PASSED")
PY

python scripts/train_varnet_m2gd_v21_stageb.py \
  --metadata_csv "$METADATA" \
  --m2gd_v21_stagea_checkpoint "$STAGEA_CHECKPOINT" \
  --stagea_audit_decision "$STAGEA_AUDIT" \
  --output_dir "$OUTPUT_DIR" \
  --acceleration 4 \
  --pd_aux_acceleration 2 \
  --epochs 3 \
  --curriculum fusioncal3 \
  --batch_size 1 \
  --grad_accum_steps 4 \
  --num_workers 4 \
  --num_cascades 12 \
  --chans 18 \
  --sens_chans 8 \
  --pools 4 \
  --sens_pools 4 \
  --initial_aux_alpha 0.1 \
  --initial_gate_probability 0.7436 \
  --reliability_lr 2e-5 \
  --fusion_lr 5e-6 \
  --clean_recon_weight 0.60 \
  --lambda_rel 0.05 \
  --rank_weight 1.0 \
  --padding_weight 0.25 \
  --padding_prob_reflect 0.70 \
  --padding_prob_replicate 0.20 \
  --padding_prob_zero 0.10 \
  --border_padding_prob_reflect 0.25 \
  --border_padding_prob_replicate 0.25 \
  --border_padding_prob_zero 0.50 \
  --seed 42

cat "$OUTPUT_DIR/final_summary.json"
