#!/bin/bash
#SBATCH --job-name=rpcrf_h4_R8_ep50
#SBATCH --output=logs/m2_prnf_rpcrf_h4/rpcrf_h4_R8_ep50_%j.out
#SBATCH --error=logs/m2_prnf_rpcrf_h4/rpcrf_h4_R8_ep50_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G

set -euo pipefail

PROJECT_ROOT="/projects/u6dm/fastmri_project/fastmri_pipeline"
METADATA="/lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv"
MANIFEST_ROOT="outputs/m2_prnf/preflight_R8_bs4"
CONDITION_MANIFEST="outputs/m2_prnf_fusion/preflight_R8_bs4/condition_manifest.json"
OUTPUT_DIR="outputs/m2_prnf_rpcrf_h4/formal_R8_b2a2/rpcrf_h4_final_seed42_ep50"

cd "$PROJECT_ROOT"
source "$HOME/miniforge3/bin/activate"
conda activate fastmri311
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs/m2_prnf_rpcrf_h4 "$OUTPUT_DIR"

for REQUIRED in \
  "$METADATA" \
  "$MANIFEST_ROOT/full_clean_manifest.json" \
  "$MANIFEST_ROOT/robustness_manifest.json" \
  "$CONDITION_MANIFEST"; do
  [[ -f "$REQUIRED" ]] || {
    echo "Missing required input: $REQUIRED"
    exit 1
  }
done

# This is a syntax check, not a model preflight and does not allocate a GPU graph.
python -m py_compile \
  src/m2_prnf_rpcrf_varnet.py \
  scripts/train_m2_prnf_rpcrf.py

RESUME_ARGS=()
if [[ -f "$OUTPUT_DIR/model_last.pt" ]]; then
  RESUME_ARGS=(--resume "$OUTPUT_DIR/model_last.pt")
  echo "Resuming from $OUTPUT_DIR/model_last.pt"
fi

printf '%s\n' "============================================================"
printf 'RPCRF H/4 formal50 | job=%s | host=%s | start=%s\n' \
  "${SLURM_JOB_ID:-unknown}" "$(hostname)" "$(date)"
printf 'Output: %s\n' "$OUTPUT_DIR"
printf '%s\n' "============================================================"

python scripts/train_m2_prnf_rpcrf.py \
  --metadata_csv "$METADATA" \
  --metadata_identity_csv "$METADATA" \
  --output_dir "$OUTPUT_DIR" \
  --full_clean_manifest "$MANIFEST_ROOT/full_clean_manifest.json" \
  --robustness_manifest "$MANIFEST_ROOT/robustness_manifest.json" \
  --condition_manifest "$CONDITION_MANIFEST" \
  --variant prnf_no_need \
  --fusion_design hybrid_direct_residual \
  --run_stage rpcrf_h4_formal50 \
  --need_scope residual \
  --residual_scale 0.1 \
  --residual_reliability_power 1.5 \
  --lambda_correction 0.05 \
  --lambda_overshoot 0.01 \
  --overshoot_kappa 1.25 \
  --correction_weight_clip 4.0 \
  --lambda_residual_gain 0.10 \
  --residual_gain_margin_relative 0.002 \
  --residual_gain_ramp_epochs 5 \
  --acceleration 8 \
  --pd_aux_acceleration 2 \
  --epochs 50 \
  --learning_rate 3e-4 \
  --batch_size 2 \
  --grad_accum_steps 2 \
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
  --seed 42 \
  "${RESUME_ARGS[@]}"

printf 'Formal training completed: %s\n' "$(date)"
cat "$OUTPUT_DIR/final_summary.json"
