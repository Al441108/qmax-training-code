#!/usr/bin/env bash
#SBATCH --job-name=qmax_final_eff
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/qmax_final_eff_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/qmax_final_eff_%j.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/projects/u6dm/fastmri_project/fastmri_pipeline
SCRATCH=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline
METADATA=/lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv
FULL_CLEAN=${SCRATCH}/outputs/m2_prnf/preflight_R8_bs4/full_clean_manifest.json
ROBUSTNESS=${SCRATCH}/outputs/m2_prnf/preflight_R8_bs4/robustness_manifest.json
CONDITIONS=${SCRATCH}/outputs/m2_prnf_fusion/preflight_R8_bs4/condition_manifest.json

SINGLE=${SCRATCH}/outputs/varnet_single/pdfs_R8_c12_ch18_ep30_bs8_nw4/model_last.pt
M2U_AUG=${SCRATCH}/outputs/m2_prnf/formal_R8_bs4/m2u_augmented_seed42_ep50/model_best.pt
FIFTH=${SCRATCH}/outputs/m2_prnf/formal_R8_bs4/quality_protected_hybrid_gain_seed42_ep50/model_best.pt
STAGEA_CORE=${SCRATCH}/outputs/qmax_stage_a/formal/qmax_core_seed42_epoch30/epoch30/model_last.pt
STAGEA_FULL=${SCRATCH}/outputs/qmax_stage_a/formal/qmax_full_seed42_epoch30/epoch60/model_last.pt
STAGEB_FULL=${SCRATCH}/outputs/qmax_stage_b_independent/formal/qmax_compactswin_qmax_full_seed42_epoch60/epoch30/model_last.pt

SCRIPT=${PROJECT_ROOT}/scripts/benchmark_qmax_final_efficiency.py
OUTPUT=${SCRATCH}/outputs/qmax_stage_a/final_efficiency/qmax_full_epoch60

cd "${PROJECT_ROOT}"
mkdir -p "${SCRATCH}/logs" "$(dirname "${OUTPUT}")"

test -f MAIN_EXPERIMENTS_COMPLETE.lock
test -f HELDOUT_TEST_COMPLETE.lock
for path in \
  "${METADATA}" "${FULL_CLEAN}" "${ROBUSTNESS}" "${CONDITIONS}" \
  "${SINGLE}" "${M2U_AUG}" "${FIFTH}" "${STAGEA_CORE}" \
  "${STAGEA_FULL}" "${STAGEB_FULL}" "${SCRIPT}" \
  "${PROJECT_ROOT}/scripts/evaluate_m2_prnf_final_comparison_R8.py" \
  "${PROJECT_ROOT}/src/m2_prnf_qmax_compactswin_varnet.py"; do
  test -f "${path}"
done

test ! -e "${OUTPUT}/qmax_final_efficiency_audit.json"
export PYTHONHASHSEED=42
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

python -u "${SCRIPT}" \
  --metadata_csv "${METADATA}" \
  --full_clean_manifest "${FULL_CLEAN}" \
  --robustness_manifest "${ROBUSTNESS}" \
  --condition_manifest "${CONDITIONS}" \
  --single_checkpoint "${SINGLE}" \
  --m2u_augmented_checkpoint "${M2U_AUG}" \
  --fifth_checkpoint "${FIFTH}" \
  --stagea_core_checkpoint "${STAGEA_CORE}" \
  --stagea_full_checkpoint "${STAGEA_FULL}" \
  --stageb_full_checkpoint "${STAGEB_FULL}" \
  --output_dir "${OUTPUT}" \
  --warmup 20 \
  --repeats 100 \
  --seed 42 \
  --single_num_cascades 12 \
  --single_chans 18 \
  --single_pools 4 \
  --single_sens_chans 8 \
  --single_sens_pools 4

test -f "${OUTPUT}/qmax_final_efficiency.csv"
test -f "${OUTPUT}/qmax_final_efficiency_audit.json"
python -c "import json; p=json.load(open('${OUTPUT}/qmax_final_efficiency_audit.json')); assert p['status']=='passed'; assert 'held-out test not accessed' in p['scope']; print(p['device'])"
echo "efficiency_output=${OUTPUT}"

