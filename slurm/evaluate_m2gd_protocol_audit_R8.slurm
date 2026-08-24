#!/bin/bash
#SBATCH --job-name=audit_M2GD_R8
#SBATCH --output=logs/evaluation/audit_M2GD_R8_%j.out
#SBATCH --error=logs/evaluation/audit_M2GD_R8_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G

set -euo pipefail

cd /projects/u6dm/fastmri_project/fastmri_pipeline

source ~/miniforge3/bin/activate
conda activate fastmri311

mkdir -p logs/evaluation
mkdir -p outputs/evaluation/m2gd_protocol_audit_R8_epoch13

SCRIPT="scripts/evaluate_m2gd_protocol_audit_R8.py"
M0="outputs/varnet_single/pdfs_R8_c12_ch18_ep30_bs8_nw4/model_best.pt"
M2U="outputs/m2u/m2u_pdR2_to_pdfs_R8_pilot5/model_best.pt"
M2GD="outputs/m2gd/m2gd_R8_pilot5/model_best.pt"
OUTPUT="outputs/evaluation/m2gd_protocol_audit_R8_epoch13"

printf '%s\n' "================================================================================================"
echo "M2-GD R=8 protocol audit before epoch 15 -> 50 continuation"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "M2-GD checkpoint: ${M2GD}"
echo "Output: ${OUTPUT}"
echo "Stage A: cardinal zero-pad shifts at 2/4/8 px"
echo "Stage B: zero/reflect/border-only 8 px + wrong patient + missing"
echo "Wrong-patient matching: exact shape preferred; nearest-shape fallback + center crop/pad"
printf '%s\n' "================================================================================================"

for path in "${SCRIPT}" "${M0}" "${M2U}" "${M2GD}"; do
    if [ ! -f "${path}" ]; then
        echo "ERROR: required file not found: ${path}"
        exit 1
    fi
done

python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise RuntimeError("No CUDA GPU visible")
print("GPU:", torch.cuda.get_device_name(0))
print(
    "GPU memory:",
    round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
    "GB",
)
PY

python -m py_compile \
  "${SCRIPT}" \
  src/m2gd_auxiliary_varnet.py \
  src/m2u_auxiliary_varnet_optimized.py

python "${SCRIPT}" \
  --metadata_csv /lus/lfs1aip2/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv \
  --single_checkpoint "${M0}" \
  --m2u_checkpoint "${M2U}" \
  --m2gd_checkpoint "${M2GD}" \
  --output_dir "${OUTPUT}" \
  --batch_size 2 \
  --num_workers 4

printf '%s\n' "================================================================================================"
echo "End: $(date)"
echo "M2-GD protocol audit completed"
echo "Output: ${OUTPUT}"
printf '%s\n' "================================================================================================"
