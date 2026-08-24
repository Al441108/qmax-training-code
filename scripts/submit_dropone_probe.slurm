#!/usr/bin/env bash
#SBATCH --job-name=dropone_probe
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/pd_mechanism/dropone_probe_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/logs/pd_mechanism/dropone_probe_%j.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT=/projects/u6dm/fastmri_project/fastmri_pipeline
CHECKPOINT=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/outputs/m2_prnf/formal_R8_bs4/global_direct_prnf_no_need_seed42_ep50/model_best.pt
OUTPUT=/lus/lfs1aip2/scratch/u6dm/albert44.u6dm/fastmri_project/fastmri_pipeline/outputs/pd_mechanism/dropone_interface_probe.json

cd "${PROJECT_ROOT}"
test -f "${CHECKPOINT}"
test -f scripts/probe_dropone_interface.py
mkdir -p "$(dirname "${OUTPUT}")"

python scripts/probe_dropone_interface.py \
  --project-root "${PROJECT_ROOT}" \
  --checkpoint "${CHECKPOINT}" \
  --output "${OUTPUT}"

echo "probe=${OUTPUT}"
