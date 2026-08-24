#!/bin/bash
#SBATCH --job-name=joint_rev_pilot_468
#SBATCH --output=/projects/u6dm/fastmri_project/fastmri_pipeline/logs/joint_varnet/joint_revised_pilot_R4_R6_R8_%j.out
#SBATCH --error=/projects/u6dm/fastmri_project/fastmri_pipeline/logs/joint_varnet/joint_revised_pilot_R4_R6_R8_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G

set -euo pipefail

cd /projects/u6dm/fastmri_project/fastmri_pipeline

source ~/miniforge3/bin/activate
conda activate fastmri311

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

METADATA=/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv
BASE_OUT=/projects/u6dm/fastmri_project/fastmri_pipeline/outputs/varnet_joint_revised/pilot_ep5_adaptive_lr1e5_bs4

echo "============================================================"
echo "M1-revised jVN-style pilot test: R=4,6,8"
echo "Config: adaptive norm, zero-init out_conv, bs4, lr=1e-5, epochs=5"
echo "============================================================"
date

for R in 4 6 8
do
  echo ""
  echo "============================================================"
  echo "Start pilot training R=${R}"
  echo "============================================================"
  date

  python scripts/train_varnet_joint_revised.py \
    --metadata_csv ${METADATA} \
    --acceleration ${R} \
    --epochs 5 \
    --num_cascades 12 \
    --chans 18 \
    --sens_chans 8 \
    --pools 4 \
    --sens_pools 4 \
    --batch_size 4 \
    --num_workers 4 \
    --learning_rate 0.00001 \
    --cross_fusion concat \
    --output_dir ${BASE_OUT}/joint_R${R}_jvn_adaptive_lr1e5_pilot_ep5_bs4

  echo "Finished pilot training R=${R}"
  date
done

echo ""
echo "============================================================"
echo "Pilot training completed for R=4,6,8"
echo "============================================================"

python - <<'PY'
from pathlib import Path
import pandas as pd

base = Path("/projects/u6dm/fastmri_project/fastmri_pipeline/outputs/varnet_joint_revised/pilot_ep5_adaptive_lr1e5_bs4")

rows = []

for R in [4, 6, 8]:
    log_path = base / f"joint_R{R}_jvn_adaptive_lr1e5_pilot_ep5_bs4" / "training_log.csv"
    if not log_path.exists():
        rows.append({
            "R": R,
            "status": "MISSING_LOG",
        })
        continue

    df = pd.read_csv(log_path)
    first = df.iloc[0]
    last = df.iloc[-1]
    best = df.loc[df["val_joint"].idxmin()]

    val_joint_drop = float(first["val_joint"] - best["val_joint"])
    val_joint_rel_drop = val_joint_drop / max(float(first["val_joint"]), 1e-8)

    rows.append({
        "R": R,
        "status": "OK",
        "first_val_joint": float(first["val_joint"]),
        "best_epoch": int(best["epoch"]),
        "best_val_joint": float(best["val_joint"]),
        "best_val_pd": float(best["val_pd"]),
        "best_val_pdfs": float(best["val_pdfs"]),
        "last_val_joint": float(last["val_joint"]),
        "last_val_pd": float(last["val_pd"]),
        "last_val_pdfs": float(last["val_pdfs"]),
        "relative_drop_from_epoch1_to_best": val_joint_rel_drop,
        "peak_gpu_memory_gb": float(df["peak_gpu_memory_gb"].max()) if "peak_gpu_memory_gb" in df.columns else None,
    })

summary = pd.DataFrame(rows)
out = base / "pilot_R4_R6_R8_summary.csv"
summary.to_csv(out, index=False)

print("")
print("Pilot summary:")
print(summary.to_string(index=False))
print("")
print(f"Saved summary to: {out}")

print("")
print("Decision guide:")
print("- PASS: no OOM, best_val_joint < 0.10 for all R, and no single contrast remains extremely high.")
print("- STRONG PASS: R4 best_val_joint <= 0.06, R6/R8 <= 0.08-0.10, with decreasing trend.")
print("- WARNING: best_val_joint around 0.10-0.15 or PD/PDFS imbalance persists.")
print("- FAIL: any R has val_joint > 0.15, unstable spikes across epochs, or one contrast collapses.")
PY

echo "Finished pilot summary"
date
