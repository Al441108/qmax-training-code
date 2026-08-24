#!/usr/bin/env python3
import subprocess
from pathlib import Path

metadata = "/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv"

patient_id = "3a1e37f17d9bdb8a1c8679ec558e3e50"
slice_idx = 34
contrast = "PD"


def find_single_ckpt(R):
    base = Path("outputs/varnet_single")

    files = []
    for pattern in ["*.pt", "*.pth", "*.ckpt"]:
        files.extend(base.rglob(pattern))

    candidates = [
        p for p in files
        if "pd" in str(p).lower()
        and "pdfs" not in str(p).lower()
        and f"r{R}" in str(p).lower()
        and ("best" in p.name.lower() or "model_best" in p.name.lower())
    ]

    if not candidates:
        candidates = [
            p for p in files
            if "pd" in str(p).lower()
            and "pdfs" not in str(p).lower()
            and f"r{R}" in str(p).lower()
        ]

    if not candidates:
        raise FileNotFoundError(f"Cannot find single PD checkpoint for R={R}")

    candidates = sorted(
        candidates,
        key=lambda p: (
            0 if p.name == "model_best.pt" else 1,
            0 if "best" in p.name.lower() else 1,
            len(str(p)),
            str(p),
        )
    )

    print(f"Using single PD checkpoint for R={R}: {candidates[0]}")
    return str(candidates[0])


joint_ckpts = {
    4: "outputs/varnet_joint_revised/joint_R4_jvn_adaptive_lr1e4_pilot_ep5_bs4/model_best.pt",
    6: "outputs/varnet_joint_revised/joint_R6_jvn_adaptive_lr1e4_ep30_bs4/model_best.pt",
    8: "outputs/varnet_joint_revised/joint_R8_jvn_adaptive_lr1e4_ep30_bs4/model_best.pt",
}

for R in [4, 6, 8]:
    print(f"\n===== Plotting slice {slice_idx}, {contrast}, R={R} =====\n")

    single_ckpt = find_single_ckpt(R)

    cmd = [
        "python", "scripts/plot_same_slice_single_joint.py",
        "--metadata_csv", metadata,
        "--patient_id", patient_id,
        "--slice_idx", str(slice_idx),
        "--contrast", contrast,
        "--R", str(R),
        "--single_checkpoint", single_ckpt,
        "--joint_checkpoint", joint_ckpts[R],
        "--output_dir", "outputs/figures/slice34_R4_R6_R8_PD"
    ]

    subprocess.run(cmd, check=True)

print("\nDone. Figures saved to outputs/figures/slice34_R4_R6_R8_PD")
