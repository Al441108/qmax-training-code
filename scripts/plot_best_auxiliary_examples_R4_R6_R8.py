#!/usr/bin/env python3
import subprocess

metadata = "/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv"

# Best same-slice examples selected from auxiliary_same_slice_figure_candidates.csv.
# Selection criterion:
# score_ssim = SSIM_gain_aux_vs_single + SSIM_gain_aux_vs_joint
examples = [
    {
        "R": 4,
        "patient_id": "375168851b63b812144c36c336fc9bc9aaa82eea40a0e82fc5c8457dfd320124",
        "slice_idx": 20,
    },
    {
        "R": 6,
        "patient_id": "5d7dc98f247e3142b52ceca4c6be1672250e79ced9befc3e88b28ceca0ca1269",
        "slice_idx": 8,
    },
    {
        "R": 8,
        "patient_id": "375168851b63b812144c36c336fc9bc9aaa82eea40a0e82fc5c8457dfd320124",
        "slice_idx": 26,
    },
]

for ex in examples:
    cmd = [
        "python",
        "scripts/plot_same_slice_single_joint_auxiliary.py",
        "--metadata_csv",
        metadata,
        "--patient_id",
        ex["patient_id"],
        "--slice_idx",
        str(ex["slice_idx"]),
        "--R",
        str(ex["R"]),
        "--output_dir",
        "outputs/figures/auxiliary_best_same_slice_examples",
        "--dpi",
        "300",
    ]

    print("=" * 100)
    print("Running:", " ".join(cmd), flush=True)
    print("=" * 100)

    subprocess.run(cmd, check=True)

print("Done. Best auxiliary same-slice figures saved to:")
print("outputs/figures/auxiliary_best_same_slice_examples")
