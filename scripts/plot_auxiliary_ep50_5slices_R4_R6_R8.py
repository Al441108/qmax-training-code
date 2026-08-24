#!/usr/bin/env python3
import subprocess
from pathlib import Path

metadata = "/projects/u6dm/fastmri_project/fastmri/multicoil_lesion_split/metadata/reorganised_dataset_split_isambard.csv"

# Selected from ep50 per-slice analysis.
# Criteria:
# - not edge slice
# - not slice 26
# - AuxPDVarNet ep50 improves over both single and symmetric joint across R=4/6/8
# - ranked by combined SSIM and PSNR gains
patient_id = "375168851b63b812144c36c336fc9bc9aaa82eea40a0e82fc5c8457dfd320124"

slice_indices = [
    20,
    22,
    21,
    30,
    31,
]

R_values = [4, 6, 8]

output_dir = "outputs/figures/auxiliary_ep50_5slices_R4_R6_R8"
Path(output_dir).mkdir(parents=True, exist_ok=True)

for slice_idx in slice_indices:
    for R in R_values:
        cmd = [
            "python",
            "scripts/plot_auxiliary_ep50_qualitative.py",
            "--metadata_csv",
            metadata,
            "--patient_id",
            patient_id,
            "--slice_idx",
            str(slice_idx),
            "--R",
            str(R),
            "--output_dir",
            output_dir,
            "--dpi",
            "300",
        ]

        print("=" * 100)
        print(f"Plotting patient={patient_id[:12]}, slice={slice_idx}, R={R}", flush=True)
        print("Running:", " ".join(cmd), flush=True)
        print("=" * 100)

        subprocess.run(cmd, check=True)

print("=" * 100)
print("Done. Saved 15 qualitative figures to:")
print(output_dir)
print("=" * 100)
