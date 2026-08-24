import h5py
from pathlib import Path
import pandas as pd

root = Path("/rds/general/user/ah725/home/fastmri/multicoil_test")
out_csv = Path("/rds/general/user/ah725/home/fastmri_pipeline/metadata/multicoil_test_metadata.csv")

files = sorted(root.rglob("*.h5"))

print(f"Found {len(files)} h5 files in {root}")

if len(files) == 0:
    raise FileNotFoundError(f"No .h5 files found in {root}")

# 先检查第一个文件结构
fname = files[0]
print("\nExample file:", fname)

with h5py.File(fname, "r") as hf:
    print("\nKeys:", list(hf.keys()))

    print("\nDatasets:")
    for k in hf.keys():
        try:
            print(f"  {k}: shape={hf[k].shape}, dtype={hf[k].dtype}")
        except Exception as e:
            print(f"  {k}: <cannot read shape/dtype> ({e})")

    print("\nAttrs:")
    for k, v in hf.attrs.items():
        print(f"  {k}: {v}")

# 批量读取 metadata
rows = []

for fname in files:
    try:
        with h5py.File(fname, "r") as hf:
            acquisition = hf.attrs.get("acquisition", "UNKNOWN")
            patient_id = hf.attrs.get("patient_id", "UNKNOWN")

            if isinstance(acquisition, bytes):
                acquisition = acquisition.decode("utf-8")
            if isinstance(patient_id, bytes):
                patient_id = patient_id.decode("utf-8")

            if "kspace" in hf:
                kspace_shape = hf["kspace"].shape
            else:
                kspace_shape = None

            if "reconstruction_rss" in hf:
                recon_shape = hf["reconstruction_rss"].shape
            elif "reconstruction_esc" in hf:
                recon_shape = hf["reconstruction_esc"].shape
            else:
                recon_shape = None

            rows.append({
                "filename": fname.name,
                "patient_id": patient_id,
                "acquisition": acquisition,
                "kspace_shape": str(kspace_shape),
                "recon_shape": str(recon_shape),
            })

    except Exception as e:
        rows.append({
            "filename": fname.name,
            "patient_id": "ERROR",
            "acquisition": "ERROR",
            "kspace_shape": "",
            "recon_shape": "",
            "error": str(e),
        })

df = pd.DataFrame(rows)

print("\nAcquisition counts:")
print(df["acquisition"].value_counts())

print("\nFirst 10 rows:")
print(df.head(10))

print("\nPatients with multiple acquisition types:")
grouped = df.groupby("patient_id")["acquisition"].apply(lambda x: sorted(set(x))).reset_index()
paired = grouped[grouped["acquisition"].apply(lambda x: len(x) > 1)]

print(paired)
print("\nNumber of possible paired patients:", len(paired))

# 保存 metadata
out_csv.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_csv, index=False)

print(f"\nSaved metadata csv to: {out_csv}")