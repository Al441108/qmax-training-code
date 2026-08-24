from pathlib import Path
import pandas as pd
import ast

metadata_csv = Path("/rds/general/user/ah725/home/fastmri_pipeline/metadata/multicoil_test_metadata.csv")
out_csv = Path("/rds/general/user/ah725/home/fastmri_pipeline/metadata/multicoil_test_pairs.csv")

df = pd.read_csv(metadata_csv)

def parse_shape(x):
    try:
        return ast.literal_eval(x)
    except Exception:
        return None

df["kspace_shape_tuple"] = df["kspace_shape"].apply(parse_shape)
df["recon_shape_tuple"] = df["recon_shape"].apply(parse_shape)

pairs = []

for patient_id, sub in df.groupby("patient_id"):
    pd_rows = sub[sub["acquisition"] == "CORPD_FBK"]
    pdfs_rows = sub[sub["acquisition"] == "CORPDFS_FBK"]

    if len(pd_rows) == 0 or len(pdfs_rows) == 0:
        continue

    # 如果同一个 patient 有多个 PD 或 PD-FS，先选 shape 最匹配的一对
    best_pair = None
    best_score = -1

    for _, pd_row in pd_rows.iterrows():
        for _, pdfs_row in pdfs_rows.iterrows():
            pd_k = pd_row["kspace_shape_tuple"]
            fs_k = pdfs_row["kspace_shape_tuple"]
            pd_r = pd_row["recon_shape_tuple"]
            fs_r = pdfs_row["recon_shape_tuple"]

            score = 0

            if pd_r is not None and fs_r is not None:
                if pd_r == fs_r:
                    score += 3
                elif pd_r[-2:] == fs_r[-2:]:
                    score += 1

            if pd_k is not None and fs_k is not None:
                if pd_k == fs_k:
                    score += 3
                elif pd_k[-2:] == fs_k[-2:]:
                    score += 1

            if score > best_score:
                best_score = score
                best_pair = (pd_row, pdfs_row, score)

    if best_pair is None:
        continue

    pd_row, pdfs_row, score = best_pair

    pairs.append({
        "patient_id": patient_id,
        "pd_file": pd_row["filename"],
        "pdfs_file": pdfs_row["filename"],
        "pd_kspace_shape": pd_row["kspace_shape"],
        "pdfs_kspace_shape": pdfs_row["kspace_shape"],
        "pd_recon_shape": pd_row["recon_shape"],
        "pdfs_recon_shape": pdfs_row["recon_shape"],
        "shape_match_score": score,
        "usable_pair": score >= 6,
    })

pairs_df = pd.DataFrame(pairs)

print("Total paired patients:", len(pairs_df))
print("Usable exact-shape pairs:", pairs_df["usable_pair"].sum())

print("\nFirst 10 pairs:")
print(pairs_df.head(10))

out_csv.parent.mkdir(parents=True, exist_ok=True)
pairs_df.to_csv(out_csv, index=False)

print(f"\nSaved pair list to: {out_csv}")

