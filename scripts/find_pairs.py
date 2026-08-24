from pathlib import Path
import h5py
import json
import csv
from collections import defaultdict


FASTMRI_ROOT = Path("/rds/general/user/ah725/home/fastmri")
OUTPUT_DIR = Path("/rds/general/user/ah725/home/fastmri_pipeline/metadata")

# 只关心这两类
PD_KEYS = {"CORPD_FBK", "CORPD"}
PDFS_KEYS = {"CORPDFS_FBK", "CORPDFS"}


def decode_attr(x):
    """Robustly decode h5 attrs."""
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    return str(x)


def classify_acquisition(acq: str):
    """Map acquisition string to PD / PDFS / OTHER."""
    acq = acq.strip().upper()
    if acq in PD_KEYS:
        return "PD"
    if acq in PDFS_KEYS:
        return "PDFS"
    return "OTHER"


def scan_split(split_name: str):
    """
    Scan one split folder, e.g. singlecoil_train
    Returns:
        records: list of per-file metadata
        patient_map: patient_id -> {"PD": [...], "PDFS": [...], "OTHER": [...]}
    """
    split_dir = FASTMRI_ROOT / split_name
    if not split_dir.exists():
        raise FileNotFoundError(f"Split folder not found: {split_dir}")

    files = sorted(split_dir.glob("*.h5"))
    records = []
    patient_map = defaultdict(lambda: {"PD": [], "PDFS": [], "OTHER": []})

    for fp in files:
        try:
            with h5py.File(fp, "r") as hf:
                patient_id = decode_attr(hf.attrs.get("patient_id", "UNKNOWN"))
                acquisition = decode_attr(hf.attrs.get("acquisition", "UNKNOWN"))
                acq_type = classify_acquisition(acquisition)

                num_slices = hf["kspace"].shape[0] if "kspace" in hf else None
                kspace_shape = tuple(hf["kspace"].shape) if "kspace" in hf else None

            rec = {
                "split": split_name,
                "file": fp.name,
                "path": str(fp),
                "patient_id": patient_id,
                "acquisition": acquisition,
                "acq_type": acq_type,
                "num_slices": num_slices,
                "kspace_shape": kspace_shape,
            }
            records.append(rec)
            patient_map[patient_id][acq_type].append(rec)

        except Exception as e:
            print(f"[WARN] Failed to read {fp.name}: {e}")

    return records, patient_map


def build_pairs(patient_map, split_name: str):
    """
    Build patient-level PD / PDFS pairs.
    If a patient has multiple PD or multiple PDFS scans, keep all combinations.
    """
    pairs = []

    for patient_id, scans in patient_map.items():
        pd_list = scans["PD"]
        pdfs_list = scans["PDFS"]

        if len(pd_list) == 0 or len(pdfs_list) == 0:
            continue

        for pd_rec in pd_list:
            for pdfs_rec in pdfs_list:
                pairs.append({
                    "split": split_name,
                    "patient_id": patient_id,
                    "pd_file": pd_rec["file"],
                    "pd_path": pd_rec["path"],
                    "pd_num_slices": pd_rec["num_slices"],
                    "pd_kspace_shape": pd_rec["kspace_shape"],
                    "pdfs_file": pdfs_rec["file"],
                    "pdfs_path": pdfs_rec["path"],
                    "pdfs_num_slices": pdfs_rec["num_slices"],
                    "pdfs_kspace_shape": pdfs_rec["kspace_shape"],
                })

    return pairs


def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_csv(rows, path: Path):
    if len(rows) == 0:
        with open(path, "w", newline="", encoding="utf-8") as f:
            pass
        return

    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_records = []
    all_pairs = []

    for split_name in ["singlecoil_train", "singlecoil_val", "singlecoil_test"]:
        split_dir = FASTMRI_ROOT / split_name
        if not split_dir.exists():
            print(f"[INFO] Skip missing split: {split_name}")
            continue

        print(f"\n=== Scanning {split_name} ===")
        records, patient_map = scan_split(split_name)
        pairs = build_pairs(patient_map, split_name)

        print(f"Files scanned: {len(records)}")
        print(f"Patients found: {len(patient_map)}")
        print(f"PD/PDFS pairs found: {len(pairs)}")

        all_records.extend(records)
        all_pairs.extend(pairs)

        save_json(records, OUTPUT_DIR / f"{split_name}_records.json")
        save_csv(records, OUTPUT_DIR / f"{split_name}_records.csv")
        save_json(pairs, OUTPUT_DIR / f"{split_name}_pairs.json")
        save_csv(pairs, OUTPUT_DIR / f"{split_name}_pairs.csv")

    print("\n=== Overall summary ===")
    print(f"Total files scanned: {len(all_records)}")
    print(f"Total PD/PDFS pairs: {len(all_pairs)}")

    save_json(all_records, OUTPUT_DIR / "all_records.json")
    save_csv(all_records, OUTPUT_DIR / "all_records.csv")
    save_json(all_pairs, OUTPUT_DIR / "all_pairs.json")
    save_csv(all_pairs, OUTPUT_DIR / "all_pairs.csv")

    # 再额外输出一个最精简的 pair list，后面 dataset 最方便直接读这个
    simple_pairs = [
        {
            "split": p["split"],
            "patient_id": p["patient_id"],
            "pd_path": p["pd_path"],
            "pdfs_path": p["pdfs_path"],
        }
        for p in all_pairs
    ]
    save_json(simple_pairs, OUTPUT_DIR / "all_pairs_simple.json")
    save_csv(simple_pairs, OUTPUT_DIR / "all_pairs_simple.csv")

    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()