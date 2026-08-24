import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from torch.utils.data import DataLoader
from src.dataset_fastmri import FastMRISinglecoilDataset

root = "/rds/general/user/ah725/home/fastmri"

ds = FastMRISinglecoilDataset(
    root=root,
    split="val",
    center_fraction=0.08,
    acceleration=4,
    sample_rate=0.02,
)

print("Dataset size:", len(ds))

sample = ds[0]
for k, v in sample.items():
    if hasattr(v, "shape"):
        print(k, v.shape, v.dtype)
    else:
        print(k, v)

loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
batch = next(iter(loader))

print("\nBatch:")
for k, v in batch.items():
    if hasattr(v, "shape"):
        print(k, v.shape, v.dtype)
    else:
        print(k, type(v))
