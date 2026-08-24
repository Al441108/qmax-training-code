from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import h5py
import numpy as np
from torch.utils.data import Sampler


RawShape = Tuple[int, int]


class RawShapeBatchSampler(Sampler[List[int]]):
    """
    Group fastMRI slices by exact raw k-space shape.

    Each yielded batch contains only one (Hraw, Wraw), so PyTorch can
    collate variable-size fastMRI acquisitions without spatial padding.

    Shuffling is deterministic for a given seed and epoch.
    """

    def __init__(
        self,
        dataset,
        batch_size: int = 4,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 1337,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

        self._path_shapes = self._read_unique_volume_shapes()
        self._buckets = self._build_index_buckets()

        if not self._buckets:
            raise RuntimeError("No raw-shape buckets were created")

    def _read_unique_volume_shapes(
        self,
    ) -> Dict[str, RawShape]:
        unique_paths = sorted(
            {row["fspd_path"] for row in self.dataset.rows}
        )

        path_shapes: Dict[str, RawShape] = {}

        for path_string in unique_paths:
            path = Path(path_string)

            if not path.is_file():
                raise FileNotFoundError(path)

            with h5py.File(path, "r") as handle:
                if "kspace" not in handle:
                    raise KeyError(f"kspace missing from {path}")

                shape = handle["kspace"].shape

            if len(shape) != 3:
                raise RuntimeError(
                    f"Expected [slices,H,W] in {path}, got {shape}"
                )

            path_shapes[path_string] = (
                int(shape[1]),
                int(shape[2]),
            )

        return path_shapes

    def _build_index_buckets(
        self,
    ) -> Dict[RawShape, List[int]]:
        buckets: Dict[RawShape, List[int]] = defaultdict(list)

        for index, row in enumerate(self.dataset.rows):
            path_string = row["fspd_path"]

            if path_string not in self._path_shapes:
                raise RuntimeError(
                    f"Shape unavailable for {path_string}"
                )

            buckets[self._path_shapes[path_string]].append(index)

        return dict(buckets)

    @property
    def shape_slice_counts(self) -> Counter:
        return Counter(
            {
                shape: len(indices)
                for shape, indices in self._buckets.items()
            }
        )

    @property
    def shape_volume_counts(self) -> Counter:
        counts = Counter()

        for path_string, shape in self._path_shapes.items():
            del path_string
            counts[shape] += 1

        return counts

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def state_dict(self) -> Dict[str, int]:
        return {"epoch": self.epoch}

    def load_state_dict(self, state: Dict[str, int]) -> None:
        self.epoch = int(state["epoch"])

    def __len__(self) -> int:
        total = 0

        for indices in self._buckets.values():
            if self.drop_last:
                total += len(indices) // self.batch_size
            else:
                total += math.ceil(
                    len(indices) / self.batch_size
                )

        return total

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.RandomState(self.seed + self.epoch)
        batches: List[List[int]] = []

        for shape in sorted(self._buckets):
            indices = list(self._buckets[shape])

            if self.shuffle:
                rng.shuffle(indices)

            for start in range(
                0,
                len(indices),
                self.batch_size,
            ):
                batch = indices[
                    start:start + self.batch_size
                ]

                if (
                    len(batch) < self.batch_size
                    and self.drop_last
                ):
                    continue

                batches.append(batch)

        if self.shuffle:
            rng.shuffle(batches)

        yield from batches