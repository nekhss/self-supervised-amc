"""Drop-in replacement for RadioMLDataset backed by tools/build_cache.py output.

Returns exactly the same item dict as RadioMLDataset:
signal float32 [2, 1024], label long, snr int16, index long (global HDF5 index).
Items follow the order of the `indices` argument, as in RadioMLDataset.
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

DEFAULT_CACHE_DIR = Path(os.environ.get("RADIOML_CACHE", r"C:\dlruns\cache"))


class _CacheSet:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.meta = json.loads((self.folder / "meta.json").read_text())
        self.indices = np.load(self.folder / "indices.npy")
        self.y = np.load(self.folder / "y.npy")
        self.snr = np.load(self.folder / "snr.npy")
        self._x = None

    @property
    def x(self):
        # Opened lazily so DataLoader workers each get their own mapping.
        if self._x is None:
            self._x = np.load(self.folder / "x.npy", mmap_mode="r")
        return self._x

    def rows_for(self, indices):
        """Row numbers for `indices`, or None if any index is not cached."""
        rows = np.searchsorted(self.indices, indices)
        rows = np.clip(rows, 0, len(self.indices) - 1)
        if np.array_equal(self.indices[rows], indices):
            return rows
        return None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_x"] = None
        return state


def _available_sets(cache_dir):
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        return []
    sets = [d for d in cache_dir.iterdir() if (d / "meta.json").is_file()]
    # Prefer the smallest set that covers the request (less disk, warmer cache).
    return sorted(sets, key=lambda d: json.loads((d / "meta.json").read_text())["count"])


class CachedRadioMLDataset(Dataset):
    def __init__(self, indices, cache_dir=DEFAULT_CACHE_DIR, **_ignored):
        self.indices = np.asarray(indices, dtype=np.int64)
        if self.indices.ndim != 1 or self.indices.size == 0:
            raise ValueError("indices must be a non-empty 1D array")
        if len(np.unique(self.indices)) != len(self.indices):
            raise ValueError("Indices out of range or duplicated")

        self.cache = None
        for folder in _available_sets(cache_dir):
            candidate = _CacheSet(folder)
            rows = candidate.rows_for(self.indices)
            if rows is not None:
                self.cache, self.rows = candidate, rows
                break
        if self.cache is None:
            raise FileNotFoundError(
                f"No cache under {cache_dir} contains all {len(self.indices):,} "
                "requested indices. Run tools/build_cache.py for the needed set."
            )
        print(f"[cache] {len(self.indices):,} samples served from "
              f"{self.cache.folder.name}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        r = int(self.rows[item])
        signal = np.array(self.cache.x[r], dtype=np.float32).T.copy()
        return {
            "signal": torch.from_numpy(signal),
            "label": torch.tensor(int(self.cache.y[r]), dtype=torch.long),
            "snr": torch.tensor(int(self.cache.snr[r]), dtype=torch.int16),
            "index": torch.tensor(int(self.indices[item]), dtype=torch.long),
        }

    # Fast path for evaluation: whole arrays for a row range of this dataset.
    def arrays(self):
        return self.cache, self.rows

    def close(self):
        self.cache._x = None
