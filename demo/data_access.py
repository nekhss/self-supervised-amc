"""Read-only access to the held-out RadioML 2018.01A test split.

Test position i means data_splits.npz["test_idx"][i], the order used by
experiments/evaluate_test.py, so a demo prediction can be compared with the stored
test-set prediction of the same model. Signals come from the project's own dataset
classes: CachedRadioMLDataset (memory-mapped cache) or, if the cache is missing,
RadioMLDataset (lazy HDF5 reads). Neither loads the full dataset into memory.
"""

import json
from dataclasses import dataclass

import numpy as np

import demo_config as cfg


class DataUnavailable(RuntimeError):
    pass


@dataclass
class Sample:
    position: int          # position in the test split
    global_index: int      # row in the original HDF5 file
    signal: np.ndarray     # float32 [2, 1024]: row 0 = I, row 1 = Q (raw values, no normalisation)
    label: int
    snr: int


class TestSplit:
    def __init__(self):
        if not cfg.SPLITS_FILE.is_file():
            raise DataUnavailable(f"Split file not found: {cfg.SPLITS_FILE}")
        with np.load(cfg.SPLITS_FILE) as s:
            self.test_idx = s["test_idx"].astype(np.int64)
        self.size = len(self.test_idx)

        self.backend, self._ds, err = None, None, []
        try:
            from src.data.cached_dataset import CachedRadioMLDataset
            ds = CachedRadioMLDataset(self.test_idx, cache_dir=cfg.CACHE_DIR)
            if ds.cache.folder.name != "test":
                raise DataUnavailable(f"cache served test indices from '{ds.cache.folder.name}'")
            self._ds, self.backend = ds, f"memory-mapped cache ({ds.cache.folder})"
        except Exception as exc:  # fall back to the original HDF5 reader
            err.append(f"cache: {exc}")
            try:
                from src.data.radioml_dataset import RadioMLDataset
                self._ds = RadioMLDataset(self.test_idx, hdf5_path=cfg.HDF5_PATH)
                self.backend = f"HDF5 ({cfg.HDF5_PATH})"
            except Exception as exc2:
                err.append(f"hdf5: {exc2}")
                raise DataUnavailable("No data source available. " + " | ".join(err))

        # Labels and SNR for browsing/filtering, in test_idx order.
        self.labels = self.snrs = None
        if self.backend.startswith("memory"):
            self.labels = np.asarray(self._ds.cache.y[self._ds.rows], dtype=np.int64)
            self.snrs = np.asarray(self._ds.cache.snr[self._ds.rows], dtype=np.int64)
        elif cfg.TEST_LABELS_FILE.is_file():
            with np.load(cfg.TEST_LABELS_FILE) as t:
                if np.array_equal(t["test_idx"], self.test_idx):
                    self.labels, self.snrs = t["y"].astype(np.int64), t["snr"].astype(np.int64)

    def get(self, position: int) -> Sample:
        if not 0 <= position < self.size:
            raise IndexError(f"Test position must be in [0, {self.size - 1}]")
        item = self._ds[position]
        g = int(item["index"])
        if g != int(self.test_idx[position]):
            raise RuntimeError("Dataset returned a sample outside the requested test position.")
        s = Sample(position, g, item["signal"].numpy().astype(np.float32),
                   int(item["label"]), int(item["snr"]))
        if self.labels is not None and (s.label != self.labels[position] or s.snr != self.snrs[position]):
            raise RuntimeError("Label/SNR mismatch between data source and stored test labels.")
        return s

    def matching(self, label=None, snr=None) -> np.ndarray:
        if self.labels is None:
            return np.arange(self.size)
        m = np.ones(self.size, bool)
        if label is not None:
            m &= self.labels == label
        if snr is not None:
            m &= self.snrs == snr
        return np.flatnonzero(m)


def load_class_names() -> list:
    """Class names exactly as recorded by evaluate_test.py in the primary result files."""
    for arm in cfg.ARMS:
        for budget, seeds in cfg.PLAN.items():
            p = cfg.primary_result(arm, budget, seeds[0])
            if p.is_file():
                names = json.loads(p.read_text())["class_names"]
                if len(names) == 24:
                    return names
    return [f"class {i}" for i in range(24)]
