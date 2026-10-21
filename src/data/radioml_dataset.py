from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


DEFAULT_HDF5_PATH = Path(
    r"C:\RadioML2018\raw\GOLD_XYZ_OSC.0001_1024.hdf5"
)


class RadioMLDataset(Dataset):
    """Lazy HDF5 dataset for RadioML 2018.01A."""

    def __init__(
        self,
        indices,
        hdf5_path=DEFAULT_HDF5_PATH,
    ):
        self.hdf5_path = Path(hdf5_path)
        self.indices = np.asarray(indices, dtype=np.int64)

        if not self.hdf5_path.is_file():
            raise FileNotFoundError(self.hdf5_path)

        if self.indices.ndim != 1 or self.indices.size == 0:
            raise ValueError("indices must be a non-empty 1D array")

        if np.any(self.indices < 0):
            raise ValueError("Indices cannot be negative")

        self._file = None

        # Validate indices against dataset size without reading signals.
        with h5py.File(self.hdf5_path, "r") as f:
            if not {"X", "Y", "Z"}.issubset(f.keys()):
                raise KeyError("HDF5 must contain X, Y, and Z")

            self.num_samples = f["X"].shape[0]

            if f["X"].shape[1:] != (1024, 2):
                raise ValueError(f"Unexpected IQ shape: {f['X'].shape}")

            if (
                np.any(self.indices >= self.num_samples)
                or len(np.unique(self.indices)) != len(self.indices)
            ):
                raise ValueError("Indices out of range or duplicated")

    def __len__(self):
        return len(self.indices)

    def _get_file(self):
        # Open a process-local handle lazily, useful with DataLoader workers.
        if self._file is None or not self._file.id.valid:
            self._file = h5py.File(self.hdf5_path, "r")
        return self._file

    def __getitem__(self, item):
        sample_idx = int(self.indices[item])
        f = self._get_file()

        # HDF5 stores [time, IQ-channel]; model receives [channel, time].
        signal = np.asarray(f["X"][sample_idx], dtype=np.float32).T.copy()
        label = int(np.argmax(f["Y"][sample_idx]))
        snr = int(f["Z"][sample_idx, 0])

        if not np.isfinite(signal).all():
            raise ValueError(f"Non-finite IQ at sample {sample_idx}")

        return {
            "signal": torch.from_numpy(signal),
            "label": torch.tensor(label, dtype=torch.long),
            "snr": torch.tensor(snr, dtype=torch.int16),
            "index": torch.tensor(sample_idx, dtype=torch.long),
        }

    def __getstate__(self):
        # Avoid serializing open HDF5 handles to worker processes.
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
