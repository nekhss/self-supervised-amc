from pathlib import Path
import h5py
import numpy as np

files = list(Path(r"C:\RadioML2018\raw").rglob("*.hdf5"))

if len(files) != 1:
    raise RuntimeError(f"Expected one HDF5 file, found {len(files)}: {files}")

path = files[0]
print("Dataset:", path)
print("File size (GB):", round(path.stat().st_size / 1e9, 2))

with h5py.File(path, "r") as f:
    print("Keys:", list(f.keys()))

    for key in f.keys():
        if isinstance(f[key], h5py.Dataset):
            print(key, f[key].shape, f[key].dtype)

    for key in ("X", "Y", "Z"):
        if key not in f:
            raise KeyError(f"Missing expected key: {key}")

    X, Y, Z = f["X"], f["Y"], f["Z"]

    print("Samples:", X.shape[0])
    print("Signal shape:", X.shape[1:])
    print("Label shape:", Y.shape[1:])
    print("SNR shape:", Z.shape[1:])

    x, y, z = X[:8], Y[:8], Z[:8]
    print("First 8 signals finite:", np.isfinite(x).all())
    print("First 8 labels finite:", np.isfinite(y).all())
    print("First 8 SNR values finite:", np.isfinite(z).all())
    print("Sample label rows:", y[:3])
    print("Sample SNR rows:", z[:8])
