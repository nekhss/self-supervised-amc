from pathlib import Path
from collections import Counter
import h5py
import numpy as np

root = Path(r"C:\RadioML2018\raw")
files = list(root.rglob("GOLD_XYZ_OSC.0001_1024.hdf5"))

if len(files) != 1:
    raise RuntimeError(f"Expected exactly one HDF5 file, found {len(files)}")

path = files[0]
class_counts = Counter()
snr_counts = Counter()
invalid_labels = 0
nonfinite_signal_samples = 0
signal_min = float("inf")
signal_max = float("-inf")
signal_count = 0

with h5py.File(path, "r") as f:
    X, Y, Z = f["X"], f["Y"], f["Z"]
    n = X.shape[0]

    for start in range(0, n, 8192):
        end = min(start + 8192, n)

        x = X[start:end]
        y = Y[start:end]
        z = Z[start:end].reshape(-1)

        nonfinite_signal_samples += int(
            (~np.isfinite(x).all(axis=(1, 2))).sum()
        )

        if not np.isfinite(x).all():
            finite = x[np.isfinite(x)]
            if finite.size:
                signal_min = min(signal_min, float(finite.min()))
                signal_max = max(signal_max, float(finite.max()))
        else:
            signal_min = min(signal_min, float(x.min()))
            signal_max = max(signal_max, float(x.max()))

        signal_count += x.size

        valid_rows = (
            np.isfinite(y).all(axis=1)
            & np.isin(y, [0, 1]).all(axis=1)
            & (y.sum(axis=1) == 1)
        )
        invalid_labels += int((~valid_rows).sum())

        class_counts.update(np.argmax(y[valid_rows], axis=1).tolist())
        snr_counts.update(z.tolist())

    print("Total samples:", n)
    print("Signal shape:", X.shape)
    print("Signal dtype:", X.dtype)
    print("Invalid one-hot label rows:", invalid_labels)
    print("Samples with non-finite IQ:", nonfinite_signal_samples)
    print("IQ minimum:", signal_min)
    print("IQ maximum:", signal_max)
    print("Class counts by index:")
    for k in sorted(class_counts):
        print(f"  {k}: {class_counts[k]}")
    print("SNR counts:")
    for k in sorted(snr_counts):
        print(f"  {k} dB: {snr_counts[k]}")

print("\nClass mapping files:")
for name in ("classes-fixed.json", "classes-fixed.txt", "classes.txt"):
    candidates = list(root.rglob(name))
    for p in candidates:
        print(f"\n--- {p.name} ---")
        print(p.read_text(encoding="utf-8")[:5000])
