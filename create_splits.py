from pathlib import Path
import h5py
import numpy as np
from sklearn.model_selection import train_test_split

SEED = 42
DATA_PATH = Path(r"C:\RadioML2018\raw\GOLD_XYZ_OSC.0001_1024.hdf5")
OUTPUT_PATH = Path("data_splits.npz")

if not DATA_PATH.is_file():
    raise FileNotFoundError(DATA_PATH)

if OUTPUT_PATH.exists():
    raise FileExistsError(
        f"{OUTPUT_PATH} already exists. Refusing to overwrite it."
    )

with h5py.File(DATA_PATH, "r") as f:
    Y = f["Y"][:]
    Z = f["Z"][:].reshape(-1)

    if len(Y) != len(Z):
        raise ValueError("Label and SNR counts differ.")

    if not np.isfinite(Y).all() or not np.isfinite(Z).all():
        raise ValueError("Non-finite label or SNR metadata.")

    if not np.all((Y == 0) | (Y == 1)) or not np.all(Y.sum(axis=1) == 1):
        raise ValueError("Labels are not valid one-hot rows.")

    labels = np.argmax(Y, axis=1).astype(np.int16)

    # Each class-SNR combination is a stratum.
    pairs = np.column_stack((labels, Z.astype(np.int16)))
    _, strata = np.unique(pairs, axis=0, return_inverse=True)

    all_indices = np.arange(len(labels), dtype=np.int32)

    train_idx, remaining_idx = train_test_split(
        all_indices,
        train_size=0.70,
        random_state=SEED,
        stratify=strata,
    )

    val_idx, test_idx = train_test_split(
        remaining_idx,
        test_size=0.50,
        random_state=SEED,
        stratify=strata[remaining_idx],
    )

    # Save only indices and reproducibility metadata, not signal data.
    np.savez_compressed(
        OUTPUT_PATH,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        seed=np.array(SEED),
        split_ratios=np.array([0.70, 0.15, 0.15]),
    )

# Verify partition integrity.
assert len(train_idx) + len(val_idx) + len(test_idx) == len(labels)
assert len(np.unique(np.concatenate([train_idx, val_idx, test_idx]))) == len(labels)

for name, idx in [
    ("Train", train_idx),
    ("Validation", val_idx),
    ("Test", test_idx),
]:
    counts = np.bincount(strata[idx], minlength=len(np.unique(strata)))
    print(
        f"{name}: {len(idx):,} samples "
        f"({len(idx) / len(labels):.2%}), "
        f"stratum min/max={counts.min()}/{counts.max()}"
    )

print("Total:", len(labels))
print("Seed:", SEED)
print("Saved:", OUTPUT_PATH.resolve())
print("Disjoint and complete partitions: PASS")
