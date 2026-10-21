from pathlib import Path
import h5py
import numpy as np

SEED = 42
HDF5_PATH = Path(r"C:\RadioML2018\raw\GOLD_XYZ_OSC.0001_1024.hdf5")
SPLIT_PATH = Path("data_splits.npz")
OUTPUT_PATH = Path("label_subsets_v2.npz")

if OUTPUT_PATH.exists():
    raise FileExistsError(f"{OUTPUT_PATH} exists; refusing to overwrite.")

with np.load(SPLIT_PATH) as split:
    train_idx = split["train_idx"].copy()

# h5py requires increasing indices for fancy indexing.
order = np.argsort(train_idx)
labels = np.empty(len(train_idx), dtype=np.int64)

with h5py.File(HDF5_PATH, "r") as f:
    labels[order] = np.argmax(f["Y"][train_idx[order]], axis=1)

rng = np.random.default_rng(SEED)
classes = np.unique(labels)
fractions = (0.01, 0.05, 0.10, 0.25, 1.00)

# Shuffle each class exactly once, then reuse its ordering for every budget.
class_orders = {}
for cls in classes:
    indices = train_idx[labels == cls].copy()
    rng.shuffle(indices)
    class_orders[int(cls)] = indices

subsets = {}
previous = None

for fraction in fractions:
    chosen = []
    for cls in classes:
        indices = class_orders[int(cls)]
        count = len(indices) if fraction == 1.0 else max(
            1, int(round(fraction * len(indices)))
        )
        chosen.extend(indices[:count])

    selected = np.array(chosen, dtype=np.int32)
    rng.shuffle(selected)
    subsets[f"train_{int(fraction * 100)}pct"] = selected

    if previous is not None:
        assert set(previous).issubset(set(selected)), (
            f"Subset nesting failed at {fraction:.0%}"
        )
    previous = selected

    selected_labels = labels[np.isin(train_idx, selected)]
    counts = np.bincount(selected_labels, minlength=len(classes))
    print(
        f"{fraction:>5.0%}: {len(selected):,} samples; "
        f"per-class min/max={counts.min()}/{counts.max()}"
    )

np.savez_compressed(
    OUTPUT_PATH,
    seed=np.array(SEED),
    fractions=np.array(fractions),
    **subsets,
)

print("Nested subsets: PASS")
print("Saved:", OUTPUT_PATH.resolve())
