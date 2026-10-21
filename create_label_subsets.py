from pathlib import Path
import numpy as np

SEED = 42
SPLIT_PATH = Path("data_splits.npz")
OUTPUT_PATH = Path("label_subsets.npz")
HDF5_PATH = Path(r"C:\RadioML2018\raw\GOLD_XYZ_OSC.0001_1024.hdf5")

if OUTPUT_PATH.exists():
    raise FileExistsError(
        f"{OUTPUT_PATH} already exists. Refusing to overwrite."
    )
if not SPLIT_PATH.is_file() or not HDF5_PATH.is_file():
    raise FileNotFoundError("Missing split file or HDF5 dataset.")

import h5py

with np.load(SPLIT_PATH) as split:
    train_idx = split["train_idx"].copy()

with h5py.File(HDF5_PATH, "r") as f:
    order = np.argsort(train_idx)
    labels = np.empty(len(train_idx), dtype=np.int64)
    labels[order] = np.argmax(f["Y"][train_idx[order]], axis=1)

rng = np.random.default_rng(SEED)
fractions = (0.01, 0.05, 0.10, 0.25, 1.00)
subsets = {}
classes = np.unique(labels)

for fraction in fractions:
    chosen = []

    for cls in classes:
        class_indices = train_idx[labels == cls].copy()
        rng.shuffle(class_indices)

        if fraction == 1.0:
            count = len(class_indices)
        else:
            count = max(1, int(round(fraction * len(class_indices))))

        chosen.extend(class_indices[:count].tolist())

    selected = np.array(chosen, dtype=np.int32)
    rng.shuffle(selected)
    subsets[f"train_{int(fraction * 100)}pct"] = selected

    selected_labels = np.concatenate([
        labels[np.isin(train_idx, selected)]
    ])
    unique, counts = np.unique(selected_labels, return_counts=True)

    print(
        f"{int(fraction * 100):>3}%: "
        f"{len(selected):,} samples; "
        f"class count min/max={counts.min()}/{counts.max()}; "
        f"classes={len(unique)}"
    )

np.savez_compressed(
    OUTPUT_PATH,
    seed=np.array(SEED),
    fractions=np.array(fractions),
    **subsets,
)

print("Saved:", OUTPUT_PATH.resolve())
print("Validation and test indices remain unchanged.")
