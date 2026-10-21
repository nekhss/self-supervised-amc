from pathlib import Path
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HDF5_PATH = Path(r"C:\RadioML2018\raw\GOLD_XYZ_OSC.0001_1024.hdf5")
SPLIT_PATH = ROOT / "data_splits.npz"
OUTPUT_PATH = ROOT / "validation_subset_v1.npz"

SEED = 42
SAMPLES_PER_STRATUM = 20

def main():
    if not HDF5_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {HDF5_PATH}")
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(f"Split file not found: {SPLIT_PATH}")

    splits = np.load(SPLIT_PATH)
    val_idx = np.asarray(splits["val_idx"], dtype=np.int64)

    rng = np.random.default_rng(SEED)
    selected = []
    selected_labels = []
    selected_snrs = []

    print("Reading labels and SNR metadata...")
    with h5py.File(HDF5_PATH, "r") as h5:
        labels_all = np.asarray(h5["Y"]).argmax(axis=1)
        snrs_all = np.asarray(h5["Z"]).reshape(-1).astype(np.int16)

    val_labels = labels_all[val_idx]
    val_snrs = snrs_all[val_idx]
    strata = sorted(
        set(zip(val_labels.tolist(), val_snrs.tolist())),
        key=lambda pair: (pair[0], pair[1]),
    )

    for label, snr in strata:
        candidates = val_idx[(val_labels == label) & (val_snrs == snr)]
        if len(candidates) < SAMPLES_PER_STRATUM:
            raise ValueError(
                f"Stratum class={label}, SNR={snr} has only "
                f"{len(candidates)} samples"
            )

        chosen = rng.choice(
            candidates, size=SAMPLES_PER_STRATUM, replace=False
        )
        selected.extend(chosen.tolist())
        selected_labels.extend([label] * SAMPLES_PER_STRATUM)
        selected_snrs.extend([snr] * SAMPLES_PER_STRATUM)

    selected = np.asarray(selected, dtype=np.int32)
    selected_labels = np.asarray(selected_labels, dtype=np.int16)
    selected_snrs = np.asarray(selected_snrs, dtype=np.int16)

    # Shuffle the final order deterministically, avoiding class/SNR ordering.
    order = rng.permutation(len(selected))
    selected = selected[order]
    selected_labels = selected_labels[order]
    selected_snrs = selected_snrs[order]

    # Confirm every selected index belongs to the original validation split.
    if not np.isin(selected, val_idx).all():
        raise RuntimeError("Selected index is outside the validation split.")
    if len(np.unique(selected)) != len(selected):
        raise RuntimeError("Duplicate validation indices detected.")

    counts = {}
    for label, snr in zip(selected_labels, selected_snrs):
        counts[(int(label), int(snr))] = counts.get((int(label), int(snr)), 0) + 1

    if len(counts) != len(strata):
        raise RuntimeError("Unexpected number of class/SNR strata.")
    if any(count != SAMPLES_PER_STRATUM for count in counts.values()):
        raise RuntimeError("Class/SNR subset is not balanced.")

    np.savez_compressed(
        OUTPUT_PATH,
        val_indices=selected,
        labels=selected_labels,
        snrs=selected_snrs,
        seed=np.int64(SEED),
        samples_per_stratum=np.int64(SAMPLES_PER_STRATUM),
        num_strata=np.int64(len(strata)),
    )

    print(f"Original validation samples: {len(val_idx):,}")
    print(f"Class/SNR strata: {len(strata)}")
    print(f"Samples per stratum: {SAMPLES_PER_STRATUM}")
    print(f"Selected validation samples: {len(selected):,}")
    print(f"Unique selected indices: {len(np.unique(selected)):,}")
    print(f"Output: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
