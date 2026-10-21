"""Build read-only NumPy caches of selected RadioML index sets.

Why: train_classifier.py reads one sample at a time from the 21 GB HDF5 file
(~385 samples/s measured). Fine-tuning and test evaluation over all label
budgets would take days at that rate. This script reads the HDF5 file once,
sequentially, and writes each requested index set to .npy files that can be
memory-mapped. Values are copied bit-for-bit (float32 IQ, argmax label, SNR).

Nothing in the project is modified. Outputs go to --cache-dir (default
C:\\dlruns\\cache, outside OneDrive). Each set is written to a temporary
folder and renamed only after it is complete and verified.

Sets:
  train_25pct  label subset (contains the nested 1/5/10% subsets)
  train_100pct the full training split (needed only for the 100% budget)
  valsub       validation_subset_v1.npz (classifier checkpoint selection)
  test         the full held-out test split (final evaluation only)
"""

import argparse
import gc
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HDF5 = Path(r"C:\RadioML2018\raw\GOLD_XYZ_OSC.0001_1024.hdf5")
DEFAULT_CACHE = Path(r"C:\dlruns\cache")
CHUNK_ROWS = 16384  # 16384 * 1024 * 2 * 4 B = 128 MiB per read
BYTES_PER_SAMPLE = 1024 * 2 * 4 + 2


def load_index_set(name):
    if name == "test":
        with np.load(ROOT / "data_splits.npz") as s:
            return s["test_idx"].astype(np.int64)
    if name == "valsub":
        with np.load(ROOT / "validation_subset_v1.npz") as s:
            return s["val_indices"].astype(np.int64)
    if name.startswith("train_"):
        with np.load(ROOT / "label_subsets_v2.npz") as s:
            return s[name].astype(np.int64)
    raise ValueError(f"Unknown set: {name}")


def digest(indices):
    return hashlib.sha256(np.ascontiguousarray(indices).tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sets", nargs="+",
                   default=["valsub", "test", "train_25pct"],
                   choices=["valsub", "test", "train_25pct", "train_100pct"])
    p.add_argument("--hdf5", default=str(DEFAULT_HDF5))
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    p.add_argument("--verify-samples", type=int, default=300)
    args = p.parse_args()

    h5_path = Path(args.hdf5)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    todo = {}
    for name in args.sets:
        final = cache_dir / name
        if (final / "meta.json").is_file():
            print(f"[skip] {name}: cache already exists at {final}")
            continue
        idx = np.sort(load_index_set(name))
        if len(np.unique(idx)) != len(idx):
            raise ValueError(f"{name}: duplicate indices")
        todo[name] = idx
    if not todo:
        print("Nothing to build.")
        return

    need = sum(len(i) for i in todo.values()) * BYTES_PER_SAMPLE
    free = shutil.disk_usage(cache_dir).free
    print(f"Required: {need / 1e9:.2f} GB, free on drive: {free / 1e9:.2f} GB")
    if need > 0.9 * free:
        sys.exit("Not enough free disk space. Build fewer --sets.")

    with h5py.File(h5_path, "r") as f:
        X, Y, Z = f["X"], f["Y"], f["Z"]
        n_total = X.shape[0]
        print(f"HDF5 X: shape={X.shape} chunks={X.chunks} "
              f"compression={X.compression}")
        for name, idx in todo.items():
            if idx[0] < 0 or idx[-1] >= n_total:
                raise ValueError(f"{name}: index out of range")

        outs = {}
        for name, idx in todo.items():
            tmp = cache_dir / f"{name}.tmp"
            if tmp.exists():
                shutil.rmtree(tmp)  # incomplete previous attempt only
            tmp.mkdir()
            outs[name] = {
                "tmp": tmp,
                "x": np.lib.format.open_memmap(
                    tmp / "x.npy", mode="w+", dtype=np.float32,
                    shape=(len(idx), 1024, 2)),
                "y": np.empty(len(idx), dtype=np.int8),
                "snr": np.empty(len(idx), dtype=np.int8),
                "filled": 0,
            }

        lo_all = min(int(i[0]) for i in todo.values())
        hi_all = max(int(i[-1]) for i in todo.values()) + 1
        started = time.time()
        for start in range(lo_all - lo_all % CHUNK_ROWS, hi_all, CHUNK_ROWS):
            end = min(start + CHUNK_ROWS, n_total)
            wanted = {}
            for name, idx in todo.items():
                a, b = np.searchsorted(idx, [start, end])
                if b > a:
                    wanted[name] = (a, b)
            if not wanted:
                continue
            xb = X[start:end]
            yb = np.argmax(Y[start:end], axis=1).astype(np.int8)
            zb = Z[start:end].reshape(-1).astype(np.int8)
            for name, (a, b) in wanted.items():
                rows = todo[name][a:b] - start
                o = outs[name]
                sel = xb[rows]
                if not np.isfinite(sel).all():
                    raise RuntimeError(f"{name}: non-finite IQ near row {start}")
                o["x"][a:b] = sel
                o["y"][a:b] = yb[rows]
                o["snr"][a:b] = zb[rows]
                o["filled"] += b - a
            done = end / hi_all
            el = time.time() - started
            print(f"\r{done:6.1%}  {el / 60:5.1f} min elapsed, "
                  f"~{el / max(done, 1e-9) * (1 - done) / 60:5.1f} min left",
                  end="", flush=True)
        print()

        rng = np.random.default_rng(0)
        for name, idx in todo.items():
            o = outs[name]
            if o["filled"] != len(idx):
                raise RuntimeError(f"{name}: filled {o['filled']} of {len(idx)}")
            o["x"].flush()
            # Spot-check against direct HDF5 reads (same as RadioMLDataset).
            for r in rng.choice(len(idx), min(args.verify_samples, len(idx)),
                                replace=False):
                g = int(idx[r])
                if not (np.array_equal(o["x"][r], X[g])
                        and o["y"][r] == np.argmax(Y[g])
                        and o["snr"][r] == int(Z[g, 0])):
                    raise RuntimeError(f"{name}: mismatch at index {g}")
            del o["x"]
            gc.collect()  # release the memmap handle before renaming (Windows)
            np.save(o["tmp"] / "indices.npy", idx)
            np.save(o["tmp"] / "y.npy", o["y"])
            np.save(o["tmp"] / "snr.npy", o["snr"])
            meta = {
                "set": name, "count": int(len(idx)),
                "indices_sha256": digest(idx),
                "source": str(h5_path),
                "source_bytes": h5_path.stat().st_size,
                "source_mtime": h5_path.stat().st_mtime,
                "layout": "x[n,1024,2] float32 (HDF5 order); y argmax int8; "
                          "snr int8; rows sorted by global index",
                "verified_samples": int(min(args.verify_samples, len(idx))),
                "built": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            (o["tmp"] / "meta.json").write_text(json.dumps(meta, indent=2))
            o["tmp"].rename(cache_dir / name)
            print(f"[done] {name}: {len(idx):,} samples, "
                  f"{meta['verified_samples']} spot-checks passed")


if __name__ == "__main__":
    main()
