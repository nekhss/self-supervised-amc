"""Create the fixed RadioML 2016.10a split protocol for the transfer experiment.

Reads the pickle (read-only) and writes, under --out-dir:
  splits_seed42.npz     train/val/test indices + nested train_{1,5,10,25,100}pct
  protocol_seed42.json  class/SNR maps, counts, hashes, amplitude multiplier,
                        RMS distributions (RML2016 raw/scaled vs RML2018 train)

Protocol (see experiments/rml2016_transfer/PROTOCOL.md):
- 220 strata (11 classes x 20 SNRs) of exactly 1000 examples; each stratum is
  permuted once (seed 42) and cut 700/150/150 into train/val/test.
- Label budgets take the first 7/35/70/175/700 train examples of each stratum's
  permutation, so they are nested and exactly class x SNR balanced (no rounding).
- Amplitude multiplier = 1 / mean per-example RMS over the training split only.

Refuses to overwrite existing outputs.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import rml2016  # noqa: E402

DEFAULT_OUT = ROOT / "experiments" / "rml2016_transfer"
DEFAULT_RML2018_CACHE = Path(r"C:\dlruns\cache\train_100pct")
RML2018_TRAIN_COUNT = 1_789_132
PERCENTILES = (1, 5, 25, 50, 75, 95, 99)


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "percentiles": {str(p): float(np.percentile(values, p)) for p in PERCENTILES},
    }


def by_snr(values, snr):
    return {str(int(s)): describe(values[snr == s]) for s in np.unique(snr)}


def rml2018_rms(cache_dir, sample_size, seed):
    """RMS distribution of a seeded random sample of the RML2018 MAE
    pretraining data (the full RML2018 training split)."""
    meta = json.loads((cache_dir / "meta.json").read_text())
    if meta.get("count") != RML2018_TRAIN_COUNT:
        raise ValueError(f"{cache_dir} is not the full RML2018 training split")
    x = np.load(cache_dir / "x.npy", mmap_mode="r")  # [n, 1024, 2]
    snr = np.load(cache_dir / "snr.npy")
    rows = np.sort(np.random.default_rng(seed).choice(len(x), sample_size, replace=False))
    sample = np.asarray(x[rows], dtype=np.float32).transpose(0, 2, 1)
    rms = rml2016.per_example_rms(sample)
    return {
        "source": "RML2018.01A training split (MAE pretraining data), cached copy",
        "indices_sha256": meta.get("indices_sha256"),
        "sample_size": sample_size,
        "sample_seed": seed,
        "overall": describe(rms),
        "by_snr": by_snr(rms, snr[rows].astype(np.int64)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", default=str(rml2016.DEFAULT_PKL_PATH))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--seed", type=int, default=rml2016.SPLIT_SEED)
    parser.add_argument("--rml2018-cache", default=str(DEFAULT_RML2018_CACHE),
                        help="Folder from tools/build_cache.py; '' to skip the comparison.")
    parser.add_argument("--rml2018-sample", type=int, default=100_000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    split_path = out_dir / f"splits_seed{args.seed}.npz"
    meta_path = out_dir / f"protocol_seed{args.seed}.json"
    for p in (split_path, meta_path):
        if p.exists():
            raise FileExistsError(f"{p} exists; refusing to overwrite.")
    out_dir.mkdir(parents=True, exist_ok=True)

    pkl = Path(args.pkl)
    started = time.time()
    data_sha = rml2016.sha256_file(pkl)
    x, y, snr = rml2016.load_rml2016(pkl)
    print(f"Loaded {len(x):,} examples, x{tuple(x.shape)} in {time.time() - started:.1f}s")

    flat = np.ascontiguousarray(x.reshape(len(x), -1))
    duplicates = len(flat) - len(np.unique(flat, axis=0))
    if duplicates:
        raise ValueError(f"{duplicates} exact duplicate examples; split would leak.")

    splits, per_budget = rml2016.make_splits(y, snr, seed=args.seed)
    rml2016.check_splits(splits, len(x))
    digest = rml2016.split_digest(splits)
    print("Split checks: PASS (disjoint, exhaustive, nested)")

    rms = rml2016.per_example_rms(x)
    train = splits["train_idx"]
    multiplier = float(1.0 / rms[train].mean())
    scaled = rms * multiplier
    print(f"Amplitude multiplier (1 / mean train RMS): {multiplier:.6f}")

    counts = {}
    for name in ("train_idx", "val_idx", "test_idx") + tuple(
            f"train_{p}pct" for p in rml2016.LABEL_BUDGETS):
        idx = splits[name]
        table = np.zeros((len(rml2016.CLASSES), len(rml2016.SNRS)), dtype=int)
        np.add.at(table, (y[idx], (snr[idx] + 20) // 2), 1)
        counts[name] = {
            "total": int(len(idx)),
            "per_stratum_min": int(table.min()),
            "per_stratum_max": int(table.max()),
        }

    meta = {
        "experiment": "rml2016_transfer",
        "description": "Cross-dataset transfer RML2018.01A (CNN-MAE pretraining) -> RML2016.10a",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_file": {"name": pkl.name, "bytes": pkl.stat().st_size, "sha256": data_sha},
        "num_examples": int(len(x)),
        "signal_shape": list(rml2016.SIGNAL_SHAPE),
        "classes": {str(i): c for i, c in enumerate(rml2016.CLASSES)},
        "snrs": list(rml2016.SNRS),
        "index_order": "classes in listed order, SNR ascending, stored example order",
        "split_seed": args.seed,
        "split_rule": "per (class, SNR) stratum: one permutation, cut 700/150/150",
        "budget_rule": "first k train examples of each stratum's permutation; nested",
        "per_stratum_budget": {str(p): n for p, n in per_budget.items()},
        "counts": counts,
        "exact_duplicate_examples": int(duplicates),
        "split_sha256": digest,
        "split_file": split_path.name,
        "amplitude": {
            "rms_definition": "sqrt(mean over time of I^2 + Q^2), per example",
            "multiplier": multiplier,
            "multiplier_rule": "1 / mean per-example RMS over the RML2016 training split only",
            "applied_to": "every RML2016 example, identically in both arms",
            "rml2016_raw_train": describe(rms[train]),
            "rml2016_scaled_train": describe(scaled[train]),
            "rml2016_scaled_train_by_snr": by_snr(scaled[train], snr[train]),
        },
    }
    if args.rml2018_cache:
        meta["amplitude"]["rml2018_pretraining"] = rml2018_rms(
            Path(args.rml2018_cache), args.rml2018_sample, args.seed)

    np.savez_compressed(split_path, **{k: v.astype(np.int32) for k, v in splits.items()})
    reloaded = rml2016.load_splits(split_path)
    if rml2016.split_digest(reloaded) != digest:
        raise RuntimeError("Saved split file does not round-trip.")
    meta["split_file_sha256"] = rml2016.sha256_file(split_path)
    meta_path.write_text(json.dumps(meta, indent=2))

    a = meta["amplitude"]
    print(f"RML2016 raw train RMS mean {a['rml2016_raw_train']['mean']:.6f}; "
          f"scaled mean {a['rml2016_scaled_train']['mean']:.4f} "
          f"(p5 {a['rml2016_scaled_train']['percentiles']['5']:.4f}, "
          f"p95 {a['rml2016_scaled_train']['percentiles']['95']:.4f})")
    if "rml2018_pretraining" in a:
        r = a["rml2018_pretraining"]["overall"]
        print(f"RML2018 train RMS mean {r['mean']:.4f} "
              f"(p5 {r['percentiles']['5']:.4f}, p95 {r['percentiles']['95']:.4f})")
    print("Wrote", split_path)
    print("Wrote", meta_path)


if __name__ == "__main__":
    main()
