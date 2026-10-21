"""Read-only loader, validation and split protocol for RadioML 2016.10a.

The pickle is a Python-2-era dict {(modulation, snr): float32 [1000, 2, 128]}.
It is unpickled with a restricted Unpickler that only allows numpy array
reconstruction, so the file cannot execute arbitrary code.

Global example index = position in the canonical order used here:
modulations in CLASSES order, SNRs ascending, examples in stored order.
"""

import hashlib
import os
import pickle
from pathlib import Path

import numpy as np

DEFAULT_PKL_PATH = Path(os.environ.get(
    "RML2016_PKL",
    r"C:\Users\admin\Downloads\rf-continual-learning\rf-continual-learning"
    r"\data\raw\RML2016.10a_dict.pkl",
))

CLASSES = ("8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK",
           "PAM4", "QAM16", "QAM64", "QPSK", "WBFM")
SNRS = tuple(range(-20, 20, 2))
EXAMPLES_PER_GROUP = 1000
SIGNAL_SHAPE = (2, 128)

SPLIT_SEED = 42
SPLIT_COUNTS = (700, 150, 150)  # train / val / test per (class, SNR) stratum
LABEL_BUDGETS = (1, 5, 10, 25, 100)  # percent of the training split

_ALLOWED_GLOBALS = {
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("_codecs", "encode"),  # bytes in protocol-2 pickles written by Python 3
}


class _NumpyOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in _ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(f"Blocked global {module}.{name}")
        if module.startswith("numpy.core."):
            module = "numpy._core." + module[len("numpy.core."):]
        return super().find_class(module, name)


def sha256_file(path, chunk=1 << 22):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def load_raw_dict(path=DEFAULT_PKL_PATH):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with open(path, "rb") as f:
        return _NumpyOnlyUnpickler(f, encoding="latin1").load()


def audit_dict(data):
    """Summarise structure without exposing IQ values. Never raises."""
    report = {"root_type": type(data).__name__}
    if not isinstance(data, dict):
        return report
    keys = list(data.keys())
    report["num_keys"] = len(keys)
    report["bad_keys"] = [repr(k) for k in keys if not (
        isinstance(k, tuple) and len(k) == 2
        and isinstance(k[0], str) and isinstance(k[1], (int, np.integer)))]
    good = [k for k in keys if repr(k) not in report["bad_keys"]]
    report["classes"] = sorted({k[0] for k in good})
    report["snrs"] = sorted({int(k[1]) for k in good})
    shapes, dtypes, nonfinite, total = {}, {}, [], 0
    for k in good:
        a = data[k]
        if not isinstance(a, np.ndarray):
            nonfinite.append(f"{k}: not ndarray ({type(a).__name__})")
            continue
        shapes[str(a.shape)] = shapes.get(str(a.shape), 0) + 1
        dtypes[str(a.dtype)] = dtypes.get(str(a.dtype), 0) + 1
        total += a.shape[0] if a.ndim else 0
        if not np.isfinite(a).all():
            nonfinite.append(str(k))
    report.update(shapes=shapes, dtypes=dtypes, total_examples=total,
                  nonfinite_groups=nonfinite)
    report["missing_groups"] = [
        f"{c}@{s}" for c in CLASSES for s in SNRS if (c, s) not in data]
    report["unexpected_classes"] = sorted(set(report["classes"]) - set(CLASSES))
    report["unexpected_snrs"] = sorted(set(report["snrs"]) - set(SNRS))
    return report


def validate_dict(data):
    """Raise ValueError listing every discrepancy from the expected layout."""
    r = audit_dict(data)
    problems = []
    if r["root_type"] != "dict":
        raise ValueError(f"Root is {r['root_type']}, expected dict")
    for field in ("bad_keys", "nonfinite_groups", "missing_groups",
                  "unexpected_classes", "unexpected_snrs"):
        if r[field]:
            problems.append(f"{field}: {r[field][:10]}")
    expected_shape = str((EXAMPLES_PER_GROUP,) + SIGNAL_SHAPE)
    if set(r["shapes"]) != {expected_shape}:
        problems.append(f"shapes {r['shapes']} != {expected_shape}")
    if set(r["dtypes"]) != {"float32"}:
        problems.append(f"dtypes {r['dtypes']} != float32")
    if r["num_keys"] != len(CLASSES) * len(SNRS):
        problems.append(f"num_keys {r['num_keys']} != {len(CLASSES) * len(SNRS)}")
    if problems:
        raise ValueError("RML2016 layout check failed: " + "; ".join(problems))
    return r


def to_arrays(data):
    """Canonical (x [N, 2, 128] float32, y int64, snr int64) arrays."""
    validate_dict(data)
    xs, ys, snrs = [], [], []
    for label, name in enumerate(CLASSES):
        for snr in SNRS:
            group = data[(name, snr)]
            xs.append(group)
            ys.append(np.full(len(group), label, dtype=np.int64))
            snrs.append(np.full(len(group), snr, dtype=np.int64))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(snrs)


def load_rml2016(path=DEFAULT_PKL_PATH):
    return to_arrays(load_raw_dict(path))


def per_example_rms(x):
    """sqrt(mean_t(I^2 + Q^2)) per example, for x shaped [N, 2, T]."""
    x = np.asarray(x, dtype=np.float64)
    return np.sqrt((x ** 2).sum(axis=1).mean(axis=-1))


def budget_count(train_count, percent):
    """Examples per stratum for a label budget; must be a whole number >= 1."""
    exact = train_count * percent / 100
    count = int(round(exact))
    if count < 1 or abs(count - exact) > 1e-9:
        raise ValueError(
            f"{percent}% of {train_count} = {exact} is not a whole number >= 1")
    return count


def make_splits(y, snr, seed=SPLIT_SEED, counts=SPLIT_COUNTS,
                budgets=LABEL_BUDGETS):
    """Stratified (class x SNR) example-level split with nested label subsets.

    Within each stratum one permutation (from a single seeded generator, strata
    visited in canonical order) is cut into train/val/test; the k% subset is the
    first budget_count(...) entries of that stratum's train part, so subsets are
    nested by construction.
    """
    y, snr = np.asarray(y), np.asarray(snr)
    rng = np.random.default_rng(seed)
    n_train, n_val, n_test = counts
    parts = {"train_idx": [], "val_idx": [], "test_idx": []}
    subsets = {p: [] for p in budgets}
    per_budget = {p: budget_count(n_train, p) for p in budgets}
    for label in range(len(CLASSES)):
        for s in SNRS:
            members = np.flatnonzero((y == label) & (snr == s))
            if len(members) != n_train + n_val + n_test:
                raise ValueError(
                    f"Stratum {CLASSES[label]}@{s} has {len(members)} examples")
            perm = members[rng.permutation(len(members))]
            train = perm[:n_train]
            parts["train_idx"].append(train)
            parts["val_idx"].append(perm[n_train:n_train + n_val])
            parts["test_idx"].append(perm[n_train + n_val:])
            for p in budgets:
                subsets[p].append(train[:per_budget[p]])
    out = {k: np.sort(np.concatenate(v)).astype(np.int64)
           for k, v in parts.items()}
    for p in budgets:
        out[f"train_{p}pct"] = np.sort(np.concatenate(subsets[p])).astype(np.int64)
    return out, per_budget


def check_splits(splits, num_examples, budgets=LABEL_BUDGETS):
    """Raise AssertionError if partitions overlap, miss examples, or subsets
    are not nested inside the training split."""
    tr, va, te = (set(splits[k].tolist()) for k in ("train_idx", "val_idx", "test_idx"))
    for name in ("train_idx", "val_idx", "test_idx"):
        assert len(np.unique(splits[name])) == len(splits[name]), f"{name} has duplicates"
    assert not (tr & va), "train/val overlap"
    assert not (tr & te), "train/test overlap"
    assert not (va & te), "val/test overlap"
    assert len(tr) + len(va) + len(te) == num_examples, "partitions not exhaustive"
    assert tr | va | te == set(range(num_examples)), "partitions not exhaustive"
    previous = set()
    for p in sorted(budgets):
        current = set(splits[f"train_{p}pct"].tolist())
        assert current <= tr, f"{p}% subset leaves the training split"
        assert previous <= current, f"{p}% subset does not contain the smaller subset"
        previous = current
    assert previous == tr, "100% subset is not the full training split"
    return True


def split_digest(splits):
    """Order-independent SHA-256 over every split array (name + int64 bytes)."""
    digest = hashlib.sha256()
    for name in sorted(splits):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(splits[name], dtype=np.int64).tobytes())
    return digest.hexdigest()


def load_splits(path):
    with np.load(path) as f:
        return {k: f[k].astype(np.int64) for k in f.files}
