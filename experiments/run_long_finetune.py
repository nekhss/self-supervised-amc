"""Run experiments/train_classifier.py unchanged for a longer budget, with a manifest
and a per-epoch metrics table.

The training code is exactly train_classifier.py. As in finetune_cached.py, the only
substitution is the dataset class: RadioMLDataset -> CachedRadioMLDataset
(bit-identical memory-mapped copies). This wrapper adds:
  * a new, non-empty-refusing output directory;
  * manifest.json: git commit and dirty flag, Python/PyTorch/CUDA/cuDNN/NumPy versions,
    GPU, full argument list, SHA-256 of the MAE init checkpoint and the index files,
    timestamps;
  * train.log: the full console output of train_classifier.py;
  * per_epoch.csv / per_epoch.json: train loss, validation loss, accuracy and macro-F1
    for every epoch, parsed from train_classifier.py's own epoch lines and checked
    against its "Best validation macro-F1" line;
  * the best checkpoint (chosen by validation macro-F1 inside train_classifier.py) and
    its SHA-256.

Learning-rate schedule: train_classifier.py's CosineAnnealingLR(T_max=--epochs),
stepped once per epoch. With --epochs 50 the lr decays from --lr to 0 over 50 epochs:
it is a full 50-epoch cosine, not a continuation of the 10-epoch schedule.

The test split is never loaded (train_classifier.py only reads the label subset and
the validation indices).

Example (1% labels, seed 42, MAE init, primary validation subset):
  python -u experiments/run_long_finetune.py --run-dir C:\\dlruns\\finetune_50ep_v1\\mae_1pct_seed42 -- ^
      --label-fraction 1 --init-checkpoint C:\\dlruns\\cnn_mae_full_v1_best.pt --epochs 50 ^
      --batch-size 128 --lr 1e-3 --weight-decay 1e-4 ^
      --validation-subset validation_subset_v1.npz --seed 42
(--output is set by the wrapper to <run-dir>/best.pt.)
"""

import argparse
import contextlib
import csv
import hashlib
import importlib.util
import json
import math
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.cached_dataset import CachedRadioMLDataset  # noqa: E402

EPOCH_RE = re.compile(
    r"Epoch (\d+)/(\d+) \| train loss=([\d.]+) \((\d+) samples\) \| val loss=([\d.]+) \| "
    r"val accuracy=([\d.]+) \| val macro-F1=([\d.]+)")
BEST_RE = re.compile(r"Best validation macro-F1: ([\d.]+) at epoch (\d+)")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 22):
            h.update(block)
    return h.hexdigest()


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True, timeout=30).stdout.strip()
    except Exception:
        return None


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("train_args", nargs=argparse.REMAINDER,
                    help="Arguments for train_classifier.py, after '--'.")
    args = ap.parse_args()
    train_args = [a for a in args.train_args if a != "--"]
    if any(a == "--output" or a.startswith("--output=") for a in train_args):
        raise SystemExit("Do not pass --output; the wrapper sets <run-dir>/best.pt.")

    run_dir = Path(args.run_dir).resolve()
    if run_dir.is_relative_to(ROOT):
        raise SystemExit("--run-dir must be outside the repository.")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"{run_dir} exists and is not empty; refusing to overwrite.")
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.pt"
    full_args = train_args + ["--output", str(best_path)]

    # Load train_classifier.py as a module (defines functions only; main() runs below).
    spec = importlib.util.spec_from_file_location(
        "train_classifier", ROOT / "experiments" / "train_classifier.py")
    tc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tc)
    tc.RadioMLDataset = CachedRadioMLDataset  # same substitution as finetune_cached.py

    def file_hash(rel):
        p = Path(rel) if Path(rel).is_absolute() else ROOT / rel
        return {"path": str(p), "sha256": sha256_file(p)} if p.is_file() else None

    init = None
    if "--init-checkpoint" in train_args:
        init = train_args[train_args.index("--init-checkpoint") + 1]
    valsub = None
    if "--validation-subset" in train_args:
        valsub = train_args[train_args.index("--validation-subset") + 1]
    status = git("status", "--porcelain")
    manifest = {
        "purpose": "longer-budget fine-tuning with train_classifier.py unchanged (cached data)",
        "train_classifier_args": full_args,
        "train_classifier_sha256": sha256_file(ROOT / "experiments" / "train_classifier.py"),
        "lr_schedule": "CosineAnnealingLR(T_max=epochs), stepped per epoch, from --lr to 0",
        "init_checkpoint": file_hash(init) if init else None,
        "label_subsets": file_hash("label_subsets_v2.npz"),
        "data_splits": file_hash("data_splits.npz"),
        "validation_subset": file_hash(valsub) if valsub else None,
        "test_split_used": False,
        "git": {"commit": git("rev-parse", "HEAD"), "dirty": bool(status) if status is not None else None,
                "status": status},
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                     "numpy": np.__version__},
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    requested_epochs = int(train_args[train_args.index("--epochs") + 1]) if "--epochs" in train_args else 10
    base_lr = float(train_args[train_args.index("--lr") + 1]) if "--lr" in train_args else 1e-3
    manifest.update(status="running", requested_epochs=requested_epochs)

    def write_manifest():
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    def fail(msg):
        manifest.update(status="failed", error=msg, failed_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        write_manifest()
        raise SystemExit(f"[wrapper] FAILED: {msg}")

    write_manifest()

    # Record the exact dict returned by train_classifier.evaluate() for every epoch
    # (the console prints only 4 decimals). Pass-through: return value is unchanged.
    val_records = []
    original_evaluate = tc.evaluate

    def recording_evaluate(*a, **k):
        result = original_evaluate(*a, **k)
        val_records.append(dict(result))
        return result

    tc.evaluate = recording_evaluate

    log_path = run_dir / "train.log"
    t0 = time.time()
    try:
        with open(log_path, "w", encoding="utf-8") as log, \
                contextlib.redirect_stdout(Tee(sys.stdout, log)):
            sys.argv = ["train_classifier.py"] + full_args
            tc.main()
    except BaseException as exc:  # includes KeyboardInterrupt; a kill leaves status "running"
        fail(f"train_classifier.main() raised {exc!r} after {len(val_records)} validation passes")
    elapsed = time.time() - t0

    text = log_path.read_text(encoding="utf-8")
    logged = [m.groups() for m in EPOCH_RE.finditer(text)]
    best = BEST_RE.search(text)
    if len(logged) != requested_epochs or len(val_records) != requested_epochs or best is None:
        fail(f"incomplete: {len(logged)} epoch lines, {len(val_records)} validation passes, "
             f"requested {requested_epochs}, best line {'found' if best else 'missing'}")
    rows = []
    for i, (g, rec) in enumerate(zip(logged, val_records), start=1):
        if int(g[0]) != i or int(g[1]) != requested_epochs:
            fail(f"unexpected epoch line {g[0]}/{g[1]} at position {i}")
        if abs(float(g[6]) - rec["macro_f1"]) > 5e-5 or abs(float(g[5]) - rec["accuracy"]) > 5e-5:
            fail(f"epoch {i}: logged and recorded validation metrics disagree")
        rows.append({
            "epoch": i,
            "lr_used": 0.5 * base_lr * (1 + math.cos(math.pi * (i - 1) / requested_epochs)),
            "train_loss": float(g[2]),  # as printed by train_classifier.py (6 decimals)
            "train_samples": int(g[3]),
            "val_loss": rec["loss"], "val_accuracy": rec["accuracy"],
            "val_macro_f1": rec["macro_f1"], "val_samples": rec["samples"],
        })

    # train_classifier.py keeps the first epoch with the strictly highest macro-F1.
    selected = 1
    for r in rows:
        if r["val_macro_f1"] > rows[selected - 1]["val_macro_f1"]:
            selected = r["epoch"]
    ckpt = torch.load(best_path, map_location="cpu", weights_only=True)
    sched = ckpt["scheduler_state_dict"]
    if not (ckpt["epoch"] == selected == int(best.group(2))
            and ckpt["best_val_macro_f1"] == rows[selected - 1]["val_macro_f1"]):
        fail(f"selection mismatch: checkpoint epoch {ckpt['epoch']}, recorded argmax {selected}, "
             f"log best {best.group(2)}")
    expected_next_lr = 0.5 * base_lr * (1 + math.cos(math.pi * selected / requested_epochs))
    if sched["T_max"] != requested_epochs or not math.isclose(
            ckpt["optimizer_state_dict"]["param_groups"][0]["lr"], expected_next_lr, rel_tol=1e-9):
        fail(f"scheduler check failed: T_max {sched['T_max']}, saved lr "
             f"{ckpt['optimizer_state_dict']['param_groups'][0]['lr']} vs expected {expected_next_lr}")

    with open(run_dir / "per_epoch.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (run_dir / "per_epoch.json").write_text(json.dumps(rows, indent=2))
    manifest.update(status="complete", completed_epochs=len(rows),
                    finished=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=elapsed,
                    best_epoch=selected,
                    best_val_macro_f1=ckpt["best_val_macro_f1"],
                    best_checkpoint={"path": str(best_path), "sha256": sha256_file(best_path)},
                    peak_gpu_memory_mib=(torch.cuda.max_memory_allocated() / 2**20
                                         if torch.cuda.is_available() else None))
    write_manifest()
    print(f"[wrapper] complete: {len(rows)}/{requested_epochs} epochs; best epoch {selected}, "
          f"val macro-F1 {ckpt['best_val_macro_f1']:.6f}; see {run_dir / 'per_epoch.csv'}")


if __name__ == "__main__":
    main()
