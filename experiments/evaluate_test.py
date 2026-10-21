"""Final held-out test evaluation of fine-tuned classifier checkpoints.

Run this only on checkpoints already selected by validation macro-F1
(train_classifier.py saves exactly that checkpoint). Nothing here trains,
tunes or selects; every checkpoint passed in is evaluated once on the full
test split (data_splits.npz:test_idx, 383,386 samples).

Outputs (in --out-dir):
  <run>.json              overall, per-class, per-SNR metrics + confusion matrix
  <run>_pred.npy          predictions in data_splits.npz:test_idx order
  test_labels_snr.npz     true labels and SNR in the same order
  test_summary.csv        one row per checkpoint

Example:
  python experiments/evaluate_test.py C:\\dlruns\\finetune\\*.pt
"""

import argparse
import csv
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.cached_dataset import CachedRadioMLDataset  # noqa: E402
from src.models.cnn import IQCNNClassifier  # noqa: E402

DEFAULT_CLASSES = Path(r"C:\RadioML2018\raw\classes-fixed.json")


def load_class_names(path):
    try:
        names = json.loads(Path(path).read_text())
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names, key=int)]
        if len(names) == 24:
            return [str(n) for n in names]
    except Exception as exc:  # report and fall back; never fatal
        print(f"[warn] class names unavailable ({exc!r}); using indices")
    return [str(i) for i in range(24)]


def predict(model, cache, rows, device, batch_size):
    order = np.argsort(rows)  # contiguous reads from the memmap
    sorted_rows = rows[order]
    preds = np.empty(len(rows), dtype=np.int64)
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(rows), batch_size):
            r = sorted_rows[s:s + batch_size]
            x = np.ascontiguousarray(cache.x[r[0]:r[-1] + 1][r - r[0]].transpose(0, 2, 1))
            x = torch.from_numpy(x).to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=(device.type == "cuda")):
                logits = model(x)
            preds[order[s:s + batch_size]] = logits.argmax(1).cpu().numpy()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return preds, time.time() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoints", nargs="+", help="Paths or glob patterns.")
    p.add_argument("--out-dir", default=r"C:\dlruns\results")
    p.add_argument("--classes", default=str(DEFAULT_CLASSES))
    p.add_argument("--batch-size", type=int, default=1024)
    args = p.parse_args()

    paths = []
    for pat in args.checkpoints:
        paths += [Path(x) for x in sorted(glob.glob(pat))] or [Path(pat)]
    paths = [x for x in paths if x.suffix == ".pt"]
    missing = [str(x) for x in paths if not x.is_file()]
    if missing or not paths:
        sys.exit(f"Missing checkpoints: {missing or args.checkpoints}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    with np.load(ROOT / "data_splits.npz") as s:
        test_idx = s["test_idx"].astype(np.int64)
    test_ds = CachedRadioMLDataset(test_idx)
    cache, rows = test_ds.arrays()
    y_true = cache.y[rows].astype(np.int64)
    snr = cache.snr[rows].astype(np.int64)
    snr_values = sorted(np.unique(snr).tolist())
    names = load_class_names(args.classes)
    np.savez(out_dir / "test_labels_snr.npz", test_idx=test_idx, y=y_true, snr=snr)

    summary = []
    for path in paths:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if "validation" not in ckpt or "model_state_dict" not in ckpt:
            print(f"[skip] {path.name}: not a classifier checkpoint")
            continue
        model = IQCNNClassifier(num_classes=24, feature_dim=128, patch_size=64)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.to(device)
        y_pred, seconds = predict(model, cache, rows, device, args.batch_size)

        cfg = ckpt.get("config", {})
        per_snr = {}
        for v in snr_values:
            m = snr == v
            per_snr[str(v)] = {
                "n": int(m.sum()),
                "accuracy": float(accuracy_score(y_true[m], y_pred[m])),
                "macro_f1": float(f1_score(y_true[m], y_pred[m], average="macro",
                                           labels=list(range(24)), zero_division=0)),
            }
        cm = confusion_matrix(y_true, y_pred, labels=list(range(24)))
        per_class_f1 = f1_score(y_true, y_pred, average=None,
                                labels=list(range(24)), zero_division=0)
        result = {
            "checkpoint": str(path),
            "initialization": ckpt.get("initialization"),
            "label_fraction_pct": cfg.get("label_fraction"),
            "seed": cfg.get("seed"),
            "init_checkpoint": cfg.get("init_checkpoint"),
            "selected_epoch": ckpt.get("epoch"),
            "val_macro_f1": ckpt["validation"].get("macro_f1"),
            "val_accuracy": ckpt["validation"].get("accuracy"),
            "test_n": int(len(y_true)),
            "test_accuracy": float(accuracy_score(y_true, y_pred)),
            "test_macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                            labels=list(range(24)), zero_division=0)),
            "test_accuracy_snr_ge_0": float(accuracy_score(y_true[snr >= 0], y_pred[snr >= 0])),
            "test_accuracy_snr_ge_10": float(accuracy_score(y_true[snr >= 10], y_pred[snr >= 10])),
            "per_snr": per_snr,
            "per_class": {names[i]: {"f1": float(per_class_f1[i]),
                                     "recall": float(cm[i, i] / max(cm[i].sum(), 1))}
                          for i in range(24)},
            "class_names": names,
            "confusion_matrix": cm.tolist(),
            "inference_seconds": seconds,
            "inference_samples_per_second": len(y_true) / seconds,
            "parameters": int(sum(t.numel() for t in model.parameters())),
            "device": str(device),
        }
        (out_dir / f"{path.stem}.json").write_text(json.dumps(result, indent=1))
        # Predictions in test_idx order, for paired significance tests.
        np.save(out_dir / f"{path.stem}_pred.npy", y_pred.astype(np.int8))
        summary.append({k: result[k] for k in (
            "checkpoint", "initialization", "label_fraction_pct", "seed",
            "selected_epoch", "val_macro_f1", "test_accuracy", "test_macro_f1",
            "test_accuracy_snr_ge_0", "test_accuracy_snr_ge_10",
            "inference_samples_per_second")})
        print(f"{path.name}: test acc={result['test_accuracy']:.4f} "
              f"macro-F1={result['test_macro_f1']:.4f} "
              f"(acc@SNR>=10: {result['test_accuracy_snr_ge_10']:.4f})")

    if summary:
        csv_path = out_dir / "test_summary.csv"
        rows_existing = []
        if csv_path.exists():
            with open(csv_path, newline="") as fh:
                rows_existing = [r for r in csv.DictReader(fh)
                                 if r["checkpoint"] not in {s["checkpoint"] for s in summary}]
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary[0]))
            w.writeheader()
            w.writerows(rows_existing + summary)
        print("Summary:", csv_path)


if __name__ == "__main__":
    main()
