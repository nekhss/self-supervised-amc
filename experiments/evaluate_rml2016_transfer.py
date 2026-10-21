"""One-time test evaluation of finished RML2016 transfer runs.

Only runs whose train.json is marked complete are evaluated, using the
checkpoint chosen by validation macro-F1 (hash-checked against train.json).
Each run is evaluated at most once; existing test.json files are left alone.

Example:
  C:\\venvs\\dlproj\\Scripts\\python.exe experiments\\evaluate_rml2016_transfer.py ^
      --out-root C:\\dlruns\\rml2016_transfer\\pilot_v1
"""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import rml2016_transfer as T  # noqa: E402
from src.data import rml2016  # noqa: E402


def evaluate_root(out_root, pkl, splits, protocol, device):
    out_root = Path(out_root)
    unfinished = [d.name for d in out_root.iterdir()
                  if d.is_dir() and not (d / "train.json").is_file()]
    if unfinished:
        raise RuntimeError(f"Run folders without train.json (crashed?): {unfinished}")
    pending = []
    for record in sorted(out_root.glob("*/train.json")):
        if (record.parent / "test.json").exists():
            print(f"[skip] {record.parent.name}: test already evaluated")
            continue
        if not json.loads(record.read_text()).get("complete"):
            raise RuntimeError(f"{record.parent} has incomplete training.")
        pending.append(record.parent)
    if not pending:
        print("No runs pending test evaluation.")
        return
    data = T.load_data(pkl, splits, protocol, device, include_test=True)
    for run_dir in pending:
        result = T.evaluate_test(run_dir, data, device)
        m = result["test"]
        print(f"[test] {run_dir.name}: acc={m['accuracy']:.4f} "
              f"macro-F1={m['macro_f1']:.4f} (epoch {result['selected_epoch']})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--pkl", default=str(rml2016.DEFAULT_PKL_PATH))
    parser.add_argument("--splits", default=str(T.DEFAULT_SPLITS))
    parser.add_argument("--protocol", default=str(T.DEFAULT_PROTOCOL))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate_root(args.out_root, args.pkl, args.splits, args.protocol, device)


if __name__ == "__main__":
    main()
