"""Train the RML2018 -> RML2016 transfer grid, then evaluate test, then summarise.

Phases run strictly in order:
  1. training + validation-based checkpoint selection for every requested run
     (test indices are not loaded in this phase);
  2. one test evaluation per finished run (experiments/evaluate_rml2016_transfer.py);
  3. summary tables (experiments/summarize_rml2016_transfer.py).

Runs whose train.json is complete are skipped, so an interrupted grid can be
resumed; a half-written run directory stops the grid instead of being overwritten.

Full suite (needs explicit approval before launch):
  C:\\venvs\\dlproj\\Scripts\\python.exe -u experiments\\run_rml2016_transfer.py ^
      --out-root C:\\dlruns\\rml2016_transfer\\final_v1

Smoke-test pilot (non-default budgets require --pilot and 'pilot' in the path):
  ... --out-root C:\\dlruns\\rml2016_transfer\\pilot_v1 --pilot --budgets 1 ^
      --seeds 42 --epochs 2
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--arms", nargs="+", default=list(T.ARMS), choices=T.ARMS)
    parser.add_argument("--budgets", nargs="+", type=int, default=list(rml2016.LABEL_BUDGETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=T.TRAIN_DEFAULTS["epochs"])
    parser.add_argument("--max-train-batches", type=int, default=0,
                        help="Pilot only; 0 means all batches.")
    parser.add_argument("--pilot", action="store_true",
                        help="Required for any non-default budget; marks outputs as smoke tests.")
    parser.add_argument("--pretrained", default=str(T.DEFAULT_PRETRAINED))
    parser.add_argument("--pkl", default=str(rml2016.DEFAULT_PKL_PATH))
    parser.add_argument("--splits", default=str(T.DEFAULT_SPLITS))
    parser.add_argument("--protocol", default=str(T.DEFAULT_PROTOCOL))
    parser.add_argument("--skip-test", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    reduced = args.epochs != T.TRAIN_DEFAULTS["epochs"] or args.max_train_batches
    if reduced and not args.pilot:
        raise ValueError("Reduced epochs/batches are only allowed with --pilot.")
    if args.pilot != ("pilot" in out_root.name.lower()):
        raise ValueError("Pilot runs must (and only pilot runs may) use an "
                         "--out-root whose name contains 'pilot'.")
    if not Path(args.pretrained).is_file():
        raise FileNotFoundError(args.pretrained)
    if not args.pilot:
        # Final runs must be traceable to an exact commit and live outside the repo.
        git = T.git_state()
        if git["commit"] is None or git["dirty"] is not False:
            raise RuntimeError(f"Final runs need a clean git working tree (got {git}).")
        if out_root.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError("--out-root must be outside the repository.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    out_root.mkdir(parents=True, exist_ok=True)

    # Phase 1: training and validation-based selection. No test indices loaded.
    data = T.load_data(args.pkl, args.splits, args.protocol, device, include_test=False)
    print(f"Data: {len(data['x']):,} examples, amplitude multiplier {data['multiplier']:.6f}")
    for seed in args.seeds:
        for budget in args.budgets:
            for arm in args.arms:
                name = T.run_name(arm, budget, seed)
                run_dir = out_root / name
                record = run_dir / "train.json"
                if record.is_file() and json.loads(record.read_text()).get("complete"):
                    print(f"[skip] {name}: already complete")
                    continue
                print(f"\n=== {name} ===")
                cfg = T.run_config(arm, budget, seed, args.epochs,
                                   args.max_train_batches, args.pretrained, args.pilot)
                T.train_run(cfg, data, run_dir, device)
    del data
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if args.skip_test:
        print("Skipping test evaluation (--skip-test).")
        return

    # Phase 2 and 3: run only after every requested run has been selected.
    from experiments.evaluate_rml2016_transfer import evaluate_root
    from experiments.summarize_rml2016_transfer import summarize_root
    evaluate_root(out_root, args.pkl, args.splits, args.protocol, device)
    summarize_root(out_root)
    missing = [T.run_name(a, b, s) for s in args.seeds for b in args.budgets
               for a in args.arms if not (out_root / T.run_name(a, b, s) / "test.json").is_file()]
    print(f"Requested runs with test results: "
          f"{len(args.seeds) * len(args.budgets) * len(args.arms) - len(missing)}; missing: {missing}")


if __name__ == "__main__":
    main()
