"""Summarise RML2016 transfer test results (reads files only).

Writes to --out-root:
  per_seed.csv   one row per run (test accuracy / macro-F1, selected epoch)
  summary.json   per budget and arm: mean/std over seeds, paired
                 transfer-minus-scratch differences per seed, per-class F1 and
                 per-SNR means
  summary.md     human-readable table (labelled SMOKE TEST if any run is a pilot)

Before pairing, checks that the two arms' configs are identical apart from
`arm` and `init_checkpoint`.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ARM_SPECIFIC = {"arm", "init_checkpoint"}


def mean_std(values):
    values = np.asarray(values, dtype=float)
    std = float(values.std(ddof=1)) if len(values) > 1 else None
    return {"n": int(len(values)), "mean": float(values.mean()), "std": std}


def shared_config(cfg):
    return {k: v for k, v in cfg.items() if k not in ARM_SPECIFIC}


def fmt(stat):
    if stat["std"] is None:
        return f"{stat['mean']:.4f} (n=1)"
    return f"{stat['mean']:.4f} +/- {stat['std']:.4f} (n={stat['n']})"


def summarize_root(out_root):
    out_root = Path(out_root)
    results = [json.loads(p.read_text()) for p in sorted(out_root.glob("*/test.json"))]
    if not results:
        print("No test results to summarise.")
        return None
    pilot = any(r["config"]["pilot"] for r in results)

    runs = {(r["config"]["budget_pct"], r["config"]["arm"], r["config"]["seed"]): r
            for r in results}
    for (budget, arm, seed), r in runs.items():
        other = runs.get((budget, "scratch" if arm == "transfer" else "transfer", seed))
        if other and shared_config(other["config"]) != shared_config(r["config"]):
            raise RuntimeError(f"Arms differ beyond initialisation at {budget}% seed {seed}")

    with open(out_root / "per_seed.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["budget_pct", "arm", "seed", "test_accuracy", "test_macro_f1",
                    "selected_epoch", "best_val_macro_f1", "pilot"])
        for key in sorted(runs):
            r = runs[key]
            w.writerow([*key, r["test"]["accuracy"], r["test"]["macro_f1"],
                        r["selected_epoch"], r["best_val_macro_f1"], r["config"]["pilot"]])

    grouped = defaultdict(dict)
    for (budget, arm, seed), r in runs.items():
        grouped[budget].setdefault(arm, {})[seed] = r

    summary = {"pilot_smoke_test": pilot, "budgets": {}}
    for budget in sorted(grouped):
        entry = {}
        for arm, by_seed in grouped[budget].items():
            seeds = sorted(by_seed)
            tests = [by_seed[s]["test"] for s in seeds]
            entry[arm] = {
                "seeds": seeds,
                "accuracy": mean_std([t["accuracy"] for t in tests]),
                "macro_f1": mean_std([t["macro_f1"] for t in tests]),
                "per_class_f1_mean": {c: float(np.mean([t["per_class_f1"][c] for t in tests]))
                                      for c in tests[0]["per_class_f1"]},
                "per_snr_mean": {s: {m: float(np.mean([t["per_snr"][s][m] for t in tests]))
                                     for m in ("accuracy", "macro_f1")}
                                 for s in tests[0]["per_snr"]},
            }
        if {"transfer", "scratch"} <= set(grouped[budget]):
            paired = sorted(set(grouped[budget]["transfer"]) & set(grouped[budget]["scratch"]))
            diffs = {m: [grouped[budget]["transfer"][s]["test"][m]
                         - grouped[budget]["scratch"][s]["test"][m] for s in paired]
                     for m in ("accuracy", "macro_f1")}
            entry["transfer_minus_scratch"] = {
                "paired_seeds": paired,
                "per_seed": {m: dict(zip(map(str, paired), v)) for m, v in diffs.items()},
                "accuracy": mean_std(diffs["accuracy"]) if paired else None,
                "macro_f1": mean_std(diffs["macro_f1"]) if paired else None,
            }
        summary["budgets"][str(budget)] = entry
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = []
    if pilot:
        lines += ["**SMOKE TEST ONLY - reduced epochs/batches; not scientific evidence.**", ""]
    lines += ["Cross-dataset transfer RML2018.01A (CNN-MAE) -> RML2016.10a; test split; "
              "mean +/- sample std over training seeds.", "",
              "| Budget | Arm | Accuracy | Macro-F1 |", "|---|---|---|---|"]
    for budget, entry in summary["budgets"].items():
        for arm in ("transfer", "scratch"):
            if arm in entry:
                lines.append(f"| {budget}% | {arm} | {fmt(entry[arm]['accuracy'])} "
                             f"| {fmt(entry[arm]['macro_f1'])} |")
        d = entry.get("transfer_minus_scratch")
        if d and d["accuracy"]:
            lines.append(f"| {budget}% | transfer - scratch | {fmt(d['accuracy'])} "
                         f"| {fmt(d['macro_f1'])} |")
    (out_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", required=True)
    summarize_root(parser.parse_args().out_root)


if __name__ == "__main__":
    main()
