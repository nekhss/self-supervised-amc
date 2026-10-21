"""Aggregate test results into tables, paired comparisons and figures.

Reads the JSON/npy outputs of evaluate_test.py; needs no GPU and no torch.

Outputs (in --results-dir):
  results_by_budget.csv   mean/std over seeds for each arm and label budget
  paired_comparison.csv   MAE minus scratch, paired by (budget, seed), with a
                          paired bootstrap 95% CI for macro-F1 and an exact
                          McNemar test for accuracy on the same test samples
  label_efficiency.png    test macro-F1 and accuracy vs label budget
  accuracy_vs_snr.png     test accuracy vs SNR, one panel per budget
  confusion_<arm>_<b>pct.png  row-normalised confusion matrices (seed 42)

Example:
  python experiments/summarize_results.py --results-dir C:\\dlruns\\results
"""

import argparse
import csv
import json
from collections import defaultdict
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ARM_STYLE = {"scratch": ("#4C72B0", "CNN from scratch"),
             "mae_pretrained": ("#DD8452", "CNN-MAE pretrained")}


def macro_f1_from_counts(tp, pred_count, true_count):
    denom = pred_count + true_count
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return f1.mean(axis=-1)


def macro_f1(y, p, k=24):
    tp = np.bincount(y[y == p], minlength=k)
    return macro_f1_from_counts(tp, np.bincount(p, minlength=k), np.bincount(y, minlength=k))


def paired_bootstrap_f1(y, pa, pb, reps, rng, k=24):
    """95% CI of macro-F1(pa) - macro-F1(pb) by resampling test samples."""
    n = len(y)
    diffs = np.empty(reps)
    ca, cb = (pa == y), (pb == y)
    for r in range(reps):
        i = rng.integers(0, n, n)
        yi = y[i]
        tc = np.bincount(yi, minlength=k)
        fa = macro_f1_from_counts(np.bincount(yi[ca[i]], minlength=k),
                                  np.bincount(pa[i], minlength=k), tc)
        fb = macro_f1_from_counts(np.bincount(yi[cb[i]], minlength=k),
                                  np.bincount(pb[i], minlength=k), tc)
        diffs[r] = fa - fb
    return np.percentile(diffs, [2.5, 97.5])


def mcnemar_exact(ca, cb):
    """Two-sided exact McNemar p-value; b = only A correct, c = only B correct."""
    b = int(np.sum(ca & ~cb))
    c = int(np.sum(~ca & cb))
    n = b + c
    if n == 0:
        return b, c, 1.0
    if n > 2000:  # normal approximation with continuity correction
        from math import erf, sqrt
        z = (abs(b - c) - 1) / sqrt(n)
        return b, c, float(2 * (1 - 0.5 * (1 + erf(z / sqrt(2)))))
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return b, c, float(min(1.0, 2 * p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=r"C:\dlruns\results")
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()
    rd = Path(args.results_dir)

    runs = []
    for js in sorted(rd.glob("*.json")):
        r = json.loads(js.read_text())
        if "test_macro_f1" in r:
            r["_stem"] = js.stem
            runs.append(r)
    if not runs:
        raise SystemExit(f"No evaluate_test.py results in {rd}")

    groups = defaultdict(list)
    for r in runs:
        groups[(r["initialization"], int(r["label_fraction_pct"]))].append(r)

    with open(rd / "results_by_budget.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["initialization", "label_fraction_pct", "n_seeds", "seeds",
                    "test_macro_f1_mean", "test_macro_f1_std",
                    "test_accuracy_mean", "test_accuracy_std",
                    "test_acc_snr_ge_10_mean", "val_macro_f1_mean"])
        for (arm, frac), rs in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            f1 = np.array([r["test_macro_f1"] for r in rs])
            acc = np.array([r["test_accuracy"] for r in rs])
            w.writerow([arm, frac, len(rs), " ".join(str(r["seed"]) for r in rs),
                        f"{f1.mean():.4f}", f"{f1.std(ddof=1) if len(rs) > 1 else 0:.4f}",
                        f"{acc.mean():.4f}", f"{acc.std(ddof=1) if len(rs) > 1 else 0:.4f}",
                        f"{np.mean([r['test_accuracy_snr_ge_10'] for r in rs]):.4f}",
                        f"{np.mean([r['val_macro_f1'] for r in rs]):.4f}"])

    # Paired comparison on identical test samples.
    labels_file = rd / "test_labels_snr.npz"
    rng = np.random.default_rng(0)
    if labels_file.exists():
        y = np.load(labels_file)["y"].astype(np.int64)
        by_key = {(r["initialization"], int(r["label_fraction_pct"]), r["seed"]): r for r in runs}
        with open(rd / "paired_comparison.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["label_fraction_pct", "seed", "macro_f1_scratch", "macro_f1_mae",
                        "delta_macro_f1", "delta_f1_ci95_low", "delta_f1_ci95_high",
                        "acc_scratch", "acc_mae", "delta_acc",
                        "mcnemar_only_mae_correct", "mcnemar_only_scratch_correct",
                        "mcnemar_p"])
            for (arm, frac, seed), rm in sorted(by_key.items(), key=lambda kv: (kv[0][1], kv[0][2])):
                if arm != "mae_pretrained" or ("scratch", frac, seed) not in by_key:
                    continue
                rs = by_key[("scratch", frac, seed)]
                pm = np.load(rd / f"{rm['_stem']}_pred.npy").astype(np.int64)
                ps = np.load(rd / f"{rs['_stem']}_pred.npy").astype(np.int64)
                lo, hi = paired_bootstrap_f1(y, pm, ps, args.bootstrap, rng)
                b, c, pval = mcnemar_exact(pm == y, ps == y)
                w.writerow([frac, seed, f"{macro_f1(y, ps):.4f}", f"{macro_f1(y, pm):.4f}",
                            f"{macro_f1(y, pm) - macro_f1(y, ps):+.4f}", f"{lo:+.4f}", f"{hi:+.4f}",
                            f"{np.mean(ps == y):.4f}", f"{np.mean(pm == y):.4f}",
                            f"{np.mean(pm == y) - np.mean(ps == y):+.4f}", b, c, f"{pval:.3g}"])

    # Label-efficiency figure.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for metric, ax, title in (("test_macro_f1", axes[0], "Test macro-F1"),
                              ("test_accuracy", axes[1], "Test accuracy")):
        for arm, (color, label) in ARM_STYLE.items():
            fr = sorted({f for (a, f) in groups if a == arm})
            if not fr:
                continue
            vals = [[r[metric] for r in groups[(arm, f)]] for f in fr]
            mean = [np.mean(v) for v in vals]
            std = [np.std(v, ddof=1) if len(v) > 1 else 0 for v in vals]
            ax.errorbar(fr, mean, yerr=std, color=color, marker="o", capsize=3, label=label)
        ax.set_xscale("log")
        ax.set_xticks([1, 5, 10, 25, 100])
        ax.set_xticklabels(["1%", "5%", "10%", "25%", "100%"])
        ax.set_xlabel("Labelled training data")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle("RadioML 2018.01A, 24 classes (mean ± std over seeds)")
    fig.tight_layout()
    fig.savefig(rd / "label_efficiency.png", dpi=150)
    plt.close(fig)

    # Accuracy vs SNR, one panel per budget.
    fracs = sorted({f for (_, f) in groups})
    fig, axes = plt.subplots(1, len(fracs), figsize=(3.6 * len(fracs), 3.4), sharey=True, squeeze=False)
    for ax, f in zip(axes[0], fracs):
        for arm, (color, label) in ARM_STYLE.items():
            rs = groups.get((arm, f))
            if not rs:
                continue
            snrs = sorted(int(s) for s in rs[0]["per_snr"])
            acc = np.mean([[r["per_snr"][str(s)]["accuracy"] for s in snrs] for r in rs], axis=0)
            ax.plot(snrs, acc, color=color, label=label)
        ax.set_title(f"{f}% labels")
        ax.set_xlabel("SNR (dB)")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("Test accuracy")
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(rd / "accuracy_vs_snr.png", dpi=150)
    plt.close(fig)

    # Confusion matrices (seed 42) for each arm and budget.
    for (arm, f), rs in groups.items():
        r = next((x for x in rs if x["seed"] == 42), rs[0])
        cm = np.array(r["confusion_matrix"], dtype=float)
        cm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(24))
        ax.set_yticks(range(24))
        ax.set_xticklabels(r["class_names"], rotation=90, fontsize=7)
        ax.set_yticklabels(r["class_names"], fontsize=7)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{ARM_STYLE.get(arm, ('', arm))[1]}, {f}% labels, seed {r['seed']} (all SNRs)")
        fig.colorbar(im, fraction=0.046)
        fig.tight_layout()
        fig.savefig(rd / f"confusion_{arm}_{f}pct.png", dpi=130)
        plt.close(fig)

    print("Wrote tables and figures to", rd)


if __name__ == "__main__":
    main()
