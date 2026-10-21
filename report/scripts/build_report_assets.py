"""Build every figure and table used in report.tex from the stored experiment outputs.

Read-only with respect to the experiments: it only reads JSON/CSV/log files under
--dlruns (default C:\\dlruns) and writes into --out (default: the report folder).
No model is loaded, no data split is touched, nothing is trained or evaluated.

  python scripts/build_report_assets.py --dlruns C:\\dlruns --out .

Outputs
  figures/*.pdf, figures/*.png    vector + raster figures
  tables/*.tex                    LaTeX table bodies \\input by report.tex
  tables/values.json              every number the tables/figures use (for checking)
"""

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PLAN = {1: [42, 43, 44], 5: [42, 43, 44], 10: [42, 43, 44], 25: [42], 100: [42]}
ARMS = {"scratch": "scratch", "mae": "mae_pretrained"}
LABEL = {"scratch": "CNN from scratch", "mae": "CNN-MAE fine-tuned"}
# Validated categorical pair (dataviz reference palette, slots 1-2) + neutral inks.
COLOR = {"scratch": "#2a78d6", "mae": "#eb6834"}
MARK = {"scratch": "o", "mae": "s"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
ORD_C, REC_C = "#52514e", "#6250d6"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.6, "lines.markersize": 5, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def best_from_log(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    m = re.search(r"Best validation macro-F1: ([\d.]+) at epoch (\d+)", text)
    return float(m.group(1)), int(m.group(2))


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / "figures" / f"{name}.{ext}", dpi=200)
    plt.close(fig)


def f4(x):
    return f"{x:.4f}"


def sci(x):
    m, e = f"{x:.0e}".split("e")
    return f"${m}\\times 10^{{{int(e)}}}$"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dlruns", default=r"C:\dlruns")
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    R, out = Path(args.dlruns), Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    V = {}

    # ------------------------------------------------------------ primary results
    runs = {}
    for arm, init in ARMS.items():
        for frac, seeds in PLAN.items():
            for s in seeds:
                j = json.loads((R / "results" / f"{arm}_{frac}pct_seed{s}_v1.json").read_text())
                assert j["initialization"] == init and int(j["label_fraction_pct"]) == frac and j["seed"] == s
                assert j["test_n"] == 383386
                runs[(arm, frac, s)] = j
    prim = {}
    for frac, seeds in PLAN.items():
        row = {"n_seeds": len(seeds)}
        for arm in ARMS:
            js = [runs[(arm, frac, s)] for s in seeds]
            for key in ("test_macro_f1", "test_accuracy", "test_accuracy_snr_ge_10", "val_macro_f1"):
                v = np.array([j[key] for j in js])
                row[f"{arm}_{key}_mean"] = float(v.mean())
                row[f"{arm}_{key}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else None
                row[f"{arm}_{key}_seeds"] = v.tolist()
            row[f"{arm}_selected_epochs"] = [j["selected_epoch"] for j in js]
        row["delta_macro_f1"] = row["mae_test_macro_f1_mean"] - row["scratch_test_macro_f1_mean"]
        row["delta_accuracy"] = row["mae_test_accuracy_mean"] - row["scratch_test_accuracy_mean"]
        row["mae_wins_f1"] = int(sum(m > s for m, s in zip(row["mae_test_macro_f1_seeds"],
                                                             row["scratch_test_macro_f1_seeds"])))
        if len(seeds) > 1:  # across-seed paired summary (n = 3; illustrative only)
            d = np.array(row["mae_test_macro_f1_seeds"]) - np.array(row["scratch_test_macro_f1_seeds"])
            sd_d = float(d.std(ddof=1))
            t = float(d.mean() / (sd_d / np.sqrt(len(d))))
            from math import sqrt
            nu = len(d) - 1
            # two-sided p for Student t with nu = 2 has a closed form
            p_two = float(1 - abs(t) / sqrt(nu + t * t)) if nu == 2 else None
            row["seed_deltas_macro_f1"] = d.tolist()
            row["seed_delta_sd"] = sd_d
            row["paired_t_across_seeds"] = t
            row["paired_t_p_two_sided_df2"] = p_two
        prim[frac] = row
    V["primary"] = prim

    def sd(x):
        return "--" if x is None else f"{x:.4f}"

    lines = []
    for frac, r in prim.items():
        lines.append(
            f"{frac}\\% & {r['n_seeds']} & {f4(r['scratch_test_macro_f1_mean'])} ({sd(r['scratch_test_macro_f1_sd'])}) & "
            f"{f4(r['mae_test_macro_f1_mean'])} ({sd(r['mae_test_macro_f1_sd'])}) & {r['delta_macro_f1']:+.4f} & "
            f"{r['mae_wins_f1']}/{r['n_seeds']} & {f4(r['scratch_test_accuracy_mean'])} & {f4(r['mae_test_accuracy_mean'])} & "
            f"{f4(r['scratch_test_accuracy_snr_ge_10_mean'])} & {f4(r['mae_test_accuracy_snr_ge_10_mean'])} \\\\")
    (out / "tables" / "primary.tex").write_text("\n".join(lines) + "\n")

    # per-seed + paired comparison (stored by summarize_results.py)
    paired = {(int(r["label_fraction_pct"]), int(r["seed"])): r
              for r in csv.DictReader(open(R / "results" / "paired_comparison.csv"))}
    V["paired"] = {f"{k[0]}_{k[1]}": v for k, v in paired.items()}
    lines = []
    for frac, seeds in PLAN.items():
        for s in seeds:
            p = paired[(frac, s)]
            sc, ma = runs[("scratch", frac, s)], runs[("mae", frac, s)]
            assert abs(float(p["macro_f1_scratch"]) - sc["test_macro_f1"]) < 6e-5
            assert abs(float(p["macro_f1_mae"]) - ma["test_macro_f1"]) < 6e-5
            pv = float(p["mcnemar_p"])
            pstr = "$<10^{-15}$" if pv == 0 else (f"{pv:.2g}" if pv < 0.01 else f"{pv:.3f}")
            lines.append(
                f"{frac}\\% & {s} & {sc['selected_epoch']} / {ma['selected_epoch']} & {f4(sc['test_macro_f1'])} & "
                f"{f4(ma['test_macro_f1'])} & {p['delta_macro_f1']} & [{p['delta_f1_ci95_low']}, {p['delta_f1_ci95_high']}] & "
                f"{p['delta_acc']} & {pstr} \\\\")
    (out / "tables" / "per_seed.tex").write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------ figure: label efficiency
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for ax, key, title in ((axes[0], "test_macro_f1", "Test macro-F1"),
                           (axes[1], "test_accuracy", "Test accuracy")):
        for i, arm in enumerate(ARMS):
            off = 0.96 if arm == "scratch" else 1.04
            xs = list(PLAN)
            means = [prim[f][f"{arm}_{key}_mean"] for f in xs]
            ax.plot([x * off for x in xs], means, color=COLOR[arm], marker=MARK[arm], label=LABEL[arm], zorder=3)
            for f in xs:
                vals = prim[f][f"{arm}_{key}_seeds"]
                ax.scatter([f * off] * len(vals), vals, s=10, facecolors="none", edgecolors=COLOR[arm],
                           linewidths=0.8, zorder=2)
        ax.set_xscale("log")
        ax.set_xticks(list(PLAN))
        ax.set_xticklabels([f"{f}%" for f in PLAN])
        ax.minorticks_off()
        ax.set_title(title)
        ax.axvspan(18, 140, color="#f1f0ec", zorder=0)
        ax.text(48, ax.get_ylim()[0] + 0.004, "1 seed", ha="center", va="bottom", fontsize=7, color=INK2)
    fig.supxlabel("Labelled training data (fraction of the training split)", fontsize=9, y=-0.04)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.06))
    save(fig, out, "fig_label_efficiency")

    # ------------------------------------------------------------ figure: accuracy vs SNR
    snrs = sorted(int(k) for k in runs[("scratch", 1, 42)]["per_snr"])
    fig, axes = plt.subplots(2, 5, figsize=(7.2, 3.6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    snr_delta = {}
    for c, frac in enumerate(PLAN):
        acc = {}
        for arm in ARMS:
            acc[arm] = np.mean([[runs[(arm, frac, s)]["per_snr"][str(v)]["accuracy"] for v in snrs]
                                for s in PLAN[frac]], axis=0)
            axes[0, c].plot(snrs, acc[arm], color=COLOR[arm], label=LABEL[arm], lw=1.3)
        d = acc["mae"] - acc["scratch"]
        snr_delta[frac] = d.tolist()
        axes[1, c].axhline(0, color=INK2, lw=0.8)
        axes[1, c].plot(snrs, d, color=INK, lw=1.2)
        axes[0, c].set_title(f"{frac}% ({len(PLAN[frac])} seed{'s' if len(PLAN[frac]) > 1 else ''})")
        axes[0, c].set_ylim(0, 0.75)
        axes[1, c].set_ylim(-0.06, 0.06)
        axes[1, c].set_xlabel("SNR (dB)")
        if c:
            axes[0, c].set_yticklabels([])
            axes[1, c].set_yticklabels([])
    axes[0, 0].set_ylabel("Test accuracy")
    axes[1, 0].set_ylabel("MAE $-$ scratch")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.05))
    V["snr_delta_accuracy"] = {"snrs": snrs, **{str(k): v for k, v in snr_delta.items()}}
    save(fig, out, "fig_accuracy_vs_snr")

    # ------------------------------------------------------------ figure: per-class F1 at 5%
    names = runs[("scratch", 5, 42)]["class_names"]
    pc = {arm: np.mean([[runs[(arm, 5, s)]["per_class"][n]["f1"] for n in names] for s in PLAN[5]], axis=0)
          for arm in ARMS}
    order = np.argsort(pc["scratch"])
    fig, ax = plt.subplots(figsize=(3.6, 4.6))
    y = np.arange(len(names))
    for a, b, yy in zip(pc["scratch"][order], pc["mae"][order], y):
        ax.plot([a, b], [yy, yy], color=GRID, lw=1.2, zorder=1)
    for arm in ARMS:
        ax.scatter(pc[arm][order], y, color=COLOR[arm], marker=MARK[arm], s=18, label=LABEL[arm], zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in order], fontsize=7)
    ax.set_xlabel("Test F1 (mean of seeds 42-44), 5% labels")
    ax.legend(loc="lower right", frameon=False, fontsize=7)
    V["per_class_f1_5pct"] = {arm: dict(zip(names, map(float, pc[arm]))) for arm in ARMS}
    save(fig, out, "fig_per_class_f1_5pct")

    # ------------------------------------------------------------ sensitivity (validation only)
    lr = {"scratch": {1e-3: best_from_log(R / "finetune/scratch_5pct_seed42_v1.log"),
                      3e-4: best_from_log(R / "tuning_lr/scratch_5pct_seed42_lr3e4.log"),
                      1e-4: best_from_log(R / "tuning_lr/scratch_5pct_seed42_lr1e4.log")},
          "mae": {1e-3: best_from_log(R / "finetune/mae_5pct_seed42_v1.log"),
                  3e-4: best_from_log(R / "tuning_lr/mae_5pct_seed42_lr3e4.log"),
                  1e-4: best_from_log(R / "tuning_lr/mae_5pct_seed42_lr1e4.log")}}
    wd = {"scratch": {1e-5: best_from_log(R / "tuning_wd/scratch_5pct_seed42_wd1e5.log"),
                      1e-4: best_from_log(R / "finetune/scratch_5pct_seed42_v1.log"),
                      1e-3: best_from_log(R / "tuning_wd/scratch_5pct_seed42_wd1e3.log")},
          "mae": {1e-5: best_from_log(R / "tuning_wd/mae_5pct_seed42_wd1e5.log"),
                  1e-4: best_from_log(R / "finetune/mae_5pct_seed42_v1.log"),
                  1e-3: best_from_log(R / "tuning_wd/mae_5pct_seed42_wd1e3.log")}}
    enc = {1e-4: best_from_log(R / "tuning_encoder_lr/mae_5pct_seed42_enc1e4.log"),
           3e-4: best_from_log(R / "tuning_encoder_lr/mae_5pct_seed42_enc3e4.log")}
    V["sensitivity"] = {"lr": {a: {str(k): v for k, v in d.items()} for a, d in lr.items()},
                        "wd": {a: {str(k): v for k, v in d.items()} for a, d in wd.items()},
                        "encoder_lr": {str(k): v for k, v in enc.items()}}
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.3), sharey=True)
    for ax, d, xl in ((axes[0], lr, "Learning rate (weight decay $10^{-4}$)"),
                      (axes[1], wd, "Weight decay (learning rate $10^{-3}$)")):
        for arm in ARMS:
            xs = sorted(d[arm])
            ax.plot(xs, [d[arm][x][0] for x in xs], color=COLOR[arm], marker=MARK[arm], label=LABEL[arm])
        ax.set_xscale("log")
        ax.set_xlabel(xl)
        ax.minorticks_off()
        ticks = sorted(d["scratch"])
        ax.set_xticks(ticks)
        ax.set_xticklabels([sci(t) for t in ticks])
    axes[0].set_ylabel("Best validation macro-F1")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.1))
    save(fig, out, "fig_sensitivity")
    lines = []
    for x in sorted(lr["scratch"], reverse=True):
        tag = " (primary)" if x == 1e-3 else ""
        lines.append(f"Learning rate & {sci(x)}{tag} & {lr['scratch'][x][0]:.4f} & {lr['mae'][x][0]:.4f} \\\\")
    lines.append("\\midrule")
    for x in sorted(wd["scratch"]):
        tag = " (primary)" if x == 1e-4 else ""
        lines.append(f"Weight decay & {sci(x)}{tag} & {wd['scratch'][x][0]:.4f} & {wd['mae'][x][0]:.4f} \\\\")
    lines.append("\\midrule")
    for x, (v, _) in sorted(enc.items()):
        lines.append(f"Encoder LR (head LR $10^{{-3}}$, WD $10^{{-5}}$) & {sci(x)} & -- & {v:.4f} \\\\")
    (out / "tables" / "sensitivity.tex").write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------ exploratory matched WD=1e-5 test
    m = list(csv.DictReader(open(R / "results_matched_wd1e5" / "test_summary.csv")))
    mt = {}
    for arm, init in ARMS.items():
        rows = sorted([r for r in m if r["initialization"] == init], key=lambda r: int(r["seed"]))
        assert [int(r["seed"]) for r in rows] == [42, 43, 44]
        mt[arm] = {k: [float(r[k]) for r in rows] for k in
                   ("test_macro_f1", "test_accuracy", "test_accuracy_snr_ge_10", "val_macro_f1")}
        mt[arm]["selected_epoch"] = [int(r["selected_epoch"]) for r in rows]
    V["matched_wd1e5"] = mt
    lines = []
    for i, s in enumerate((42, 43, 44)):
        a, b = mt["scratch"]["test_macro_f1"][i], mt["mae"]["test_macro_f1"][i]
        lines.append(f"{s} & {f4(a)} & {f4(b)} & {b - a:+.4f} \\\\")
    a, b = np.mean(mt["scratch"]["test_macro_f1"]), np.mean(mt["mae"]["test_macro_f1"])
    lines.append("\\midrule")
    lines.append(f"Mean & {f4(a)} & {f4(b)} & {b - a:+.4f} \\\\")
    (out / "tables" / "matched_wd1e5.tex").write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------ BatchNorm diagnostics
    bn = {arm: json.loads((R / "diagnostics" / f"bn_{arm}_5pct_seed42" / "results.json").read_text())
          for arm in ARMS}
    V["bn_checkpoint"] = {arm: {k: (d[k] if k in d else None) for k in
                                ("A1_evaluate", "A_fp32", "B", "B_fp32", "C_summary", "A_reproduces_log", "B_vs_original_bn",
                                 "torch", "gpu")} for arm, d in bn.items()}
    lines = []
    for arm in ARMS:
        d = bn[arm]
        cs = d["C_summary"]
        lines.append(
            f"{LABEL[arm]} & {f4(d['A1_evaluate']['accuracy'])} / {f4(d['A1_evaluate']['macro_f1'])} & "
            f"{f4(d['A_fp32']['accuracy'])} / {f4(d['A_fp32']['macro_f1'])} & "
            f"{f4(d['B']['accuracy'])} / {f4(d['B']['macro_f1'])} & "
            f"{cs['accuracy_min_median_max'][0]:.4f}--{cs['accuracy_min_median_max'][2]:.4f} & "
            f"{cs['macro_f1_min_median_max'][0]:.4f}--{cs['macro_f1_min_median_max'][2]:.4f} \\\\")
    (out / "tables" / "bn_checkpoint.tex").write_text("\n".join(lines) + "\n")

    pe, summ, repro = {}, {}, {}
    for arm in ARMS:
        base = R / "diagnostics" / "perepoch_5pct_seed42" / arm
        pe[arm] = [json.loads(l) for l in (base / "epoch_metrics.jsonl").read_text().splitlines() if l.strip()]
        summ[arm] = json.loads((base / "summary.json").read_text())
        repro[arm] = json.loads((base / "reproduction.json").read_text())
        assert repro[arm]["log_consistent_at_printed_precision"]
        assert repro[arm]["epoch5_weights_vs_original_checkpoint"]["bitwise_identical"]
        assert all(r["training_state_unchanged"] for r in pe[arm])
    V["perepoch"] = {arm: {"summary": summ[arm],
                           "records": [{k: r[k] for k in ("epoch", "ordinary", "recalibrated",
                                                          "relative_weight_change")} for r in pe[arm]],
                           "reproduction": {k: repro[arm][k] for k in (
                               "log_consistent_at_printed_precision", "first_log_mismatch_epoch",
                               "original_selected_epoch", "new_selected_epoch", "original_artifacts_unchanged")},
                           "epoch5_bitwise": repro[arm]["epoch5_weights_vs_original_checkpoint"]["bitwise_identical"],
                           "selected_ckpt_identical": {k: (v if not isinstance(v, dict) else v["bitwise_identical"])
                                                       for k, v in repro[arm].get("selected_checkpoint_vs_original", {}).items()}}
                     for arm in ARMS}
    lines = []
    for arm in ARMS:
        for metric, name in (("accuracy", "Accuracy"), ("macro_f1", "Macro-F1")):
            s = summ[arm][metric]
            lines.append(
                f"{LABEL[arm]} & {name} & {s['ordinary']['mean_abs_delta']:.4f} & {s['recalibrated']['mean_abs_delta']:.4f} & "
                f"{s['ordinary']['range']:.4f} & {s['recalibrated']['range']:.4f} & "
                f"{', '.join(map(str, s['ordinary']['dips'])) or 'none'} & "
                f"{', '.join(map(str, s['recalibrated']['dips'])) or 'none'} \\\\")
    (out / "tables" / "bn_perepoch.tex").write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.2), sharex=True)
    for r_i, arm in enumerate(ARMS):
        ep = [r["epoch"] for r in pe[arm]]
        for c_i, (metric, name) in enumerate((("accuracy", "Validation accuracy"), ("macro_f1", "Validation macro-F1"))):
            ax = axes[r_i, c_i]
            ax.axvspan(3.5, 10.5, color="#f1f0ec", zorder=0)
            ax.plot(ep, [r["ordinary"][metric] for r in pe[arm]], color=ORD_C, marker="o",
                    label="Stored running statistics (as logged)")
            ax.plot(ep, [r["recalibrated"][metric] for r in pe[arm]], color=REC_C, marker="s", ls="--",
                    label="Exact BatchNorm recalibration (copy)")
            ax.axvline(5, color=INK2, lw=0.8, ls=":")
            ax.set_title(f"{LABEL[arm]}: {name}")
            ax.set_xticks(range(1, 11))
            if r_i == 1:
                ax.set_xlabel("Epoch")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    save(fig, out, "fig_bn_perepoch")

    # ------------------------------------------------------------ MAE pretraining log
    text = (R / "cnn_mae_full_v1.log").read_bytes()
    text = text.decode("utf-16") if text[:2] in (b"\xff\xfe", b"\xfe\xff") else text.decode("utf-8", "replace")
    pre = [dict(epoch=int(a), train=float(b), val=float(c), lr=float(d)) for a, b, c, d in re.findall(
        r"Epoch (\d+)/5 \| train MSE=([\d.]+) \(\d+ samples\) \| val MSE=([\d.]+) \(\d+ samples\) \| lr_used=([\d.e+-]+)", text)]
    el = re.search(r"Elapsed: ([\d.]+)s", text)
    assert len(pre) == 5
    V["pretraining"] = {"epochs": pre, "elapsed_s": float(el.group(1))}
    lines = [f"{p['epoch']} & {p['train']:.6f} & {p['val']:.6f} & ${p['lr']:.2e}$ \\\\".replace("e-05", "\\times10^{-5}")
             .replace("e-06", "\\times10^{-6}").replace("e-04", "\\times10^{-4}") for p in pre]
    (out / "tables" / "pretraining.tex").write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------ run times (wall clock, from logs)
    times = {}
    for arm in ARMS:
        for frac, seeds in PLAN.items():
            for s in seeds:
                t = (R / "finetune" / f"{arm}_{frac}pct_seed{s}_v1.log").read_text(encoding="utf-8", errors="replace")
                times[f"{arm}_{frac}_{s}"] = float(re.search(r"Elapsed: ([\d.]+)s", t).group(1))
    V["finetune_elapsed_s"] = times
    sps = [runs[k]["inference_samples_per_second"] for k in runs]
    V["inference_samples_per_second"] = {"median": float(np.median(sps)), "min": float(min(sps)), "max": float(max(sps))}
    V["parameters_classifier"] = runs[("scratch", 5, 42)]["parameters"]

    (out / "tables" / "values.json").write_text(json.dumps(V, indent=1, default=float))
    print("Wrote figures/ and tables/ to", out.resolve())


if __name__ == "__main__":
    main()
