"""Final figures and tables from verified saved artifacts (read-only; no training, no evaluation).

Run:  C:\\venvs\\dlproj\\Scripts\\python.exe final_artifacts\\make_final_figures.py
Writes only into final_artifacts\\{figures,tables}\\ and final_artifacts\\VERIFICATION.md.

Sources (via demo/results_data.py and demo/exploratory_data.py, which validate metadata):
  primary     C:\\dlruns\\results\\{arm}_{budget}pct_seed{seed}_v1.json, paired_comparison.csv
  50 epochs   C:\\dlruns\\finetune_50ep_v1\\* (+ misnamed MAE seed-43 folder), results_50ep_v1_exploratory\\*\\best.json
  100 epochs  C:\\dlruns\\finetune_100ep_v2\\runs\\*, test_results\\*\\best.json
Every plotted number is re-read from the raw JSON/CSV files and asserted equal to the loader value.
"""
import csv, json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "demo"))
import demo_config as cfg, results_data as R, exploratory_data as E  # noqa: E402

FIG, TAB = HERE / "figures", HERE / "tables"; FIG.mkdir(exist_ok=True); TAB.mkdir(exist_ok=True)
C = {"scratch": "#2a78d6", "mae": "#eb6834"}; LBL = {"scratch": "Scratch CNN", "mae": "CNN-MAE"}
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})
CHECKS = []


def check(ok, msg):
    CHECKS.append((bool(ok), msg))
    if not ok: print("VERIFY FAIL:", msg)


def save(fig, name):
    for ext in ("png", "pdf"): fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)


def jl(p): return json.loads(Path(p).read_text())


# ------------------------------------------------------------------ primary
runs = R.load_runs(); summ = R.summarise(runs)
check(len(runs) == 22, "22 primary result files")
for _, r in runs.iterrows():
    j = jl(r.file); check(j["test_macro_f1"] == r.test_macro_f1 and j["test_accuracy"] == r.test_accuracy, f"primary {Path(r.file).name} values")
check(bool(summ.matches_reference.all()), "primary budget means equal the reference values (4 d.p.)")
budgets = list(cfg.PLAN); xs = np.arange(len(budgets))
runs.drop(columns=["per_snr"]).to_csv(TAB / "primary_per_seed.csv", index=False)
summ.to_csv(TAB / "primary_summary.csv", index=False)

for metric, ylab, name in (("test_macro_f1", "Test macro-F1", "fig1_primary_macro_f1_vs_budget"),
                           ("test_accuracy", "Test accuracy", "fig2_primary_accuracy_vs_budget")):
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for k, arm in enumerate(("scratch", "mae")):
        off = (k - 0.5) * 0.12
        m = [runs[(runs.budget == b) & (runs.arm == arm)][metric].mean() for b in budgets]
        sd = [runs[(runs.budget == b) & (runs.arm == arm)][metric].std(ddof=1) for b in budgets]
        ax.errorbar(xs + off, m, yerr=[0 if np.isnan(v) else v for v in sd], color=C[arm], marker="o" if arm == "scratch" else "s",
                    lw=1.8, capsize=3, label=f"{LBL[arm]} (mean ± SD over seeds)")
        for i, b in enumerate(budgets):
            v = runs[(runs.budget == b) & (runs.arm == arm)][metric]
            ax.scatter([xs[i] + off] * len(v), v, facecolors="none", edgecolors=C[arm], s=22, lw=0.9)
    ax.set_xticks(xs, [f"{b}%\n({len(cfg.PLAN[b])} seed{'s' if len(cfg.PLAN[b]) > 1 else ''})" for b in budgets])
    ax.set_xlabel("Labelled fraction of the training split"); ax.set_ylabel(ylab); ax.set_ylim(0, 0.45)
    ax.set_title(f"Primary experiment: {ylab.lower()} vs label budget (383,386 test signals)")
    ax.legend(loc="lower right"); ax.text(0.01, 0.97, "Hollow markers: individual seeds. 25% and 100%: one seed (no SD).",
                                          transform=ax.transAxes, fontsize=8, va="top", color="#555")
    save(fig, name)

# fig3: primary per-seed differences with test-set bootstrap CIs; paired t-test across seeds at 5%
pc = pd.DataFrame(list(csv.DictReader(open(cfg.PAIRED_CSV, newline=""))))
for c_ in ("label_fraction_pct", "seed"): pc[c_] = pc[c_].astype(int)
for c_ in ("delta_macro_f1", "delta_f1_ci95_low", "delta_f1_ci95_high", "mcnemar_p"): pc[c_] = pc[c_].astype(float)
pc.to_csv(TAB / "primary_paired_comparison.csv", index=False)
fig, ax = plt.subplots(figsize=(7.4, 4))
for i, b in enumerate(budgets):
    sub = pc[pc.label_fraction_pct == b].sort_values("seed")
    for k, (_, r) in enumerate(sub.iterrows()):
        x = i + (k - (len(sub) - 1) / 2) * 0.16
        d = runs[(runs.budget == b) & (runs.seed == r.seed) & (runs.arm == "mae")].test_macro_f1.iloc[0] - \
            runs[(runs.budget == b) & (runs.seed == r.seed) & (runs.arm == "scratch")].test_macro_f1.iloc[0]
        check(abs(d - r.delta_macro_f1) <= 5e-5 + 1e-12, f"paired_comparison delta {b}% seed {r.seed} equals result files (CSV is rounded to 4 d.p.)")
        ax.errorbar(x, d, yerr=[[d - r.delta_f1_ci95_low], [r.delta_f1_ci95_high - d]], fmt="o", color="#333", ms=4, capsize=2)
    row = summ[summ.budget == b].iloc[0]
    ax.hlines(row.delta_test_macro_f1, i - 0.3, i + 0.3, color=C["mae"], lw=2)
ax.axhline(0, color="#777", lw=1); ax.set_xticks(xs, [f"{b}%" for b in budgets])
p5 = float(summ[summ.budget == 5].paired_p.iloc[0]); check(abs(round(p5, 2) - cfg.REFERENCE_P_5PCT) < 1e-9, "5% paired p = 0.29")
ax.set_ylabel("CNN-MAE − scratch (test macro-F1)"); ax.set_xlabel("Labelled fraction")
ax.set_title(f"Primary: per-seed paired differences (5%: paired t-test over 3 seeds, p = {p5:.2f})")
fig.text(0.01, -0.07, "Dots: exact per-seed differences with 95% bootstrap CI over test signals (paired_comparison.csv, 4 d.p.;\n"
         "fixed models, ignores training randomness). Orange bar: mean over seeds. 25% and 100%: one seed.", fontsize=8, color="#555")
save(fig, "fig3_primary_per_seed_differences")

# fig4: per-SNR accuracy, all budgets, full SNR range (no selection)
snr_rows = []
fig, axes = plt.subplots(1, len(budgets), figsize=(15, 3.4), sharey=True)
for ax, b in zip(axes, budgets):
    df = R.per_snr(runs, b)
    for arm in ("scratch", "mae"):
        d = df[df.arm == cfg.ARMS[arm]]; ax.plot(d.snr, d.accuracy, color=C[arm], lw=1.6, marker=".", label=LBL[arm])
        snr_rows += [{"budget": b, "arm": arm, "snr_db": int(s), "accuracy_mean_over_seeds": a, "seeds": int(n)} for s, a, n in zip(d.snr, d.accuracy, d.seeds)]
    ax.axhline(1 / 24, ls=":", color="#777"); ax.axvspan(-20, -10, color="#eee", zorder=0)
    ax.set_title(f"{b}% labels ({len(cfg.PLAN[b])} seed{'s' if len(cfg.PLAN[b]) > 1 else ''})"); ax.set_xlabel("SNR (dB)"); ax.set_ylim(0, 1)
axes[0].set_ylabel("Test accuracy"); axes[0].legend(loc="upper left")
fig.suptitle("Primary: per-SNR test accuracy over the full SNR range (shaded: −20 to −10 dB, near chance; dotted: chance 1/24)", y=1.03)
save(fig, "fig4_primary_per_snr_accuracy")
pd.DataFrame(snr_rows).to_csv(TAB / "primary_per_snr_accuracy.csv", index=False)
j0 = jl(cfg.primary_result("mae", 5, 42))
check(abs(pd.DataFrame(snr_rows).query("budget==5 and arm=='mae'").accuracy_mean_over_seeds.iloc[0] -
          np.mean([jl(cfg.primary_result("mae", 5, s))["per_snr"]["-20"]["accuracy"] for s in (42, 43, 44)])) < 1e-12, "per-SNR value re-read from JSON")

# fig9: per-class F1 and confusion matrices at 5% (mean over 3 seeds), from saved evaluations
names = j0["class_names"]
pcf = {arm: np.mean([[jl(cfg.primary_result(arm, 5, s))["per_class"][n]["f1"] for n in names] for s in (42, 43, 44)], axis=0) for arm in ("scratch", "mae")}
order = np.argsort(pcf["scratch"])
fig, ax = plt.subplots(figsize=(7, 6))
yy = np.arange(len(names))
for k, arm in enumerate(("scratch", "mae")):
    ax.barh(yy + (k - 0.5) * 0.38, pcf[arm][order], height=0.38, color=C[arm], label=LBL[arm])
ax.set_yticks(yy, [names[i] for i in order]); ax.set_xlabel("Test F1 (mean of seeds 42–44)"); ax.set_xlim(0, 1)
ax.set_title("Primary 5% labels: per-class test F1"); ax.legend(loc="lower right")
save(fig, "fig9a_primary_per_class_f1_5pct")
pd.DataFrame({"class": names, "scratch_f1": pcf["scratch"], "mae_f1": pcf["mae"]}).to_csv(TAB / "primary_per_class_f1_5pct.csv", index=False)
fig, axes = plt.subplots(1, 2, figsize=(15, 7))
for ax, arm in zip(axes, ("scratch", "mae")):
    cm = np.mean([np.array(jl(cfg.primary_result(arm, 5, s))["confusion_matrix"], float) for s in (42, 43, 44)], axis=0)
    check(abs(cm.sum() - 383386) < 1e-6, f"{arm} 5% confusion matrix sums to the test size")
    cmn = cm / cm.sum(1, keepdims=True)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(24), names, rotation=90, fontsize=7); ax.set_yticks(range(24), names, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"{LBL[arm]}, 5% labels (row-normalised, mean of 3 seeds)"); ax.grid(False)
fig.colorbar(im, ax=axes, shrink=0.7, label="Fraction of true class")
save(fig, "fig9b_primary_confusion_5pct")

# ------------------------------------------------------------------ exploratory studies
ex = {}
for st_ in ("50", "100"):
    r_, e_, prob = E.load(st_); check(not prob and len(r_) == 6, f"{st_}-epoch study: 6 verified runs, no problems")
    check(not E.reference_mismatches(r_, st_), f"{st_}-epoch study matches its recorded cross-check values")
    for _, r in r_.iterrows():
        j = jl(r.result_file); m = jl(r.manifest)
        check(j["test_macro_f1"] == r.test_macro_f1 and m["best_val_macro_f1"] == r.val_macro_f1, f"{st_} ep {r.arm} seed {r.seed} values")
    ex[st_] = (r_, e_)
    r_.to_csv(TAB / f"exploratory_{st_}ep_per_seed.csv", index=False)
    E.summary(r_).to_csv(TAB / f"exploratory_{st_}ep_summary.csv", index=False)
an = jl(cfg.FINETUNE_100EP_DIR / "analysis" / "summary.json")["metrics"]["test_f1"]
s100 = E.summary(ex["100"][0]).set_index("metric").loc["test_macro_f1"]
check(abs(an["mae"][0] - s100.mae_mean) < 1e-12 and abs(an["diff"][1] - s100.diff_sd) < 1e-12, "100-epoch summary equals analysis/summary.json")

for st_, fname in (("50", "fig5_exploratory_50ep_per_seed"), ("100", "fig6_exploratory_100ep_per_seed")):
    r_ = ex[st_][0]; sm = E.summary(r_).set_index("metric")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4), sharey=True)
    for ax, metric, ttl in zip(axes, ("val_macro_f1", "test_macro_f1"), ("Validation (12,480-signal subset, selected epoch)", "Test (383,386 signals)")):
        p = E.paired(r_, metric)
        for i, s in enumerate(p.index):
            ax.plot([i, i], [p.loc[s, "scratch"], p.loc[s, "mae"]], color="#bbb", lw=2, zorder=1)
            for arm, mk in (("scratch", "o"), ("mae", "s")):
                ax.scatter(i, p.loc[s, arm], color=C[arm], marker=mk, s=45, zorder=2, label=LBL[arm] if i == 0 else None)
            ax.annotate(f"Δ {p.loc[s, 'diff']:+.4f}", (i + 0.08, (p.loc[s, 'mae'] + p.loc[s, 'scratch']) / 2), ha="left", va="center", fontsize=8)
        mm, ms_, d, ds = sm.loc[metric, ["mae_mean", "scratch_mean", "diff_mean", "diff_sd"]]
        ax.set_xticks(range(3), [f"seed {s}" for s in p.index]); ax.set_xlim(-0.4, 2.7)
        ax.set_title(f"{ttl}\nmean: MAE {mm:.4f}, scratch {ms_:.4f}; paired Δ {d:+.4f} ± {ds:.4f} (SD)", fontsize=9.5)
    axes[0].set_ylabel("Macro-F1"); axes[0].legend(loc="upper right")
    fig.suptitle(f"EXPLORATORY {st_}-epoch follow-up (1% labels): CNN-MAE vs scratch per seed", y=1.02)
    save(fig, fname)

# fig7: stages
st_tab = E.stage_table(runs, {50: ex["50"][0], 100: ex["100"][0]}); st_tab.to_csv(TAB / "stage_comparison_10_50_100.csv", index=False)
check(len(st_tab) == 18, "stage table has 18 rows (2 arms × 3 seeds × 3 stages)")
fig, ax = plt.subplots(figsize=(6.4, 4))
for (arm, seed), g in st_tab.groupby(["arm", "seed"]):
    ax.plot(g.epochs, g.test_macro_f1, color=C[arm], marker="o" if arm == "scratch" else "s", lw=1.2, alpha=0.85)
for arm in ("scratch", "mae"):
    mm = st_tab[st_tab.arm == arm].groupby("epochs").test_macro_f1.mean()
    ax.plot(mm.index, mm.values, color=C[arm], lw=3, alpha=0.35, label=f"{LBL[arm]} (thin: seeds; thick: mean)")
ax.axvspan(5, 20, color="#eef4ff", zorder=0); ax.text(12.5, 0.02, "primary", ha="center", fontsize=8)
ax.text(75, 0.02, "exploratory follow-ups", ha="center", fontsize=8)
ax.set_xticks([10, 50, 100]); ax.set_xlabel("Fine-tuning epochs (1% labels)"); ax.set_ylabel("Test macro-F1"); ax.set_ylim(0, 0.4)
ax.set_title("Same arm and seed across fine-tuning schedules"); ax.legend(loc="center right")
save(fig, "fig7_stages_10_50_100")

# fig8: curves (loss, macro-F1, accuracy) for each study and seed
for st_ in ("50", "100"):
    r_, e_ = ex[st_]
    fig, axes = plt.subplots(3, 4, figsize=(15, 8.5), sharex=True)
    for row, s in enumerate((42, 43, 44)):
        for arm in ("scratch", "mae"):
            g = e_[(e_.seed == s) & (e_.arm == arm)].sort_values("epoch")
            be = int(r_[(r_.seed == s) & (r_.arm == arm)].best_epoch.iloc[0])
            for col, (k, ttl) in enumerate((("train_loss", "Training loss"), ("val_loss", "Validation loss"),
                                            ("val_macro_f1", "Validation macro-F1"), ("val_accuracy", "Validation accuracy"))):
                ax = axes[row, col]; ax.plot(g.epoch, g[k], color=C[arm], lw=1.1, label=LBL[arm])
                ax.axvline(be, color=C[arm], ls="--", lw=0.8)
                if row == 0: ax.set_title(ttl)
                if col == 0: ax.set_ylabel(f"seed {s}")
    for ax in axes[-1]: ax.set_xlabel("Fine-tuning epoch")
    axes[0, 0].legend()
    fig.suptitle(f"EXPLORATORY {st_}-epoch runs: curves on the 12,480-signal validation subset (dashed: selected epoch; "
                 "training loss as logged, 6 d.p.)", y=1.01)
    save(fig, f"fig8_curves_{st_}ep")

ok = all(c for c, _ in CHECKS)
(HERE / "VERIFICATION.md").write_text(
    "# Verification of final figures and tables\n\n"
    f"All checks passed: **{ok}** ({len(CHECKS)} checks; values re-read from the raw result files).\n\n" +
    "\n".join(f"- {'PASS' if c else 'FAIL'}: {m}" for c, m in CHECKS) +
    "\n\nExploratory studies (50 and 100 epochs) were designed after the primary test results were known; they are shown"
    " separately and never pooled with the primary grid. Three seeds do not support claims of statistical significance.\n",
    encoding="utf-8")
print(f"{'ALL VERIFICATION CHECKS PASSED' if ok else 'VERIFICATION FAILED'}: {len(CHECKS)} checks; figures in {FIG}")
sys.exit(0 if ok else 1)
