"""Verified primary results, recomputed from the stored test-result files.

Source of truth: C:\\dlruns\\results\\{scratch,mae}_{budget}pct_seed{seed}_v1.json (written once
per selected checkpoint by experiments/evaluate_test.py) and results\\paired_comparison.csv.
Nothing here trains or evaluates a model.
"""

import csv
import json
from math import sqrt

import numpy as np
import pandas as pd

import demo_config as cfg


def load_runs() -> pd.DataFrame:
    rows = []
    for budget, seeds in cfg.PLAN.items():
        for seed in seeds:
            for arm in cfg.ARMS:
                p = cfg.primary_result(arm, budget, seed)
                if not p.is_file():
                    continue
                j = json.loads(p.read_text())
                if (j.get("initialization") != cfg.ARM_INIT[arm] or int(j["label_fraction_pct"]) != budget
                        or int(j["seed"]) != seed):
                    raise ValueError(f"{p.name}: metadata does not match its file name")
                rows.append({"arm": arm, "budget": budget, "seed": seed,
                             "test_macro_f1": j["test_macro_f1"], "test_accuracy": j["test_accuracy"],
                             "test_accuracy_snr_ge_10": j["test_accuracy_snr_ge_10"],
                             "val_macro_f1": j["val_macro_f1"], "selected_epoch": j["selected_epoch"],
                             "test_n": j["test_n"], "per_snr": j["per_snr"], "file": str(p)})
    return pd.DataFrame(rows)


def _p_two_sided_df2(t):
    # Student t with 2 degrees of freedom has a closed-form CDF.
    return 1 - abs(t) / sqrt(2 + t * t)


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    out = []
    for budget in cfg.PLAN:
        sub = runs[runs.budget == budget]
        row = {"budget": budget}
        for arm in cfg.ARMS:
            a = sub[sub.arm == arm].sort_values("seed")
            row[f"n_{arm}"] = len(a)
            for m in ("test_macro_f1", "test_accuracy"):
                row[f"{arm}_{m}"] = a[m].mean() if len(a) else np.nan
                row[f"{arm}_{m}_sd"] = a[m].std(ddof=1) if len(a) > 1 else np.nan
        row["n_seeds"] = min(row["n_scratch"], row["n_mae"])
        for m in ("test_macro_f1", "test_accuracy"):
            row[f"delta_{m}"] = row[f"mae_{m}"] - row[f"scratch_{m}"]
        s = sub[sub.arm == "scratch"].set_index("seed")["test_macro_f1"]
        m_ = sub[sub.arm == "mae"].set_index("seed")["test_macro_f1"]
        common = sorted(set(s.index) & set(m_.index))
        d = np.array([m_[k] - s[k] for k in common])
        row["mae_wins"] = int((d > 0).sum())
        row["paired_t"] = row["paired_p"] = np.nan
        if len(d) == 3 and d.std(ddof=1) > 0:
            t = d.mean() / (d.std(ddof=1) / sqrt(3))
            row["paired_t"], row["paired_p"] = t, _p_two_sided_df2(t)
        ref = cfg.REFERENCE_MACRO_F1.get(budget)
        row["matches_reference"] = bool(ref and abs(round(row["scratch_test_macro_f1"], 4) - ref[0]) < 1e-9
                                        and abs(round(row["mae_test_macro_f1"], 4) - ref[1]) < 1e-9)
        out.append(row)
    return pd.DataFrame(out)


def load_paired() -> pd.DataFrame:
    if not cfg.PAIRED_CSV.is_file():
        return pd.DataFrame()
    with open(cfg.PAIRED_CSV, newline="") as fh:
        return pd.DataFrame(list(csv.DictReader(fh)))


def report_p_value(budget: int = 5):
    """Paired-test p recorded by the report build (report/tables/values.json), if present."""
    try:
        v = json.loads(cfg.REPORT_VALUES.read_text())
        return v["primary"][str(budget)]["paired_t_p_two_sided_df2"]
    except Exception:
        return None


def per_snr(runs: pd.DataFrame, budget: int) -> pd.DataFrame:
    rows = []
    for arm in cfg.ARMS:
        sub = runs[(runs.budget == budget) & (runs.arm == arm)]
        if sub.empty:
            continue
        snrs = sorted(int(k) for k in sub.iloc[0]["per_snr"])
        acc = np.mean([[r["per_snr"][str(s)]["accuracy"] for s in snrs] for _, r in sub.iterrows()], axis=0)
        rows += [{"arm": cfg.ARMS[arm], "snr": s, "accuracy": float(a), "seeds": len(sub)} for s, a in zip(snrs, acc)]
    return pd.DataFrame(rows)
