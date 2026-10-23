"""Exploratory 50-epoch fine-tuning results (1% labels, seeds 42-44), read from the run artifacts.

EXPLORATORY: these runs were designed after the primary test results were known. They are kept
completely separate from results_data.py (the primary grid): nothing here is pooled with,
averaged with or ranked against the primary results.

Sources (read-only):
  training runs   <RUNS>/finetune_50ep_v1/<run>/{manifest.json, per_epoch.csv, best.pt}
                  (written by experiments/run_long_finetune.py, which runs train_classifier.py unchanged)
  test results    <RUNS>/results_50ep_v1_exploratory/<run>/best.json
                  (experiments/evaluate_test.py, one call per selected checkpoint)

Runs are identified from their manifests (seed argument, presence of an MAE init checkpoint,
status, epoch count), never from folder names. One valid run, MAE seed 43, sits in a misnamed
sibling folder, <RUNS>/finetune_50ep_v1${a}_1pct_seed43, because of a quoting error in its launch
script. Its manifest records that path, and it is used in place.

Each test result is matched to its run through the checkpoint path stored in best.json. It must
agree with the manifest on seed, initialisation, selected epoch and validation macro-F1, and
best.pt must still have the SHA-256 recorded in the manifest. Any disagreement is reported.
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import demo_config as cfg

SEEDS = (42, 43, 44)
ARMS = ("mae", "scratch")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _norm(p) -> str:
    return str(Path(p)).lower()


def candidate_run_dirs(study="50"):
    """Run folders of a study. For the 50-epoch study this includes siblings whose name starts with
    the base folder name (the misnamed MAE seed-43 folder). Only folders with a manifest.json count."""
    base = cfg.STUDIES[study]["run_base"]
    dirs = []
    if base.is_dir():
        dirs += [d for d in base.iterdir() if d.is_dir()]
    if cfg.STUDIES[study]["siblings"] and base.parent.is_dir():
        dirs += [d for d in base.parent.iterdir()
                 if d.is_dir() and d.name.startswith(base.name) and d.name != base.name]
    return sorted(d for d in dirs if (d / "manifest.json").is_file())


def _arg(args, flag):
    return args[args.index(flag) + 1] if flag in args else None


def load(study="50"):
    """Return (runs, epochs, problems).

    runs:   one row per (arm, seed) with validation (from the manifest and per_epoch.csv) and test
            (from best.json) metrics and the source paths.
    epochs: per-epoch rows from per_epoch.csv for every run.
    problems: human-readable list of every inconsistency found (empty when all checks pass).
    """
    problems, rows, epoch_rows = [], [], []
    by_ckpt = {}
    S = cfg.STUDIES[study]; E = S["epochs"]
    for d in candidate_run_dirs(study):
        m = json.loads((d / "manifest.json").read_text())
        args = m.get("train_classifier_args", [])
        try:
            seed = int(_arg(args, "--seed"))
        except (TypeError, ValueError):
            problems.append(f"{d}: manifest has no --seed")
            continue
        arm = "mae" if m.get("init_checkpoint") else "scratch"
        label = f"{arm} seed {seed}"
        if m.get("status") != "complete" or m.get("completed_epochs") != E or _arg(args, "--epochs") != str(E):
            problems.append(f"{label}: not a complete {E}-epoch run ({d})")
            continue
        if _arg(args, "--label-fraction") != "1" or _arg(args, "--validation-subset") != "validation_subset_v1.npz":
            problems.append(f"{label}: unexpected label fraction or validation set ({d})")
        ckpt = d / "best.pt"
        rec = m.get("best_checkpoint", {})
        if _norm(rec.get("path", "")) != _norm(ckpt):
            problems.append(f"{label}: manifest records checkpoint {rec.get('path')}, found {ckpt}")
        sha_ok = ckpt.is_file() and _sha256(ckpt) == rec.get("sha256")
        if not sha_ok:
            problems.append(f"{label}: best.pt missing or SHA-256 differs from the manifest")
        if (arm, seed) in {(r["arm"], r["seed"]) for r in rows}:
            problems.append(f"{label}: more than one run folder")
            continue
        with open(d / "per_epoch.csv", newline="") as fh:
            ep = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(fh)]
        if [int(r["epoch"]) for r in ep] != list(range(1, E + 1)):
            problems.append(f"{label}: per_epoch.csv does not hold epochs 1-{E}")
        best_ep = int(m["best_epoch"])
        f1 = [r["val_macro_f1"] for r in ep]
        first_max = int(np.argmax(f1)) + 1  # train_classifier keeps the first strictly-best epoch
        if first_max != best_ep or f1[best_ep - 1] != m["best_val_macro_f1"]:
            problems.append(f"{label}: per_epoch.csv disagrees with the manifest's selected epoch")
        for r in ep:
            epoch_rows.append({"arm": arm, "seed": seed, **r})
        row = {"arm": arm, "seed": seed, "run_dir": str(d), "misnamed_dir": d.parent != S["run_base"],
               "manifest": str(d / "manifest.json"), "checkpoint": str(ckpt), "checkpoint_sha_ok": sha_ok,
               "best_epoch": best_ep, "val_macro_f1": m["best_val_macro_f1"],
               "val_accuracy": ep[best_ep - 1]["val_accuracy"], "val_n": int(ep[0]["val_samples"]),
               "train_seconds": m.get("seconds")}
        rows.append(row)
        by_ckpt[_norm(ckpt)] = row

    if S["results"].is_dir():
        for d in sorted(p for p in S["results"].iterdir() if p.is_dir()):
            f = d / "best.json"
            if not f.is_file():
                problems.append(f"{d}: no best.json")
                continue
            j = json.loads(f.read_text())
            row = by_ckpt.get(_norm(j.get("checkpoint", "")))
            if row is None:
                problems.append(f"{f}: checkpoint {j.get('checkpoint')} matches no verified run")
                continue
            exp_init = "mae_pretrained" if row["arm"] == "mae" else "scratch"
            if (int(j["seed"]) != row["seed"] or j["initialization"] != exp_init
                    or int(j["selected_epoch"]) != row["best_epoch"]
                    or abs(j["val_macro_f1"] - row["val_macro_f1"]) > 1e-12):
                problems.append(f"{f}: metadata disagrees with {row['manifest']}")
                continue
            if "test_macro_f1" in row:
                problems.append(f"{row['arm']} seed {row['seed']}: more than one test result")
                continue
            with open(d / "test_summary.csv", newline="") as fh:
                n_rows = len(list(csv.DictReader(fh)))
            if n_rows != 1:
                problems.append(f"{d}: test_summary.csv has {n_rows} rows (expected exactly one evaluation)")
            row.update(test_macro_f1=j["test_macro_f1"], test_accuracy=j["test_accuracy"],
                       test_accuracy_snr_ge_10=j["test_accuracy_snr_ge_10"], test_n=int(j["test_n"]),
                       result_file=str(f))
    else:
        problems.append(f"{S['results']} not found")

    runs = pd.DataFrame(rows)
    have = {(r["arm"], r["seed"]) for r in rows}
    for arm in ARMS:
        for seed in SEEDS:
            if (arm, seed) not in have:
                problems.append(f"{arm} seed {seed}: no verified run found")
    if not runs.empty:
        for col in ("test_macro_f1", "test_accuracy", "test_accuracy_snr_ge_10", "test_n", "result_file"):
            if col not in runs:
                runs[col] = np.nan
        for _, r in runs[runs.test_macro_f1.isna()].iterrows():
            problems.append(f"{r.arm} seed {r.seed}: no matching test result")
        runs = runs.sort_values(["seed", "arm"]).reset_index(drop=True)
    return runs, pd.DataFrame(epoch_rows), problems


METRICS = {"val_macro_f1": "Validation macro-F1", "test_macro_f1": "Test macro-F1",
           "test_accuracy": "Test accuracy", "test_accuracy_snr_ge_10": "Test accuracy, SNR ≥ 10 dB"}


def paired(runs: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per-seed MAE, scratch and MAE − scratch for one metric."""
    p = runs.pivot(index="seed", columns="arm", values=metric).reindex(columns=list(ARMS))
    p["diff"] = p["mae"] - p["scratch"]
    return p


def summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean and sample SD across seeds per arm, and of the paired difference, for each metric."""
    out = []
    for metric, name in METRICS.items():
        p = paired(runs, metric)
        row = {"metric": metric, "name": name, "n_seeds": int(p["diff"].notna().sum())}
        for col in ("mae", "scratch", "diff"):
            row[f"{col}_mean"] = float(p[col].mean())
            row[f"{col}_sd"] = float(p[col].std(ddof=1))
        row["mae_ahead"] = int((p["diff"] > 0).sum())
        out.append(row)
    return pd.DataFrame(out)


def ten_vs_fifty(primary_runs: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """Matched 1% runs: primary 10-epoch vs exploratory 50-epoch, same arm and seed.
    Side by side only; never averaged together."""
    rows = []
    prim = primary_runs[primary_runs.budget == 1] if not primary_runs.empty else primary_runs
    for _, r in runs.iterrows():
        p = prim[(prim.arm == r.arm) & (prim.seed == r.seed)] if not prim.empty else prim
        rows.append({"arm": r.arm, "seed": int(r.seed),
                     "test_macro_f1_10": float(p.test_macro_f1.iloc[0]) if len(p) else np.nan,
                     "val_macro_f1_10": float(p.val_macro_f1.iloc[0]) if len(p) else np.nan,
                     "epoch_10": int(p.selected_epoch.iloc[0]) if len(p) else None,
                     "source_10": p.file.iloc[0] if len(p) else None,
                     "test_macro_f1_50": r.test_macro_f1, "val_macro_f1_50": r.val_macro_f1,
                     "epoch_50": int(r.best_epoch), "source_50": r.result_file})
    df = pd.DataFrame(rows)
    df["gain"] = df.test_macro_f1_50 - df.test_macro_f1_10
    return df


def stage_table(primary_runs: pd.DataFrame, studies: dict) -> pd.DataFrame:
    """Long table of test macro-F1 for the same arm and seed at each stage: primary 10 epochs and every
    exploratory study given as {epochs: runs}. Rows are kept separate per stage; nothing is averaged."""
    rows = []
    prim = primary_runs[primary_runs.budget == 1] if not primary_runs.empty else primary_runs
    for _, r in prim.iterrows():
        rows.append({"arm": r.arm, "seed": int(r.seed), "epochs": 10, "stage": "primary",
                     "test_macro_f1": float(r.test_macro_f1), "selected_epoch": int(r.selected_epoch), "source": r.file})
    for E, runs in sorted(studies.items()):
        for _, r in runs.iterrows():
            rows.append({"arm": r.arm, "seed": int(r.seed), "epochs": int(E), "stage": "exploratory",
                         "test_macro_f1": float(r.test_macro_f1), "selected_epoch": int(r.best_epoch), "source": r.result_file})
    return pd.DataFrame(rows).sort_values(["seed", "arm", "epochs"]).reset_index(drop=True)


def reference_mismatches(runs: pd.DataFrame, study="50"):
    """Compare with the cross-check values in demo_config (4 d.p.): per-seed values for the 50-epoch
    study, the reported test macro-F1 mean/SD summary for the 100-epoch study."""
    bad = []
    if study == "100":
        p = paired(runs, "test_macro_f1")
        for col, (mu, sd) in cfg.EXPLORATORY_100_REFERENCE.items():
            got = (float(p[col].mean()), float(p[col].std(ddof=1)))
            if abs(round(got[0], 4) - mu) > 1e-9 or abs(round(got[1], 4) - sd) > 1e-9:
                bad.append(f"test_macro_f1 {col}: files {got[0]:.4f} ± {got[1]:.4f}, reference {mu:.4f} ± {sd:.4f}")
        return bad
    for metric, ref in cfg.EXPLORATORY_REFERENCE.items():
        p = paired(runs, metric)
        for seed, (m_ref, s_ref) in ref.items():
            for arm, v in (("mae", m_ref), ("scratch", s_ref)):
                got = p.loc[seed, arm] if seed in p.index else np.nan
                if not abs(round(float(got), 4) - v) < 1e-9:
                    bad.append(f"{metric} {arm} seed {seed}: files {got:.4f}, reference {v:.4f}")
    return bad
