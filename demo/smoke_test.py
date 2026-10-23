"""Smoke test for the demo. Read-only: it never trains, never writes into C:\\dlruns.

  C:\\venvs\\dlproj\\Scripts\\python.exe demo\\smoke_test.py            # all checks, all models
  C:\\venvs\\dlproj\\Scripts\\python.exe demo\\smoke_test.py --quick    # 5% seed-42 pair + supplementary only

Checks
  1. paths and the held-out test split (sample loads from test_idx, labels match stored test labels)
  2. invalid sample index is rejected
  3. verified results: 22 result files, budget means equal the reference values, 5% paired p ~ 0.29
  4. every available checkpoint loads; real inference returns a valid class, probabilities sum to 1,
     latency > 0; predictions on N test signals are compared with the stored test-set predictions
  5. a missing checkpoint raises ModelUnavailable (no crash)
  6. every page of the Streamlit app renders without an exception (streamlit.testing.AppTest)
  7. exploratory 50-epoch artifacts: six runs mapped from their manifests (incl. the misnamed MAE seed-43
     folder), means / SDs / paired differences recomputed independently from the raw files, matched
     10- vs 50-epoch pairs, and separation from the primary results and the Live Prediction models
  8. no file in the primary or exploratory artifact folders was created, changed or removed by this run
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import demo_config as cfg  # noqa: E402

FAILS = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)


def snapshot():
    """(path -> (size, mtime_ns)) for every file in the artifact folders the demo reads."""
    roots = [cfg.RESULTS_DIR, cfg.FINETUNE_DIR, cfg.FINETUNE_50EP_DIR, cfg.EXPLORATORY_RESULTS_DIR,
             cfg.STUDIES["100"]["run_base"], cfg.STUDIES["100"]["results"]]
    roots += [d for d in cfg.RUNS_DIR.iterdir() if d.is_dir() and d.name.startswith(cfg.FINETUNE_50EP_DIR.name)
              and d.name != cfg.FINETUNE_50EP_DIR.name] if cfg.RUNS_DIR.is_dir() else []
    snap = {}
    for r in roots:
        if r.is_dir():
            for p in r.rglob("*"):
                if p.is_file():
                    st = p.stat()
                    snap[str(p)] = (st.st_size, st.st_mtime_ns)
    return snap


def check_exploratory(study="50"):
    import json
    import pandas as pd
    import exploratory_data as E
    import results_data
    runs, epochs, problems = E.load(study)
    EP = cfg.STUDIES[study]["epochs"]; base = cfg.STUDIES[study]["run_base"]
    check(not problems, f"[{EP} ep] exploratory artifacts load without problems" + (f": {problems}" if problems else ""))
    check(len(runs) == 6 and set(zip(runs.arm, runs.seed)) == {(a, s) for a in E.ARMS for s in E.SEEDS},
          f"six exploratory runs found, one per arm x seed ({len(runs)})")
    misnamed = cfg.RUNS_DIR / (cfg.FINETUNE_50EP_DIR.name + "${a}_1pct_seed43")
    for _, r in runs.iterrows():
        expected = misnamed if (study == "50" and (r.arm, r.seed) == ("mae", 43)) else base / f"{r.arm}_1pct_seed{r.seed}"
        check(Path(r.run_dir) == expected and bool(r.misnamed_dir) == (expected == misnamed),
              f"{r.arm} seed {r.seed} -> {r.run_dir}")
        m = json.loads(Path(r.manifest).read_text())
        args = m["train_classifier_args"]
        check(int(args[args.index("--seed") + 1]) == r.seed and bool(m["init_checkpoint"]) == (r.arm == "mae")
              and m["status"] == "complete" and m["completed_epochs"] == EP and r.checkpoint_sha_ok,
              f"[{EP} ep] {r.arm} seed {r.seed}: manifest seed/init/status/{EP} epochs and checkpoint SHA-256 verified")
        res = json.loads(Path(r.result_file).read_text())
        check(Path(r.result_file).parent.name == f"{r.arm}_1pct_seed{r.seed}" and Path(res["checkpoint"]) == Path(r.checkpoint)
              and res["test_n"] == 383386, f"{r.arm} seed {r.seed}: test result {Path(r.result_file).parent.name} "
                                           "belongs to this checkpoint (383,386 test signals)")
    check(len(epochs) == 6 * EP and (epochs.groupby(["arm", "seed"]).size() == EP).all(), f"{6 * EP} per-epoch rows, {EP} per run")
    check((runs.val_n == 12480).all(), "every run selected its epoch on the 12,480-signal validation subset")

    # independent recomputation from the raw files (not via exploratory_data's aggregation)
    raw = {}
    for _, r in runs.iterrows():
        m = json.loads(Path(r.manifest).read_text()); j = json.loads(Path(r.result_file).read_text())
        raw[(r.arm, r.seed)] = {"val_macro_f1": m["best_val_macro_f1"], "test_macro_f1": j["test_macro_f1"],
                                "test_accuracy": j["test_accuracy"], "test_accuracy_snr_ge_10": j["test_accuracy_snr_ge_10"]}
    summ = E.summary(runs).set_index("metric")
    for metric in E.METRICS:
        mae = pd.Series([raw[("mae", s)][metric] for s in E.SEEDS]); scr = pd.Series([raw[("scratch", s)][metric] for s in E.SEEDS])
        d = mae - scr
        ok = all(abs(a - b) < 1e-12 for a, b in [(mae.mean(), summ.loc[metric, "mae_mean"]), (mae.std(), summ.loc[metric, "mae_sd"]),
                                                  (scr.mean(), summ.loc[metric, "scratch_mean"]), (scr.std(), summ.loc[metric, "scratch_sd"]),
                                                  (d.mean(), summ.loc[metric, "diff_mean"]), (d.std(), summ.loc[metric, "diff_sd"])])
        check(ok, f"{metric}: MAE {mae.mean():.4f}±{mae.std():.4f}, scratch {scr.mean():.4f}±{scr.std():.4f}, "
                  f"paired Δ {d.mean():+.4f}±{d.std():.4f} agree with the raw files")
    mism = E.reference_mismatches(runs, study)
    check(not mism, f"[{EP} ep] values equal the recorded cross-check values" + (f": {mism}" if mism else ""))

    prim = results_data.load_runs()
    if study == "100":
        st_ = E.stage_table(prim, {50: E.load("50")[0], 100: runs})
        ok = len(st_) == 18 and st_.groupby(["arm", "seed"]).epochs.apply(lambda v: list(v) == [10, 50, 100]).all()
        for _, r in st_.iterrows():
            want = (cfg.primary_result(r.arm, 1, r.seed) if r.epochs == 10 else None)
            ok &= (Path(r.source) == want) if want else (Path(r.source).parent.name == f"{r.arm}_1pct_seed{r.seed}")
        check(ok, "10/50/100-epoch stage table: 18 rows, same arm and seed matched to the correct source files")
        an = json.loads((cfg.FINETUNE_100EP_DIR / "analysis" / "summary.json").read_text())["metrics"]["test_f1"]
        check(abs(an["mae"][0] - summ.loc["test_macro_f1", "mae_mean"]) < 1e-12 and abs(an["diff"][0] - summ.loc["test_macro_f1", "diff_mean"]) < 1e-12,
              "100-epoch test macro-F1 summary equals the saved analysis/summary.json")
        return
    tvf = E.ten_vs_fifty(prim, runs)
    for _, r in tvf.iterrows():
        check(Path(r.source_10) == cfg.primary_result(r.arm, 1, r.seed) and
              Path(r.source_50).parent.name == f"{r.arm}_1pct_seed{r.seed}" and pd.notna(r.test_macro_f1_10),
              f"10 vs 50 epochs, {r.arm} seed {r.seed}: {r.test_macro_f1_10:.4f} ({Path(r.source_10).name}) "
              f"-> {r.test_macro_f1_50:.4f}")
    expl_files = set(runs.result_file) | set(runs.checkpoint)
    check(not expl_files & set(prim.file) and all(not Path(f).is_relative_to(cfg.RESULTS_DIR) and
                                                  not Path(f).is_relative_to(cfg.FINETUNE_DIR) for f in expl_files),
          "exploratory files are disjoint from the 22 primary result files and folders")
    check(len(prim) == 22 and set(prim.budget) == set(cfg.PLAN), "primary grid still has 22 runs over all five budgets")
    from inference import discover_models
    specs, _ = discover_models()
    ok = all(Path(s.path).parent == cfg.FINETUNE_DIR or Path(s.path) == cfg.SUPPLEMENTARY_CKPT for s in specs)
    check(ok and not any("finetune_50ep" in str(s.path) for s in specs),
          f"Live Prediction uses only the primary checkpoints and the supplementary model ({len(specs)} models)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n", type=int, default=25, help="test signals per model for stored-prediction check")
    ap.add_argument("--no-stored-check", action="store_true",
                    help="skip comparison with stored predictions (only for synthetic test fixtures)")
    ap.add_argument("--skip-app", action="store_true")
    args = ap.parse_args()

    print(f"Project root: {cfg.PROJECT_ROOT}\nRuns dir:     {cfg.RUNS_DIR}\nCache dir:    {cfg.CACHE_DIR}")
    before = snapshot()

    print("\n[1] Held-out test split")
    from data_access import TestSplit, load_class_names
    split = TestSplit()
    print(f"      backend: {split.backend}; {split.size:,} test signals")
    check(split.size == 383386, "test split has 383,386 signals")
    names = load_class_names()
    check(len(names) == 24 and not names[0].startswith("class "), f"24 class names loaded ({names[:3]}...)")
    stored = np.load(cfg.TEST_LABELS_FILE) if cfg.TEST_LABELS_FILE.is_file() else None
    for pos in (0, 1, 12345, split.size - 1):
        s = split.get(pos)
        ok = s.signal.shape == (2, 1024) and s.signal.dtype == np.float32 and np.isfinite(s.signal).all()
        check(ok, f"test index {pos}: signal [2,1024] float32 finite (class {names[s.label]}, SNR {s.snr} dB)")
        check(s.global_index == int(split.test_idx[pos]), f"test index {pos} maps to test_idx row {s.global_index}")
        if stored is not None:
            check(s.label == stored["y"][pos] and s.snr == stored["snr"][pos],
                  f"test index {pos}: label/SNR equal results/test_labels_snr.npz")

    print("\n[2] Invalid index handling")
    try:
        split.get(split.size)
        check(False, "out-of-range index rejected")
    except IndexError:
        check(True, "out-of-range index rejected with IndexError")

    print("\n[3] Verified results")
    import results_data
    runs = results_data.load_runs()
    check(len(runs) == 22, f"{len(runs)} primary result files found (expected 22)")
    summ = results_data.summarise(runs)
    for _, r in summ.iterrows():
        check(bool(r.matches_reference), f"{int(r.budget)}%: scratch {r.scratch_test_macro_f1:.4f} / "
              f"MAE {r.mae_test_macro_f1:.4f} (Δ {r.delta_test_macro_f1:+.4f}, {int(r.n_seeds)} seed(s)) "
              f"match reference")
    p5 = float(summ[summ.budget == 5].paired_p.iloc[0])
    check(abs(round(p5, 2) - cfg.REFERENCE_P_5PCT) < 1e-9, f"5% paired t-test p = {p5:.3f} (reference 0.29)")

    print("\n[4] Models and real inference")
    import torch
    from inference import ModelSpec, ModelUnavailable, discover_models, load_model, pick_device, predict
    device = pick_device()
    print(f"      torch {torch.__version__}; device {device}")
    specs, missing = discover_models()
    check(len(missing) == 0, f"all 22 primary checkpoints present ({len(missing)} missing)")
    if args.quick:
        specs = [s for s in specs if (s.budget == 5 and s.seed == 42) or s.kind == "supplementary"]
    rng = np.random.default_rng(0)
    positions = sorted(rng.choice(split.size, size=args.n, replace=False).tolist())
    signals = {p: split.get(p) for p in positions}
    for spec in specs:
        t0 = time.perf_counter()
        try:
            loaded = load_model(spec, device)
        except ModelUnavailable as exc:
            check(False, f"{spec.label}: load failed: {exc}")
            continue
        load_s = time.perf_counter() - t0
        agree, lat = 0, []
        for p in positions:
            pr = predict(loaded, signals[p].signal, p if spec.kind == "primary" else None, repeat=0)
            assert 0 <= pr.pred < 24
            assert abs(pr.probs.sum() - 1) < 1e-4 and pr.latency_ms > 0
            lat.append(pr.latency_ms)
            if pr.stored_pred is not None:
                agree += int(pr.stored_pred == pr.pred)
        msg = (f"{spec.label}: loaded in {load_s:.2f}s, {len(positions)} predictions, valid probs, "
               f"median latency {np.median(lat):.2f} ms")
        check(True, msg)
        if spec.kind == "primary" and not args.no_stored_check:
            check(agree == len(positions), f"{spec.label}: {agree}/{len(positions)} predictions equal the "
                                           "stored test-set predictions")

    print("\n[5] Missing checkpoint handling")
    ghost = ModelSpec(key="ghost", label="ghost", arm="mae", budget=5, seed=99,
                      path=cfg.RUNS_DIR / "does_not_exist.pt", kind="primary")
    try:
        load_model(ghost, device)
        check(False, "missing checkpoint raises ModelUnavailable")
    except ModelUnavailable as exc:
        check(True, f"missing checkpoint raises ModelUnavailable ({exc})")

    if not args.skip_app:
        print("\n[6] Streamlit pages render")
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(Path(__file__).with_name("app.py")), default_timeout=300)
        at.run()
        check(not at.exception, "app starts (Signal Explorer)")
        for page in ("Live Prediction", "Experimental Results", "About the Experiment"):
            at.sidebar.radio[0].set_value(page).run()
            errs = [e.value for e in at.error]
            check(not at.exception and not errs, f"page '{page}' renders" + (f" — errors: {errs}" if errs else ""))
            if page == "Experimental Results":
                labels = [t.label for t in at.tabs]
                text = " ".join(m.value for m in at.markdown)
                check(any("Primary" in l for l in labels) and any("Exploratory" in l for l in labels),
                      f"results page has separate primary and exploratory tabs ({labels})")
                check("EXPLORATORY" in text and "designed after the primary test" in text,
                      "exploratory tab carries the exploratory label and design-timing note")
                for k in ("expl_seed_50", "expl_seed_100"):
                    sb = [s for s in at.selectbox if s.key == k]
                    check(len(sb) == 1, f"training-curve seed selector {k} present")
                    if sb:
                        sb[0].set_value(44).run()
                        check(not at.exception and not [e.value for e in at.error], f"{k}: training curves re-render for seed 44")
                check(len(at.tabs) >= 3 and any("100-epoch" in t.label for t in at.tabs), "results page has a separate 100-epoch tab")
        at.sidebar.radio[0].set_value("Signal Explorer").run()
        at.sidebar.number_input[0].set_value(split.size - 1).run()
        check(not at.exception and not at.error, "changing the test index re-renders the explorer")

    print("\n[7] Exploratory 50- and 100-epoch artifacts")
    check_exploratory("50")
    check_exploratory("100")

    print("\n[8] No writes to artifact folders")
    after = snapshot()
    changed = sorted(set(before) ^ set(after) | {k for k in before.keys() & after.keys() if before[k] != after[k]})
    check(not changed and len(before) > 0, f"{len(before)} artifact files unchanged" + (f"; changed: {changed[:5]}" if changed else ""))

    print("\n" + ("ALL CHECKS PASSED" if not FAILS else f"{len(FAILS)} CHECK(S) FAILED"))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
