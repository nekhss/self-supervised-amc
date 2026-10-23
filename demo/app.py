"""CNN-MAE for Automatic Modulation Classification: research demo.

Launch from the project root:
    C:\\venvs\\dlproj\\Scripts\\python.exe -m streamlit run demo\\app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

import demo_config as cfg  # noqa: E402
import charts  # noqa: E402
import results_data  # noqa: E402
import exploratory_data  # noqa: E402
from data_access import DataUnavailable, TestSplit, load_class_names  # noqa: E402

st.set_page_config(page_title="CNN-MAE for AMC", page_icon="📡", layout="wide",
                   initial_sidebar_state="expanded")

CSS = """
<style>
.block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px;}
h1.app-title {font-size: 1.85rem; font-weight: 650; letter-spacing: -0.01em; margin: 0 0 0.15rem 0;}
p.app-sub {color: #6b6a65; font-size: 1.02rem; margin: 0 0 0.4rem 0;}
.section-label {text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.72rem; color: #6b6a65;
                font-weight: 600; margin-bottom: 0.2rem;}
.card {border: 1px solid #e4e3df; border-radius: 10px; padding: 0.9rem 1.1rem; background: #ffffff;}
.kv {display: flex; flex-direction: column; gap: 0.1rem;}
.kv .k {font-size: 0.78rem; color: #6b6a65;}
.kv .v {font-size: 1.25rem; font-weight: 600; color: #1d1c1a;}
.badge {display: inline-block; padding: 0.12rem 0.55rem; border-radius: 999px; font-size: 0.78rem;
        font-weight: 600; border: 1px solid; margin-right: 0.3rem;}
.b-ok {color: #0f6b3f; border-color: #9fd3b5; background: #eef8f2;}
.b-bad {color: #9b2c2c; border-color: #f0b4b4; background: #fdf1f1;}
.b-info {color: #3d3c38; border-color: #d6d5cf; background: #f6f5f2;}
.b-warn {color: #7a4b00; border-color: #f0cf8c; background: #fff7e6;}
.note {color: #4a4945; font-size: 0.9rem; line-height: 1.5;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def show_chart(chart):
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:
        st.altair_chart(chart, use_container_width=True)


def show_table(df, **kw):
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def kv(label, value):
    return f'<div class="kv"><span class="k">{label}</span><span class="v">{value}</span></div>'


def badge(text, kind="info"):
    return f'<span class="badge b-{kind}">{text}</span>'


# ------------------------------------------------------------------ cached resources
@st.cache_resource(show_spinner="Opening the held-out test split…")
def get_split():
    return TestSplit()


@st.cache_data(show_spinner=False)
def get_class_names():
    return load_class_names()


@st.cache_data(show_spinner="Reading verified result files…")
def get_results():
    runs = results_data.load_runs()
    return runs, results_data.summarise(runs) if not runs.empty else pd.DataFrame(), results_data.load_paired()


@st.cache_data(show_spinner="Reading exploratory run artifacts…")
def get_exploratory(study: str = "50"):
    return exploratory_data.load(study)


@st.cache_resource(show_spinner=False)
def get_registry():
    from inference import discover_models
    return discover_models()


@st.cache_resource(show_spinner="Loading model weights…")
def get_model(key: str):
    from inference import load_model, pick_device
    specs, _ = get_registry()
    spec = next(s for s in specs if s.key == key)
    return load_model(spec, pick_device())


@st.cache_data(show_spinner=False, max_entries=64)
def get_sample(position: int):
    s = get_split().get(position)
    return {"position": s.position, "global_index": s.global_index, "signal": s.signal,
            "label": s.label, "snr": s.snr}


names = get_class_names()

# ------------------------------------------------------------------ header + navigation
st.markdown('<h1 class="app-title">CNN-MAE for Automatic Modulation Classification</h1>'
            '<p class="app-sub">Evaluating self-supervised pretraining under limited labeled data · '
            'RadioML 2018.01A · 24 classes · input [2, 1024]</p>', unsafe_allow_html=True)

PAGES = ["Signal Explorer", "Live Prediction", "Experimental Results", "About the Experiment"]
with st.sidebar:
    st.markdown('<div class="section-label">Navigate</div>', unsafe_allow_html=True)
    page = st.radio("Section", PAGES, label_visibility="collapsed")
    st.divider()

# ------------------------------------------------------------------ sample selector (shared)
split, split_error = None, None
try:
    split = get_split()
except DataUnavailable as exc:
    split_error = str(exc)
except Exception as exc:  # unexpected data error: show, don't crash
    split_error = f"{type(exc).__name__}: {exc}"

if "pos" not in st.session_state:
    st.session_state.pos = 0

with st.sidebar:
    st.markdown('<div class="section-label">Held-out test signal</div>', unsafe_allow_html=True)
    if split is None:
        st.error("Test data unavailable.")
    else:
        mode = st.radio("Choose by", ["Test index", "Class & SNR"], horizontal=True)
        if mode == "Test index":
            pos = st.number_input(f"Test index (0 – {split.size - 1:,})", 0, split.size - 1,
                                  value=int(st.session_state.pos), step=1)
            st.session_state.pos = int(pos)
            if st.button("Random test signal", use_container_width=True):
                st.session_state.pos = int(np.random.default_rng().integers(split.size))
                st.rerun()
        else:
            if split.labels is None:
                st.info("Class/SNR filtering needs the cached labels; use the test index instead.")
            else:
                cls = st.selectbox("Modulation", range(24), format_func=lambda i: names[i])
                snr = st.select_slider("SNR (dB)", options=list(range(-20, 32, 2)), value=10)
                matches = split.matching(cls, snr)
                if len(matches) == 0:
                    st.warning("No test signals match.")
                else:
                    k = st.number_input(f"Match (0 – {len(matches) - 1})", 0, len(matches) - 1, 0, step=1)
                    st.session_state.pos = int(matches[int(k)])
        st.caption(f"Selected test index **{st.session_state.pos:,}** · source: {split.backend}")

sample, sample_error = None, None
if split is not None:
    try:
        sample = get_sample(int(st.session_state.pos))
    except Exception as exc:
        sample_error = f"{type(exc).__name__}: {exc}"


def require_sample():
    if split_error:
        st.error(f"The held-out test split could not be opened.\n\n{split_error}")
        st.stop()
    if sample_error:
        st.error(f"Could not read this test signal: {sample_error}")
        st.stop()


# ------------------------------------------------------------------ pages
def page_explorer():
    require_sample()
    s = sample
    sig = s["signal"]
    st.markdown('<div class="section-label">Signal Explorer</div>', unsafe_allow_html=True)
    cols = st.columns(5)
    items = [("True modulation", names[s["label"]]), ("True SNR", f'{s["snr"]} dB'),
             ("Test index", f'{s["position"]:,}'), ("HDF5 row", f'{s["global_index"]:,}'),
             ("Input", f"{sig.shape[0]} × {sig.shape[1]} · {sig.dtype}")]
    for c, (k, v) in zip(cols, items):
        c.markdown(f'<div class="card">{kv(k, v)}</div>', unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([1.75, 1])
    with left:
        st.markdown("**I/Q components over time**")
        show_chart(charts.iq_waveform(sig))
        st.caption("Raw values exactly as fed to the models (no normalisation). Drag to zoom along time.")
    with right:
        st.markdown("**Constellation (raw samples)**")
        show_chart(charts.constellation(sig))
        st.caption("All 1,024 samples plotted as I vs Q; no carrier or symbol synchronisation is applied.")
    rms = float(np.sqrt((sig ** 2).sum(0).mean()))
    st.markdown(
        f'<p class="note">Signal statistics: mean I = {sig[0].mean():+.4f}, mean Q = {sig[1].mean():+.4f}, '
        f'RMS magnitude = {rms:.4f}, peak |x| = {np.abs(sig).max():.4f}. This signal belongs to the '
        f'held-out test split (data_splits.npz → test_idx), which no model was trained or tuned on.</p>',
        unsafe_allow_html=True)


def render_prediction(col, key, s):
    from inference import ModelUnavailable, predict
    specs, _ = get_registry()
    spec = next(x for x in specs if x.key == key)
    color = cfg.COLOR[spec.arm]
    with col:
        with st.container(border=True):
            tag = badge("primary grid", "info") if spec.kind == "primary" else badge("supplementary", "warn")
            st.markdown(f'<div style="border-left:4px solid {color};padding-left:0.6rem">'
                        f'<b>{spec.label}</b><br>{tag}</div>', unsafe_allow_html=True)
            try:
                loaded = get_model(key)
                pr = predict(loaded, s["signal"], s["position"] if spec.kind == "primary" else None)
            except ModelUnavailable as exc:
                st.error(f"Model unavailable: {exc}")
                return
            except Exception as exc:
                st.error(f"Inference failed ({type(exc).__name__}): {exc}")
                return
            correct = pr.pred == s["label"]
            st.markdown(kv("Predicted modulation", names[pr.pred]), unsafe_allow_html=True)
            st.markdown(badge("correct", "ok") if correct else badge(f"incorrect · true {names[s['label']]}", "bad"),
                        unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.metric("Model confidence", f"{pr.probs[pr.pred]:.1%}",
                      help="Softmax probability of the predicted class. It is the model's own score, "
                           "not a guarantee of correctness.")
            c2.metric("Inference latency", f"{pr.latency_ms:.2f} ms",
                      help="Measured for this prediction: host-to-device copy, forward pass and softmax "
                           f"({pr.device}, {pr.precision}). Excludes model loading and data reading.")
            c2.caption(f"median of 10 repeat passes: {pr.latency_median_ms:.2f} ms")
            show_chart(charts.top_probs(pr.probs, names, s["label"], color))
            m = loaded.meta
            lines = [f"Device {pr.device} · {pr.precision} · {m['parameters']:,} parameters",
                     f"Selected epoch {m['selected_epoch']} · validation macro-F1 {m['val_macro_f1']:.4f} "
                     f"(on {m['val_samples']:,} validation signals)"]
            if "test_macro_f1" in m:
                lines.append(f"Whole test set (stored evaluation): macro-F1 {m['test_macro_f1']:.4f}, "
                             f"accuracy {m['test_accuracy']:.4f}")
            if pr.stored_pred is not None:
                ok = pr.stored_pred == pr.pred
                lines.append(("Matches" if ok else "Differs from") + " the stored test-set prediction for this "
                             f"signal ({names[pr.stored_pred]})")
            if spec.kind == "supplementary":
                lines.append(spec.note)
            st.caption("  \n".join(lines))
            st.caption(f"Checkpoint: `{spec.path}`")


def page_prediction():
    require_sample()
    s = sample
    try:
        specs, missing = get_registry()
    except Exception as exc:
        st.error(f"PyTorch or the model code could not be imported: {exc}")
        return
    st.markdown('<div class="section-label">Live Prediction</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="note">Real inference on the selected held-out signal '
                f'(test index {s["position"]:,}, true class <b>{names[s["label"]]}</b>, SNR {s["snr"]} dB). '
                'Both models are the fine-tuned classifiers from the primary grid for the chosen label budget '
                'and seed; they share the architecture and differ only in encoder initialisation.</p>',
                unsafe_allow_html=True)
    primary = [x for x in specs if x.kind == "primary"]
    if not primary:
        st.error(f"No primary-grid checkpoints found in {cfg.FINETUNE_DIR}.")
        return
    c1, c2, c3 = st.columns([1, 1, 2])
    budgets = sorted({x.budget for x in primary})
    budget = c1.selectbox("Label budget", budgets, index=budgets.index(5) if 5 in budgets else 0,
                          format_func=lambda b: f"{b}% labels")
    seeds = sorted({x.seed for x in primary if x.budget == budget})
    seed = c2.selectbox("Seed", seeds)
    show_supp = c3.checkbox("Also show the supplementary 1% CNN-MAE model",
                            value=False, disabled=not any(x.kind == "supplementary" for x in specs),
                            help="Trained after the primary grid with a different validation set for epoch "
                                 "selection; not part of the primary results.")
    keys = {x.arm: x.key for x in primary if x.budget == budget and x.seed == seed}
    cols = st.columns(3 if show_supp else 2)
    for i, arm in enumerate(cfg.ARMS):
        if arm in keys:
            render_prediction(cols[i], keys[arm], s)
        else:
            cols[i].warning(f"No {cfg.ARMS[arm]} checkpoint for {budget}% / seed {seed}: "
                            f"`{cfg.primary_ckpt(arm, budget, seed)}` is missing.")
    if show_supp:
        supp = next(x for x in specs if x.kind == "supplementary")
        render_prediction(cols[2], supp.key, s)
    if missing:
        with st.expander(f"{len(missing)} primary checkpoint(s) missing"):
            st.code("\n".join(map(str, missing)))
    st.info("A single prediction illustrates the models; it is not evidence that one approach is better. "
            "Aggregate test-set results are in Experimental Results.", icon="ℹ️")


def page_results():
    st.markdown('<div class="section-label">Experimental Results</div>', unsafe_allow_html=True)
    tab_p, tab_50, tab_100 = st.tabs(["Primary experiments (authoritative)", "Exploratory · 50-epoch follow-up (1% labels)",
                                      "Exploratory · 100-epoch follow-up (1% labels)"])
    with tab_p:
        results_primary()
    with tab_50:
        results_exploratory("50")
    with tab_100:
        results_exploratory("100")


def results_primary():
    st.markdown('<p class="note">The pre-specified grid: scratch vs CNN-MAE at five label budgets, 10 fine-tuning '
                'epochs, each selected model evaluated <b>once</b> on the 383,386-signal held-out test set. '
                'These results are the basis of the project\'s conclusions.</p>', unsafe_allow_html=True)
    try:
        runs, summ, paired = get_results()
    except Exception as exc:
        st.error(f"Could not read the result files in {cfg.RESULTS_DIR}: {exc}")
        return
    if runs.empty:
        st.error(f"No primary result files found in {cfg.RESULTS_DIR}.")
        return
    n_runs = len(runs)
    if n_runs != 22:
        st.warning(f"Expected 22 result files, found {n_runs}; summaries use what is available.")
    bad = summ[~summ.matches_reference]
    if bad.empty:
        st.markdown(badge("All budget means match the reference values (4 d.p.)", "ok") +
                    badge(f"{n_runs} runs · {int(runs.test_n.iloc[0]):,} test signals each", "info"),
                    unsafe_allow_html=True)
    else:
        st.warning("Discrepancy between result files and reference values at budgets "
                   + ", ".join(f"{b}%" for b in bad.budget) + ". The values shown come from the result files.")

    metric = st.radio("Metric", ["Macro-F1", "Accuracy"], horizontal=True)
    mcol = "test_macro_f1" if metric == "Macro-F1" else "test_accuracy"
    left, right = st.columns([1.35, 1])
    with left:
        st.markdown(f"**Test {metric} vs. label budget**")
        show_chart(charts.budget_lines(runs, mcol, f"Test {metric}"))
        st.caption("Filled markers: mean over seeds. Hollow markers: individual seeds. "
                   "25% and 100% have one seed each. The y-axis starts at zero.")
    with right:
        st.markdown(f"**CNN-MAE − scratch ({metric})**")
        show_chart(charts.delta_chart(runs, mcol))
        st.caption("Bars: mean difference. Dots: per-seed paired difference (same seed, same label subset).")

    table = pd.DataFrame({
        "Label budget": summ.budget.map(lambda b: f"{b}%"),
        "Seeds": summ.n_seeds,
        "Scratch CNN": summ[f"scratch_{mcol}"].map("{:.4f}".format),
        "CNN-MAE": summ[f"mae_{mcol}"].map("{:.4f}".format),
        "MAE − scratch": summ[f"delta_{mcol}"].map("{:+.4f}".format),
        "MAE better (seeds)": [f"{w}/{n}" for w, n in zip(summ.mae_wins, summ.n_seeds)],
        "Scratch SD": summ[f"scratch_{mcol}_sd"].map(lambda v: "—" if pd.isna(v) else f"{v:.4f}"),
        "MAE SD": summ[f"mae_{mcol}_sd"].map(lambda v: "—" if pd.isna(v) else f"{v:.4f}"),
    })
    st.markdown(f"**Measured test {metric}**")
    show_table(table)
    st.caption("“MAE better (seeds)” always counts macro-F1. SD: sample standard deviation over seeds "
               "(not defined for single-seed budgets).")

    r5 = summ[summ.budget == 5].iloc[0]
    p_rep = results_data.report_p_value(5)
    p_txt = f"{r5.paired_p:.2f}"
    if p_rep is not None and abs(p_rep - r5.paired_p) > 1e-6:
        p_txt += f" (report records {p_rep:.3f})"
    largest = summ.loc[summ.delta_test_macro_f1.abs().idxmax()]
    st.markdown("**Interpretation**")
    st.markdown(f"""
<div class="card note">
<ul style="margin:0 0 0 1rem;padding:0">
<li><b>MAE is not consistently better.</b> Mean macro-F1 differences range from
{summ.delta_test_macro_f1.min():+.4f} to {summ.delta_test_macro_f1.max():+.4f}; at 1%, 10% and 25% the
MAE-initialised model is marginally <i>lower</i>.</li>
<li><b>Largest difference: {int(largest.budget)}% labels, {largest.delta_test_macro_f1:+.4f} macro-F1</b>
({r5.mae_test_macro_f1:.4f} vs {r5.scratch_test_macro_f1:.4f}). Two of three seeds favour MAE; the per-seed
differences vary in sign, and a paired t-test across the three seeds gives p = {p_txt} (2 degrees of freedom).</li>
<li><b>25% and 100% use one seed each</b>, so those comparisons cannot show a robust advantage either way.</li>
<li>Seed-to-seed standard deviations (up to {np.nanmax(summ[['scratch_test_macro_f1_sd','mae_test_macro_f1_sd']].values):.4f})
are larger than most of the measured differences.</li>
<li><b>Cautious conclusion:</b> pretraining may help in some low-label settings, but the observed improvement is
modest and was not consistently demonstrated in these experiments.</li>
</ul></div>""", unsafe_allow_html=True)

    with st.expander("Per-seed paired comparison (test set)"):
        if paired.empty:
            st.info(f"{cfg.PAIRED_CSV} not found.")
        else:
            show = paired.rename(columns={
                "label_fraction_pct": "Labels %", "seed": "Seed", "macro_f1_scratch": "Scratch F1",
                "macro_f1_mae": "MAE F1", "delta_macro_f1": "Δ F1", "delta_f1_ci95_low": "CI low",
                "delta_f1_ci95_high": "CI high", "delta_acc": "Δ accuracy", "mcnemar_p": "McNemar p"})
            show_table(show[["Labels %", "Seed", "Scratch F1", "MAE F1", "Δ F1", "CI low", "CI high",
                             "Δ accuracy", "McNemar p"]])
            st.caption("Bootstrap intervals and McNemar tests resample test signals for two fixed trained models. "
                       "They ignore training randomness, so a per-seed interval that excludes zero does not "
                       "show a reproducible effect (they occur with both signs). McNemar p printed as 0 is below "
                       "floating-point resolution.")

    with st.expander("Accuracy by SNR (descriptive)"):
        b = st.selectbox("Label budget", list(cfg.PLAN), index=1, format_func=lambda x: f"{x}% labels",
                         key="snr_budget")
        df = results_data.per_snr(runs, b)
        if df.empty:
            st.info("No per-SNR data for this budget.")
        else:
            show_chart(charts.snr_chart(df))
            st.caption(f"Mean over {df.seeds.max()} seed(s), from the stored per-SNR test metrics. Dashed line: "
                       "chance (1/24). Descriptive only; no per-SNR uncertainty is estimated.")

    st.caption(f"Sources: {cfg.RESULTS_DIR}\\{{scratch,mae}}_{{budget}}pct_seed{{seed}}_v1.json and "
               f"{cfg.PAIRED_CSV.name}. Exploratory runs, sweeps and diagnostics are not mixed into these results.")


def _d(x):
    """Signed difference, 4 d.p.; values that round to zero are shown as ≈0 with their sign kept."""
    return f"≈0 ({x:+.5f})" if abs(x) < 5e-5 else f"{x:+.4f}"


def _ms(mean, sd, signed=False):
    f = "{:+.4f}" if signed else "{:.4f}"
    return f"{f.format(mean)} ± {sd:.4f}"


def results_exploratory(study="50"):
    E = cfg.STUDIES[study]["epochs"]
    restart = (" All six runs were trained anew from the same initialisation (not continued from the 50-epoch runs), and this follow-up was designed after the 50-epoch test results were also known." if study == "100" else "")
    st.markdown(
        '<div class="card note" style="border-color:#f0cf8c;background:#fffaf0">'
        + badge("EXPLORATORY", "warn") +
        ' These six runs (1% labels, seeds 42–44, CNN-MAE and scratch) were <b>designed after the primary test '
        'results were known</b>. Their test evaluation is therefore <b>not an untouched confirmatory test</b>. They '
        'are shown separately and are never averaged, pooled or ranked together with the primary grid. The only '
        f'change from the primary protocol is a longer fine-tuning schedule: <b>{E} classifier epochs</b> (cosine '
        f'learning rate over {E} epochs) instead of 10. Both arms got the same schedule, and the MAE encoder is the same '
        f'5-epoch pretraining checkpoint.{restart}</div>', unsafe_allow_html=True)
    st.write("")
    try:
        runs, epochs, problems = get_exploratory(study)
        prev = {E2: get_exploratory(s2)[0] for s2, E2 in ((s, cfg.STUDIES[s]['epochs']) for s in cfg.STUDIES)
                if E2 <= E}
        primary_runs = get_results()[0]
    except Exception as exc:
        st.error(f"Could not read the exploratory artifacts: {type(exc).__name__}: {exc}")
        return
    if problems:
        st.warning("Artifact checks reported problems; affected runs are excluded:\n\n- " + "\n- ".join(problems))
    if runs.empty or runs.test_macro_f1.isna().any() or len(runs) != 6:
        st.error("The six exploratory runs are not all available with verified test results.")
        return
    mism = exploratory_data.reference_mismatches(runs, study)
    badges = badge(f"6 runs verified: manifest, checkpoint SHA-256, {E}/{E} epochs, one test evaluation each", "info")
    badges += (badge("Per-seed macro-F1 matches the recorded cross-check values (4 d.p.)" if study == "50" else
                     "Test macro-F1 mean ± SD matches the reported summary (4 d.p.)", "ok") if not mism else "")
    st.markdown(badges, unsafe_allow_html=True)
    if mism:
        st.warning("Values differ from the cross-check table; the values shown come from the files:\n\n- "
                   + "\n- ".join(mism))

    val = exploratory_data.paired(runs, "val_macro_f1")
    test = exploratory_data.paired(runs, "test_macro_f1")
    ep = runs.pivot(index="seed", columns="arm", values="best_epoch")
    seed_rows = [{"Seed": str(s), "Selected epoch (MAE / scratch)": f"{int(ep.loc[s, 'mae'])} / {int(ep.loc[s, 'scratch'])}",
                  "Val F1 · MAE": f"{val.loc[s, 'mae']:.4f}", "Val F1 · scratch": f"{val.loc[s, 'scratch']:.4f}",
                  "Val Δ": _d(val.loc[s, 'diff']),
                  "Test F1 · MAE": f"{test.loc[s, 'mae']:.4f}", "Test F1 · scratch": f"{test.loc[s, 'scratch']:.4f}",
                  "Test Δ": _d(test.loc[s, 'diff'])} for s in val.index]
    seed_rows.append({"Seed": "Mean ± SD", "Selected epoch (MAE / scratch)": "—",
                      "Val F1 · MAE": _ms(val.mae.mean(), val.mae.std(ddof=1)),
                      "Val F1 · scratch": _ms(val.scratch.mean(), val.scratch.std(ddof=1)),
                      "Val Δ": _ms(val["diff"].mean(), val["diff"].std(ddof=1), True),
                      "Test F1 · MAE": _ms(test.mae.mean(), test.mae.std(ddof=1)),
                      "Test F1 · scratch": _ms(test.scratch.mean(), test.scratch.std(ddof=1)),
                      "Test Δ": _ms(test["diff"].mean(), test["diff"].std(ddof=1), True)})
    st.markdown(f"**Macro-F1 by seed: CNN-MAE vs scratch ({E} fine-tuning epochs)**")
    show_table(pd.DataFrame(seed_rows))
    st.caption(f"Val = best validation macro-F1 on the fixed {int(runs.val_n.iloc[0]):,}-signal validation subset, "
               f"the score used to choose the epoch. Test = that checkpoint evaluated once on "
               f"{int(runs.test_n.iloc[0]):,} test signals. Δ = CNN-MAE − scratch for the same seed (same label "
               "subset and data order). SD: sample standard deviation over the three seeds.")

    summ = exploratory_data.summary(runs)
    where = {"val_macro_f1": f"validation subset ({int(runs.val_n.iloc[0]):,})",
             "test_macro_f1": "test set", "test_accuracy": "test set", "test_accuracy_snr_ge_10": "test set, SNR ≥ 10 dB"}
    st.markdown("**All metrics, mean ± SD over seeds (each metric labelled with the data it was measured on)**")
    show_table(pd.DataFrame({
        "Metric": summ.name, "Measured on": summ.metric.map(where),
        "CNN-MAE": [_ms(a, b) for a, b in zip(summ.mae_mean, summ.mae_sd)],
        "Scratch CNN": [_ms(a, b) for a, b in zip(summ.scratch_mean, summ.scratch_sd)],
        "MAE − scratch (paired)": [_ms(a, b, True) for a, b in zip(summ.diff_mean, summ.diff_sd)],
        "Seeds with MAE ahead": [f"{k}/{n}" for k, n in zip(summ.mae_ahead, summ.n_seeds)]}))

    left, right = st.columns([1, 1.15])
    with left:
        st.markdown("**Paired difference per seed (macro-F1)**")
        show_chart(charts.expl_paired_diff(val, test))
        st.caption(f"Each point is CNN-MAE − scratch for one seed: circles on the {int(runs.val_n.iloc[0]):,}-signal "
                   f"validation subset, diamonds on the {int(runs.test_n.iloc[0]):,}-signal test set. Above zero favours CNN-MAE.")
    stages = exploratory_data.stage_table(primary_runs, prev)
    epochs_shown = sorted(stages.epochs.unique())
    with right:
        st.markdown(f"**Same arm and seed across {', '.join(map(str, epochs_shown))} fine-tuning epochs (test macro-F1, 1% labels)**")
        show_chart(charts.stages_chart(stages))
        st.caption("10 epochs = primary experiment; longer schedules = exploratory follow-ups. Each line joins the same arm "
                   "and seed. Stages are shown side by side only, never averaged together. The y-axis starts at zero.")
    wide = stages.pivot_table(index=["arm", "seed"], columns="epochs", values="test_macro_f1").reset_index()
    sel_ep = stages.pivot_table(index=["arm", "seed"], columns="epochs", values="selected_epoch").reset_index()
    tbl = {"Approach": wide.arm.map(cfg.ARMS), "Seed": wide.seed.astype(str)}
    for e_ in epochs_shown:
        tbl[f"{e_} epochs: test F1 (selected epoch)"] = [f"{a:.4f} ({int(b)})" for a, b in zip(wide[e_], sel_ep[e_])]
    p_ep = epochs_shown[-2]
    gain = wide[E] - wide[p_ep]
    tbl[f"Change {p_ep} → {E}"] = gain.map("{:+.4f}".format)
    show_table(pd.DataFrame(tbl))
    st.caption(f"All stages select their epoch on the same validation subset and are evaluated on the same test set.")

    t = summ.set_index("metric")
    neg = [str(s) for s in test.index if test.loc[s, "diff"] < 0]
    st.markdown("**Interpretation**")
    st.markdown(f"""
<div class="card note">
<ul style="margin:0 0 0 1rem;padding:0">
<li><b>{"Longer fine-tuning helped both approaches." if (gain > 0).all() else "Longer fine-tuning did not help uniformly."}</b>
With {E} instead of {p_ep} classifier epochs, test macro-F1 changed by {gain.min():+.4f} to {gain.max():+.4f}
across the six matched runs{" (higher for every seed in both arms)" if (gain > 0).all() else ""}. Both arms
shared the schedule, so this change is not an effect of MAE pretraining.</li>
<li><b>{"CNN-MAE had a small mean lead" if t.loc['test_macro_f1','diff_mean'] > 0 else "Scratch had a small mean lead"}:</b> test macro-F1 {t.loc['test_macro_f1','mae_mean']:.4f} (CNN-MAE) vs
{t.loc['test_macro_f1','scratch_mean']:.4f} (scratch), a paired difference of {_ms(t.loc['test_macro_f1','diff_mean'], t.loc['test_macro_f1','diff_sd'], True)}.
It was not consistent: {int(t.loc['test_macro_f1','mae_ahead'])} of 3 test seeds favour CNN-MAE
{f"(seed {', '.join(neg)} favours scratch)" if neg else ""}.</li>
<li><b>The difference is about as large as its seed-to-seed spread</b>, so three seeds do not establish a statistically
significant or general advantage of MAE pretraining.</li>
<li><b>The selected scores are optimistic.</b> Each run keeps the best of {E} noisy validation scores, which flatters
validation macro-F1. The test values are not affected by that selection, but these runs were designed after the
primary test results were known.</li>
<li><b>These results do not replace the primary experiment.</b> They suggest that 10 fine-tuning epochs were not
enough for either arm at 1% labels.</li>
</ul></div>""", unsafe_allow_html=True)

    with st.expander("Training curves of the six exploratory runs"):
        seed = st.selectbox("Seed", list(exploratory_data.SEEDS), key=f"expl_seed_{study}",
                            help="Compare CNN-MAE and scratch trained with the same seed, label subset and data order.")
        e = epochs[epochs.seed == seed]
        sel = {a: int(runs[(runs.arm == a) & (runs.seed == seed)].best_epoch.iloc[0]) for a in exploratory_data.ARMS}
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Training loss**")
            show_chart(charts.training_curve(e, sel, "train_loss", "Cross-entropy"))
        with c2:
            st.markdown("**Validation loss**")
            show_chart(charts.training_curve(e, sel, "val_loss", "Cross-entropy"))
        with c3:
            st.markdown("**Validation macro-F1**")
            show_chart(charts.training_curve(e, sel, "val_macro_f1", "Macro-F1"))
        st.caption(f"Dashed lines: the epoch each run selected (CNN-MAE {sel['mae']}, scratch {sel['scratch']}), the "
                   f"maximum validation macro-F1 over {E} epochs. Validation metrics are the exact values returned by the "
                   f"evaluation function on the {int(runs.val_n.iloc[0]):,}-signal subset. Training loss is the value "
                   "printed by train_classifier.py (6 decimals), the only stored record of it.")

    with st.expander("How these runs relate to the other 1% models"):
        st.markdown(f"""
* **Primary 1% models** (10 epochs) select their epoch on the same {int(runs.val_n.iloc[0]):,}-signal validation subset,
  so their validation scores are comparable with these runs.
* **The supplementary 1% CNN-MAE checkpoint** shown in Live Prediction selected its epoch on the **full 383,386-signal
  validation split**. Its validation score is therefore not directly comparable with the validation scores here.
* The {E}-epoch checkpoints are **not** used in Live Prediction, which keeps the primary models.
* MAE pretraining lasted **5 epochs**. The "{E} epochs" are classifier fine-tuning epochs.
{"* One valid 50-epoch run (CNN-MAE, seed 43) is stored in a misnamed folder caused by a launch-script quoting error. It is identified from its manifest and used in place." if study == "50" else "* These runs were restarted from the verified initialisation; exact continuation of the 50-epoch runs was impossible (only selected-epoch states were saved)."}
""")
    srcs = runs[["arm", "seed", "run_dir", "result_file"]].copy()
    srcs["arm"] = srcs.arm.map(cfg.ARMS)
    with st.expander("Source artifacts for every number on this tab"):
        show_table(srcs.rename(columns={"arm": "Approach", "seed": "Seed", "run_dir": "Training run (manifest, per_epoch.csv, best.pt)",
                                        "result_file": "Test result (best.json)"}))
        st.caption("Read-only. Generated by experiments/run_long_finetune.py and experiments/evaluate_test.py; "
                   "loaded and cross-checked by demo/exploratory_data.py.")


def page_about():
    st.markdown('<div class="section-label">About the Experiment</div>', unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        st.markdown("""
#### Task
**Automatic modulation classification (AMC)**: identify which of 24 modulation schemes produced a received
radio signal, from raw samples, at signal-to-noise ratios between −20 and +30 dB.

#### Data
**RadioML 2018.01A**: 2,555,904 recordings. Each input is **[2, 1024]**: 1,024 time samples of the
in-phase (I) and quadrature (Q) components, used as raw values without normalisation.
A fixed 70/15/15 split stratified by class and SNR (seed 42) gives 1,789,132 training,
383,386 validation and 383,386 **held-out test** signals.

#### Models
* **Scratch CNN**: a patch-based 1-D CNN. The signal is cut into 16 patches of 64 samples;
  a shared 3-layer convolutional encoder embeds each patch; patch features are averaged and a
  linear layer gives 24 class scores (55,544 parameters).
* **CNN-MAE**: the same classifier, but its encoder is first pretrained as a **masked autoencoder** on
  all training signals *without labels* (half of the patches hidden; a small Transformer decoder
  reconstructs them). The pretrained encoder is then fine-tuned with labels exactly like the scratch model.

#### Two training stages (not to be confused)
1. **MAE pretraining** (CNN-MAE arm only): **5 epochs** on all 1,789,132 training signals, without labels.
   A single run; the checkpoint with the lowest validation reconstruction error (epoch 5) is used.
2. **Classifier fine-tuning** (both arms): supervised training on the labelled subset. **10 epochs** in the
   primary grid, **50 epochs** in the exploratory runs. The longer schedule refers only to this stage.

#### Research question
Does initialising the encoder from MAE pretraining improve held-out classification compared with training the
same CNN from scratch, when only 1–100% of the training labels are available?
""")
    with b:
        st.markdown("""
#### Primary experiment (authoritative)
* Labelled data: **1%, 5%, 10%, 25%, 100%** of the training split (nested, class-balanced subsets).
* Seeds **42, 43, 44** at 1/5/10%; **seed 42 only** at 25% and 100% (22 runs).
* Identical fine-tuning for both: AdamW (lr 1e-3, weight decay 1e-4), cosine schedule, 10 epochs, batch 128.
* The best epoch is chosen on a fixed 12,480-signal validation subset; each chosen model is evaluated
  **once** on the 383,386-signal test set.

#### Exploratory follow-up (clearly separated)
* 1% labels, seeds 42–44, both arms, **50 fine-tuning epochs** (cosine over 50); otherwise the primary protocol.
* Designed **after** the primary test results were known, so its test evaluation is not confirmatory. It is
  shown in its own tab and never pooled with the primary grid.
* Both arms improved over their 10-epoch versions. CNN-MAE's mean advantage stayed small, and its sign was not
  the same for every seed.

#### Limitations
* Few seeds: seed-to-seed variation is about as large as most MAE-vs-scratch differences.
* 25% and 100% rest on a single seed.
* One pretraining run and one encoder design. The encoder only sees patches in isolation, which limits
  absolute accuracy (≈0.38 macro-F1 with all labels in the primary grid).
* Picking the best of many validation epochs gives optimistic validation scores.
* A follow-up diagnostic (5% labels, seed 42) found validation metrics sensitive to BatchNorm running
  statistics, which adds noise to checkpoint selection.

#### Conclusion (cautious)
Across the primary grid, MAE pretraining did not give a consistent improvement. The exploratory 50-epoch runs show
that longer fine-tuning helps both approaches at 1% labels, with at most a small and uncertain edge for CNN-MAE.
Neither result establishes a significant or general benefit of MAE pretraining.

#### How to read this demo
Live predictions show what individual models do on individual signals. Only the aggregate,
multi-seed test results speak to the research question.
""")
    st.caption(f"Project root: {cfg.PROJECT_ROOT} · experiment outputs: {cfg.RUNS_DIR}")


{"Signal Explorer": page_explorer, "Live Prediction": page_prediction,
 "Experimental Results": page_results, "About the Experiment": page_about}[page]()
