"""Altair charts (Altair ships with Streamlit, so no extra dependency)."""

import altair as alt
import numpy as np
import pandas as pd

import demo_config as cfg

ARM_DOMAIN = list(cfg.ARMS.values())
ARM_RANGE = [cfg.COLOR["scratch"], cfg.COLOR["mae"]]
BUDGET_ORDER = [f"{b}%" for b in cfg.PLAN]


def _base(chart, height):
    return (chart.properties(height=height)
            .configure_view(strokeWidth=0)
            .configure_axis(gridColor=cfg.COLOR["grid"], labelColor=cfg.COLOR["muted"],
                            titleColor=cfg.COLOR["ink"], labelFontSize=12, titleFontSize=12,
                            titleFontWeight="normal", domainColor="#c9c8c3")
            .configure_legend(orient="top", title=None, labelFontSize=12, symbolSize=90))


def iq_waveform(signal: np.ndarray, height=260):
    n = signal.shape[1]
    df = pd.DataFrame({"sample": np.tile(np.arange(n), 2),
                       "value": np.concatenate([signal[0], signal[1]]),
                       "component": ["I (in-phase)"] * n + ["Q (quadrature)"] * n})
    ch = alt.Chart(df).mark_line(strokeWidth=1.4).encode(
        x=alt.X("sample:Q", title="Sample index (time)", scale=alt.Scale(domain=[0, n - 1], nice=False)),
        y=alt.Y("value:Q", title="Amplitude"),
        color=alt.Color("component:N", scale=alt.Scale(domain=["I (in-phase)", "Q (quadrature)"],
                                                         range=[cfg.COLOR["I"], cfg.COLOR["Q"]])),
        tooltip=[alt.Tooltip("component:N"), alt.Tooltip("sample:Q"), alt.Tooltip("value:Q", format=".4f")],
    ).interactive(bind_y=False)
    return _base(ch, height)


def constellation(signal: np.ndarray, height=300):
    df = pd.DataFrame({"I": signal[0], "Q": signal[1], "sample": np.arange(signal.shape[1])})
    lim = float(np.abs(signal).max()) * 1.08 or 1.0
    sc = alt.Scale(domain=[-lim, lim], nice=False)
    ch = alt.Chart(df).mark_circle(size=14, opacity=0.55, color=cfg.COLOR["I"]).encode(
        x=alt.X("I:Q", title="I", scale=sc), y=alt.Y("Q:Q", title="Q", scale=sc),
        tooltip=["sample:Q", alt.Tooltip("I:Q", format=".4f"), alt.Tooltip("Q:Q", format=".4f")],
    ).properties(width=height)
    return _base(ch, height)


def top_probs(probs: np.ndarray, names, true_label: int, color: str, k=5, height=195):
    idx = np.argsort(probs)[::-1][:k]
    df = pd.DataFrame({"class": [names[i] for i in idx], "p": probs[idx],
                       "is_true": ["true class" if i == true_label else "other" for i in idx]})
    ch = alt.Chart(df).mark_bar(cornerRadiusEnd=3, height=18).encode(
        y=alt.Y("class:N", sort=None, title=None, axis=alt.Axis(labelOverlap=False, labelLimit=120)),
        x=alt.X("p:Q", title="Model probability (softmax)", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("is_true:N", scale=alt.Scale(domain=["true class", "other"],
                                                     range=[color, "#c9c8c3"]), legend=None),
        tooltip=["class:N", alt.Tooltip("p:Q", format=".3f")],
    )
    text = ch.mark_text(align="left", dx=4, color=cfg.COLOR["ink"], fontSize=11).encode(
        text=alt.Text("p:Q", format=".3f"), color=alt.value(cfg.COLOR["ink"]))
    return _base(ch + text, height)


def budget_lines(runs: pd.DataFrame, metric: str, title: str, height=320):
    """Mean per budget (line) + every seed (hollow points). Y axis starts at 0: honest scale."""
    df = runs.copy()
    df["Labels"] = df["budget"].map(lambda b: f"{b}%")
    df["Approach"] = df["arm"].map(cfg.ARMS)
    mean = df.groupby(["Labels", "Approach"], as_index=False)[metric].mean()
    n = df.groupby(["Labels", "Approach"], as_index=False)["seed"].count().rename(columns={"seed": "seeds"})
    mean = mean.merge(n)
    x = alt.X("Labels:N", sort=BUDGET_ORDER, title="Labelled training data (fraction of training split)",
              axis=alt.Axis(labelAngle=0))
    color = alt.Color("Approach:N", scale=alt.Scale(domain=ARM_DOMAIN, range=ARM_RANGE))
    y = alt.Y(f"{metric}:Q", title=title, scale=alt.Scale(domain=[0, max(0.45, float(df[metric].max()) + 0.05)]))
    lines = alt.Chart(mean).mark_line(strokeWidth=2.2).encode(x=x, y=y, color=color)
    pts = alt.Chart(mean).mark_point(filled=True, size=70).encode(
        x=x, y=y, color=color, shape=alt.Shape("Approach:N", scale=alt.Scale(domain=ARM_DOMAIN, range=["circle", "square"])),
        tooltip=["Approach:N", "Labels:N", alt.Tooltip(f"{metric}:Q", format=".4f", title="mean"), "seeds:Q"])
    seeds = alt.Chart(df).mark_point(size=28, filled=False, opacity=0.8).encode(
        x=x, y=y, color=color, tooltip=["Approach:N", "Labels:N", "seed:Q", alt.Tooltip(f"{metric}:Q", format=".4f")])
    return _base(lines + pts + seeds, height)


def delta_chart(runs: pd.DataFrame, metric: str, height=260):
    """MAE minus scratch per seed (points) and mean (bar), with a zero reference line."""
    s = runs[runs.arm == "scratch"].set_index(["budget", "seed"])[metric]
    m = runs[runs.arm == "mae"].set_index(["budget", "seed"])[metric]
    d = (m - s).dropna().reset_index().rename(columns={metric: "delta"})
    d["Labels"] = d["budget"].map(lambda b: f"{b}%")
    mean = d.groupby("Labels", as_index=False).agg(delta=("delta", "mean"), seeds=("seed", "count"))
    mean["n"] = mean["seeds"].map(lambda k: f"{k} seed" + ("s" if k > 1 else ""))
    x = alt.X("Labels:N", sort=BUDGET_ORDER, title=None, axis=alt.Axis(labelAngle=0))
    lim = max(0.03, float(d["delta"].abs().max()) * 1.2)
    y = alt.Y("delta:Q", title="CNN-MAE − scratch", scale=alt.Scale(domain=[-lim, lim]))
    bars = alt.Chart(mean).mark_bar(size=34, cornerRadius=3, color="#b9b8b2", opacity=0.75).encode(
        x=x, y=y, tooltip=["Labels:N", alt.Tooltip("delta:Q", format="+.4f", title="mean difference"), "n:N"])
    pts = alt.Chart(d).mark_point(filled=True, size=55, color=cfg.COLOR["ink"]).encode(
        x=x, y=y, tooltip=["Labels:N", "seed:Q", alt.Tooltip("delta:Q", format="+.4f")])
    zero = alt.Chart(pd.DataFrame({"z": [0]})).mark_rule(color=cfg.COLOR["muted"]).encode(y="z:Q")
    label = alt.Chart(mean).mark_text(dy=-10, fontSize=11, color=cfg.COLOR["muted"]).encode(
        x=x, y=alt.value(12), text="n:N")
    return _base(bars + zero + pts + label, height)


def snr_chart(df: pd.DataFrame, height=300):
    color = alt.Color("arm:N", scale=alt.Scale(domain=ARM_DOMAIN, range=ARM_RANGE), title=None)
    ch = alt.Chart(df).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=22)).encode(
        x=alt.X("snr:Q", title="SNR (dB)"), y=alt.Y("accuracy:Q", title="Test accuracy", scale=alt.Scale(domain=[0, 1])),
        color=color, tooltip=["arm:N", "snr:Q", alt.Tooltip("accuracy:Q", format=".4f"), "seeds:Q"])
    chance = alt.Chart(pd.DataFrame({"y": [1 / 24]})).mark_rule(strokeDash=[4, 4], color=cfg.COLOR["muted"]).encode(y="y:Q")
    return _base(ch + chance, height)


# ---------------------------------------------------------------- exploratory 50-epoch runs
EXPL_SPLITS = ["Validation subset", "Test set"]


def expl_paired_diff(val: pd.DataFrame, test: pd.DataFrame, height=260):
    """Per-seed MAE − scratch macro-F1 on validation and on test, with a zero line.
    val/test: exploratory_data.paired(...) frames (index = seed, column 'diff')."""
    df = pd.concat([
        pd.DataFrame({"seed": val.index.astype(str), "diff": val["diff"].values, "split": EXPL_SPLITS[0]}),
        pd.DataFrame({"seed": test.index.astype(str), "diff": test["diff"].values, "split": EXPL_SPLITS[1]})])
    lim = max(0.015, float(df["diff"].abs().max()) * 1.35)
    x = alt.X("seed:N", title="Seed", axis=alt.Axis(labelAngle=0))
    y = alt.Y("diff:Q", title="CNN-MAE − scratch (macro-F1)", scale=alt.Scale(domain=[-lim, lim]))
    color = alt.Color("split:N", scale=alt.Scale(domain=EXPL_SPLITS, range=["#8d8c86", cfg.COLOR["ink"]]))
    pts = alt.Chart(df).mark_point(filled=True, size=95).encode(
        x=x, y=y, color=color, xOffset=alt.XOffset("split:N", sort=EXPL_SPLITS),
        shape=alt.Shape("split:N", scale=alt.Scale(domain=EXPL_SPLITS, range=["circle", "diamond"])),
        tooltip=["split:N", "seed:N", alt.Tooltip("diff:Q", format="+.4f", title="MAE − scratch")])
    zero = alt.Chart(pd.DataFrame({"z": [0]})).mark_rule(color=cfg.COLOR["muted"]).encode(y="z:Q")
    return _base(zero + pts, height)


def ten_vs_fifty(df: pd.DataFrame, height=300):
    """Test macro-F1 of the matched 1% runs: primary 10-epoch vs exploratory 50-epoch (same arm, seed)."""
    cats = ["10 ep · primary", "50 ep · exploratory"]
    long = pd.concat([
        pd.DataFrame({"schedule": cats[0], "f1": df.test_macro_f1_10, "arm": df.arm, "seed": df.seed}),
        pd.DataFrame({"schedule": cats[1], "f1": df.test_macro_f1_50, "arm": df.arm, "seed": df.seed})])
    long["Approach"] = long["arm"].map(cfg.ARMS)
    long["run"] = long["Approach"] + " · seed " + long["seed"].astype(str)
    x = alt.X("schedule:N", sort=cats, title="Fine-tuning schedule (1% labels)",
              axis=alt.Axis(labelAngle=0, labelLimit=220))
    y = alt.Y("f1:Q", title="Test macro-F1", scale=alt.Scale(domain=[0, 0.4]))
    color = alt.Color("Approach:N", scale=alt.Scale(domain=ARM_DOMAIN, range=ARM_RANGE))
    lines = alt.Chart(long).mark_line(strokeWidth=1.6, opacity=0.75).encode(x=x, y=y, color=color, detail="run:N")
    pts = alt.Chart(long).mark_point(filled=True, size=55).encode(
        x=x, y=y, color=color,
        shape=alt.Shape("Approach:N", scale=alt.Scale(domain=ARM_DOMAIN, range=["circle", "square"])),
        tooltip=["run:N", "schedule:N", alt.Tooltip("f1:Q", format=".4f", title="test macro-F1")])
    return _base(lines + pts, height)


def stages_chart(long: pd.DataFrame, height=300):
    """Test macro-F1 for the same arm and seed across stages (10 = primary; others exploratory)."""
    df = long.copy()
    df["schedule"] = df.apply(lambda r: f"{r.epochs} ep · {'primary' if r.stage == 'primary' else 'exploratory'}", axis=1)
    order = [f"{e} ep · {'primary' if e == 10 else 'exploratory'}" for e in sorted(df.epochs.unique())]
    df["Approach"] = df["arm"].map(cfg.ARMS)
    df["run"] = df["Approach"] + " · seed " + df["seed"].astype(str)
    x = alt.X("schedule:N", sort=order, title="Fine-tuning schedule (1% labels)", axis=alt.Axis(labelAngle=0, labelLimit=220))
    y = alt.Y("test_macro_f1:Q", title="Test macro-F1", scale=alt.Scale(domain=[0, 0.4]))
    color = alt.Color("Approach:N", scale=alt.Scale(domain=ARM_DOMAIN, range=ARM_RANGE))
    lines = alt.Chart(df).mark_line(strokeWidth=1.6, opacity=0.75).encode(x=x, y=y, color=color, detail="run:N")
    pts = alt.Chart(df).mark_point(filled=True, size=55).encode(
        x=x, y=y, color=color, shape=alt.Shape("Approach:N", scale=alt.Scale(domain=ARM_DOMAIN, range=["circle", "square"])),
        tooltip=["run:N", "schedule:N", alt.Tooltip("test_macro_f1:Q", format=".4f"), "selected_epoch:Q"])
    return _base(lines + pts, height)


def training_curve(ep: pd.DataFrame, selected: dict, metric: str, title: str, height=230):
    """One metric over epochs for MAE and scratch of one seed; dashed rule at each selected epoch.
    selected: {'mae': epoch, 'scratch': epoch}."""
    df = ep.copy()
    df["Approach"] = df["arm"].map(cfg.ARMS)
    color = alt.Color("Approach:N", scale=alt.Scale(domain=ARM_DOMAIN, range=ARM_RANGE),
                      legend=alt.Legend(orient="bottom", direction="vertical"))  # fits narrow columns
    x = alt.X("epoch:Q", title="Fine-tuning epoch", scale=alt.Scale(domain=[1, 50], nice=False))
    line = alt.Chart(df).mark_line(strokeWidth=1.8).encode(
        x=x, y=alt.Y(f"{metric}:Q", title=title, scale=alt.Scale(zero=False)), color=color,
        tooltip=["Approach:N", "epoch:Q", alt.Tooltip(f"{metric}:Q", format=".4f")])
    sel = pd.DataFrame({"Approach": [cfg.ARMS[a] for a in selected], "epoch": list(selected.values())})
    rules = alt.Chart(sel).mark_rule(strokeDash=[5, 4], strokeWidth=1.4).encode(
        x="epoch:Q", color=color, tooltip=["Approach:N", alt.Tooltip("epoch:Q", title="selected epoch")])
    return _base(line + rules, height)
