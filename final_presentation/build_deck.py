"""Build the final Group 41 presentation (editable .pptx) from verified project artifacts.

  C:\\venvs\\pptx\\Scripts\\python.exe final_presentation\\build_deck.py

Every result number shown on a slide is read from final_artifacts\\tables\\*.csv (themselves verified
against the raw result files by final_artifacts\\make_final_figures.py) and asserted against the values
recorded in the project before the deck is written. Diagrams, tables and the primary chart are native
PowerPoint objects. Figures are the verified PNGs from final_artifacts\\figures. Nothing outside
final_presentation\\ is written.
"""
import csv, json
from pathlib import Path
from statistics import mean
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_MARKER_STYLE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt
from lxml import etree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TAB, FIG = ROOT / "final_artifacts" / "tables", ROOT / "final_artifacts" / "figures"
OUT = HERE / "Group41_CNN-MAE_Final_Presentation.pptx"

INK, MUTED, LIGHT, LINE = "1F2328", "5B6470", "F2F4F6", "D5DAE0"
ACCENT = "0E7C7B"                       # restrained teal: headings, primary tag
SCR, MAE = "2A78D6", "EB6834"           # same colours as the final figures
EXP, EXP_BG = "9A6200", "FFF3DC"        # exploratory tag
OK, NO, PART = "1E7B4A", "A1372E", "8A6A00"
HEAD, BODY = "Cambria", "Calibri"
CHECKS = []


def check(ok, msg):
    CHECKS.append((bool(ok), msg))
    if not ok:
        raise SystemExit(f"VERIFY FAIL: {msg}")


def rgb(h): return RGBColor.from_string(h)


def rows(name):
    with open(TAB / name, newline="") as fh:
        return list(csv.DictReader(fh))


# ------------------------------------------------------------------ verified data
prim = {int(r["budget"]): r for r in rows("primary_summary.csv")}
REF = {1: (0.2577, 0.2566, -0.0011), 5: (0.3199, 0.3320, +0.0121), 10: (0.3583, 0.3571, -0.0012),
       25: (0.3630, 0.3629, -0.0002), 100: (0.3821, 0.3838, +0.0017)}
for b, (s, m, d) in REF.items():
    r = prim[b]
    check(round(float(r["scratch_test_macro_f1"]), 4) == s and round(float(r["mae_test_macro_f1"]), 4) == m
          and round(float(r["delta_test_macro_f1"]), 4) == d, f"primary {b}% macro-F1 {s}/{m}/{d:+.4f}")
P5 = float(prim[5]["paired_p"]); check(round(P5, 2) == 0.29, "5% paired t-test p = 0.29")
check(prim[25]["n_seeds"] == "1" and prim[100]["n_seeds"] == "1", "25% and 100% have one seed")


def expl(ep):
    s = {r["metric"]: r for r in rows(f"exploratory_{ep}ep_summary.csv")}["test_macro_f1"]
    per = rows(f"exploratory_{ep}ep_per_seed.csv")
    d = {}
    for seed in (42, 43, 44):
        v = {r["arm"]: float(r["test_macro_f1"]) for r in per if int(r["seed"]) == seed}
        d[seed] = v["mae"] - v["scratch"]
    return {k: float(s[k]) for k in ("mae_mean", "mae_sd", "scratch_mean", "scratch_sd", "diff_mean", "diff_sd")} | \
        {"ahead": int(s["mae_ahead"]), "per_seed": d}


E50, E100 = expl(50), expl(100)
for E, ref, tag in ((E50, (0.3277, 0.0079, 0.3227, 0.0024, 0.0051, 0.0062), "50"),
                    (E100, (0.3281, 0.0090, 0.3299, 0.0100, -0.0018, 0.0170), "100")):
    got = tuple(round(E[k], 4) for k in ("mae_mean", "mae_sd", "scratch_mean", "scratch_sd", "diff_mean", "diff_sd"))
    check(got == ref and E["ahead"] == 2, f"{tag}-epoch test summary {got}, MAE ahead in 2/3 seeds")

snr = rows("primary_per_snr_accuracy.csv")


def band(b, arm, lo, hi):
    return mean(float(r["accuracy_mean_over_seeds"]) for r in snr
                if int(r["budget"]) == b and r["arm"] == arm and lo <= int(r["snr_db"]) <= hi)


BAND = {b: {"low": (band(b, "scratch", -20, -10), band(b, "mae", -20, -10)),
            "mid": band(b, "mae", -8, 0) - band(b, "scratch", -8, 0),
            "high": band(b, "mae", 10, 30) - band(b, "scratch", 10, 30)} for b in REF}
check(abs(BAND[5]["mid"] - 0.0277) < 5e-5 and abs(BAND[1]["mid"] - 0.0258) < 5e-5, "low-SNR band differences")
sig = json.loads((HERE / "assets" / "signals.json").read_text())
stg = rows("stage_comparison_10_50_100.csv")
_st = {(r["arm"], int(r["seed"]), int(r["epochs"])): float(r["test_macro_f1"]) for r in stg}
_g = [_st[(a, sd, e)] - _st[(a, sd, 10)] for a in ("mae", "scratch") for sd in (42, 43, 44) for e in (50, 100)]
GAIN = (min(_g), max(_g)); check(len(stg) == 18 and GAIN[0] > 0, f"every longer run beats its 10-epoch run ({GAIN[0]:.4f}..{GAIN[1]:.4f})")

# ------------------------------------------------------------------ helpers
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
W = 13.333


def tb(slide, x, y, w, h, paras, size=16, color=INK, font=BODY, bold=False, align=PP_ALIGN.LEFT,
       anchor=MSO_ANCHOR.TOP, space_after=6, name=None):
    """paras: list of str or list of (text, {opts}) runs; a para may be ('bullet', runs)."""
    s = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name: s.name = name
    tf = s.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.02); tf.margin_top = tf.margin_bottom = Inches(0.02)
    for i, p in enumerate(paras):
        bullet = isinstance(p, tuple) and p[0] == "bullet"
        runs = p[1] if bullet else p
        if isinstance(runs, str): runs = [(runs, {})]
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align; para.space_after = Pt(space_after)
        if bullet:
            pPr = para._p.get_or_add_pPr(); pPr.set("marL", str(Inches(0.22))); pPr.set("indent", str(-Inches(0.2)))
            bu = etree.SubElement(pPr, qn("a:buChar")); bu.set("char", "•")
        for text, o in runs:
            r = para.add_run(); r.text = text
            f = r.font; f.size = Pt(o.get("size", size)); f.name = o.get("font", font)
            f.bold = o.get("bold", bold); f.italic = o.get("italic", False); f.color.rgb = rgb(o.get("color", color))
    return s


def box(slide, x, y, w, h, fill=LIGHT, line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08, name=None):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if name: s.name = name
    s.fill.solid(); s.fill.fore_color.rgb = rgb(fill)
    if line: s.line.color.rgb = rgb(line); s.line.width = Pt(1)
    else: s.line.fill.background()
    s.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE: s.adjustments[0] = radius
    return s


def label_box(slide, x, y, w, h, text, fill=LIGHT, color=INK, size=14, bold=False, line=None, sub=None, name=None):
    s = box(slide, x, y, w, h, fill=fill, line=line, name=name)
    tf = s.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for m in ("margin_left", "margin_right"): setattr(tf, m, Inches(0.08))
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = text; r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = rgb(color); r.font.name = BODY
    if sub:
        p2 = tf.add_paragraph(); p2.alignment = PP_ALIGN.CENTER
        r2 = p2.add_run(); r2.text = sub; r2.font.size = Pt(size - 3); r2.font.color.rgb = rgb(MUTED if fill == LIGHT or fill == "FFFFFF" else color); r2.font.name = BODY
    return s


def tag(slide, x, y, text, fg, bg, w=None):
    w = w or 0.13 * len(text) + 0.35
    s = box(slide, x, y, w, 0.32, fill=bg, radius=0.5)
    tf = s.text_frame; tf.margin_left = tf.margin_right = Inches(0.05); tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = text; r.font.size = Pt(11); r.font.bold = True; r.font.color.rgb = rgb(fg); r.font.name = BODY
    return s


def arrow(slide, a, b, ax="r", bx="l", color=MUTED, width=1.6, dash=False):
    """Straight connector glued to shape a (side ax) and shape b (side bx) with an arrowhead."""
    idx = {"t": 0, "l": 1, "b": 2, "r": 3}
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, 0, 0, Inches(1), Inches(1))
    c.begin_connect(a, idx[ax]); c.end_connect(b, idx[bx])
    c.line.color.rgb = rgb(color); c.line.width = Pt(width)
    ln = c.line._get_or_add_ln()
    if dash:
        d = etree.SubElement(ln, qn("a:prstDash")); d.set("val", "dash")
    t = etree.SubElement(ln, qn("a:tailEnd")); t.set("type", "triangle"); t.set("w", "med"); t.set("len", "med")
    return c


def waveform(slide, x, y, w, h, values, color, width=1.25, vmax=None, name=None):
    """Native freeform polyline of real IQ samples (editable)."""
    vmax = vmax or max(abs(v) for v in values) or 1.0
    n = len(values)
    pts = [(Inches(x + w * i / (n - 1)), Inches(y + h / 2 - (h / 2) * 0.95 * v / vmax)) for i, v in enumerate(values)]
    fb = slide.shapes.build_freeform(pts[0][0], pts[0][1], scale=1.0)
    fb.add_line_segments(pts[1:], close=False)
    s = fb.convert_to_shape()
    s.fill.background(); s.line.color.rgb = rgb(color); s.line.width = Pt(width)
    if name: s.name = name
    return s


def title(slide, text, kicker=None):
    if kicker:
        tb(slide, 0.6, 0.32, 9, 0.3, [[(kicker.upper(), {"size": 11, "bold": True, "color": ACCENT})]])
    tb(slide, 0.6, 0.58, 12.1, 0.8, [[(text, {"size": 28, "bold": True, "font": HEAD})]], name="Title")


def footer(slide, n):
    tb(slide, 0.6, 7.05, 8, 0.3, [[("Group 41 · 22AIE304 Deep Learning · CNN-MAE for low-label AMC", {"size": 10, "color": MUTED})]])
    tb(slide, 11.9, 7.05, 0.85, 0.3, [[(str(n), {"size": 10, "color": MUTED})]], align=PP_ALIGN.RIGHT)


def takeaway(slide, x, y, w, h, runs, fill="E7F3F2", color=INK, size=16):
    s = box(slide, x, y, w, h, fill=fill, name="Takeaway")
    tf = s.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.2)
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.LEFT
    for text, o in runs:
        r = p.add_run(); r.text = text; r.font.size = Pt(o.get("size", size)); r.font.bold = o.get("bold", False)
        r.font.color.rgb = rgb(o.get("color", color)); r.font.name = BODY
    return s


def table(slide, x, y, w, col_w, data, header_fill=INK, size=13, row_h=0.38, bold_first_col=False, fills=None, left_cols=(0,)):
    nr, nc = len(data), len(data[0])
    t = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr)).table
    for j, cw in enumerate(col_w): t.columns[j].width = Inches(cw)
    for i in range(nr):
        t.rows[i].height = Inches(row_h)
        for j in range(nc):
            c = t.cell(i, j); val = data[i][j]
            txt, o = (val if isinstance(val, tuple) else (val, {}))
            c.text = ""; p = c.text_frame.paragraphs[0]
            r = p.add_run(); r.text = str(txt); r.font.name = BODY; r.font.size = Pt(o.get("size", size))
            r.font.bold = i == 0 or o.get("bold", False) or (bold_first_col and j == 0)
            r.font.color.rgb = rgb("FFFFFF" if i == 0 else o.get("color", INK))
            p.alignment = o.get("align", PP_ALIGN.LEFT if j in left_cols else PP_ALIGN.CENTER)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.margin_left = c.margin_right = Inches(0.08); c.margin_top = c.margin_bottom = Inches(0.03)
            c.fill.solid(); c.fill.fore_color.rgb = rgb(header_fill if i == 0 else (fills[i] if fills and fills[i] else ("FFFFFF" if i % 2 else "F7F8FA")))
    return t


def notes(slide, text): slide.notes_slide.notes_text_frame.text = text


def f4(v, signed=False): return f"{v:+.4f}" if signed else f"{v:.4f}"


slides = []


def new():
    s = prs.slides.add_slide(BLANK); slides.append(s)
    bg = s.background.fill; bg.solid(); bg.fore_color.rgb = rgb("FFFFFF")
    return s


# ------------------------------------------------------------------ 1 title
s = new()
box(s, 7.35, 0, 5.983, 7.5, fill="F2F6F6", shape=MSO_SHAPE.RECTANGLE, name="Signal panel")
t = sig["title_qpsk_30db"]
vmax = max(max(abs(v) for v in t["I"]), max(abs(v) for v in t["Q"]))
waveform(s, 7.75, 1.45, 5.2, 1.9, t["I"][:160], SCR, 1.5, vmax, "I trace")
waveform(s, 7.75, 3.55, 5.2, 1.9, t["Q"][:160], MAE, 1.5, vmax, "Q trace")
tb(s, 7.75, 1.05, 3, 0.3, [[("I  (in-phase)", {"size": 12, "bold": True, "color": SCR})]])
tb(s, 7.75, 3.15, 3, 0.3, [[("Q  (quadrature)", {"size": 12, "bold": True, "color": MAE})]])
tb(s, 7.75, 5.75, 5.2, 0.6, [[(f"A real held-out test signal: {t['class']} at {t['snr_db']:+d} dB SNR, first 160 of 1,024 samples "
                               f"(RadioML 2018.01A, test index {t['test_index']:,})", {"size": 11, "color": MUTED})]])
tb(s, 0.7, 0.75, 6.3, 0.35, [[("22AIE304 DEEP LEARNING · FINAL REVIEW", {"size": 12, "bold": True, "color": ACCENT})]])
tb(s, 0.7, 1.25, 6.4, 2.6, [[("Self-Supervised Masked Autoencoder Pretraining for Low-Label Automatic Modulation "
                               "Classification Using IQ Signals", {"size": 32, "bold": True, "font": HEAD})]], space_after=0)
tb(s, 0.7, 4.15, 6.3, 0.35, [[("Group 41", {"size": 18, "bold": True})]])
team = [("P V Nekha", "AM.SC.U4AIE24037"), ("Vinayak S", "AM.SC.U4AIE24056"), ("Monish T", "AM.SC.U4AIE24064")]
for i, (n, roll) in enumerate(team):
    tb(s, 0.7, 4.6 + i * 0.42, 6.3, 0.4, [[(n, {"size": 16, "bold": True}), (f"    {roll}", {"size": 15, "color": MUTED})]])
tb(s, 0.7, 6.05, 6.3, 0.35, [[("B.Tech CSE (Artificial Intelligence)", {"size": 15, "color": MUTED})]])
notes(s, "Introduce the team and the question: does masked-autoencoder pretraining on raw radio signals help when labels are scarce?")

# ------------------------------------------------------------------ 2 problem and motivation
s = new(); title(s, "Modulation is hard to recognise with few labels and low SNR", "Problem statement and motivation")
tb(s, 0.6, 1.65, 6.2, 3.6, [
    ("bullet", [("Automatic modulation classification (AMC): ", {"bold": True}), ("identify how a received radio signal was modulated, e.g. QPSK or 16QAM, directly from its I/Q samples", {})]),
    ("bullet", [("Labels are expensive: ", {"bold": True}), ("each training signal needs a known transmitter setting, RF equipment and expert effort", {})]),
    ("bullet", [("Low signal-to-noise ratio (SNR) ", {"bold": True}), ("buries the modulation pattern in noise, so accuracy drops sharply", {})]),
    ("bullet", [("Unlabelled signals are plentiful. ", {"bold": True}), ("Self-supervised pretraining can learn signal structure from them first, then fine-tune with few labels", {})]),
], size=17, space_after=12)
q18, qm10 = sig["qpsk_18db"], sig["qpsk_m10db"]
box(s, 7.3, 1.65, 5.45, 3.6, fill=LIGHT, name="SNR comparison")
for k, (sg, yy) in enumerate(((q18, 2.05), (qm10, 3.55))):
    vm = max(abs(v) for v in sg["I"][:120])
    waveform(s, 7.6, yy + 0.35, 4.85, 1.0, sg["I"][:120], SCR if k == 0 else MUTED, 1.2, vm)
    tb(s, 7.6, yy, 4.9, 0.3, [[(f"Same class (QPSK), I component, SNR {sg['snr_db']:+d} dB", {"size": 12, "bold": True})]])
tb(s, 7.6, 4.8, 4.9, 0.35, [[("Real test signals: the pattern visible at +18 dB is lost in noise at −10 dB", {"size": 11, "color": MUTED})]])
takeaway(s, 0.6, 5.65, 12.15, 0.95, [("Research question: ", {"bold": True, "color": ACCENT}),
         ("does self-supervised CNN masked-autoencoder (MAE) pretraining improve AMC when labelled data is limited, compared with training the same CNN from scratch?", {})])
footer(s, 2)
notes(s, "AMC is used in spectrum monitoring and cognitive radio. Labelling is the bottleneck; low SNR is the hard regime.")

# ------------------------------------------------------------------ 3 objectives
s = new(); title(s, "Objectives: what we set out to do, and what was completed", "Objectives")
obj = [
    ("Build a CNN classifier trained from scratch (the baseline)", "Completed", OK),
    ("Pretrain a CNN masked autoencoder on unlabelled raw IQ (1.79 M training signals)", "Completed", OK),
    ("Compare both at 1%, 5%, 10%, 25% and 100% labelled data", "Completed", OK),
    ("Analyse accuracy, macro-F1 and per-SNR behaviour", "Completed", OK),
    ("Investigate longer classifier fine-tuning (50 and 100 epochs, 1% labels)", "Completed · exploratory", EXP),
    ("Cross-dataset transfer to RadioML 2016.10a", "Completed · separate study", PART),
    ("Linear probe with a frozen encoder; test whether reconstruction error predicts accuracy", "Not done", NO),
]
for i, (txt, st, col) in enumerate(obj):
    y = 1.6 + i * 0.66
    box(s, 0.6, y, 12.15, 0.56, fill=LIGHT if i < 6 else "FBEDEB")
    tb(s, 0.8, y + 0.1, 9.0, 0.4, [[(f"{i + 1}.  ", {"bold": True, "color": ACCENT}), (txt, {})]], size=16)
    tag(s, 12.75 - 2.45, y + 0.12, st, "FFFFFF", col, w=2.3)
tb(s, 0.6, 6.32, 12.1, 0.6, [[("Objectives 6–7 came from the mid-review plan. The transfer study is reported separately "
                              "(11 classes, 128-sample signals): the transferred encoder was 0.009–0.012 macro-F1 below scratch at every budget.",
                              {"size": 12, "color": MUTED})]])
footer(s, 3)
notes(s, "Be explicit that the linear probe and the reconstruction-vs-accuracy link were planned but not run.")

# ------------------------------------------------------------------ 4 literature
s = new(); title(s, "Raw-IQ MAE exists; a controlled evaluation is missing", "Literature review and research gap")
lit = [["Study", "Approach", "Relevance / limitation for our question"],
       ["O'Shea, Roy & Clancy (2018)", "Supervised deep CNN on raw IQ; introduced RadioML 2018.01A", "Strong baseline, but needs fully labelled data"],
       ["Liu et al. (2021)", "Self-contrastive pretraining + fine-tuning", "Uses unlabelled signals; contrastive, not masked reconstruction"],
       ["He et al. (2022)", "Masked autoencoders (ViT) for images", "Established masked reconstruction as SSL; built for images"],
       ["Shi et al. (2024), GAF-MAE", "MAE on Gramian Angular Field images of signals", "Signals converted to images, not raw IQ"],
       ["Liu et al. (2025), RIS-MAE", "MAE directly on raw IQ for AMC", "Closest prior work: proposes the method; not our controlled multi-budget comparison"]]
table(s, 0.6, 1.6, 12.15, [3.0, 4.15, 5.0], lit, size=14, row_h=0.56, bold_first_col=True, left_cols=(0, 1, 2))
takeaway(s, 0.6, 5.3, 12.15, 1.3, [("Gap we address: ", {"bold": True, "color": ACCENT}),
         ("does raw-IQ MAE pretraining help ", {}), ("consistently", {"bold": True}),
         (" across label budgets, random seeds and SNR levels, when compared with a like-for-like scratch baseline that shares the same CNN backbone and training recipe?", {})])
footer(s, 4)
notes(s, "Kong 2023, Xiao 2024 (MCLHN) and Ma 2024 are in the reference list: contrastive or pseudo-label methods.")

# ------------------------------------------------------------------ 5 contributions
s = new(); title(s, "Our contribution: a careful experiment, not a new architecture", "Novelty and contributions")
contrib = [("Controlled comparison", "Scratch vs CNN-MAE with the same backbone, data and fine-tuning recipe"),
           ("Five label budgets", "1%, 5%, 10%, 25%, 100%; nested, class-balanced subsets"),
           ("Seed-level analysis", "3 paired seeds at 1–10%; paired differences and a paired t-test"),
           ("Per-SNR analysis", "Accuracy across all 26 SNR levels, with the low-SNR band examined"),
           ("Longer fine-tuning", "Exploratory 50- and 100-epoch follow-ups at 1%, three seeds each"),
           ("Reproducible evidence", "Manifests and hashes, verified figures, a working Streamlit demo")]
for i, (h, d) in enumerate(contrib):
    cx, cy = 0.6 + (i % 3) * 4.1, 1.65 + (i // 3) * 1.95
    box(s, cx, cy, 3.85, 1.7, fill=LIGHT)
    tb(s, cx + 0.25, cy + 0.2, 3.4, 0.4, [[(h, {"size": 18, "bold": True, "color": ACCENT})]])
    tb(s, cx + 0.25, cy + 0.68, 3.4, 0.95, [[(d, {"size": 15})]])
box(s, 0.6, 5.65, 12.15, 1.0, fill="FFFFFF", line=LINE)
tb(s, 0.85, 5.78, 11.7, 0.8, [[("Not claimed: ", {"bold": True, "color": NO}),
    ("a new MAE architecture, the first raw-IQ MAE (RIS-MAE, 2025, came first), or state-of-the-art accuracy. Our encoder is a small patch-based CNN (55,544 parameters).", {})]], size=15)
footer(s, 5)

# ------------------------------------------------------------------ 6 dataset
s = new(); title(s, "RadioML 2018.01A: 2.56 M labelled IQ signals, 24 classes", "Dataset")
stats = [("2,555,904", "IQ signals"), ("24", "modulation classes"), ("26", "SNR levels, −20 to +30 dB (2 dB steps)"), ("2 × 1,024", "I/Q channels × time samples")]
for i, (v, l) in enumerate(stats):
    x = 0.6 + i * 3.08
    tb(s, x, 1.6, 2.9, 0.75, [[(v, {"size": 34, "bold": True, "color": ACCENT, "font": HEAD})]])
    tb(s, x, 2.35, 2.9, 0.6, [[(l, {"size": 14, "color": MUTED})]])
tb(s, 0.6, 3.2, 6, 0.35, [[("Fixed split (70 / 15 / 15)", {"size": 16, "bold": True})]])
for (lbl, n, w, col) in (("Train 1,789,132", 0, 8.5, SCR), ("Val 383,386", 8.5, 1.82, "6F8FA8"), ("Test 383,386", 10.32, 1.83, INK)):
    label_box(s, 0.6 + n, 3.6, w - 0.04, 0.5, lbl, fill=col, color="FFFFFF", size=13, bold=True)
budgets = [["Label budget", "1%", "5%", "10%", "25%", "100%"],
           ["Labelled training signals", "17,883", "89,451", "178,914", "447,281", "1,789,132"],
           ["Seeds (primary)", "3", "3", "3", "1", "1"]]
table(s, 0.6, 4.45, 8.4, [2.6, 1.1, 1.1, 1.2, 1.2, 1.2], budgets, size=14, row_h=0.42)
box(s, 9.3, 4.45, 3.45, 1.26, fill=EXP_BG)
tb(s, 9.45, 4.53, 3.2, 1.15, [[("Secondary: RadioML 2016.10a", {"size": 13, "bold": True, "color": EXP})],
                              [("220,000 signals · 11 classes · 2 × 128. Used only for the separate transfer study.", {"size": 12})]], space_after=2)
tb(s, 0.6, 6.0, 12.1, 0.9, [[("Source: DeepSig RadioML 2018.01A (O'Shea, Roy & Clancy, 2018; deepsig.ai/datasets). "
                             "Validation for checkpoint selection uses a fixed 12,480-signal subset (20 per class × SNR) of the validation split.", {"size": 12, "color": MUTED})]])
footer(s, 6)

# ------------------------------------------------------------------ 7 preprocessing
s = new(); title(s, "Raw IQ input, fixed partitions, no test data in training", "Data preprocessing and leakage prevention")
steps = [("Load", "HDF5 → float32 [1,024 × 2], transposed to [2 × 1,024]"),
         ("Labels", "One-hot → class index; SNR kept for splitting and analysis"),
         ("Split", "70/15/15, stratified by class × SNR, seed 42"),
         ("Label budgets", "Nested, class-balanced subsets of the training split, drawn once"),
         ("Model input", "Raw values: no normalisation, augmentation or hand-crafted features")]
prev = None
for i, (h, d) in enumerate(steps):
    x = 0.6 + i * 2.5
    b_ = label_box(s, x, 1.65, 2.2, 1.85, h, fill=LIGHT, color=ACCENT, size=16, bold=True, sub=d)
    if prev is not None: arrow(s, prev, b_)
    prev = b_
tb(s, 0.6, 3.85, 6, 0.4, [[("Leakage controls", {"size": 18, "bold": True})]])
tb(s, 0.6, 4.3, 7.4, 2.4, [
    ("bullet", [("Train, validation and test indices are disjoint (checked), and the same for every budget and seed", {})]),
    ("bullet", [("MAE pretraining sees only training signals, without labels; its checkpoint is chosen by validation MSE", {})]),
    ("bullet", [("Classifier checkpoints are chosen by validation macro-F1 on the fixed 12,480-signal subset", {})]),
    ("bullet", [("The 383,386 test signals are evaluated once per selected model, after training", {})]),
], size=15, space_after=8)
box(s, 8.4, 3.9, 4.35, 2.7, fill="FFFFFF", line=LINE)
tb(s, 8.6, 4.0, 4.0, 2.5, [[("Checked vs not claimed", {"size": 15, "bold": True})],
    [("Loader rejects non-finite samples and invalid one-hot labels. ", {"size": 13}), ("Label subsets are balanced by class, not by SNR. ", {"size": 13}),
     ("No duplicate or missing-value filtering is performed.", {"size": 13})]], space_after=6)
footer(s, 7)

# ------------------------------------------------------------------ 8 architecture (native, editable)
s = new(); title(s, "One CNN backbone; the two paths differ only in encoder start", "Proposed system architecture")
tb(s, 0.6, 1.45, 9, 0.3, [[("STAGE 1 · SELF-SUPERVISED PRETRAINING (CNN-MAE only, no labels, 5 epochs)", {"size": 12, "bold": True, "color": MAE})]])
r1 = [("Raw IQ", "2 × 1,024, train split"), ("16 patches × 64", "mask 50%, keep 8"), ("CNN patch encoder", "visible patches only"),
      ("Transformer decoder", "2 layers + mask tokens"), ("Reconstruct masked", "MSE on masked only")]
row1 = []
for i, (h, d) in enumerate(r1):
    row1.append(label_box(s, 0.6 + i * 2.5, 1.85, 2.1, 0.95, h, fill="FDEEE6", color=INK, size=14, bold=True, sub=d, name=f"pre_{i}"))
for a, b in zip(row1, row1[1:]): arrow(s, a, b, color=MAE)
tb(s, 0.6, 3.1, 9, 0.3, [[("STAGE 2 · SUPERVISED FINE-TUNING (both paths, same labelled subset and recipe)", {"size": 12, "bold": True, "color": INK})]])
for yy, txt, col in ((3.55, "Path B\nCNN-MAE", MAE), (4.85, "Path A\nScratch", SCR)):
    tb(s, 0.6, yy + 0.12, 1.5, 0.7, [[(line_, {"size": 15, "bold": True, "color": col})] for line_ in txt.split("\n")], space_after=0)
rowB = [label_box(s, 1.85, 3.55, 2.1, 0.85, "Labelled subset", sub="1–100% of train", size=14, bold=True),
        label_box(s, 5.6, 3.55, 2.1, 0.85, "Pretrained encoder", fill=MAE, color="FFFFFF", size=14, bold=True, sub="weights from stage 1"),
        label_box(s, 8.1, 3.55, 2.1, 0.85, "Mean-pool → linear", sub="16 patches → 24 classes", size=14, bold=True)]
rowA = [label_box(s, 1.85, 4.85, 2.1, 0.85, "Labelled subset", sub="same indices", size=14, bold=True),
        label_box(s, 5.6, 4.85, 2.1, 0.85, "Random-init encoder", fill=SCR, color="FFFFFF", size=14, bold=True, sub="same architecture"),
        label_box(s, 8.1, 4.85, 2.1, 0.85, "Mean-pool → linear", sub="16 patches → 24 classes", size=14, bold=True)]
for row in (rowB, rowA):
    for a, b in zip(row, row[1:]): arrow(s, a, b)
ev = label_box(s, 10.7, 3.55, 2.05, 2.15, "Evaluate & compare", fill=INK, color="FFFFFF", size=15, bold=True,
               sub="val macro-F1 → pick epoch; test once (383,386)")
arrow(s, rowB[-1], ev); arrow(s, rowA[-1], ev)
arrow(s, row1[2], rowB[1], ax="b", bx="t", color=MAE, width=2.2, dash=True)
tb(s, 6.85, 3.0, 3.2, 0.3, [[("transfer encoder weights", {"size": 12, "bold": True, "color": MAE, "italic": True})]])
tb(s, 0.6, 6.0, 12.1, 0.9, [[("Encoder per 64-sample patch: 3 × (Conv1d → BatchNorm → GELU), max-pooling, global average pool, linear → 128-d. "
                             "The whole classifier (encoder + head) is fine-tuned in both paths. The decoder and positional embeddings are used only in pretraining.",
                             {"size": 13, "color": MUTED})]])
footer(s, 8)
notes(s, "Built from src/models/cnn.py, src/models/cnn_mae.py and experiments/train_classifier.py.")

# ------------------------------------------------------------------ 9 methodology and setup
s = new(); title(s, "Same fine-tuning recipe; only the initialisation changes", "Methodology and experimental setup")
tb(s, 0.6, 1.6, 4.9, 4.6, [
    ("bullet", [("Why a CNN: ", {"bold": True}), ("1-D convolutions learn local phase/amplitude patterns directly from raw I/Q", {})]),
    ("bullet", [("Self-supervised signal: ", {"bold": True}), ("hide half of the 16 patches and reconstruct them from the visible half", {})]),
    ("bullet", [("Transfer: ", {"bold": True}), ("copy the pretrained encoder; add a fresh linear head; fine-tune everything", {})]),
    ("bullet", [("Scratch baseline: ", {"bold": True}), ("same model and recipe from random initialisation, so any gap is due to pretraining", {})]),
], size=16, space_after=12)
cfg = [["Setting", "MAE pretraining", "Classifier fine-tuning"],
       ["Data", "1,789,132 train signals, no labels", "labelled subset (1–100%)"],
       ["Loss", "MSE on masked patches", "cross-entropy"],
       ["Optimiser", "AdamW, lr 1e-4, wd 1e-4", "AdamW, lr 1e-3, wd 1e-4"],
       ["Schedule", "cosine, 5 epochs, batch 64", "cosine, batch 128, grad-clip 1.0, FP16"],
       ["Epochs", "5", ("10 primary · 50 / 100 exploratory", {"bold": True})],
       ["Selection", "lowest validation MSE (epoch 5)", "best validation macro-F1 (12,480)"]]
table(s, 5.8, 1.6, 6.95, [1.35, 2.75, 2.85], cfg, size=13, row_h=0.52, left_cols=(0, 1, 2))
tb(s, 5.8, 5.4, 6.95, 0.4, [[("Environment: Python 3.11, PyTorch 2.10.0 + CUDA 12.6, NVIDIA RTX 4050 Laptop GPU (6 GB). Pretraining took 4.9 h.", {"size": 12, "color": MUTED})]])
tag(s, 0.6, 6.25, "PRIMARY = 10 epochs, pre-specified", "FFFFFF", ACCENT, w=3.7)
tag(s, 4.5, 6.25, "EXPLORATORY = 50 / 100 epochs, added later", EXP, EXP_BG, w=4.3)
footer(s, 9)

# ------------------------------------------------------------------ 10 primary results (native chart + table)
s = new(); title(s, "Primary result: modest gain at 5%, no consistent advantage", "Primary experiment · test set (383,386 signals)")
cd = CategoryChartData(); cd.categories = [f"{b}%" for b in REF]
cd.add_series("Scratch", [round(float(prim[b]["scratch_test_macro_f1"]), 4) for b in REF])
cd.add_series("CNN-MAE", [round(float(prim[b]["mae_test_macro_f1"]), 4) for b in REF])
gf = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(0.6), Inches(1.6), Inches(6.2), Inches(4.25), cd)
ch = gf.chart; ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.TOP; ch.legend.include_in_layout = False
ch.legend.font.size = Pt(13); ch.font.name = BODY
va = ch.value_axis; va.minimum_scale, va.maximum_scale, va.major_unit = 0.20, 0.40, 0.05
va.has_major_gridlines = True; va.major_gridlines.format.line.color.rgb = rgb("E3E6EA"); va.tick_labels.font.size = Pt(12)
va.tick_labels.number_format = "0.00"; va.tick_labels.number_format_is_linked = False
va.has_title = True; va.axis_title.text_frame.text = "Test macro-F1"
va.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(12)
ca = ch.category_axis; ca.tick_labels.font.size = Pt(12); ca.has_title = True
ca.axis_title.text_frame.text = "Labelled fraction of the training split"; ca.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(12)
for ser, col, mk in zip(ch.plots[0].series, (SCR, MAE), (XL_MARKER_STYLE.CIRCLE, XL_MARKER_STYLE.SQUARE)):
    ser.format.line.color.rgb = rgb(col); ser.format.line.width = Pt(2.5); ser.smooth = False
    ser.marker.style = mk; ser.marker.size = 8; ser.marker.format.fill.solid(); ser.marker.format.fill.fore_color.rgb = rgb(col)
    ser.marker.format.line.color.rgb = rgb(col)
tb(s, 0.6, 5.9, 6.2, 0.5, [[("Mean over seeds (3 at 1–10%, 1 at 25% and 100%). The y-axis starts at 0.20.", {"size": 11, "color": MUTED})]])
tdat = [["Labels", "Seeds", "Scratch", "CNN-MAE", "Δ F1", "Δ acc."]]
fills = [None]
for b in REF:
    r = prim[b]
    d = float(r["delta_test_macro_f1"])
    tdat.append([f"{b}%", r["n_seeds"], f4(float(r["scratch_test_macro_f1"])), f4(float(r["mae_test_macro_f1"])),
                 (f4(d, True), {"bold": b == 5, "color": OK if d > 0 else NO}), f4(float(r["delta_test_accuracy"]), True)])
    fills.append("E7F3F2" if b == 5 else None)
table(s, 7.15, 1.6, 5.6, [0.85, 0.75, 1.0, 1.05, 0.95, 1.0], tdat, size=14, row_h=0.46, fills=fills)
tb(s, 7.15, 4.45, 5.6, 0.4, [[("Δ = CNN-MAE − scratch (F1 = macro-F1, acc. = accuracy). Macro-F1 = unweighted mean of per-class F1 over 24 classes.", {"size": 11, "color": MUTED})]])
takeaway(s, 7.15, 4.95, 5.6, 1.45, [("5%: +0.0121 macro-F1", {"bold": True, "color": ACCENT}),
         (f" (2 of 3 seeds; paired t-test p = {P5:.2f}, not significant). At 1%, 10% and 25% MAE is marginally lower.", {"size": 15})], size=16)
footer(s, 10)
notes(s, "Values from final_artifacts/tables/primary_summary.csv (recomputed from C:/dlruns/results). Accuracy deltas in the last column.")

# ------------------------------------------------------------------ 11 per-SNR
s = new(); title(s, "Pretraining helps most at −8 to 0 dB with few labels", "Primary experiment · per-SNR test accuracy")
pic = s.shapes.add_picture(str(FIG / "fig4_primary_per_snr_accuracy.png"), Inches(0.9), Inches(1.4), width=Inches(11.5))
TY = 1.4 + pic.height / 914400 + 0.12
bt = [["Labels", "−20 to −10 dB acc. (scratch / MAE)", "−8 to 0 dB: Δ acc.", "≥ 10 dB: Δ acc."]]
for b in REF:
    lo = BAND[b]["low"]
    bt.append([f"{b}%", f"{lo[0]:.3f} / {lo[1]:.3f}", (f"{100 * BAND[b]['mid']:+.1f} pp", {"bold": True, "color": OK if BAND[b]['mid'] > 0 else NO}),
               f"{100 * BAND[b]['high']:+.1f} pp"])
table(s, 0.6, TY, 8.1, [1.0, 2.6, 2.25, 2.25], bt, size=12, row_h=0.3)
takeaway(s, 8.95, TY, 3.8, min(2.4, 6.9 - TY), [("Descriptive only: ", {"bold": True, "color": ACCENT}),
         ("MAE's accuracy edge is largest at low-moderate SNR with few labels (+2.6 to +2.8 percentage points at 1–5%) and fades with more labels. "
          "Below −10 dB both are at chance (1/24 ≈ 0.042). Means over seeds; no per-SNR uncertainty.", {"size": 13})], size=13)
footer(s, 11)
notes(s, "Band means from final_artifacts/tables/primary_per_snr_accuracy.csv (mean over seeds; pp = percentage points).")

# ------------------------------------------------------------------ 12 longer fine-tuning
s = new(); title(s, "Longer fine-tuning lifts both; the MAE edge stays inconsistent", "Exploratory follow-ups · 1% labels · 3 seeds")
tag(s, 0.6, 1.5, "EXPLORATORY · designed after primary test results were seen", EXP, EXP_BG, w=6.1)
s.shapes.add_picture(str(FIG / "fig7_stages_10_50_100.png"), Inches(0.6), Inches(2.0), width=Inches(6.4))
for k, (E, ep, x) in enumerate(((E50, 50, 7.35), (E100, 100, 10.1))):
    box(s, x, 1.95, 2.6, 4.3, fill=LIGHT)
    tb(s, x + 0.2, 2.05, 2.3, 0.4, [[(f"{ep} epochs", {"size": 20, "bold": True, "font": HEAD})]])
    tb(s, x + 0.2, 2.55, 2.3, 1.3, [[("CNN-MAE ", {"size": 13, "color": MUTED}), (f"{E['mae_mean']:.4f} ± {E['mae_sd']:.4f}", {"size": 15, "bold": True, "color": MAE})],
                                    [("Scratch ", {"size": 13, "color": MUTED}), (f"{E['scratch_mean']:.4f} ± {E['scratch_sd']:.4f}", {"size": 15, "bold": True, "color": SCR})]], space_after=4)
    tb(s, x + 0.2, 3.55, 2.3, 0.75, [[("Paired Δ", {"size": 13, "color": MUTED})],
                                     [(f"{E['diff_mean']:+.4f} ± {E['diff_sd']:.4f}", {"size": 18, "bold": True, "color": OK if E['diff_mean'] > 0 else NO})]], space_after=0)
    tb(s, x + 0.2, 4.45, 2.3, 1.7, [[("Per seed (42 / 43 / 44)", {"size": 12, "color": MUTED})]] +
       [[(f"{sd}: {v:+.4f}", {"size": 14, "color": OK if v > 0 else NO})] for sd, v in E["per_seed"].items()], space_after=2)
tb(s, 7.35, 6.32, 5.4, 0.4, [[("Test macro-F1, mean ± SD over 3 seeds; Δ = CNN-MAE − scratch for the same seed.", {"size": 11, "color": MUTED})]])
takeaway(s, 0.6, 6.0, 6.4, 0.75, [(f"Every run gains {GAIN[0]:+.3f} to {GAIN[1]:+.3f} macro-F1 over its 10-epoch run; MAE leads in 2 of 3 seeds at both lengths.", {"size": 14})], size=14)
footer(s, 12)
notes(s, "The 100-epoch runs were trained anew (not continued). Seed 44 favours scratch by -0.0214 at 100 epochs, which flips the mean.")

# ------------------------------------------------------------------ 13 limitations, future, conclusion
s = new(); title(s, "Limitations, future work and conclusion", "Wrapping up")
cols = [("Limitations", NO, [
            "No consistent MAE advantage in the primary grid",
            f"5% gain not significant (paired p = {P5:.2f}, 3 seeds)",
            "25% and 100%: one seed each",
            "Longer runs: exploratory, 3 seeds, one label budget",
            "Noisy validation (BatchNorm statistics) affects epoch selection",
            "Benchmark data, not over-the-air deployment"]),
        ("Future work", ACCENT, [
            "More paired seeds at every budget",
            "Stabilise validation and checkpoint selection",
            "Pre-specified study of mask ratio and pretraining length",
            "Test whether reconstruction quality predicts accuracy",
            "Per-SNR uncertainty estimates; controlled transfer studies"])]
for i, (h, col, items) in enumerate(cols):
    x = 0.6 + i * 4.2
    box(s, x, 1.6, 3.95, 4.0, fill=LIGHT)
    tb(s, x + 0.25, 1.75, 3.5, 0.4, [[(h, {"size": 19, "bold": True, "color": col})]])
    tb(s, x + 0.25, 2.3, 3.55, 3.25, [("bullet", [(t_, {})]) for t_ in items], size=14, space_after=7)
box(s, 9.0, 1.6, 3.75, 4.0, fill=INK)
tb(s, 9.25, 1.75, 3.3, 0.4, [[("Conclusion", {"size": 19, "bold": True, "color": "FFFFFF"})]])
tb(s, 9.25, 2.3, 3.3, 3.25, [[("MAE pretraining gave modest gains in some settings: +0.012 macro-F1 at 5% labels, "
                               "the low-SNR band at small budgets, and the 50-epoch follow-up. "
                               "The evidence does not show a consistent advantage over training from scratch.", {"size": 15, "color": "FFFFFF"})]])
takeaway(s, 0.6, 5.9, 12.15, 0.8, [("Take-home: ", {"bold": True, "color": ACCENT}),
         ("with a small patch CNN, pretraining helped at the margins, but not reliably. Seeds and controls matter as much as the method.", {})], size=17)
footer(s, 13)

# ------------------------------------------------------------------ 14 references and questions
s = new(); title(s, "References and questions", "References")
refs = ["T. J. O'Shea, T. Roy, T. C. Clancy, \"Over-the-Air Deep Learning Based Radio Signal Classification,\" IEEE J. Sel. Topics Signal Process., vol. 12, no. 1, pp. 168–179, 2018.",
        "D. Liu, P. Wang, T. Wang, T. Abdelzaher, \"Self-Contrastive Learning based Semi-Supervised Radio Modulation Classification,\" MILCOM 2021, pp. 777–782.",
        "K. He, X. Chen, S. Xie, Y. Li, P. Dollár, R. Girshick, \"Masked Autoencoders Are Scalable Vision Learners,\" IEEE/CVF CVPR, 2022, pp. 16000–16009.",
        "W. Kong et al., \"A Transformer-Based Contrastive Semi-Supervised Learning Framework for Automatic Modulation Recognition,\" IEEE TCCN, 2023.",
        "Y. Shi et al., \"GAF-MAE: A Self-Supervised AMC Method Based on Gramian Angular Field and Masked Autoencoder,\" IEEE TCCN, 2024.",
        "C. Xiao et al., \"MCLHN: Towards Automatic Modulation Classification via Masked Contrastive Learning With Hard Negatives,\" IEEE TWC, 2024.",
        "M. Ma, S. Liu, \"Refined Semi-Supervised Modulation Classification: Integrating Consistency Regularization and Pseudo-Labeling,\" Future Internet, vol. 16, no. 2, art. 38, 2024.",
        "Y. Liu et al., \"RIS-MAE: A Self-Supervised Modulation Classification Method Based on Raw IQ Signals and Masked Autoencoder,\" arXiv:2508.00274, 2025.",
        "DeepSig Inc., RadioML 2018.01A and 2016.10a datasets, https://www.deepsig.ai/datasets/."]
tb(s, 0.6, 1.5, 8.4, 5.4, [[(f"[{i + 1}] ", {"bold": True, "color": ACCENT}), (r_, {})] for i, r_ in enumerate(refs)], size=12, space_after=6)
box(s, 9.4, 1.6, 3.35, 5.0, fill="F2F6F6")
tb(s, 9.4, 3.3, 3.35, 1.0, [[("Questions?", {"size": 36, "bold": True, "font": HEAD, "color": ACCENT})]], align=PP_ALIGN.CENTER)
tb(s, 9.6, 4.4, 2.95, 1.5, [[("Live demo: Streamlit app (demo\\app.py)", {"size": 13, "color": MUTED})]], align=PP_ALIGN.CENTER)
footer(s, 14)

prs.core_properties.title = "Self-Supervised MAE Pretraining for Low-Label AMC (Group 41)"
prs.core_properties.author = "Group 41: P V Nekha, Vinayak S, Monish T"
prs.save(OUT)
(HERE / "build_checks.json").write_text(json.dumps([{"ok": o, "check": m} for o, m in CHECKS], indent=1))
print(f"wrote {OUT} ({len(prs.slides)} slides); {len(CHECKS)} data checks passed")
