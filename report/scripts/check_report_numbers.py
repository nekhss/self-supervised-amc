"""Consistency check: every number in the prose of report.tex must match a value
derived from tables/values.json (or a listed constant). Prints unmatched numbers."""
import json, re, sys
from pathlib import Path

V = json.loads(Path("tables/values.json").read_text())
vals = set()

def walk(x):
    if isinstance(x, dict):
        for v in x.values(): walk(v)
    elif isinstance(x, (list, tuple)):
        for v in x: walk(v)
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        vals.add(float(x))
    elif isinstance(x, str):
        try: vals.add(float(x))
        except ValueError: pass
walk(V)
# derived quantities used in prose
P = V["primary"]
for f in P.values():
    vals.add(f["mae_test_accuracy_snr_ge_10_mean"] - f["scratch_test_accuracy_snr_ge_10_mean"])
vals.add(P["100"]["scratch_test_macro_f1_mean"] - P["10"]["scratch_test_macro_f1_mean"])
for a in V["perepoch"]:
    for m in ("accuracy", "macro_f1"):
        vals.add(100 * (1 - V["perepoch"][a]["summary"][m]["swing_ratio_recal_over_ordinary"]))
S, M = V["perepoch"]["scratch"]["records"], V["perepoch"]["mae"]["records"]
for s, m in zip(S, M):
    for k in ("ordinary", "recalibrated"):
        for met in ("accuracy", "macro_f1"):
            vals.add(m[k][met] - s[k][met])
for a in V["bn_checkpoint"]:
    d = V["bn_checkpoint"][a]
    for met in ("accuracy", "macro_f1"):
        vals.add(d["B"][met] - d["A1_evaluate"][met])
snr = V["snr_delta_accuracy"]
for f in ("1", "5", "10", "25", "100"):
    v = snr[f]; s_ = snr["snrs"]
    vals.add(sum(x for x, t in zip(v, s_) if -8 <= t <= 0) / 5)
    vals.add(sum(x for x, t in zip(v, s_) if t >= 10) / sum(1 for t in s_ if t >= 10))
pc = V["per_class_f1_5pct"]
for k in pc["mae"]:
    vals.add(pc["mae"][k] - pc["scratch"][k])
mt = V["matched_wd1e5"]
for i in range(3):
    for a, b in (("scratch", "scratch"), ("mae", "mae")):
        pass
import statistics
for arm in ("scratch", "mae"):
    for k in ("test_macro_f1", "test_accuracy", "test_accuracy_snr_ge_10"):
        vals.add(statistics.mean(mt[arm][k]))
vals.add(statistics.mean(mt["mae"]["test_macro_f1"]) - statistics.mean(mt["scratch"]["test_macro_f1"]))
CONST = {2555904, 1024, 24, 26, 20, 30, 70, 15, 1789132, 383386, 17883, 89451, 178914, 447281, 12480,
         16, 64, 128, 32, 7, 5, 3, 2, 1, 4, 8, 0.5, 512, 10, 42, 43, 44, 22, 55544, 52448, 1000, 0.1, 0.9, 65,
         100, 25, 50, 6, 9, 11, 2026, 21, 48, 300, 16000, 50000, 0.15, 0.03, 0.01, 0.02, 0.037, 0.038,
         2.418, 0.458, 0.497, 0.07, 0.09, 1.0, 12, 1.43, 0.29, 17730, 3.11, 2.10, 0.0018, 0.001, 0.006}
vals |= {float(c) for c in CONST}

tex = Path("report.tex").read_text()
body = re.sub(r"\\tabinput\{[^}]*\}", "", tex)
body = re.sub(r"%.*", "", body)
nums = re.findall(r"(?<![\w.])[-+]?\d[\d,]*\.?\d*", body)
bad = []
for n in nums:
    x = n.replace(",", "").lstrip("+")
    if x in ("-",) or x.endswith("-"): continue
    try: f = float(x)
    except ValueError: continue
    dec = len(x.split(".")[1]) if "." in x else 0
    tol = 0.5 * 10 ** (-dec) + 1e-12
    if not any(abs(f - v) <= tol or abs(-f - v) <= tol for v in vals):
        bad.append(n)
print("numbers checked:", len(nums)); print("unmatched:", sorted(set(bad), key=lambda s: float(s.replace(',', '').lstrip('+'))))
