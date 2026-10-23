"""Extract a few REAL held-out test signals (read-only) for the slide visuals.

Uses the demo's verified test-split reader (data_splits.npz -> test_idx, memory-mapped cache).
Run with the project environment:
  C:\\venvs\\dlproj\\Scripts\\python.exe final_presentation\\assets\\extract_signals.py
Writes final_presentation\\assets\\signals.json (the first 256 of 1,024 samples of I and Q for each signal).
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "demo"))
from data_access import TestSplit, load_class_names  # noqa: E402

split, names = TestSplit(), load_class_names()
qpsk = names.index("QPSK")
out = {}
for key, snr in (("title_qpsk_30db", 30), ("qpsk_18db", 18), ("qpsk_m10db", -10)):
    pos = int(split.matching(qpsk, snr)[0])
    s = split.get(pos)
    assert s.label == qpsk and s.snr == snr
    out[key] = {"test_index": pos, "hdf5_row": int(s.global_index), "class": names[s.label], "snr_db": int(s.snr),
                "I": np.round(s.signal[0, :256], 5).tolist(), "Q": np.round(s.signal[1, :256], 5).tolist()}
Path(__file__).with_name("signals.json").write_text(json.dumps(out))
print({k: (v["class"], v["snr_db"], v["test_index"]) for k, v in out.items()})
