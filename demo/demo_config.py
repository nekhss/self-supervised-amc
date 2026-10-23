"""Paths and constants for the demo. Every path can be overridden by an environment variable.

  DLPROJ_ROOT     project root (default: the parent folder of demo/)
  DLPROJ_RUNS     experiment outputs (default: C:\\dlruns)
  RADIOML_CACHE   memory-mapped data cache built by tools/build_cache.py (default: <runs>/cache)
  RADIOML_HDF5    original RadioML 2018.01A HDF5 file (fallback when the cache is missing)
"""

import os
import sys
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("DLPROJ_ROOT", DEMO_DIR.parent)).resolve()
RUNS_DIR = Path(os.environ.get("DLPROJ_RUNS", r"C:\dlruns"))
CACHE_DIR = Path(os.environ.get("RADIOML_CACHE", str(RUNS_DIR / "cache")))
HDF5_PATH = Path(os.environ.get("RADIOML_HDF5", r"C:\RadioML2018\raw\GOLD_XYZ_OSC.0001_1024.hdf5"))

SPLITS_FILE = PROJECT_ROOT / "data_splits.npz"
FINETUNE_DIR = RUNS_DIR / "finetune"            # primary-grid classifier checkpoints
RESULTS_DIR = RUNS_DIR / "results"              # primary-grid test results (evaluate_test.py)
PAIRED_CSV = RESULTS_DIR / "paired_comparison.csv"
TEST_LABELS_FILE = RESULTS_DIR / "test_labels_snr.npz"
MAE_PRETRAIN_CKPT = RUNS_DIR / "cnn_mae_full_v1_best.pt"
REPORT_VALUES = PROJECT_ROOT / "report" / "tables" / "values.json"

# A 1% MAE classifier trained after the primary grid. It selects its epoch on the FULL
# validation split (not the 12,480-signal subset used by the primary protocol), so it is
# shown only as a clearly labelled supplementary model.
SUPPLEMENTARY_CKPT = RUNS_DIR / "classifier_mae_fullpretrain_1pct_seed42.pt"

# EXPLORATORY 50-epoch fine-tuning (1% labels, seeds 42-44), run after the primary test results
# were known. Read by exploratory_data.py only and never pooled with the primary grid.
FINETUNE_50EP_DIR = RUNS_DIR / "finetune_50ep_v1"                    # run folders (+ one misnamed sibling)
EXPLORATORY_RESULTS_DIR = RUNS_DIR / "results_50ep_v1_exploratory"   # one test evaluation per checkpoint
# Cross-check values (seed -> (MAE, scratch), 4 d.p.) from the experiment notes. The app recomputes
# everything from the files and uses these only to flag discrepancies.
EXPLORATORY_REFERENCE = {
    "val_macro_f1": {42: (0.3346, 0.3261), 43: (0.3278, 0.3240), 44: (0.3185, 0.3185)},
    "test_macro_f1": {42: (0.3349, 0.3255), 43: (0.3291, 0.3212), 44: (0.3192, 0.3213)},
}

# EXPLORATORY 100-epoch follow-up (same design, 100 fine-tuning epochs, all runs restarted).
FINETUNE_100EP_DIR = RUNS_DIR / "finetune_100ep_v2"
# Reported summary (test macro-F1 mean, SD) used only to flag discrepancies; recomputed from files.
EXPLORATORY_100_REFERENCE = {"mae": (0.3281, 0.0090), "scratch": (0.3299, 0.0100), "diff": (-0.0018, 0.0170)}

# Exploratory studies: name -> where its run folders and test results live.
STUDIES = {
    "50": {"epochs": 50, "run_base": FINETUNE_50EP_DIR, "results": EXPLORATORY_RESULTS_DIR, "siblings": True},
    "100": {"epochs": 100, "run_base": FINETUNE_100EP_DIR / "runs", "results": FINETUNE_100EP_DIR / "test_results",
            "siblings": False},
}

# Primary experiment plan: label budget (%) -> seeds.
PLAN = {1: [42, 43, 44], 5: [42, 43, 44], 10: [42, 43, 44], 25: [42], 100: [42]}
ARMS = {"scratch": "Scratch CNN", "mae": "CNN-MAE"}
ARM_INIT = {"scratch": "scratch", "mae": "mae_pretrained"}

# Reference values from the project brief. The app recomputes everything from the result
# files and only uses these to flag discrepancies.
REFERENCE_MACRO_F1 = {1: (0.2577, 0.2566), 5: (0.3199, 0.3320), 10: (0.3583, 0.3571),
                      25: (0.3630, 0.3629), 100: (0.3821, 0.3838)}
REFERENCE_P_5PCT = 0.29

COLOR = {"scratch": "#2a78d6", "mae": "#eb6834", "I": "#6250d6", "Q": "#199e70",
         "ink": "#1d1c1a", "muted": "#6b6a65", "grid": "#e4e3df"}

# The project's own model and data code is imported from src/.
os.environ.setdefault("RADIOML_CACHE", str(CACHE_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def primary_ckpt(arm: str, budget: int, seed: int) -> Path:
    return FINETUNE_DIR / f"{arm}_{budget}pct_seed{seed}_v1.pt"


def primary_result(arm: str, budget: int, seed: int) -> Path:
    return RESULTS_DIR / f"{arm}_{budget}pct_seed{seed}_v1.json"


def primary_pred(arm: str, budget: int, seed: int) -> Path:
    return RESULTS_DIR / f"{arm}_{budget}pct_seed{seed}_v1_pred.npy"
