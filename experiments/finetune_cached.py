"""Run experiments/train_classifier.py unchanged, but read data from the cache.

Every argument is passed straight through to train_classifier.py, so the
training loop, label subsets, validation subset, optimiser, schedule and
checkpoint-selection rule are identical. The only substitution is the dataset
class: RadioMLDataset (per-sample HDF5 reads) -> CachedRadioMLDataset
(memory-mapped copy of the same values, built by tools/build_cache.py).

Example:
  python experiments/finetune_cached.py --label-fraction 5 --epochs 10 \
      --batch-size 128 --validation-subset validation_subset_v1.npz \
      --seed 42 --output C:\\dlruns\\finetune\\scratch_5pct_seed42.pt
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.cached_dataset import CachedRadioMLDataset  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "train_classifier", ROOT / "experiments" / "train_classifier.py")
train_classifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_classifier)

train_classifier.RadioMLDataset = CachedRadioMLDataset

if __name__ == "__main__":
    train_classifier.main()
