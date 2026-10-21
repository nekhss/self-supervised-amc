"""Focused tests for the RML2018 -> RML2016 transfer experiment.

Run:  C:\\venvs\\dlproj\\Scripts\\python.exe -m unittest tests.test_rml2016_transfer -v

Tests that need local artifacts (the git history, the real MAE checkpoint, an
RML2018 fine-tuned checkpoint, the generated split files) skip when absent.
"""

import io
import json
import pickle
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import rml2016_transfer as T  # noqa: E402
from src.data import rml2016  # noqa: E402
from src.models.cnn import IQCNNClassifier  # noqa: E402
from src.models.cnn_mae import IQCNNMaskedAutoencoder  # noqa: E402

BASE_COMMIT = "5101e68"  # last commit before signal_length was added
REAL_MAE = T.DEFAULT_PRETRAINED
REAL_RML2018_FINETUNED = Path(r"C:\dlruns\finetune\mae_1pct_seed42_v1.pt")


def original_cnn_module():
    try:
        source = subprocess.run(
            ["git", "show", f"{BASE_COMMIT}:src/models/cnn.py"], cwd=ROOT,
            capture_output=True, text=True, check=True, encoding="utf-8").stdout
    except Exception:
        return None
    module = types.ModuleType("cnn_original")
    exec(compile(source.lstrip("\ufeff"), "cnn_original.py", "exec"), module.__dict__)
    return module


class BackwardCompatibilityTest(unittest.TestCase):
    """The 1024-sample RML2018 path must be numerically identical."""

    @classmethod
    def setUpClass(cls):
        cls.old = original_cnn_module()
        if cls.old is None:
            raise unittest.SkipTest("original cnn.py not available from git")

    def _pair(self, state=None):
        torch.manual_seed(0)
        old = self.old.IQCNNClassifier(num_classes=24, feature_dim=128, patch_size=64)
        torch.manual_seed(0)
        new = IQCNNClassifier(num_classes=24, feature_dim=128, patch_size=64)
        if state is not None:
            old.load_state_dict(state, strict=True)
            new.load_state_dict(state, strict=True)
        return old, new

    def test_same_init_keys_and_outputs(self):
        old, new = self._pair()
        self.assertEqual(list(old.state_dict()), list(new.state_dict()))
        for k, v in old.state_dict().items():
            self.assertTrue(torch.equal(v, new.state_dict()[k]), k)
        x = torch.randn(8, 2, 1024, generator=torch.Generator().manual_seed(1))
        for mode in ("eval", "train"):
            getattr(old, mode)()
            getattr(new, mode)()
            self.assertTrue(torch.equal(old(x), new(x)), mode)

    def test_real_rml2018_checkpoint_identical(self):
        if not REAL_RML2018_FINETUNED.is_file():
            self.skipTest("RML2018 fine-tuned checkpoint not present")
        state = torch.load(REAL_RML2018_FINETUNED, map_location="cpu",
                           weights_only=False)["model_state_dict"]
        old, new = self._pair(state)
        old.eval()
        new.eval()
        x = torch.randn(16, 2, 1024, generator=torch.Generator().manual_seed(2))
        with torch.no_grad():
            self.assertTrue(torch.equal(old(x), new(x)))

    def test_default_still_requires_1024(self):
        model = IQCNNClassifier()
        self.assertEqual(model.signal_length, 1024)
        with self.assertRaises(ValueError):
            model(torch.zeros(2, 2, 128))


class ShapeTest(unittest.TestCase):
    def test_128_sample_signals(self):
        model = T.build_model().eval()
        self.assertEqual(model.signal_length, 128)
        self.assertEqual(model.patch_size, 64)
        out = model(torch.randn(5, 2, 128))
        self.assertEqual(tuple(out.shape), (5, 11))
        with self.assertRaises(ValueError):
            model(torch.randn(5, 2, 1024))

    def test_patch_must_divide_length(self):
        with self.assertRaises(ValueError):
            IQCNNClassifier(patch_size=48, signal_length=128)
        with self.assertRaises(ValueError):
            IQCNNClassifier(patch_size=8, signal_length=128)

    def test_patch_means_match_manual(self):
        model = T.build_model().eval()
        x = torch.randn(3, 2, 128)
        manual = torch.stack([model.encoder(x[:, :, :64]), model.encoder(x[:, :, 64:])]).mean(0)
        self.assertTrue(torch.allclose(model.encode_signal(x), manual, atol=1e-6))


def synthetic_mae_checkpoint(path, patch_size=64, extra=None):
    torch.manual_seed(123)
    mae = IQCNNMaskedAutoencoder(signal_length=1024, patch_size=patch_size)
    state = mae.state_dict()
    if extra:
        state.update(extra)
    torch.save({"epoch": 1, "model_state_dict": state,
                "config": {"feature_dim": 128, "patch_size": patch_size}}, path)
    return mae


class EncoderLoadingTest(unittest.TestCase):
    def _check(self, model, mae_state, report):
        n = len(model.encoder.state_dict())
        self.assertEqual(report["loaded_encoder_tensors"], n)
        self.assertEqual(report["missing_keys"], [])
        self.assertEqual(report["unexpected_keys"], [])
        self.assertIn("encoder_pos_embed", report["excluded_mae_only_keys"])
        self.assertIn("decoder_pos_embed", report["excluded_mae_only_keys"])
        self.assertFalse(any(k.startswith("encoder.") for k in report["excluded_mae_only_keys"]))
        for k, v in model.encoder.state_dict().items():
            self.assertTrue(torch.equal(v, mae_state[f"encoder.{k}"]), k)

    def test_synthetic_checkpoint_strict_load_and_fresh_head(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "mae.pt"
            mae = synthetic_mae_checkpoint(path)
            torch.manual_seed(7)
            model = T.build_model()
            head_before = {k: v.clone() for k, v in model.classifier.state_dict().items()}
            report = T.load_pretrained_encoder(model, path)
            self._check(model, mae.state_dict(), report)
            for k, v in model.classifier.state_dict().items():
                self.assertTrue(torch.equal(v, head_before[k]), "head changed by loading")
            self.assertEqual(tuple(model.classifier.weight.shape), (11, 128))

    def test_rejects_classifier_head_and_wrong_patch(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bad.pt"
            synthetic_mae_checkpoint(path, extra={"classifier.weight": torch.zeros(24, 128)})
            with self.assertRaises(ValueError):
                T.load_pretrained_encoder(T.build_model(), path)
            path2 = Path(d) / "bad2.pt"
            synthetic_mae_checkpoint(path2, patch_size=32)
            with self.assertRaises(ValueError):
                T.load_pretrained_encoder(T.build_model(), path2)

    def test_real_checkpoint(self):
        if not REAL_MAE.is_file():
            self.skipTest("real CNN-MAE checkpoint not present")
        model = T.build_model()
        report = T.load_pretrained_encoder(model, REAL_MAE)
        state = torch.load(REAL_MAE, map_location="cpu", weights_only=False)["model_state_dict"]
        self._check(model, state, report)
        self.assertEqual(report["loaded_encoder_tensors"], 23)
        self.assertEqual(len(report["excluded_mae_only_keys"]), 33)

    def test_head_init_matched_across_arms(self):
        torch.manual_seed(42)
        a = T.build_model()
        torch.manual_seed(42)
        b = T.build_model()
        torch.manual_seed(43)
        c = T.build_model()
        self.assertTrue(torch.equal(a.classifier.weight, b.classifier.weight))
        self.assertFalse(torch.equal(a.classifier.weight, c.classifier.weight))


def synthetic_labels():
    y = np.repeat(np.arange(len(rml2016.CLASSES)), len(rml2016.SNRS) * 1000)
    snr = np.tile(np.repeat(np.array(rml2016.SNRS), 1000), len(rml2016.CLASSES))
    return y, snr


class SplitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.y, cls.snr = synthetic_labels()
        cls.splits, cls.per_budget = rml2016.make_splits(cls.y, cls.snr, seed=42)

    def test_disjoint_exhaustive_nested(self):
        self.assertTrue(rml2016.check_splits(self.splits, len(self.y)))
        self.assertEqual(self.per_budget, {1: 7, 5: 35, 10: 70, 25: 175, 100: 700})
        self.assertEqual(len(self.splits["train_idx"]), 154_000)
        self.assertEqual(len(self.splits["val_idx"]), 33_000)
        self.assertEqual(len(self.splits["test_idx"]), 33_000)

    def test_stratified_counts(self):
        for name, per in [("val_idx", 150), ("test_idx", 150), ("train_1pct", 7),
                          ("train_25pct", 175)]:
            idx = self.splits[name]
            table = np.zeros((11, 20), dtype=int)
            np.add.at(table, (self.y[idx], (self.snr[idx] + 20) // 2), 1)
            self.assertTrue((table == per).all(), name)

    def test_deterministic_and_seed_dependent(self):
        again, _ = rml2016.make_splits(self.y, self.snr, seed=42)
        other, _ = rml2016.make_splits(self.y, self.snr, seed=43)
        self.assertEqual(rml2016.split_digest(again), rml2016.split_digest(self.splits))
        self.assertNotEqual(rml2016.split_digest(other), rml2016.split_digest(self.splits))

    def test_check_catches_leakage(self):
        bad = dict(self.splits)
        bad["val_idx"] = np.concatenate([self.splits["val_idx"][1:], self.splits["test_idx"][:1]])
        with self.assertRaises(AssertionError):
            rml2016.check_splits(bad, len(self.y))
        bad = dict(self.splits)
        bad["train_5pct"] = self.splits["train_5pct"][1:]
        bad["train_1pct"] = np.union1d(self.splits["train_1pct"], self.splits["train_10pct"][:1])
        with self.assertRaises(AssertionError):
            rml2016.check_splits(bad, len(self.y))

    def test_budget_rounding_is_exact(self):
        self.assertEqual(rml2016.budget_count(700, 1), 7)
        with self.assertRaises(ValueError):
            rml2016.budget_count(650, 1)

    def test_generated_split_file(self):
        protocol = T.DEFAULT_PROTOCOL
        if not protocol.is_file():
            self.skipTest("split files not generated yet")
        meta = json.loads(protocol.read_text())
        splits = rml2016.load_splits(T.DEFAULT_SPLITS)
        self.assertEqual(rml2016.split_digest(splits), meta["split_sha256"])
        self.assertTrue(rml2016.check_splits(splits, meta["num_examples"]))
        self.assertEqual(meta["split_seed"], 42)


class LoaderValidationTest(unittest.TestCase):
    def _tiny(self):
        return {(c, s): np.zeros((1000, 2, 128), np.float32)
                for c in rml2016.CLASSES for s in rml2016.SNRS}

    def test_valid_dict_and_canonical_order(self):
        data = self._tiny()
        data[("BPSK", 4)][:] = 1.0
        x, y, snr = rml2016.to_arrays(data)
        self.assertEqual(x.shape, (220_000, 2, 128))
        i = (rml2016.CLASSES.index("BPSK") * 20 + rml2016.SNRS.index(4)) * 1000
        self.assertTrue((x[i:i + 1000] == 1).all())
        self.assertTrue((y[i:i + 1000] == rml2016.CLASSES.index("BPSK")).all())
        self.assertTrue((snr[i:i + 1000] == 4).all())

    def test_discrepancies_raise(self):
        for mutate in (
            lambda d: d.pop(("QPSK", 0)),
            lambda d: d.__setitem__(("QPSK", 0), np.zeros((1000, 2, 1024), np.float32)),
            lambda d: d.__setitem__(("QPSK", 0), np.full((1000, 2, 128), np.nan, np.float32)),
            lambda d: d.__setitem__(("FOO", 0), np.zeros((1000, 2, 128), np.float32)),
        ):
            data = self._tiny()
            mutate(data)
            with self.assertRaises(ValueError):
                rml2016.validate_dict(data)
        with self.assertRaises(ValueError):
            rml2016.validate_dict([1, 2])

    def test_restricted_unpickler_blocks_code(self):
        class Evil:
            def __reduce__(self):
                return (print, ("pwned",))
        payload = pickle.dumps(Evil())
        with self.assertRaises(pickle.UnpicklingError):
            rml2016._NumpyOnlyUnpickler(io.BytesIO(payload), encoding="latin1").load()
        ok = pickle.dumps({("BPSK", 0): np.zeros((2, 2, 128), np.float32)}, protocol=2)
        loaded = rml2016._NumpyOnlyUnpickler(io.BytesIO(ok), encoding="latin1").load()
        self.assertEqual(loaded[("BPSK", 0)].shape, (2, 2, 128))


class MatchedArmsTest(unittest.TestCase):
    def test_configs_differ_only_in_initialisation(self):
        a = T.run_config("transfer", 5, 42, 10, 0, REAL_MAE, False)
        b = T.run_config("scratch", 5, 42, 10, 0, REAL_MAE, False)
        diff = {k for k in a if a[k] != b[k]}
        self.assertEqual(diff, {"arm", "init_checkpoint"})
        self.assertIsNone(b["init_checkpoint"])
        self.assertEqual(a["train"], T.TRAIN_DEFAULTS)


if __name__ == "__main__":
    unittest.main()
