"""Validation-instability diagnostic: BatchNorm statistics and FP16 (review before running).

Checkpoint (read-only): C:\\dlruns\\finetune\\scratch_5pct_seed42_v1.pt
Validation data:        validation_subset_v1.npz (12,480 samples, val split only)
Calibration data:       label_subsets_v2.npz["train_5pct"] (89,451 samples, the exact
                        labelled subset this checkpoint was trained on)
Test set:               never loaded. Only train_idx / val_idx keys of data_splits.npz
                        are read, for membership checks; cache\\test is never opened.
Output:                 C:\\dlruns\\diagnostics\\bn_scratch_5pct_seed42\\ (refuses to run
                        if it already exists). No other file is written.

Conditions (same weights, same 12,480 validation samples, same order, batch 128):
  A1, A2   existing validation path (train_classifier.evaluate), run twice
  A-fp32   same pass with autocast disabled
  B        exact BatchNorm recalibration on train_5pct, then FP16 validation
  B-fp32   B with autocast disabled
  C        20 draws: starting from the checkpoint statistics, 50 random training
           batches (128 each, train_5pct) pass through in train mode under FP16
           autocast with no gradients and no optimiser, so only BatchNorm running
           statistics update (PyTorch momentum 0.1). Then FP16 validation.
           C estimates how sensitive validation metrics are to BatchNorm
           running-statistic updates under sampled training batches AT FIXED
           WEIGHTS. It does NOT reproduce the training trajectory: weights do
           not change, and batches are random draws, not the real epoch order.

Run:  C:\\venvs\\dlproj\\Scripts\\python.exe tools\\diag_bn_scratch_5pct.py
"""

import copy
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.cached_dataset import CachedRadioMLDataset  # noqa: E402
from src.models.cnn import IQCNNClassifier  # noqa: E402

CKPT = Path(r"C:\dlruns\finetune\scratch_5pct_seed42_v1.pt")
OUT = Path(r"C:\dlruns\diagnostics\bn_scratch_5pct_seed42")

# Verified read-only on 20 Oct before writing this script.
EXPECTED_TRAIN5_SHA = "14af9fce7b3d011fef3eae64d35144e9374dd01a60545d6bdd09fa167a502f02"
EXPECTED_VAL_SHA = "dfa90efbbf19a41a7c29145dd0c0e0d9dd9dc3beff7c40a5dce9da5d20531c76"
EXPECTED_N_TRAIN5, EXPECTED_N_VAL = 89451, 12480
LOGGED = {"loss": 2.089304475295238, "accuracy": 0.32900641025641025,
          "macro_f1": 0.3118944782303938}
BN_LAYERS = ("encoder.features.1", "encoder.features.5", "encoder.features.9")
C_DRAWS, C_BATCHES, BATCH = 20, 50, 128


def sha_indices(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, np.int64)).tobytes()).hexdigest()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_train_classifier():
    """Import experiments/train_classifier.py unchanged, to reuse evaluate()."""
    spec = importlib.util.spec_from_file_location(
        "train_classifier", ROOT / "experiments" / "train_classifier.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def non_bn_state(model):
    """Every parameter and buffer except BatchNorm running statistics."""
    skip = {f"{l}.{b}" for l in BN_LAYERS
            for b in ("running_mean", "running_var", "num_batches_tracked")}
    return {k: v.detach().clone() for k, v in model.state_dict().items() if k not in skip}


def assert_unchanged(model, reference, label):
    now = non_bn_state(model)
    if now.keys() != reference.keys():
        raise RuntimeError(f"{label}: state keys changed")
    bad = [k for k in reference if not torch.equal(now[k], reference[k])]
    if bad:
        raise RuntimeError(f"{label}: non-BatchNorm state changed: {bad}")


def bn_stats(model):
    mods = dict(model.named_modules())
    return {l: (mods[l].running_mean.detach().double().cpu().clone(),
                mods[l].running_var.detach().double().cpu().clone()) for l in BN_LAYERS}


def compare_bn(reference, other):
    """Per layer: standardised mean shift and variance ratio vs the reference statistics."""
    out = {}
    for l in BN_LAYERS:
        m0, v0 = reference[l]
        m1, v1 = other[l]
        shift = ((m1 - m0) / v0.sqrt()).abs()
        ratio = v1 / v0
        out[l] = {"abs_std_mean_shift_median": float(shift.median()),
                  "abs_std_mean_shift_max": float(shift.max()),
                  "var_ratio_median": float(ratio.median()),
                  "var_ratio_min": float(ratio.min()),
                  "var_ratio_max": float(ratio.max()),
                  "channels_var_ratio_outside_0.5_2": int(((ratio < 0.5) | (ratio > 2)).sum())}
    return out


def validation_pass(model, loader, device, autocast, y_snr):
    """Mirror of train_classifier.evaluate() that also returns predictions."""
    model.eval()
    total_loss, n_total, preds, targets = 0.0, 0, [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["signal"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite validation loss.")
            total_loss += loss.float().item() * y.size(0)
            n_total += y.size(0)
            preds.append(logits.argmax(1).cpu())
            targets.append(y.cpu())
    p, t = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    if not np.array_equal(t, y_snr[0]):
        raise RuntimeError("Validation label order differs from the cached order.")
    snr = y_snr[1]
    counts = np.bincount(p, minlength=24)
    return {
        "loss": total_loss / n_total,
        "accuracy": float(accuracy_score(t, p)),
        "macro_f1": float(f1_score(t, p, average="macro", zero_division=0)),
        "samples": int(n_total),
        "pred_counts": counts.tolist(),
        "max_class_share": float(counts.max() / n_total),
        "classes_never_predicted": int((counts == 0).sum()),
        "acc_by_snr": {int(s): float((p[snr == s] == t[snr == s]).mean()) for s in np.unique(snr)},
    }, p


def recalibrate_exact(model, calib_x, device):
    """Exact per-channel BatchNorm statistics on calib_x, one layer at a time, in eval mode.

    For layer k, layers < k already carry recalibrated statistics, so layer k's
    statistics are computed on exactly the inputs it receives at evaluation time.
    Moments are combined across batches with Chan's parallel algorithm in float64,
    over all (sample, position) elements of each channel.
    Only running_mean / running_var are assigned. FP32, no autocast, no gradients.
    """
    mods = dict(model.named_modules())
    model.eval()
    report = {}
    for layer in BN_LAYERS:
        bn = mods[layer]
        acc = {"n": 0, "mean": None, "m2": None}

        def hook(_module, inputs):
            x = inputs[0].detach().double()            # [N_patches, C, L]
            n_b = x.shape[0] * x.shape[2]
            mean_b = x.mean(dim=(0, 2))
            m2_b = ((x - mean_b[None, :, None]) ** 2).sum(dim=(0, 2))
            if acc["mean"] is None:
                acc.update(n=n_b, mean=mean_b, m2=m2_b)
            else:
                n_a = acc["n"]
                delta = mean_b - acc["mean"]
                tot = n_a + n_b
                acc["mean"] = acc["mean"] + delta * (n_b / tot)
                acc["m2"] = acc["m2"] + m2_b + delta ** 2 * (n_a * n_b / tot)
                acc["n"] = tot

        handle = bn.register_forward_pre_hook(hook)
        try:
            with torch.no_grad():
                for s in range(0, len(calib_x), 1024):
                    model(torch.from_numpy(calib_x[s:s + 1024]).to(device))
        finally:
            handle.remove()
        var = acc["m2"] / (acc["n"] - 1)   # unbiased, PyTorch's running_var convention
        with torch.no_grad():
            bn.running_mean.copy_(acc["mean"].to(bn.running_mean.dtype))
            bn.running_var.copy_(var.to(bn.running_var.dtype))
        report[layer] = {"elements_per_channel": int(acc["n"]), "channels": int(var.numel())}
    return report


def main():
    if not torch.cuda.is_available():
        sys.exit("CUDA is required to reproduce the FP16 validation path.")
    device = torch.device("cuda")
    if OUT.exists():
        sys.exit(f"{OUT} already exists; refusing to overwrite. Choose a new folder.")
    started = time.time()
    ckpt_sha_before = sha_file(CKPT)

    # ---------- identity checks (no test data loaded) ----------
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    expect = {"label_fraction": "5", "seed": 42, "init_checkpoint": None,
              "validation_subset": "validation_subset_v1.npz", "batch_size": 128,
              "max_val_batches": 0}
    for k, v in expect.items():
        if cfg.get(k) != v:
            raise RuntimeError(f"Checkpoint config {k}={cfg.get(k)!r}, expected {v!r}")
    if ckpt["epoch"] != 5 or ckpt["initialization"] != "scratch":
        raise RuntimeError("Unexpected checkpoint epoch/initialisation.")

    with np.load(ROOT / "label_subsets_v2.npz") as s:
        train5 = s["train_5pct"].astype(np.int64)
    with np.load(ROOT / "validation_subset_v1.npz") as s:
        val_idx = s["val_indices"].astype(np.int64)
    with np.load(ROOT / "data_splits.npz") as s:     # only these two keys are read
        split_train, split_val = s["train_idx"], s["val_idx"]
    checks = {
        "train5_sha": sha_indices(train5) == EXPECTED_TRAIN5_SHA,
        "train5_n": len(train5) == EXPECTED_N_TRAIN5,
        "val_sha": sha_indices(val_idx) == EXPECTED_VAL_SHA,
        "val_n": len(val_idx) == EXPECTED_N_VAL,
        "train5_in_train_split": bool(np.isin(train5, split_train).all()),
        "val_in_val_split": bool(np.isin(val_idx, split_val).all()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Identity checks failed: {checks}")
    del split_train, split_val

    # ---------- data (same classes and loader settings as train_classifier) ----------
    val_ds = CachedRadioMLDataset(val_idx)
    train_ds = CachedRadioMLDataset(train5)
    for ds in (val_ds, train_ds):
        if ds.cache.folder.name == "test":
            raise RuntimeError("Refusing to read the test cache.")
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0,
                            pin_memory=True)
    y_snr = (val_ds.cache.y[val_ds.rows].astype(np.int64),
             val_ds.cache.snr[val_ds.rows].astype(np.int64))
    calib_rows = np.sort(train_ds.rows)               # deterministic, index order
    calib_x = np.ascontiguousarray(
        train_ds.cache.x[calib_rows].transpose(0, 2, 1))   # float32, ~733 MB

    # ---------- model ----------
    model = IQCNNClassifier(num_classes=24, feature_dim=128, patch_size=64)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    reference_state = non_bn_state(model)
    original_bn = bn_stats(model)
    original_full = {k: v.detach().clone() for k, v in model.state_dict().items()}

    tc = load_train_classifier()
    results = {"checkpoint": str(CKPT), "checkpoint_sha256": ckpt_sha_before,
               "identity_checks": checks, "logged_reference": LOGGED,
               "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}

    # ---------- A1 / A2: existing path ----------
    a1_eval = tc.evaluate(model, val_loader, device)
    a2_eval = tc.evaluate(model, val_loader, device)
    a1, _ = validation_pass(model, val_loader, device, True, y_snr)
    results["A1_evaluate"], results["A2_evaluate"] = a1_eval, a2_eval
    results["A_detail"] = a1
    results["A_reproduces_log"] = {
        "accuracy_abs_diff": abs(a1_eval["accuracy"] - LOGGED["accuracy"]),
        "macro_f1_abs_diff": abs(a1_eval["macro_f1"] - LOGGED["macro_f1"]),
        "loss_abs_diff": abs(a1_eval["loss"] - LOGGED["loss"]),
        "within_one_sample": abs(a1_eval["accuracy"] - LOGGED["accuracy"]) <= 1 / EXPECTED_N_VAL + 1e-12,
        "A1_equals_A2": a1_eval == a2_eval,
        "detail_pass_matches_evaluate": (a1["accuracy"] == a1_eval["accuracy"]
                                         and a1["macro_f1"] == a1_eval["macro_f1"]),
    }
    assert_unchanged(model, reference_state, "after A")

    # ---------- A-fp32 ----------
    results["A_fp32"], _ = validation_pass(model, val_loader, device, False, y_snr)
    assert_unchanged(model, reference_state, "after A-fp32")

    # ---------- B / B-fp32: exact recalibration on a copy ----------
    model_b = copy.deepcopy(model)
    results["B_recalibration"] = recalibrate_exact(model_b, calib_x, device)
    assert_unchanged(model_b, reference_state, "after recalibration")
    results["B_vs_original_bn"] = compare_bn(original_bn, bn_stats(model_b))
    results["B"], _ = validation_pass(model_b, val_loader, device, True, y_snr)
    results["B_fp32"], _ = validation_pass(model_b, val_loader, device, False, y_snr)
    assert_unchanged(model_b, reference_state, "after B")
    recal_bn = bn_stats(model_b)
    del model_b

    # ---------- C: momentum-0.1 BN updates from sampled batches, fixed weights ----------
    draws = []
    n_needed = C_BATCHES * BATCH
    for r in range(C_DRAWS):
        m = copy.deepcopy(model)
        rng = np.random.default_rng(r)
        chosen = rng.choice(len(train_ds.rows), size=n_needed, replace=False)
        m.train()                                    # BN uses batch stats and updates EMA
        with torch.no_grad():
            for b in range(C_BATCHES):
                rows = np.sort(train_ds.rows[chosen[b * BATCH:(b + 1) * BATCH]])
                x = torch.from_numpy(np.ascontiguousarray(
                    train_ds.cache.x[rows].transpose(0, 2, 1))).to(device)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    m(x)
        assert_unchanged(m, reference_state, f"after C draw {r}")
        metrics, _ = validation_pass(m, val_loader, device, True, y_snr)
        metrics.pop("pred_counts")
        metrics["bn_vs_original"] = compare_bn(original_bn, bn_stats(m))
        metrics["bn_vs_recalibrated"] = compare_bn(recal_bn, bn_stats(m))
        draws.append(metrics)
        del m
    acc = np.array([d["accuracy"] for d in draws])
    f1 = np.array([d["macro_f1"] for d in draws])
    results["C_draws"] = draws
    results["C_summary"] = {
        "draws": C_DRAWS, "batches_per_draw": C_BATCHES, "batch_size": BATCH,
        "accuracy_min_median_max": [float(acc.min()), float(np.median(acc)), float(acc.max())],
        "macro_f1_min_median_max": [float(f1.min()), float(np.median(f1)), float(f1.max())],
        "checkpoint_accuracy_percentile_among_draws": float((acc < a1_eval["accuracy"]).mean() * 100),
        "interpretation_scope": "Sensitivity of validation metrics to BatchNorm running-statistic "
                                "updates under sampled training batches at fixed weights; not a "
                                "reproduction of the training trajectory.",
    }

    # ---------- final integrity checks ----------
    final = model.state_dict()
    if any(not torch.equal(final[k], original_full[k]) for k in original_full):
        raise RuntimeError("Original model state changed during the diagnostic.")
    if sha_file(CKPT) != ckpt_sha_before:
        raise RuntimeError("Checkpoint file changed on disk.")
    results["integrity"] = {"original_model_unchanged": True, "checkpoint_file_unchanged": True}
    results["elapsed_seconds"] = time.time() - started

    OUT.mkdir(parents=True, exist_ok=False)          # created only after all checks pass
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    rows = [("A1 (evaluate)", a1_eval), ("A2 (evaluate)", a2_eval),
            ("A-fp32", results["A_fp32"]), ("B (recalibrated)", results["B"]),
            ("B-fp32", results["B_fp32"])]
    lines = [f"Checkpoint: {CKPT}", f"Logged epoch-5: loss {LOGGED['loss']:.6f} "
             f"acc {LOGGED['accuracy']:.6f} F1 {LOGGED['macro_f1']:.6f}", "",
             f"{'condition':18s} {'loss':>9s} {'acc':>8s} {'macroF1':>8s}"]
    lines += [f"{n:18s} {r['loss']:9.6f} {r['accuracy']:8.4f} {r['macro_f1']:8.4f}" for n, r in rows]
    cs = results["C_summary"]
    lines += ["", f"C ({C_DRAWS} draws x {C_BATCHES} batches): acc min/med/max "
              + " / ".join(f"{v:.4f}" for v in cs["accuracy_min_median_max"])
              + ", F1 min/med/max " + " / ".join(f"{v:.4f}" for v in cs["macro_f1_min_median_max"]),
              f"Checkpoint accuracy percentile among C draws: {cs['checkpoint_accuracy_percentile_among_draws']:.0f}",
              "", "Reproduces log: " + json.dumps(results["A_reproduces_log"]),
              "B vs original BN: " + json.dumps(results["B_vs_original_bn"]),
              "", cs["interpretation_scope"], f"Elapsed: {results['elapsed_seconds']:.1f}s"]
    (OUT / "summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
