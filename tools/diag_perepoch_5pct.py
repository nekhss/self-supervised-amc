"""Per-epoch validation diagnostic: ordinary vs BatchNorm-recalibrated validation.

Re-runs the ORIGINAL 5% seed-42 fine-tuning (experiments/train_classifier.py,
imported unmodified) for one arm, and after every epoch additionally scores a
deep copy of the model whose BatchNorm statistics were recalibrated exactly on
the labelled training subset (train_5pct). The copy never touches training state.

  python tools/diag_perepoch_5pct.py --arm scratch
  python tools/diag_perepoch_5pct.py --arm mae

Output (refuses to run if it exists): C:\\dlruns\\diagnostics\\perepoch_5pct_seed42\\<arm>\\
  train_log.txt               console output of the training run
  epoch_metrics.jsonl         one record per epoch (ordinary, recalibrated, BN diffs, checks)
  epoch_XX_model_state.pt     training-model state at the end of each epoch
  <arm>_5pct_seed42_diag.pt   checkpoint written by train_classifier.main (best val macro-F1)
  reproduction.json           comparison with the original log and checkpoint
  summary.json / summary.txt  swing and dip summaries with the fixed criteria below

Namespace substitutions in the imported train_classifier module (no file is edited):
  RadioMLDataset -> CachedRadioMLDataset (as in finetune_cached.py)
  DataLoader     -> function returning a genuine DataLoader and recording it
  evaluate       -> wrapper: calls the original evaluate(), returns its result object
                    unchanged, then measures the recalibrated copy
During tc.main() only, torch.optim.AdamW, torch.optim.lr_scheduler.CosineAnnealingLR
and torch.amp.GradScaler are replaced by functions that call the original class and
record the genuine instance they return; the originals are restored afterwards.

The test split is never loaded. Original logs and checkpoints are only read.
"""

import argparse
import contextlib
import copy
import importlib.util
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.cached_dataset import CachedRadioMLDataset  # noqa: E402


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Reviewed helpers (recalibration, BN comparison, hashing) reused, not copied.
diag = load_module("diag_bn_scratch_5pct", ROOT / "tools" / "diag_bn_scratch_5pct.py")

ARMS = {
    "scratch": {"orig_ckpt": Path(r"C:\dlruns\finetune\scratch_5pct_seed42_v1.pt"),
                "orig_log": Path(r"C:\dlruns\finetune\scratch_5pct_seed42_v1.log"),
                "init": None},
    "mae": {"orig_ckpt": Path(r"C:\dlruns\finetune\mae_5pct_seed42_v1.pt"),
            "orig_log": Path(r"C:\dlruns\finetune\mae_5pct_seed42_v1.log"),
            "init": r"C:\dlruns\cnn_mae_full_v1_best.pt"},
}
OUT_ROOT = Path(r"C:\dlruns\diagnostics\perepoch_5pct_seed42")
TRAIN_ARGS = ["--label-fraction", "5", "--epochs", "10", "--batch-size", "128",
              "--lr", "0.001", "--workers", "0", "--seed", "42",
              "--max-train-batches", "0", "--max-val-batches", "0",
              "--validation-subset", "validation_subset_v1.npz"]
EXPECTED_CONFIG = {"label_fraction": "5", "epochs": 10, "batch_size": 128, "lr": 0.001,
                   "workers": 0, "seed": 42, "max_train_batches": 0, "max_val_batches": 0,
                   "validation_subset": "validation_subset_v1.npz"}

# Fixed analysis criteria (set before any run).
WINDOW = (4, 10)            # epochs analysed, inclusive
ACC_DIP = 0.03              # accuracy dip: drop > 0.03 (3 points) from the previous epoch
F1_DIP = 0.03               # macro-F1 dip: drop > 0.03 from the previous epoch
SWING_REDUCTION = 0.5       # recalibrated mean |delta| <= 0.5 x ordinary mean |delta|

EPOCH_RE = re.compile(
    r"^Epoch (\d+)/\d+ \| train loss=([\d.]+) \(\d+ samples\) \| val loss=([\d.]+) \| "
    r"val accuracy=([\d.]+) \| val macro-F1=([\d.]+)")


# ---------------------------------------------------------------- helpers
def parse_log(path):
    rows = {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        m = EPOCH_RE.match(line.strip())
        if m:
            rows[int(m.group(1))] = {"train_loss": m.group(2), "val_loss": m.group(3),
                                     "accuracy": m.group(4), "macro_f1": m.group(5)}
    return rows


def deep_equal(a, b):
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
                and a.dtype == b.dtype and a.shape == b.shape
                and torch.equal(a.cpu(), b.cpu()))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(deep_equal(x, y) for x, y in zip(a, b))
    return a == b


def clone_state(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().clone()
    if isinstance(obj, dict):
        return {k: clone_state(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(clone_state(v) for v in obj)
    return copy.deepcopy(obj)


def tensor_diffs(ref, new):
    """Exact equality plus max abs / rel differences, per entry and overall."""
    out, all_equal = {}, ref.keys() == new.keys()
    for k in ref:
        a, b = ref[k].detach().cpu(), new[k].detach().cpu()
        same_meta = a.dtype == b.dtype and a.shape == b.shape
        equal = same_meta and torch.equal(a, b)
        entry = {"equal": bool(equal), "dtype_shape_match": bool(same_meta)}
        if same_meta and a.is_floating_point():
            d = (a.double() - b.double()).abs()
            entry["max_abs_diff"] = float(d.max()) if d.numel() else 0.0
            entry["max_rel_diff"] = float((d / a.double().abs().clamp_min(1e-12)).max()) if d.numel() else 0.0
        elif same_meta:
            entry["values"] = [a.tolist(), b.tolist()] if a.numel() == 1 else None
        out[k] = entry
        all_equal = all_equal and equal
    floats = [v for v in out.values() if "max_abs_diff" in v]
    return {"bitwise_identical": bool(all_equal),
            "entries_equal": sum(v["equal"] for v in out.values()), "entries": len(out),
            "max_abs_diff": max((v["max_abs_diff"] for v in floats), default=0.0),
            "max_rel_diff": max((v["max_rel_diff"] for v in floats), default=0.0),
            "per_entry": out}


def series_stats(values, threshold):
    """values: {epoch: metric}. Swing over WINDOW; dips = epochs whose value fell
    by more than `threshold` from the previous epoch (both epochs inside WINDOW)."""
    lo, hi = WINDOW
    v = [values[e] for e in range(lo, hi + 1)]
    deltas = np.diff(v)
    dips = [lo + i + 1 for i, d in enumerate(deltas) if -d > threshold]
    return {"values": v, "mean_abs_delta": float(np.mean(np.abs(deltas))),
            "range": float(max(v) - min(v)), "dips": dips}


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


# ---------------------------------------------------------------- RNG handling
def rng_snapshot(train_gen):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all(),
            "train_loader_generator": train_gen.get_state()}


def rng_restore(state, train_gen):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])
    train_gen.set_state(state["train_loader_generator"])


def rng_changed(a, b):
    return sorted(k for k in a if not deep_equal(a[k], b[k]))


# ---------------------------------------------------------------- instance recording
@contextlib.contextmanager
def recording_constructors(registry):
    """Within the block, AdamW / CosineAnnealingLR / GradScaler attribute lookups
    return functions that build the GENUINE class instance and record it."""
    originals = {("optim", "AdamW"): torch.optim.AdamW,
                 ("sched", "CosineAnnealingLR"): torch.optim.lr_scheduler.CosineAnnealingLR,
                 ("amp", "GradScaler"): torch.amp.GradScaler}

    def recorder(key, cls):
        def build(*args, **kwargs):
            obj = cls(*args, **kwargs)
            registry[key].append(obj)
            return obj
        return build

    torch.optim.AdamW = recorder("optimizer", originals[("optim", "AdamW")])
    torch.optim.lr_scheduler.CosineAnnealingLR = recorder(
        "scheduler", originals[("sched", "CosineAnnealingLR")])
    torch.amp.GradScaler = recorder("scaler", originals[("amp", "GradScaler")])
    try:
        yield
    finally:
        torch.optim.AdamW = originals[("optim", "AdamW")]
        torch.optim.lr_scheduler.CosineAnnealingLR = originals[("sched", "CosineAnnealingLR")]
        torch.amp.GradScaler = originals[("amp", "GradScaler")]
    if not (torch.optim.AdamW is originals[("optim", "AdamW")]
            and torch.amp.GradScaler is originals[("amp", "GradScaler")]
            and torch.optim.lr_scheduler.CosineAnnealingLR is originals[("sched", "CosineAnnealingLR")]):
        raise RuntimeError("Failed to restore torch constructors.")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    arm = ap.parse_args().arm
    cfg_arm = ARMS[arm]
    out = OUT_ROOT / arm

    # ---------- preflight (nothing is written until all checks pass) ----------
    if not torch.cuda.is_available():
        sys.exit("CUDA is required (the original runs used FP16 autocast on CUDA).")
    if out.exists():
        sys.exit(f"{out} already exists; refusing to overwrite.")
    for p in (cfg_arm["orig_ckpt"], cfg_arm["orig_log"]):
        if not p.is_file():
            sys.exit(f"Missing original artifact: {p}")
    if cfg_arm["init"] and not Path(cfg_arm["init"]).is_file():
        sys.exit(f"Missing MAE checkpoint: {cfg_arm['init']}")
    guarded = [cfg_arm["orig_ckpt"], cfg_arm["orig_log"]] + ([Path(cfg_arm["init"])] if cfg_arm["init"] else [])
    orig_sha = {str(p): diag.sha_file(p) for p in guarded}

    orig = torch.load(cfg_arm["orig_ckpt"], map_location="cpu", weights_only=False)
    ocfg = orig["config"]
    mismatched = {k: ocfg.get(k) for k, v in EXPECTED_CONFIG.items() if ocfg.get(k) != v}
    if ocfg.get("init_checkpoint") != cfg_arm["init"]:
        mismatched["init_checkpoint"] = ocfg.get("init_checkpoint")
    if ocfg.get("weight_decay", 1e-4) != 1e-4 or ocfg.get("encoder_lr") is not None:
        mismatched["optimizer"] = (ocfg.get("weight_decay"), ocfg.get("encoder_lr"))
    if mismatched:
        sys.exit(f"Original checkpoint config differs from the planned run: {mismatched}")
    orig_log = parse_log(cfg_arm["orig_log"])
    if sorted(orig_log) != list(range(1, 11)):
        sys.exit(f"Could not parse 10 epoch lines from {cfg_arm['orig_log']}")

    with np.load(ROOT / "label_subsets_v2.npz") as s:
        train5 = s["train_5pct"].astype(np.int64)
    with np.load(ROOT / "validation_subset_v1.npz") as s:
        val_idx = s["val_indices"].astype(np.int64)
    with np.load(ROOT / "data_splits.npz") as s:        # only these two keys are read
        split_train, split_val = s["train_idx"], s["val_idx"]
    checks = {
        "train5_sha": diag.sha_indices(train5) == diag.EXPECTED_TRAIN5_SHA,
        "val_sha": diag.sha_indices(val_idx) == diag.EXPECTED_VAL_SHA,
        "train5_in_train_split": bool(np.isin(train5, split_train).all()),
        "val_in_val_split": bool(np.isin(val_idx, split_val).all()),
    }
    if not all(checks.values()):
        sys.exit(f"Identity checks failed: {checks}")
    del split_train, split_val

    # Calibration data: train_5pct only, index order, read directly (no loader, no RNG).
    calib_ds = CachedRadioMLDataset(train5)
    if calib_ds.cache.folder.name == "test":
        sys.exit("Refusing to read the test cache.")
    calib_rows = np.sort(calib_ds.rows)
    calib_x = np.ascontiguousarray(calib_ds.cache.x[calib_rows].transpose(0, 2, 1))

    # ---------- import the original training module and substitute names ----------
    tc = load_module("train_classifier", ROOT / "experiments" / "train_classifier.py")
    orig_evaluate = tc.evaluate
    real_dataloader = tc.DataLoader
    registry = {"optimizer": [], "scheduler": [], "scaler": [], "loaders": []}

    def recording_dataloader(*args, **kwargs):
        dl = real_dataloader(*args, **kwargs)
        registry["loaders"].append(dl)
        return dl

    out.mkdir(parents=True, exist_ok=False)           # all preflight checks passed
    metrics_file = open(out / "epoch_metrics.jsonl", "w", encoding="utf-8")
    state = {"epoch": 0, "prev_params": None, "records": []}

    def wrapped_evaluate(model, loader, device, max_batches=0):
        loaders = registry["loaders"]
        if len(loaders) != 2 or loader is not loaders[1] or loaders[0].generator is None:
            raise RuntimeError("Unexpected loader setup; expected (train, val) loaders.")
        if not (len(registry["optimizer"]) == len(registry["scheduler"])
                == len(registry["scaler"]) == 1):
            raise RuntimeError(f"Expected one optimizer/scheduler/scaler: "
                               f"{ {k: len(v) for k, v in registry.items()} }")
        train_gen = loaders[0].generator
        optimizer, scheduler = registry["optimizer"][0], registry["scheduler"][0]
        scaler = registry["scaler"][0]

        # 1. Original validation, exactly as in train_classifier (its RNG use is kept).
        rng_pre_eval = rng_snapshot(train_gen)
        result = orig_evaluate(model, loader, device, max_batches=max_batches)
        rng_post_eval = rng_snapshot(train_gen)
        state["epoch"] += 1
        epoch = state["epoch"]
        t0 = time.time()

        # 2. Snapshot everything training depends on.
        fp = {"model": clone_state(model.state_dict()), "training_mode": model.training,
              "optimizer": clone_state(optimizer.state_dict()),
              "scheduler": clone_state(scheduler.state_dict()),
              "scaler": clone_state(scaler.state_dict())}

        # 3. Recalibrated copy (independent storage, never seen by the optimizer).
        twin = copy.deepcopy(model)
        model_ptrs = {t.data_ptr() for t in list(model.parameters()) + list(model.buffers())}
        twin_ptrs = {t.data_ptr() for t in list(twin.parameters()) + list(twin.buffers())}
        opt_param_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
        if model_ptrs & twin_ptrs or any(id(p) in opt_param_ids for p in twin.parameters()):
            raise RuntimeError("Recalibration copy shares storage or optimizer membership.")
        for p in twin.parameters():
            p.requires_grad_(False)
        diag.recalibrate_exact(twin, calib_x, device)
        recal = orig_evaluate(twin, loader, device, max_batches=max_batches)
        bn_diff = diag.compare_bn(diag.bn_stats(model), diag.bn_stats(twin))
        del twin

        # 4. Weight change and per-epoch model snapshot (reads only).
        params = torch.cat([p.detach().double().flatten() for p in model.parameters()])
        prev = state["prev_params"]
        rel_change = None if prev is None else float((params - prev).norm() / prev.norm())
        state["prev_params"] = params.clone()
        torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                   out / f"epoch_{epoch:02d}_model_state.pt")

        # 5. Restore RNG to its post-validation state and verify nothing else moved.
        rng_before_restore = rng_snapshot(train_gen)
        changed_by_measurement = rng_changed(rng_post_eval, rng_before_restore)
        rng_restore(rng_post_eval, train_gen)
        if rng_changed(rng_post_eval, rng_snapshot(train_gen)):
            raise RuntimeError("RNG restore failed.")
        now = {"model": model.state_dict(), "training_mode": model.training,
               "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
               "scaler": scaler.state_dict()}
        moved = [k for k in fp if not deep_equal(fp[k], now[k])]
        if moved:
            raise RuntimeError(f"Measurement changed training state: {moved}")

        rec = {"epoch": epoch,
               "ordinary": {k: result[k] for k in ("loss", "accuracy", "macro_f1", "samples")},
               "recalibrated": {k: recal[k] for k in ("loss", "accuracy", "macro_f1", "samples")},
               "bn_running_vs_recalibrated": bn_diff,
               "relative_weight_change": rel_change,
               "rng_used_by_original_evaluate": rng_changed(rng_pre_eval, rng_post_eval),
               "rng_used_by_measurement_restored": changed_by_measurement,
               "training_state_unchanged": True,
               "measurement_seconds": time.time() - t0}
        state["records"].append(rec)
        metrics_file.write(json.dumps(rec) + "\n")
        metrics_file.flush()
        return result                                   # the original object, unchanged

    tc.RadioMLDataset = CachedRadioMLDataset
    tc.DataLoader = recording_dataloader
    tc.evaluate = wrapped_evaluate

    new_ckpt = out / f"{arm}_5pct_seed42_diag.pt"
    argv = ["train_classifier.py", *TRAIN_ARGS, "--output", str(new_ckpt)]
    if cfg_arm["init"]:
        argv += ["--init-checkpoint", cfg_arm["init"]]
    old_argv, old_stdout = sys.argv, sys.stdout
    with open(out / "train_log.txt", "w", encoding="utf-8") as log_fh:
        sys.argv, sys.stdout = argv, Tee(old_stdout, log_fh)
        try:
            with recording_constructors(registry):
                tc.main()
        finally:
            sys.argv, sys.stdout = old_argv, old_stdout
            metrics_file.close()

    # ---------- reproduction checks ----------
    new_log = parse_log(out / "train_log.txt")
    log_cmp, first_mismatch = {}, None
    for e in range(1, 11):
        same = new_log.get(e) == orig_log[e]
        log_cmp[e] = {"original": orig_log[e], "new": new_log.get(e), "identical_at_printed_precision": same}
        if not same and first_mismatch is None:
            first_mismatch = e
    new = torch.load(new_ckpt, map_location="cpu", weights_only=False)
    epoch5 = torch.load(out / "epoch_05_model_state.pt", map_location="cpu", weights_only=False)
    repro = {
        "arm": arm,
        "log_consistent_at_printed_precision": first_mismatch is None,
        "first_log_mismatch_epoch": first_mismatch,
        "note": "Matching rounded log values is consistency, not proof of exact reproduction.",
        "epoch5_weights_vs_original_checkpoint": tensor_diffs(orig["model_state_dict"], epoch5),
        "original_selected_epoch": orig["epoch"], "new_selected_epoch": new["epoch"],
        "per_epoch_log": log_cmp,
    }
    if new["epoch"] == orig["epoch"]:
        repro["selected_checkpoint_vs_original"] = {
            "model": tensor_diffs(orig["model_state_dict"], new["model_state_dict"]),
            "optimizer_state_identical": deep_equal(orig["optimizer_state_dict"], new["optimizer_state_dict"]),
            "scheduler_state_identical": deep_equal(orig["scheduler_state_dict"], new["scheduler_state_dict"]),
            "scaler_state_identical": deep_equal(orig["scaler_state_dict"], new["scaler_state_dict"]),
            "train_loss_identical": orig["train_loss"] == new["train_loss"],
            "validation_identical": orig["validation"] == new["validation"],
        }
    repro["original_artifacts_unchanged"] = all(
        diag.sha_file(Path(p)) == h for p, h in orig_sha.items())
    if not repro["original_artifacts_unchanged"]:
        raise RuntimeError("An original artifact changed on disk.")
    (out / "reproduction.json").write_text(json.dumps(repro, indent=1))

    # ---------- swing / dip summary with the fixed criteria ----------
    recs = {r["epoch"]: r for r in state["records"]}
    summary = {"arm": arm, "window": WINDOW, "acc_dip_threshold": ACC_DIP,
               "f1_dip_threshold": F1_DIP, "swing_reduction_threshold": SWING_REDUCTION}
    for metric, thr in (("accuracy", ACC_DIP), ("macro_f1", F1_DIP)):
        o = series_stats({e: recs[e]["ordinary"][metric] for e in recs}, thr)
        c = series_stats({e: recs[e]["recalibrated"][metric] for e in recs}, thr)
        ratio = c["mean_abs_delta"] / o["mean_abs_delta"] if o["mean_abs_delta"] > 0 else None
        summary[metric] = {
            "ordinary": o, "recalibrated": c, "swing_ratio_recal_over_ordinary": ratio,
            "swing_reduced": ratio is not None and ratio <= SWING_REDUCTION,
            "ordinary_dips_absent_in_recalibrated": not (set(o["dips"]) & set(c["dips"])),
        }
        # Not applicable (None) when the ordinary series has no dips to remove.
        summary[metric]["meets_bn_artifact_criterion"] = (
            None if not o["dips"] else
            summary[metric]["swing_reduced"] and summary[metric]["ordinary_dips_absent_in_recalibrated"])
    summary["scope"] = ("One arm, 5% labels, seed 42. The cross-arm interpretation requires both "
                        "arms; selection or gap differences under recalibration are exploratory.")
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    lines = [f"Arm: {arm}   output: {out}",
             f"Log consistent at printed precision: {repro['log_consistent_at_printed_precision']} "
             f"(first mismatch: {first_mismatch})",
             f"Epoch-5 weights bitwise identical to original: "
             f"{repro['epoch5_weights_vs_original_checkpoint']['bitwise_identical']} "
             f"(max abs {repro['epoch5_weights_vs_original_checkpoint']['max_abs_diff']:.3g}, "
             f"max rel {repro['epoch5_weights_vs_original_checkpoint']['max_rel_diff']:.3g})",
             f"Selected epoch: original {orig['epoch']}, new {new['epoch']}", "",
             f"{'ep':>3} {'ord acc':>8} {'rec acc':>8} {'ord F1':>8} {'rec F1':>8} {'dW/W':>8}"]
    for e in sorted(recs):
        r = recs[e]
        dw = "-" if r["relative_weight_change"] is None else f"{r['relative_weight_change']:.4f}"
        lines.append(f"{e:>3} {r['ordinary']['accuracy']:8.4f} {r['recalibrated']['accuracy']:8.4f} "
                     f"{r['ordinary']['macro_f1']:8.4f} {r['recalibrated']['macro_f1']:8.4f} {dw:>8}")
    for metric in ("accuracy", "macro_f1"):
        s = summary[metric]
        lines.append(f"{metric}: mean|d| ord {s['ordinary']['mean_abs_delta']:.4f} rec "
                     f"{s['recalibrated']['mean_abs_delta']:.4f}; dips ord {s['ordinary']['dips']} "
                     f"rec {s['recalibrated']['dips']}; criterion met: {s['meets_bn_artifact_criterion']}")
    (out / "summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
