"""Exploratory low-learning-rate extension of a finished classifier fine-tuning run.

This branches from the *saved best* checkpoint of a train_classifier.py run (for the
intended use: epoch 9 of 10). The original epoch-10 state was never saved and is not
used. Model, AdamW and GradScaler states are restored exactly; the run then trains for
at most 3 more epochs at a deliberately chosen low learning rate.

Nothing in experiments/train_classifier.py is changed. Its seed_everything() and
evaluate() are imported unchanged, and the model, data split, label subset, dataset
class, preprocessing, DataLoader settings, AMP, loss, gradient clipping and
validation-based selection are reproduced from it.

Order of operations
  1. Load the checkpoint and check its config is a plain train_classifier.py run.
  2. Rebuild IQCNNClassifier(24, 128, 64) and load the full model state (strict).
  3. Rebuild AdamW exactly as the original. Check the parameter groups, parameter
     count and per-parameter state shapes before loading the optimizer state.
  4. Load the GradScaler state. Check the saved cosine-scheduler state for
     consistency (T_max, last_epoch, last lr), but do NOT step it beyond T_max.
  5. Evaluate the restored model on the full validation split before any update.
     Abort (exit code 2) unless the saved validation macro-F1 is reproduced within
     --f1-tolerance and accuracy within --acc-tolerance.
  6. Train for --extra-epochs (1-3) with a full validation pass after each epoch.
     Save a checkpoint whenever validation macro-F1 beats the starting best.
The test split is never loaded.

Learning-rate extension (documented here and recorded in every output):
  lr_k = lr0 * 0.5 * (1 + cos(pi * k / E)),  k = 0 .. E-1,  E = --extra-epochs
  where lr0 is the lr stored in the checkpoint's optimizer: 2.4472e-05, the lr the
  original schedule would have used for epoch 10. For E = 3 this gives
  2.447e-05, 1.836e-05, 6.118e-06. It is held constant within an epoch, as in the
  original per-epoch cosine schedule.

Shuffle order: the original DataLoader shuffled with torch.Generator().manual_seed(seed)
and nothing else drew from it. The script replays the generator through the
<starting epoch> completed epochs (on an index-only dataset of the same length), so
extension epoch k uses the order the original would have used for epoch
<starting epoch> + k. This replay cannot be verified from saved artifacts. Python,
NumPy, global torch and CUDA RNG states were not saved, but they are not drawn during
training (no dropout or augmentation). cuDNN/AMP kernels are not deterministic, so
exact reproduction of the original trajectory is not possible; treat this as an
exploratory extension.

Examples:
  # restore and compatibility checks only (no data, no training)
  python experiments/resume_classifier.py --check-only
  # bounded run (full validation x 4, 3 training epochs)
  python -u experiments/resume_classifier.py --extra-epochs 3 ^
      --output-dir C:\\dlruns\\continued\\mae_fullpretrain_1pct_seed42_ext3
"""

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.radioml_dataset import RadioMLDataset  # noqa: E402
from src.models.cnn import IQCNNClassifier  # noqa: E402

# Reuse the original training script's helpers unchanged (it only defines them).
_spec = importlib.util.spec_from_file_location(
    "train_classifier", ROOT / "experiments" / "train_classifier.py")
train_classifier = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_classifier)
seed_everything = train_classifier.seed_everything
evaluate = train_classifier.evaluate

DEFAULT_CHECKPOINT = Path(r"C:\dlruns\classifier_mae_fullpretrain_1pct_seed42.pt")
DEFAULT_OUTPUT = Path(r"C:\dlruns\continued\mae_fullpretrain_1pct_seed42_ext3")
MAX_EXTRA_EPOCHS = 3

# The checkpoint must come from a plain train_classifier.py run with these settings;
# anything else would make "the same training" ambiguous.
EXPECTED_CONFIG = {
    "label_fraction": "1", "batch_size": 128, "lr": 1e-3, "weight_decay": 1e-4,
    "encoder_lr": None, "seed": 42, "max_train_batches": 0,
    "validation_subset": None, "max_val_batches": 0, "epochs": 10,
}


class _IndexOnly(Dataset):
    """Same length as the training set; used only to replay the shuffle generator."""

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return i


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 22):
            h.update(block)
    return h.hexdigest()


def extension_lrs(lr0, epochs):
    return [lr0 * 0.5 * (1.0 + math.cos(math.pi * k / epochs)) for k in range(epochs)]


def fail(msg):
    raise SystemExit(f"ABORT: {msg}")


def restore(checkpoint_path, device):
    """Rebuild model, optimizer and scaler from the checkpoint, checking everything.
    Returns (model, optimizer, scaler, checkpoint, report)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    for key in ("epoch", "model_state_dict", "optimizer_state_dict",
                "scheduler_state_dict", "scaler_state_dict", "validation",
                "best_val_macro_f1", "config", "initialization"):
        if key not in ckpt:
            fail(f"checkpoint lacks '{key}'")
    cfg = ckpt["config"]
    for k, v in EXPECTED_CONFIG.items():
        if cfg.get(k) != v:
            fail(f"checkpoint config {k}={cfg.get(k)!r}, expected {v!r}")
    if ckpt["initialization"] != "mae_pretrained":
        fail(f"initialization is {ckpt['initialization']!r}, expected 'mae_pretrained'")
    start_epoch = int(ckpt["epoch"])
    if not 1 <= start_epoch <= cfg["epochs"]:
        fail(f"checkpoint epoch {start_epoch} outside 1..{cfg['epochs']}")
    if ckpt["validation"]["macro_f1"] != ckpt["best_val_macro_f1"]:
        fail("saved validation macro-F1 differs from best_val_macro_f1")

    # Model: same constructor call as train_classifier.py; full strict load.
    model = IQCNNClassifier(num_classes=24, feature_dim=128, patch_size=64)
    state = ckpt["model_state_dict"]
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        fail(f"model state mismatch: {result}")
    for k, v in model.state_dict().items():
        if not torch.equal(v, state[k]):
            fail(f"model tensor {k} differs after loading")
    model = model.to(device)

    # Optimizer: same construction as the original single-group AdamW.
    params = list(model.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    saved = ckpt["optimizer_state_dict"]
    groups = saved["param_groups"]
    if len(groups) != len(optimizer.param_groups) or len(groups) != 1:
        fail(f"optimizer has {len(groups)} saved groups, expected 1")
    g = groups[0]
    if len(g["params"]) != len(params):
        fail(f"saved group has {len(g['params'])} params, model has {len(params)}")
    for name in ("betas", "eps", "weight_decay", "amsgrad", "maximize"):
        if g[name] != optimizer.param_groups[0][name]:
            fail(f"optimizer hyperparameter {name}: saved {g[name]!r} "
                 f"vs rebuilt {optimizer.param_groups[0][name]!r}")
    if g.get("initial_lr") != cfg["lr"]:
        fail(f"saved initial_lr {g.get('initial_lr')} != config lr {cfg['lr']}")
    steps = set()
    for pid, p in zip(g["params"], params):
        st = saved["state"].get(pid)
        if st is None:
            fail(f"no optimizer state for parameter {pid}")
        if st["exp_avg"].shape != p.shape or st["exp_avg_sq"].shape != p.shape:
            fail(f"optimizer state shape mismatch for parameter {pid}")
        steps.add(float(st["step"]))
    if len(steps) != 1:
        fail(f"inconsistent AdamW step counts {steps}")
    optimizer.load_state_dict(saved)
    for pid, p in zip(g["params"], params):
        restored = optimizer.state[p]
        if not torch.equal(restored["exp_avg"].cpu(), saved["state"][pid]["exp_avg"]):
            fail(f"exp_avg not restored for parameter {pid}")

    # Scheduler: consistency check only, never stepped past T_max.
    sched = ckpt["scheduler_state_dict"]
    lr0 = float(optimizer.param_groups[0]["lr"])
    if sched.get("T_max") != cfg["epochs"] or sched.get("last_epoch") != start_epoch:
        fail(f"scheduler state T_max={sched.get('T_max')} last_epoch={sched.get('last_epoch')} "
             f"inconsistent with config epochs={cfg['epochs']} / checkpoint epoch={start_epoch}")
    expected_lr0 = 0.5 * cfg["lr"] * (1 + math.cos(math.pi * start_epoch / cfg["epochs"]))
    if not math.isclose(lr0, expected_lr0, rel_tol=1e-9) or not math.isclose(
            lr0, sched["_last_lr"][0], rel_tol=1e-12):
        fail(f"saved lr {lr0} inconsistent with cosine schedule ({expected_lr0})")

    # GradScaler.
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    scaler.load_state_dict(ckpt["scaler_state_dict"])
    if device.type == "cuda" and scaler.get_scale() != ckpt["scaler_state_dict"]["scale"]:
        fail("GradScaler scale not restored")

    report = {
        "start_epoch": start_epoch,
        "model_tensors_loaded": len(state),
        "optimizer_params": len(params),
        "adamw_step": steps.pop(),
        "saved_lr": lr0,
        "saved_scheduler": {k: sched[k] for k in ("T_max", "eta_min", "base_lrs", "last_epoch", "_last_lr")},
        "scaler_scale": ckpt["scaler_state_dict"]["scale"],
        "classifier_head_shape": list(state["classifier.weight"].shape),
    }
    return model, optimizer, scaler, ckpt, report


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--extra-epochs", type=int, default=3)
    ap.add_argument("--workers", type=int, default=0,
                    help="DataLoader workers (original run: 0). Does not change data order or values.")
    ap.add_argument("--f1-tolerance", type=float, default=1e-4,
                    help="Max |reproduced - saved| validation macro-F1 before training.")
    ap.add_argument("--acc-tolerance", type=float, default=1e-4)
    ap.add_argument("--check-only", action="store_true",
                    help="Run restore/compatibility checks and print the plan; no data, no training.")
    args = ap.parse_args()

    if not 1 <= args.extra_epochs <= MAX_EXTRA_EPOCHS:
        fail(f"--extra-epochs must be 1..{MAX_EXTRA_EPOCHS}")
    ckpt_path = Path(args.checkpoint).resolve()
    out_dir = Path(args.output_dir).resolve()
    if not ckpt_path.is_file():
        fail(f"checkpoint not found: {ckpt_path}")
    if out_dir == ckpt_path.parent or ckpt_path.is_relative_to(out_dir):
        fail("--output-dir must be a new directory that does not contain the starting checkpoint")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    start_sha = sha256_file(ckpt_path)
    model, optimizer, scaler, ckpt, report = restore(ckpt_path, device)
    cfg = ckpt["config"]
    lrs = extension_lrs(report["saved_lr"], args.extra_epochs)
    schedule = {
        "type": "cosine decay over the extension, per epoch, from the saved lr",
        "formula": "lr_k = lr0 * 0.5 * (1 + cos(pi * k / E)), k = 0..E-1",
        "lr0": report["saved_lr"], "E": args.extra_epochs, "lrs": lrs,
        "original_scheduler_stepped": False,
    }
    print(f"Restored epoch-{report['start_epoch']} best checkpoint (val macro-F1 "
          f"{ckpt['best_val_macro_f1']:.6f}); model {report['model_tensors_loaded']} tensors, "
          f"AdamW {report['optimizer_params']} params at step {report['adamw_step']:.0f}, "
          f"lr {report['saved_lr']:.4e}, GradScaler scale {report['scaler_scale']:g}")
    print("Saved scheduler (checked, not continued):", report["saved_scheduler"])
    print("Extension lrs:", ", ".join(f"{x:.4e}" for x in lrs))
    if args.check_only:
        print("Check-only: all restore checks passed; no data loaded, nothing trained.")
        return

    if out_dir.exists() and any(out_dir.iterdir()):
        fail(f"{out_dir} exists and is not empty; refusing to overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "resume_log.json"

    # Same index files, dataset class and loader settings as train_classifier.py.
    with np.load(ROOT / "label_subsets_v2.npz") as s:
        train_indices = s[f"train_{cfg['label_fraction']}pct"].copy()
    with np.load(ROOT / "data_splits.npz") as s:
        val_indices = s["val_idx"].copy()
    if len(val_indices) != ckpt["validation"]["samples"]:
        fail(f"validation split has {len(val_indices)} samples, checkpoint recorded "
             f"{ckpt['validation']['samples']}")
    train_ds, val_ds = RadioMLDataset(train_indices), RadioMLDataset(val_indices)

    seed_everything(cfg["seed"])
    generator = torch.Generator().manual_seed(cfg["seed"])
    # Replay the shuffle stream through the completed epochs (index-only, no data read).
    replay = DataLoader(_IndexOnly(len(train_ds)), batch_size=cfg["batch_size"],
                        shuffle=True, generator=generator)
    for _ in range(report["start_epoch"]):
        for _ in replay:
            pass
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=args.workers, pin_memory=pin, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=args.workers, pin_memory=pin)

    log = {
        "exploratory_extension": True,
        "starting_checkpoint": str(ckpt_path),
        "starting_checkpoint_sha256": start_sha,
        "branch": (f"from the saved best checkpoint (epoch {report['start_epoch']} of "
                   f"{cfg['epochs']}); the original epoch-{cfg['epochs']} state was not saved and is not used"),
        "original_config": cfg,
        "restore_report": report,
        "extension_schedule": schedule,
        "optimizer": {"type": "AdamW", "weight_decay": cfg["weight_decay"],
                      "betas": optimizer.param_groups[0]["betas"], "eps": optimizer.param_groups[0]["eps"],
                      "state": "restored from checkpoint"},
        "seed": cfg["seed"],
        "shuffle": "torch.Generator(seed) replayed through the completed epochs (unverifiable)",
        "reproducibility": ("Python/NumPy/torch/CUDA RNG states were not saved; cuDNN and AMP "
                            "are not deterministic. Not an exact continuation of the original run."),
        "test_split_used": False,
        "tolerances": {"macro_f1": args.f1_tolerance, "accuracy": args.acc_tolerance},
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "epochs": [],
    }

    def write_log():
        log_path.write_text(json.dumps(log, indent=2))

    # Baseline validation before any optimizer update.
    t0 = time.time()
    base = evaluate(model, val_loader, device)
    saved = ckpt["validation"]
    d_f1 = abs(base["macro_f1"] - saved["macro_f1"])
    d_acc = abs(base["accuracy"] - saved["accuracy"])
    ok = (base["samples"] == saved["samples"] and d_f1 <= args.f1_tolerance
          and d_acc <= args.acc_tolerance)
    log["baseline_validation"] = {**base, "saved": saved, "abs_diff_macro_f1": d_f1,
                                  "abs_diff_accuracy": d_acc, "passed": ok,
                                  "seconds": time.time() - t0}
    write_log()
    print(f"Baseline validation: macro-F1 {base['macro_f1']:.6f} (saved {saved['macro_f1']:.6f}, "
          f"|diff| {d_f1:.2e}), accuracy {base['accuracy']:.6f} (|diff| {d_acc:.2e}), "
          f"loss {base['loss']:.6f} (saved {saved['loss']:.6f}); {time.time() - t0:.0f}s")
    if not ok:
        print("ABORT: restored model does not reproduce the saved validation metrics; "
              "no optimizer update was made.")
        sys.exit(2)

    best_f1 = float(ckpt["best_val_macro_f1"])
    for k, lr in enumerate(lrs, start=1):
        global_epoch = report["start_epoch"] + k
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        t0 = time.time()
        total_loss, total_samples = 0.0, 0
        for batch_idx, batch in enumerate(train_loader):  # loop body as in train_classifier.py
            signals = batch["signal"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=(device.type == "cuda")):
                logits = model(signals)
                loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at batch {batch_idx}.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            n = labels.size(0)
            total_loss += loss.detach().float().item() * n
            total_samples += n
        train_seconds = time.time() - t0
        val = evaluate(model, val_loader, device)
        improved = val["macro_f1"] > best_f1
        row = {"extension_epoch": k, "global_epoch": global_epoch, "lr": lr,
               "train_loss": total_loss / total_samples, "train_samples": total_samples,
               "validation": val, "improved": improved,
               "train_seconds": train_seconds, "seconds": time.time() - t0}
        print(f"Extension epoch {k}/{len(lrs)} (global {global_epoch}) | lr {lr:.4e} | "
              f"train loss {row['train_loss']:.6f} | val loss {val['loss']:.6f} | "
              f"val acc {val['accuracy']:.4f} | val macro-F1 {val['macro_f1']:.4f}"
              f"{' (improved)' if improved else ''} | {row['seconds']:.0f}s")
        if improved:
            best_f1 = val["macro_f1"]
            path = out_dir / f"best_ext_epoch{k}.pt"
            torch.save({
                "epoch": global_epoch, "extension_epoch": k,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": None, "extension_schedule": schedule,
                "scaler_state_dict": scaler.state_dict(),
                "train_loss": row["train_loss"], "validation": val,
                "best_val_macro_f1": best_f1, "config": cfg,
                "initialization": ckpt["initialization"],
                "starting_checkpoint": str(ckpt_path), "starting_checkpoint_sha256": start_sha,
                "starting_epoch": report["start_epoch"], "exploratory_extension": True,
            }, path)
            row["checkpoint"] = str(path)
        log["epochs"].append(row)
        write_log()

    log["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    log["best_val_macro_f1"] = best_f1
    log["best"] = (next((r["checkpoint"] for r in reversed(log["epochs"]) if r.get("checkpoint")), None)
                   or f"no improvement; starting checkpoint remains best ({ckpt_path.name})")
    write_log()
    print(f"Best validation macro-F1: {best_f1:.6f} -> {log['best']}")
    print("Log:", log_path)


if __name__ == "__main__":
    main()
