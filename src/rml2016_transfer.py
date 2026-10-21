"""Shared code for the RML2018 -> RML2016 cross-dataset transfer experiment.

Two arms, identical except for encoder initialisation:
  transfer  encoder from the RML2018 CNN-MAE checkpoint, fresh 11-class head
  scratch   whole classifier randomly initialised

Training mirrors experiments/train_classifier.py (AdamW, cosine by epoch, AMP
fp16, grad-clip 1.0, checkpoint = best validation macro-F1, no early stopping).
The training phase never receives test indices; test evaluation is a separate
function that only accepts runs whose training finished.
"""

import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.data import rml2016
from src.models.cnn import IQCNNClassifier

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRETRAINED = Path(r"C:\dlruns\cnn_mae_full_v1_best.pt")
DEFAULT_SPLITS = ROOT / "experiments" / "rml2016_transfer" / "splits_seed42.npz"
DEFAULT_PROTOCOL = ROOT / "experiments" / "rml2016_transfer" / "protocol_seed42.json"

ARMS = ("transfer", "scratch")
MODEL_CONFIG = {
    "num_classes": len(rml2016.CLASSES),
    "feature_dim": 128,
    "patch_size": 64,  # same patch length the encoder was pretrained on
    "signal_length": rml2016.SIGNAL_SHAPE[1],
}
# Matched to experiments/train_classifier.py defaults used for RML2018.
TRAIN_DEFAULTS = {
    "epochs": 10,
    "batch_size": 128,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealingLR(T_max=epochs), stepped per epoch",
    "grad_clip_norm": 1.0,
    "amp": "float16 autocast + GradScaler on CUDA",
    "selection": "best validation macro-F1 over all epochs (strictly greater)",
    "early_stopping": False,
}
ENCODER_PREFIX = "encoder."
_NUMPY_SAFE_GLOBALS = [np._core.multiarray._reconstruct, np.ndarray, np.dtype,
                       type(np.dtype(np.uint32))]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model():
    return IQCNNClassifier(**MODEL_CONFIG)


def sha256_file(path):
    return rml2016.sha256_file(path)


# --------------------------------------------------------------------------
# Pretrained encoder loading
# --------------------------------------------------------------------------

def load_pretrained_encoder(model, checkpoint_path):
    """Load every encoder tensor from a CNN-MAE checkpoint (strict).

    Only `encoder.*` tensors are taken; everything else in the checkpoint is
    MAE-specific (positional embeddings, mask token, decoder, reconstruction
    head) and is listed as intentionally excluded. The classifier head is never
    touched. Raises if any encoder tensor is missing, unexpected or misshaped.
    """
    checkpoint_path = Path(checkpoint_path)
    # MAE checkpoints also store numpy RNG state; allow just the numpy types
    # needed to rebuild those arrays so the load stays weights_only.
    with torch.serialization.safe_globals(_NUMPY_SAFE_GLOBALS):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint["model_state_dict"]
    config = checkpoint.get("config", {})

    if config.get("feature_dim") != MODEL_CONFIG["feature_dim"]:
        raise ValueError(f"Checkpoint feature_dim {config.get('feature_dim')} "
                         f"!= {MODEL_CONFIG['feature_dim']}")
    if config.get("patch_size") != MODEL_CONFIG["patch_size"]:
        raise ValueError(f"Checkpoint patch_size {config.get('patch_size')} "
                         f"!= {MODEL_CONFIG['patch_size']}")
    if any(k.startswith("classifier.") for k in state):
        raise ValueError("Checkpoint contains a classifier head; expected an MAE "
                         "checkpoint. Refusing to risk reusing a 24-class head.")

    encoder_state = {k[len(ENCODER_PREFIX):]: v for k, v in state.items()
                     if k.startswith(ENCODER_PREFIX)}
    excluded = sorted(k for k in state if not k.startswith(ENCODER_PREFIX))
    expected = model.encoder.state_dict()
    if not encoder_state:
        raise ValueError("No encoder tensors in checkpoint.")
    for name, tensor in encoder_state.items():
        if name in expected and tensor.shape != expected[name].shape:
            raise ValueError(f"Shape mismatch for encoder.{name}: "
                             f"{tuple(tensor.shape)} vs {tuple(expected[name].shape)}")

    result = model.encoder.load_state_dict(encoder_state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"Encoder load incomplete: {result}")
    if len(encoder_state) != len(expected):
        raise RuntimeError("Encoder tensor count mismatch.")

    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_val_mse": checkpoint.get("best_val"),
        "pretraining_config": config,
        "loaded_encoder_tensors": len(encoder_state),
        "model_encoder_tensors": len(expected),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
        "excluded_mae_only_keys": excluded,
        "head": "fresh nn.Linear(128, 11), not loaded from checkpoint",
    }


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load_data(pkl_path, splits_path, protocol_path, device, include_test):
    """Load RML2016, apply the protocol's fixed amplitude multiplier, verify
    hashes. test_idx is only returned when include_test is True."""
    protocol = json.loads(Path(protocol_path).read_text())
    pkl_path = Path(pkl_path)
    if pkl_path.stat().st_size != protocol["data_file"]["bytes"]:
        raise ValueError("RML2016 file size differs from the protocol.")
    data_sha = sha256_file(pkl_path)
    if data_sha != protocol["data_file"]["sha256"]:
        raise ValueError("RML2016 file SHA-256 differs from the protocol.")

    splits = rml2016.load_splits(splits_path)
    if rml2016.split_digest(splits) != protocol["split_sha256"]:
        raise ValueError("Split file does not match the protocol split hash.")

    x, y, snr = rml2016.load_rml2016(pkl_path)
    rml2016.check_splits(splits, len(x))
    multiplier = float(protocol["amplitude"]["multiplier"])
    x_dev = torch.from_numpy(x).to(device) * multiplier

    keep = ["train_idx", "val_idx"] + [f"train_{p}pct" for p in rml2016.LABEL_BUDGETS]
    if include_test:
        keep.append("test_idx")
    return {
        "x": x_dev,
        "y": torch.from_numpy(y).to(device),
        "snr": snr,
        "y_np": y,
        "splits": {k: splits[k] for k in keep},
        "multiplier": multiplier,
        "data_sha256": data_sha,
        "data_bytes": pkl_path.stat().st_size,
        "split_sha256": protocol["split_sha256"],
        "split_file_sha256": sha256_file(splits_path),
    }


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def predict(model, x, indices, device, batch_size):
    model.eval()
    preds, all_logits = [], []
    idx_t = torch.as_tensor(indices, device=device)
    with torch.no_grad():
        for s in range(0, len(indices), batch_size):
            batch = idx_t[s:s + batch_size]
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=(device.type == "cuda")):
                logits = model(x[batch])
            preds.append(logits.argmax(dim=1))
            all_logits.append(logits.float())
    logits = torch.cat(all_logits)
    return torch.cat(preds).cpu().numpy(), logits


def full_metrics(y_true, y_pred, snr):
    labels = list(range(len(rml2016.CLASSES)))
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_snr = {}
    for s in sorted(np.unique(snr).tolist()):
        m = snr == s
        per_snr[str(s)] = {
            "n": int(m.sum()),
            "accuracy": float(accuracy_score(y_true[m], y_pred[m])),
            "macro_f1": float(f1_score(y_true[m], y_pred[m], labels=labels,
                                       average="macro", zero_division=0)),
        }
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels,
                                   average="macro", zero_division=0)),
        "per_class_f1": {c: float(v) for c, v in zip(rml2016.CLASSES, per_class)},
        "per_snr": per_snr,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "class_order": list(rml2016.CLASSES),
    }


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------

def git_state():
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                                  text=True, timeout=30).stdout.strip()
        except Exception:
            return None
    status = run("status", "--porcelain")
    return {"commit": run("rev-parse", "HEAD"),
            "dirty": bool(status) if status is not None else None}


def environment():
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "platform": platform.platform(),
    }


def run_name(arm, budget, seed):
    return f"{arm}_{budget}pct_seed{seed}"


def run_config(arm, budget, seed, epochs, max_train_batches, pretrained, pilot):
    """Config for one run. Both arms get the same dict apart from `arm` and
    `init_checkpoint`; summarize checks this."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    if budget not in rml2016.LABEL_BUDGETS:
        raise ValueError(f"budget must be one of {rml2016.LABEL_BUDGETS}")
    train = dict(TRAIN_DEFAULTS, epochs=epochs)
    return {
        "arm": arm,
        "budget_pct": budget,
        "seed": seed,
        "init_checkpoint": str(pretrained) if arm == "transfer" else None,
        "model": dict(MODEL_CONFIG),
        "train": train,
        "max_train_batches": max_train_batches,
        "preprocessing": "x * protocol amplitude multiplier (global, from RML2016 train split)",
        "pilot": bool(pilot),
    }


# --------------------------------------------------------------------------
# Training (validation only) and test evaluation (separate)
# --------------------------------------------------------------------------

def train_run(cfg, data, run_dir, device, log=print):
    run_dir = Path(run_dir)
    ckpt_path = run_dir / "best.pt"
    record_path = run_dir / "train.json"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"{run_dir} is not empty; refusing to overwrite.")
    run_dir.mkdir(parents=True, exist_ok=True)
    if "test_idx" in data["splits"]:
        raise RuntimeError("Training must not receive test indices.")

    started = time.strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.time()
    seed = cfg["seed"]
    tc = cfg["train"]

    seed_everything(seed)
    model = build_model()  # same RNG draws in both arms -> identical head init
    head_init_sha = hashlib.sha256(
        model.classifier.weight.detach().numpy().tobytes()).hexdigest()
    load_report = None
    if cfg["arm"] == "transfer":
        load_report = load_pretrained_encoder(model, cfg["init_checkpoint"])
        log(f"[load] {load_report['loaded_encoder_tensors']}/"
            f"{load_report['model_encoder_tensors']} encoder tensors loaded strictly; "
            f"missing={load_report['missing_keys']} unexpected={load_report['unexpected_keys']}; "
            f"excluded {len(load_report['excluded_mae_only_keys'])} MAE-only tensors")
    else:
        log("[load] scratch: encoder and head randomly initialised")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=tc["lr"],
                                  weight_decay=tc["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tc["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    generator = torch.Generator().manual_seed(seed)

    train_idx = torch.as_tensor(data["splits"][f"train_{cfg['budget_pct']}pct"], device=device)
    val_idx = data["splits"]["val_idx"]
    x, y = data["x"], data["y"]
    y_val = data["y_np"][val_idx]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    history, best_f1, best_epoch = [], -1.0, 0
    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        e0 = time.time()
        perm = torch.randperm(len(train_idx), generator=generator).to(device)
        total_loss, total_n, batches = 0.0, 0, 0
        for start in range(0, len(perm), tc["batch_size"]):
            if cfg["max_train_batches"] and batches >= cfg["max_train_batches"]:
                break
            batch = train_idx[perm[start:start + tc["batch_size"]]]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=(device.type == "cuda")):
                logits = model(x[batch])
                loss = F.cross_entropy(logits, y[batch])
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}, batch {batches}.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip_norm"])
            scaler.step(optimizer)
            scaler.update()
            n = len(batch)
            total_loss += loss.detach().float().item() * n
            total_n += n
            batches += 1
        if total_n == 0:
            raise RuntimeError("No training batches processed.")

        val_pred, val_logits = predict(model, x, val_idx, device, tc["batch_size"])
        val_loss = float(F.cross_entropy(val_logits, y[torch.as_tensor(val_idx, device=device)]))
        if not np.isfinite(val_loss):
            raise RuntimeError("Non-finite validation loss.")
        val_acc = float(accuracy_score(y_val, val_pred))
        val_f1 = float(f1_score(y_val, val_pred, average="macro", zero_division=0))
        scheduler.step()

        row = {"epoch": epoch, "train_loss": total_loss / total_n,
               "train_samples": total_n, "train_batches": batches,
               "val_loss": val_loss, "val_accuracy": val_acc, "val_macro_f1": val_f1,
               "seconds": time.time() - e0}
        history.append(row)
        log(f"Epoch {epoch}/{tc['epochs']} | train loss={row['train_loss']:.6f} "
            f"({total_n} samples) | val loss={val_loss:.6f} | val acc={val_acc:.4f} "
            f"| val macro-F1={val_f1:.4f} | {row['seconds']:.1f}s")

        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "config": cfg, "validation": row}, ckpt_path)

    peak = (torch.cuda.max_memory_allocated(device) / 2**20
            if device.type == "cuda" else None)
    record = {
        "run": run_name(cfg["arm"], cfg["budget_pct"], cfg["seed"]),
        "config": cfg,
        "complete": True,
        "selection": {"metric": "val_macro_f1", "best_epoch": best_epoch,
                      "best_val_macro_f1": best_f1,
                      "test_used_for_selection": False},
        "history": history,
        "encoder_load": load_report,
        "head_init_weight_sha256": head_init_sha,
        "labeled_train_examples": int(len(train_idx)),
        "val_examples": int(len(val_idx)),
        "best_checkpoint": str(ckpt_path),
        "best_checkpoint_sha256": sha256_file(ckpt_path),
        "manifest": {
            "git": git_state(),
            "pretrained_checkpoint": cfg["init_checkpoint"],
            "pretrained_checkpoint_sha256": (load_report["checkpoint_sha256"]
                                             if load_report else None),
            "environment": environment(),
            "data_bytes": data["data_bytes"],
            "data_sha256": data["data_sha256"],
            "split_sha256": data["split_sha256"],
            "split_file_sha256": data["split_file_sha256"],
            "amplitude_multiplier": data["multiplier"],
            "started": started,
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seconds": time.time() - t0,
            "peak_gpu_memory_mib": peak,
            "argv": sys.argv,
        },
    }
    record_path.write_text(json.dumps(record, indent=2))
    log(f"Best val macro-F1 {best_f1:.4f} at epoch {best_epoch}; "
        f"{record['manifest']['seconds']:.1f}s; peak GPU {peak and round(peak, 1)} MiB")
    return record


def evaluate_test(run_dir, data, device, batch_size=512):
    """Evaluate a finished run's selected checkpoint once on the test split."""
    run_dir = Path(run_dir)
    record = json.loads((run_dir / "train.json").read_text())
    if not record.get("complete"):
        raise RuntimeError(f"{run_dir}: training not complete; refusing test evaluation.")
    out_path = run_dir / "test.json"
    if out_path.exists():
        raise FileExistsError(f"{out_path} exists; test is evaluated once per run.")
    ckpt_path = run_dir / "best.pt"
    if sha256_file(ckpt_path) != record["best_checkpoint_sha256"]:
        raise RuntimeError(f"{ckpt_path} changed after selection.")
    if record["manifest"]["amplitude_multiplier"] != data["multiplier"]:
        raise RuntimeError("Amplitude multiplier differs from training.")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device)

    test_idx = data["splits"]["test_idx"]
    pred, _ = predict(model, data["x"], test_idx, device, batch_size)
    metrics = full_metrics(data["y_np"][test_idx], pred, data["snr"][test_idx])
    result = {
        "run": record["run"],
        "config": record["config"],
        "selected_epoch": checkpoint["epoch"],
        "best_val_macro_f1": record["selection"]["best_val_macro_f1"],
        "evaluated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test": metrics,
    }
    out_path.write_text(json.dumps(result, indent=2))
    np.save(run_dir / "test_pred.npy", pred.astype(np.int8))
    return result
