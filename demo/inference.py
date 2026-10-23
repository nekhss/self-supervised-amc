"""Model discovery, checkpoint loading and single-signal inference.

Uses the project's IQCNNClassifier with the configuration used for every primary run
(24 classes, feature_dim 128, patch_size 64). Inference mirrors experiments/evaluate_test.py:
model.eval(), no gradients, FP16 autocast on CUDA and FP32 on CPU, raw [2, 1024] IQ input.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

import demo_config as cfg


class ModelUnavailable(RuntimeError):
    pass


@dataclass
class ModelSpec:
    key: str
    label: str
    arm: str
    budget: int
    seed: int
    path: Path
    kind: str                      # "primary" or "supplementary"
    result_json: Optional[Path] = None
    pred_file: Optional[Path] = None
    note: str = ""


@dataclass
class Prediction:
    pred: int
    probs: np.ndarray
    latency_ms: float
    latency_median_ms: float
    device: str
    precision: str
    stored_pred: Optional[int] = None


@dataclass
class LoadedModel:
    spec: ModelSpec
    model: object
    device: str
    meta: dict = field(default_factory=dict)


def discover_models():
    """Return (available specs, list of missing primary checkpoint paths)."""
    specs, missing = [], []
    for budget, seeds in cfg.PLAN.items():
        for seed in seeds:
            for arm, name in cfg.ARMS.items():
                p = cfg.primary_ckpt(arm, budget, seed)
                if p.is_file():
                    specs.append(ModelSpec(
                        key=f"{arm}_{budget}_{seed}", label=f"{name} · {budget}% labels · seed {seed}",
                        arm=arm, budget=budget, seed=seed, path=p, kind="primary",
                        result_json=cfg.primary_result(arm, budget, seed),
                        pred_file=cfg.primary_pred(arm, budget, seed)))
                else:
                    missing.append(p)
    if cfg.SUPPLEMENTARY_CKPT.is_file():
        specs.append(ModelSpec(
            key="supp_mae_1_42", label="CNN-MAE · 1% labels · seed 42 (supplementary)",
            arm="mae", budget=1, seed=42, path=cfg.SUPPLEMENTARY_CKPT, kind="supplementary",
            note="Trained after the primary grid; epoch selected on the full validation split "
                 "(383,386 signals) instead of the 12,480-signal subset. Not part of the primary "
                 "results; no stored test evaluation."))
    return specs, missing


def pick_device():
    import torch
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            return "cuda"
        except Exception:
            pass
    return "cpu"


def load_model(spec: ModelSpec, device: str) -> LoadedModel:
    import torch
    from src.models.cnn import IQCNNClassifier

    if not spec.path.is_file():
        raise ModelUnavailable(f"Checkpoint not found: {spec.path}")
    try:
        # weights_only=True: tensors and plain Python containers only (no arbitrary code).
        ckpt = torch.load(spec.path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ModelUnavailable(f"Could not load {spec.path.name} safely: {exc}") from exc
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt or "validation" not in ckpt:
        raise ModelUnavailable(f"{spec.path.name} is not a classifier checkpoint.")
    conf = ckpt.get("config", {})
    expected_init = cfg.ARM_INIT[spec.arm]
    if ckpt.get("initialization") != expected_init or str(conf.get("label_fraction")) != str(spec.budget) \
            or int(conf.get("seed", -1)) != spec.seed:
        raise ModelUnavailable(
            f"{spec.path.name}: metadata (init={ckpt.get('initialization')}, "
            f"labels={conf.get('label_fraction')}%, seed={conf.get('seed')}) does not match the file name.")

    model = IQCNNClassifier(num_classes=24, feature_dim=128, patch_size=64)
    try:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
    except RuntimeError as exc:
        raise ModelUnavailable(f"{spec.path.name} is incompatible with IQCNNClassifier: {exc}") from exc
    model.eval()
    try:
        model.to(device)
    except Exception:
        device = "cpu"
        model.to(device)

    with torch.inference_mode():   # warm-up (same path as predict) so timings exclude one-off initialisation
        for _ in range(3):
            torch.softmax(_forward(model, np.zeros((2, 1024), np.float32), device), dim=1)
        if device == "cuda":
            torch.cuda.synchronize()
    val = ckpt["validation"]
    meta = {
        "selected_epoch": ckpt.get("epoch"),
        "val_macro_f1": val.get("macro_f1"), "val_accuracy": val.get("accuracy"),
        "val_samples": val.get("samples"),
        "init_checkpoint": conf.get("init_checkpoint"),
        "lr": conf.get("lr"), "weight_decay": conf.get("weight_decay", 1e-4),
        "validation_subset": conf.get("validation_subset"),
        "parameters": int(sum(p.numel() for p in model.parameters())),
    }
    if spec.result_json and spec.result_json.is_file():
        r = json.loads(spec.result_json.read_text())
        meta.update(test_macro_f1=r["test_macro_f1"], test_accuracy=r["test_accuracy"])
    return LoadedModel(spec, model, device, meta)


def _forward(model, signal: np.ndarray, device: str):
    import torch
    x = torch.from_numpy(np.ascontiguousarray(signal, dtype=np.float32)).unsqueeze(0).to(device)
    with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.float16,
                        enabled=(device == "cuda")):
        logits = model(x)
    return logits.float()


def predict(loaded: LoadedModel, signal: np.ndarray, position: Optional[int] = None,
            repeat: int = 10) -> Prediction:
    """Run one forward pass. Latency covers host-to-device copy, forward pass and softmax,
    synchronised on CUDA; it excludes model loading and data reading. `repeat` extra passes on the
    same input give a median latency that is less affected by background load."""
    import torch
    if signal.shape != (2, 1024):
        raise ValueError(f"Expected a [2, 1024] signal, got {signal.shape}")
    dev = loaded.device

    def timed():
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = torch.softmax(_forward(loaded.model, signal, dev), dim=1)[0]
        if dev == "cuda":
            torch.cuda.synchronize()
        return out, (time.perf_counter() - t0) * 1000.0

    with torch.inference_mode():
        probs, latency = timed()                       # the prediction that is shown
        repeats = [timed()[1] for _ in range(max(0, int(repeat)))]
    probs = probs.cpu().numpy().astype(np.float64)
    if not np.isfinite(probs).all():
        raise RuntimeError("Model produced non-finite outputs.")
    stored = None
    spec = loaded.spec
    if position is not None and spec.pred_file is not None and spec.pred_file.is_file():
        stored = int(np.load(spec.pred_file, mmap_mode="r")[position])
    return Prediction(pred=int(probs.argmax()), probs=probs, latency_ms=latency,
                      latency_median_ms=float(np.median(repeats)) if repeats else latency, device=dev,
                      precision="FP16 autocast" if dev == "cuda" else "FP32", stored_pred=stored)
