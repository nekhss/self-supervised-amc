import argparse
import random
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

from src.data.radioml_dataset import RadioMLDataset
from src.models.cnn import IQCNNClassifier


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device, max_batches=0):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches and batch_idx >= max_batches:
                break

            signals = batch["signal"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                logits = model(signals)
                loss = F.cross_entropy(logits, labels)

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite validation loss.")

            predictions = logits.argmax(dim=1)
            n = labels.size(0)
            total_loss += loss.float().item() * n
            total_samples += n
            all_targets.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    if total_samples == 0:
        raise RuntimeError("No validation batches processed.")

    return {
        "loss": total_loss / total_samples,
        "accuracy": accuracy_score(all_targets, all_predictions),
        "macro_f1": f1_score(
            all_targets, all_predictions, average="macro", zero_division=0
        ),
        "samples": total_samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-fraction", choices=["1", "5", "10", "25", "100"],
                        default="1")
    parser.add_argument("--init-checkpoint", default=None,
                        help="MAE checkpoint; omit for scratch baseline.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="AdamW weight decay (default: 1e-4).")
    parser.add_argument("--encoder-lr", type=float, default=None,
                        help="Optional separate learning rate for model.encoder; "
                             "requires --init-checkpoint. The --lr value is used "
                             "for all non-encoder (classifier-head) parameters.")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-batches", type=int, default=0,
                        help="0 means use all training batches.")
    parser.add_argument("--validation-subset", type=str, default=None, help="NPZ file under the project root containing val_indices; defaults to the full validation split")
    parser.add_argument("--max-val-batches", type=int, default=0,
                        help="0 means use all validation batches.")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive.")
    if args.lr <= 0:
        raise ValueError("lr must be positive.")
    if args.weight_decay < 0:
        raise ValueError("weight-decay must be non-negative.")
    if args.encoder_lr is not None and args.encoder_lr <= 0:
        raise ValueError("encoder-lr must be positive.")
    if args.encoder_lr is not None and not args.init_checkpoint:
        raise ValueError("--encoder-lr requires --init-checkpoint (MAE initialization).")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    output_path = ROOT / args.output
    if output_path.exists():
        raise FileExistsError(
            f"{output_path} already exists. Choose a new output filename."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    subset_path = ROOT / "label_subsets_v2.npz"
    split_path = ROOT / "data_splits.npz"

    with np.load(subset_path) as subsets:
        train_indices = subsets[f"train_{args.label_fraction}pct"].copy()

    if args.validation_subset:
        subset_path = ROOT / args.validation_subset
        if not subset_path.is_file():
            raise FileNotFoundError(
                f"Validation subset not found: {subset_path}"
            )
        with np.load(subset_path) as subset:
            val_indices = subset["val_indices"].copy()
        print(
            f"Using fixed validation subset: {subset_path.name} "
            f"({len(val_indices):,} samples)"
        )
    else:
        with np.load(split_path) as splits:
            val_indices = splits["val_idx"].copy()
        print(
            f"Using full validation split ({len(val_indices):,} samples)"
        )

    train_ds = RadioMLDataset(train_indices)
    val_ds = RadioMLDataset(val_indices)

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    model = IQCNNClassifier(num_classes=24, feature_dim=128, patch_size=64)

    if args.init_checkpoint:
        checkpoint_path = Path(args.init_checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = ROOT / checkpoint_path
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)

        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        state = checkpoint["model_state_dict"]
        encoder_state = {
            key.removeprefix("encoder."): value
            for key, value in state.items()
            if key.startswith("encoder.")
        }
        if not encoder_state:
            raise ValueError("No encoder weights found in the checkpoint.")

        model.encoder.load_state_dict(encoder_state, strict=True)
        print("Loaded pretrained encoder from:", checkpoint_path)
        print("MAE checkpoint epoch:", checkpoint.get("epoch", "unknown"))
    else:
        print("Using randomly initialized encoder (scratch baseline).")

    model = model.to(device)
    if args.encoder_lr is None:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        print(f"Optimizer: single learning rate={args.lr:g}, "
              f"weight_decay={args.weight_decay:g}")
    else:
        encoder_params = list(model.encoder.parameters())
        encoder_param_ids = {id(param) for param in encoder_params}
        head_params = [
            param for param in model.parameters()
            if id(param) not in encoder_param_ids
        ]
        if not encoder_params or not head_params:
            raise RuntimeError(
                "Could not form non-empty encoder and classifier-head parameter groups."
            )
        optimizer = torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": args.encoder_lr},
                {"params": head_params, "lr": args.lr},
            ],
            weight_decay=args.weight_decay,
        )
        print(f"Optimizer: encoder_lr={args.encoder_lr:g}, "
              f"head_lr={args.lr:g}, weight_decay={args.weight_decay:g}")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda")
    )

    best_f1 = -1.0
    best_epoch = 0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for batch_idx, batch in enumerate(train_loader):
            if args.max_train_batches and batch_idx >= args.max_train_batches:
                break

            signals = batch["signal"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                logits = model(signals)
                loss = F.cross_entropy(logits, labels)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite training loss at batch {batch_idx}."
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            n = labels.size(0)
            total_loss += loss.detach().float().item() * n
            total_samples += n

        if total_samples == 0:
            raise RuntimeError("No training batches processed.")

        val = evaluate(
            model, val_loader, device, max_batches=args.max_val_batches
        )
        scheduler.step()

        train_loss = total_loss / total_samples
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train loss={train_loss:.6f} ({total_samples} samples) | "
            f"val loss={val['loss']:.6f} | "
            f"val accuracy={val['accuracy']:.4f} | "
            f"val macro-F1={val['macro_f1']:.4f}"
        )

        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "train_loss": train_loss,
                    "validation": val,
                    "best_val_macro_f1": best_f1,
                    "config": vars(args),
                    "initialization": (
                        "mae_pretrained" if args.init_checkpoint else "scratch"
                    ),
                },
                output_path,
            )

    print(f"Best validation macro-F1: {best_f1:.4f} at epoch {best_epoch}")
    print("Best checkpoint:", output_path)
    print(f"Elapsed: {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
