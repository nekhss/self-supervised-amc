import argparse
import random
import time
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.radioml_dataset import RadioMLDataset
from src.models.cnn_mae import IQCNNMaskedAutoencoder


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, optimizer, scaler, device, train, max_batches):
    model.train(train)
    total_loss = 0.0
    samples = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break

        signals = batch["signal"].to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                loss = model(signals)["loss"]

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at batch {batch_idx}: {loss.item()}"
                )

            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

        batch_size = signals.size(0)
        total_loss += loss.detach().float().item() * batch_size
        samples += batch_size

    if samples == 0:
        raise RuntimeError("No batches processed. Check dataset and batch limits.")

    return total_loss / samples, samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2,
                        help="Total target epoch, not additional epochs.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-train-batches", type=int, default=20)
    parser.add_argument("--max-val-batches", type=int, default=5)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", type=str,
                        default="checkpoints/cnn_mae_resumable.pt")
    parser.add_argument("--resume", type=str, default=None,
                        help="Existing latest checkpoint to resume.")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive.")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    output_path = ROOT / args.output
    best_path = output_path.with_name(
        output_path.stem + "_best" + output_path.suffix
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.resume is None and (output_path.exists() or best_path.exists()):
        raise FileExistsError(
            f"Output already exists: {output_path}. "
            "Choose another --output or explicitly resume."
        )

    with np.load(ROOT / "data_splits.npz") as splits:
        train_idx = splits["train_idx"].copy()
        val_idx = splits["val_idx"].copy()

    train_ds = RadioMLDataset(train_idx)
    val_ds = RadioMLDataset(val_idx)

    train_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    config = {
        "signal_length": 1024,
        "patch_size": 64,
        "mask_ratio": 0.5,
        "feature_dim": 128,
        "decoder_dim": 128,
        "decoder_depth": 2,
        "decoder_heads": 4,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "lr": args.lr,
        "max_train_batches": args.max_train_batches,
        "max_val_batches": args.max_val_batches,
        "workers": args.workers,
        "scheduler": "manual_cosine_by_epoch",
        "target_epochs": args.epochs,
    }

    model = IQCNNMaskedAutoencoder(
        signal_length=1024,
        patch_size=64,
        mask_ratio=0.5,
        feature_dim=128,
        decoder_dim=128,
        decoder_depth=2,
        decoder_heads=4,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    # The learning rate is set explicitly at the start of each epoch.
    # This avoids scheduler-state inconsistencies when extending a run.
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda")
    )

    start_epoch = 1
    best_val = float("inf")

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = ROOT / resume_path
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")

        checkpoint = torch.load(
            resume_path, map_location=device, weights_only=False
        )

        saved_config = checkpoint.get("config", {})
        current_config = dict(config)

        # The total epoch target may increase when resuming.
        saved_config = dict(saved_config)
        saved_config.pop("target_epochs", None)
        current_config.pop("target_epochs", None)
        # Older pilot checkpoints used a stateful scheduler. This version
        # recalculates the learning rate explicitly at each epoch.
        saved_config.pop("scheduler", None)
        current_config.pop("scheduler", None)

        if saved_config != current_config:
            raise ValueError(
                "Resume configuration mismatch. Keep the model, seed, "
                "batch size, learning rate, data limits, and workers unchanged."
            )

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        saved_target = checkpoint["config"]["target_epochs"]

        if args.epochs < checkpoint["epoch"]:
            raise ValueError(
                "Target epochs cannot be less than the saved checkpoint epoch."
            )

        if args.epochs != saved_target:
            print(
                f"Updating cosine schedule target from {saved_target} "
                f"to {args.epochs} total epochs."
            )
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        best_val = checkpoint["best_val"]

        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda" and checkpoint["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all([state.detach().cpu().to(torch.uint8) for state in checkpoint["cuda_rng_state"]])
        np.random.set_state(checkpoint["numpy_rng_state"])
        random.setstate(checkpoint["python_rng_state"])
        train_generator.set_state(checkpoint["loader_rng_state"].cpu())

        print(f"Resuming from epoch {checkpoint['epoch']}")
        print(f"Previous best validation MSE: {best_val:.6f}")

    if start_epoch > args.epochs:
        print(f"Checkpoint already reached target epoch {args.epochs}.")
        return

    started = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        # Epoch 1 starts at base LR; later epochs follow a cosine decay.
        progress = (epoch - 1) / args.epochs
        epoch_lr = args.lr * 0.5 * (1.0 + np.cos(np.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = float(epoch_lr)

        train_loss, train_samples = run_epoch(
            model, train_loader, optimizer, scaler, device,
            train=True, max_batches=args.max_train_batches,
        )

        model.eval()
        with torch.random.fork_rng(
            devices=[device] if device.type == "cuda" else []
        ):
            torch.manual_seed(args.seed + 10000)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(args.seed + 10000)
            with torch.no_grad():
                val_loss, val_samples = run_epoch(
                    model, val_loader, optimizer, scaler, device,
                    train=False, max_batches=args.max_val_batches,
                )

        improved = val_loss < best_val
        if improved:
            best_val = val_loss

        # No scheduler.step(): the next epoch's LR is calculated explicitly.
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None,
            "scaler_state_dict": scaler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val": best_val,
            "config": config,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all()
                if device.type == "cuda" else None
            ),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "loader_rng_state": train_generator.get_state(),
        }

        torch.save(checkpoint, output_path)
        if improved:
            torch.save(checkpoint, best_path)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train MSE={train_loss:.6f} ({train_samples} samples) | "
            f"val MSE={val_loss:.6f} ({val_samples} samples) | "
            f"lr_used={epoch_lr:.2e}"
        )

    print("Latest checkpoint:", output_path)
    print("Best checkpoint:", best_path)
    print(f"Elapsed: {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()

