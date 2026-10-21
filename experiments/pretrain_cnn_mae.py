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
                result = model(signals)
                loss = result["loss"]

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite {'train' if train else 'validation'} loss "
                    f"at batch {batch_idx}: {loss.item()}"
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
        raise RuntimeError("No batches were processed.")

    return total_loss / samples, samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-train-batches", type=int, default=20)
    parser.add_argument("--max-val-batches", type=int, default=5)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/cnn_mae_pilot.pt",
    )
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive.")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    split_path = ROOT / "data_splits.npz"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")

    with np.load(split_path) as splits:
        train_idx = splits["train_idx"].copy()
        val_idx = splits["val_idx"].copy()

    # Reuse the existing, fixed split. Never include validation/test in pretraining.
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
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda")
    )

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_path = output_path.with_name(
        output_path.stem + "_best" + output_path.suffix
    )

    if output_path.exists() or best_path.exists():
        raise FileExistsError(
            "Pilot output already exists. Choose a new --output path "
            "to avoid overwriting a checkpoint."
        )

    best_val = float("inf")
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_samples = run_epoch(
            model, train_loader, optimizer, scaler, device,
            train=True, max_batches=args.max_train_batches,
        )

        # Fixed seed gives a repeatable validation mask sequence per epoch.
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

        scheduler.step()
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "seed": args.seed,
            "model_config": {
                "signal_length": 1024,
                "patch_size": 64,
                "mask_ratio": 0.5,
                "feature_dim": 128,
                "decoder_dim": 128,
                "decoder_depth": 2,
                "decoder_heads": 4,
            },
        }

        torch.save(checkpoint, output_path)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, best_path)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train masked MSE={train_loss:.6f} ({train_samples} samples) | "
            f"val masked MSE={val_loss:.6f} ({val_samples} samples)"
        )

    if device.type == "cuda":
        torch.cuda.synchronize()
        print(
            f"Peak allocated GPU memory: "
            f"{torch.cuda.max_memory_allocated() / (1024**3):.3f} GiB"
        )

    print(f"Latest checkpoint: {output_path}")
    print(f"Best checkpoint:   {best_path}")
    print(f"Elapsed: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()



