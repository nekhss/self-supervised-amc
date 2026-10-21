
import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.radioml_dataset import RadioMLDataset
from src.models.mae import IQMaskedAutoencoder


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])

    torch_state = state["torch"]
    if not isinstance(torch_state, torch.Tensor):
        torch_state = torch.as_tensor(torch_state, dtype=torch.uint8)
    torch.set_rng_state(torch_state.detach().cpu().to(torch.uint8))

    if torch.cuda.is_available() and "cuda" in state:
        cuda_states = [
            item.detach().cpu().to(torch.uint8)
            if isinstance(item, torch.Tensor)
            else torch.as_tensor(item, dtype=torch.uint8)
            for item in state["cuda"]
        ]
        torch.cuda.set_rng_state_all(cuda_states)


def run_epoch(
    model,
    loader,
    device,
    optimizer=None,
    scaler=None,
    max_batches=None,
    mask_seed=None,
):
    training = optimizer is not None
    model.train(training)

    # Reset this generator for each validation epoch. With an ordered
    # validation loader, the same samples receive the same masks each epoch.
    mask_generator = None
    if mask_seed is not None:
        mask_generator = torch.Generator(device=device)
        mask_generator.manual_seed(mask_seed)

    total_loss = 0.0
    total_samples = 0

    for batch_number, batch in enumerate(loader):
        if max_batches is not None and batch_number >= max_batches:
            break

        signals = batch["signal"].to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                output = model(signals, generator=mask_generator)
                loss = output["loss"]

            if not torch.isfinite(loss).item():
                raise FloatingPointError(
                    f"Non-finite loss at batch {batch_number}"
                )

            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                # Fail visibly instead of silently accepting bad gradients.
                for name, parameter in model.named_parameters():
                    if (
                        parameter.grad is not None
                        and not torch.isfinite(parameter.grad).all().item()
                    ):
                        raise FloatingPointError(
                            f"Non-finite gradient: {name}"
                        )

                scaler.step(optimizer)
                scaler.update()

        count = signals.shape[0]
        total_loss += loss.detach().float().item() * count
        total_samples += count

    if total_samples == 0:
        raise RuntimeError("No batches were processed.")

    return total_loss / total_samples, total_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-train-batches", type=int, default=20)
    parser.add_argument("--max-val-batches", type=int, default=5)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", default="checkpoints/mae_run.pt")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    if args.max_train_batches < 1 or args.max_val_batches < 1:
        parser.error("batch limits must be positive")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    with np.load(PROJECT_ROOT / "data_splits.npz", allow_pickle=False) as data:
        train_idx = data["train_idx"].copy()
        val_idx = data["val_idx"].copy()

    train_ds = RadioMLDataset(train_idx)
    val_ds = RadioMLDataset(val_idx)

    # Separate generator makes DataLoader shuffling reproducible and
    # its state available for checkpointing.
    loader_generator = torch.Generator().manual_seed(args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.workers > 0),
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.workers > 0),
    )

    model = IQMaskedAutoencoder().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda")
    )

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_path = output_path.with_name(
        output_path.stem + "_best" + output_path.suffix
    )

    start_epoch = 1
    best_val_loss = float("inf")
    history = []

    if args.resume:
        if not output_path.is_file():
            raise FileNotFoundError(
                f"Cannot resume: checkpoint not found: {output_path}"
            )

        checkpoint = torch.load(
            output_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        if "rng_state" in checkpoint:
            restore_rng_state(checkpoint["rng_state"])
        if "loader_generator_state" in checkpoint:
            loader_state = checkpoint["loader_generator_state"]
            loader_generator.set_state(
                loader_state.detach().cpu().clone().to(torch.uint8)
            )

        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        history = checkpoint.get("history", [])
        print(f"Resuming at epoch {start_epoch}")
        print(f"Best recorded validation loss: {best_val_loss:.6f}")

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            train_loss, train_count = run_epoch(
                model,
                train_loader,
                device,
                optimizer=optimizer,
                scaler=scaler,
                max_batches=args.max_train_batches,
            )

            val_loss, val_count = run_epoch(
                model,
                val_loader,
                device,
                max_batches=args.max_val_batches,
                mask_seed=args.seed + 100_000,
            )

            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_samples": train_count,
                "val_samples": val_count,
            }
            history.append(record)
            print(json.dumps(record))

            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "rng_state": rng_state(),
                "loader_generator_state": loader_generator.get_state(),
                "best_val_loss": best_val_loss,
                "history": history,
                "config": vars(args),
            }

            # Latest checkpoint supports resuming, even if it is not best.
            torch.save(checkpoint, output_path)

            if improved:
                torch.save(checkpoint, best_path)
                print(f"New best checkpoint: {best_path}")

            if device.type == "cuda":
                peak = torch.cuda.max_memory_allocated(device) / 1024**3
                print(f"Peak allocated GPU memory: {peak:.2f} GiB")

            print(f"Latest checkpoint: {output_path}")

    finally:
        train_ds.close()
        val_ds.close()


if __name__ == "__main__":
    main()
