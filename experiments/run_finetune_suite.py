"""Run the scratch or MAE fine-tuning grid with one fixed protocol.

Protocol (identical for both arms, matching the existing 1% runs):
  label subsets   label_subsets_v2.npz (nested, class-balanced)
  validation      validation_subset_v1.npz (12,480 samples, balanced by class/SNR)
  selection       best validation macro-F1 epoch (train_classifier.py)
  optimiser       AdamW lr 1e-3, wd 1e-4, cosine over epochs, batch 128, 10 epochs
  test set        never touched here

Each run is a separate process with its own log. Existing outputs are skipped,
so the suite can be stopped and restarted safely.

Examples:
  python experiments/run_finetune_suite.py --arm scratch
  python experiments/run_finetune_suite.py --arm mae --mae-checkpoint C:\\dlruns\\cnn_mae_full_v1_best.pt
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_GRID = {  # label fraction (%) -> seeds
    "1": [42, 43, 44],
    "5": [42, 43, 44],
    "10": [42, 43, 44],
    "25": [42],
    "100": [42],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["scratch", "mae"], required=True)
    p.add_argument("--mae-checkpoint", default=None)
    p.add_argument("--fractions", nargs="+", default=list(DEFAULT_GRID))
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="Override seeds for every fraction.")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   help="AdamW weight decay (default: 1e-4).")
    p.add_argument("--encoder-lr", type=float, default=None,
                   help="Optional separate MAE encoder learning rate.")
    p.add_argument("--out-dir", default=r"C:\dlruns\finetune")
    p.add_argument("--tag", default="v1",
                   help="Suffix that ties outputs to one MAE checkpoint/protocol.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.arm == "mae":
        if not args.mae_checkpoint or not Path(args.mae_checkpoint).is_file():
            sys.exit("--mae-checkpoint must point to an existing MAE checkpoint.")
    elif args.mae_checkpoint:
        sys.exit("The scratch arm must not use --mae-checkpoint.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for frac in args.fractions:
        for seed in (args.seeds or DEFAULT_GRID[frac]):
            name = f"{args.arm}_{frac}pct_seed{seed}_{args.tag}"
            cmd = [sys.executable, "-u", str(ROOT / "experiments" / "finetune_cached.py"),
                   "--label-fraction", frac,
                   "--epochs", str(args.epochs),
                   "--batch-size", str(args.batch_size),
                   "--lr", str(args.lr),
                   "--weight-decay", str(args.weight_decay),
                   "--workers", "0",
                   "--seed", str(seed),
                   "--max-train-batches", "0",
                   "--max-val-batches", "0",
                   "--validation-subset", "validation_subset_v1.npz",
                   "--output", str(out_dir / f"{name}.pt")]
            if args.encoder_lr is not None:
                cmd += ["--encoder-lr", str(args.encoder_lr)]
            if args.arm == "mae":
                cmd += ["--init-checkpoint", str(Path(args.mae_checkpoint).resolve())]
            elif args.encoder_lr is not None:
                sys.exit("--encoder-lr is only supported for the MAE arm.")
            jobs.append((name, cmd))

    for i, (name, cmd) in enumerate(jobs, 1):
        ckpt, log = out_dir / f"{name}.pt", out_dir / f"{name}.log"
        if ckpt.exists() and log.exists() and "Elapsed:" in log.read_text(errors="ignore"):
            print(f"[{i}/{len(jobs)}] skip {name} (finished)")
            continue
        if ckpt.exists():
            sys.exit(f"{ckpt} exists but its log shows no completed run. "
                     "Inspect it and move it aside before rerunning.")
        print(f"[{i}/{len(jobs)}] {name}")
        if args.dry_run:
            print("   ", " ".join(cmd))
            continue
        t0 = time.time()
        with open(log, "w", encoding="utf-8") as fh:
            rc = subprocess.call(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
        tail = log.read_text(errors="ignore").strip().splitlines()[-3:]
        print(f"    exit={rc} in {(time.time() - t0) / 60:.1f} min | " + " | ".join(tail))
        if rc != 0:
            sys.exit(f"Run failed: see {log}")


if __name__ == "__main__":
    main()
