"""Read-only summary of checkpoints in DL-Project/checkpoints and HDF5 layout."""
from pathlib import Path
import h5py
import torch

ROOT = Path(__file__).resolve().parents[1]
H5 = Path(r"C:\RadioML2018\raw\GOLD_XYZ_OSC.0001_1024.hdf5")

for path in sorted((ROOT / "checkpoints").glob("*.pt")):
    try:
        c = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # report, do not hide
        print(f"{path.name}: LOAD FAILED {exc!r}")
        continue
    cfg = c.get("config", {})
    keep = {k: cfg.get(k) for k in ("label_fraction", "init_checkpoint", "epochs", "target_epochs",
                                    "batch_size", "lr", "max_train_batches", "max_val_batches",
                                    "validation_subset", "seed") if k in cfg}
    val = c.get("validation")
    print(f"{path.name}: epoch={c.get('epoch')} init={c.get('initialization', '-')} "
          f"train={c.get('train_loss')} val_loss={c.get('val_loss', val and val.get('loss'))} "
          f"val_acc={val and val.get('accuracy')} val_f1={val and val.get('macro_f1')} "
          f"val_n={val and val.get('samples')} cuda_rng={'yes' if c.get('cuda_rng_state') is not None else 'no'} cfg={keep}")

with h5py.File(H5, "r") as f:
    for k in ("X", "Y", "Z"):
        d = f[k]
        print(f"HDF5 {k}: shape={d.shape} dtype={d.dtype} chunks={d.chunks} compression={d.compression}")
