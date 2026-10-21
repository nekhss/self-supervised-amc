"""Check that an MAE checkpoint actually learned to reconstruct masked patches.

Compares the model's masked-patch MSE on the balanced validation subset with
two trivial predictors computed on the same masked patches:
  zero      predict 0 for every masked value
  mean      predict each signal's mean over its visible patches (per channel)
Also verifies that the encoder weights transfer into the classifier.
Read-only: loads the checkpoint, writes nothing.

Example:
  python tools/check_mae.py C:\\dlruns\\cnn_mae_full_v1_best.pt
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.cached_dataset import CachedRadioMLDataset  # noqa: E402
from src.models.cnn import IQCNNClassifier  # noqa: E402
from src.models.cnn_mae import IQCNNMaskedAutoencoder  # noqa: E402


def main():
    path = Path(sys.argv[1])
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    print(f"{path.name}: epoch {ckpt['epoch']} of {cfg.get('target_epochs')}, "
          f"train MSE {ckpt['train_loss']:.4f}, val MSE {ckpt['val_loss']:.4f}, "
          f"max_train_batches={cfg.get('max_train_batches')}")

    model = IQCNNMaskedAutoencoder(
        signal_length=1024, patch_size=cfg["patch_size"], mask_ratio=cfg["mask_ratio"],
        feature_dim=cfg["feature_dim"], decoder_dim=cfg["decoder_dim"],
        decoder_depth=cfg["decoder_depth"], decoder_heads=cfg["decoder_heads"])
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    with np.load(ROOT / "validation_subset_v1.npz") as s:
        idx = s["val_indices"]
    cache, rows = CachedRadioMLDataset(idx).arrays()
    rows = np.sort(rows)
    snr_all = cache.snr[rows]

    torch.manual_seed(0)
    sums = {"model": 0.0, "zero": 0.0, "mean": 0.0}
    per_snr = {}
    count = 0
    with torch.no_grad():
        for s in range(0, len(rows), 512):
            r = rows[s:s + 512]
            x = torch.from_numpy(np.ascontiguousarray(
                cache.x[r[0]:r[-1] + 1][r - r[0]].transpose(0, 2, 1))).to(device)
            out = model(x)
            m = out["mask"]                       # [B, N], 1 = masked
            t, pr = out["targets"], out["predictions"].float()
            per_patch = {
                "model": (pr - t).pow(2).mean(-1),
                "zero": t.pow(2).mean(-1),
            }
            vis = (1 - m).unsqueeze(-1)
            ps = model.patch_size
            tc = t.reshape(t.shape[0], t.shape[1], 2, ps)
            ch_mean = (tc * vis.unsqueeze(-1)).sum((1, 3)) / (vis.sum(1) * ps)  # [B, 2]
            guess = ch_mean[:, None, :, None].expand_as(tc).reshape_as(t)
            per_patch["mean"] = (guess - t).pow(2).mean(-1)
            for k, v in per_patch.items():
                sums[k] += float((v * m).sum())
            sample_mse = ((per_patch["model"] * m).sum(1) / m.sum(1)).cpu().numpy()
            sample_zero = ((per_patch["zero"] * m).sum(1) / m.sum(1)).cpu().numpy()
            for snr_v, a, b in zip(snr_all[s:s + 512], sample_mse, sample_zero):
                d = per_snr.setdefault(int(snr_v), [0.0, 0.0, 0])
                d[0] += a
                d[1] += b
                d[2] += 1
            count += float(m.sum())

    res = {k: v / count for k, v in sums.items()}
    print(f"Masked-patch MSE on {len(rows):,} validation samples:")
    for k in ("model", "zero", "mean"):
        print(f"  {k:5s}: {res[k]:.5f}")
    print(f"  model / zero = {res['model'] / res['zero']:.3f}  "
          f"(< 1 means it beats predicting zeros)")
    print("  SNR  model/zero")
    for snr_v in sorted(per_snr):
        a, b, n = per_snr[snr_v]
        print(f"  {snr_v:4d}  {a / max(b, 1e-12):.3f}")

    clf = IQCNNClassifier(num_classes=24, feature_dim=cfg["feature_dim"], patch_size=cfg["patch_size"])
    enc = {k.removeprefix("encoder."): v for k, v in ckpt["model_state_dict"].items()
           if k.startswith("encoder.")}
    clf.encoder.load_state_dict(enc, strict=True)
    same = all(torch.equal(clf.encoder.state_dict()[k], enc[k]) for k in enc)
    print(f"Encoder transfer: {len(enc)} tensors loaded strictly, identical={same}")


if __name__ == "__main__":
    main()
