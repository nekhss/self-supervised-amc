# Cross-dataset transfer: RML2018.01A CNN-MAE -> RML2016.10a

Question: does the CNN-MAE encoder pretrained on RadioML2018.01A improve
low-label modulation classification on RadioML2016.10a compared with the same
CNN trained from scratch? This is cross-dataset transfer between these two
datasets only, not evidence of general transfer. RML2016 results are a separate
benchmark and are not merged into the RML2018 tables.

## Fixed protocol

| Item | Setting | Source |
|---|---|---|
| Target data | RML2016.10a pickle (read-only, outside the repo); 11 classes x 20 SNRs (-20..18 dB) x 1000 = 220,000 examples, [2, 128] float32; layout validated, 0 non-finite values, 0 exact duplicates | `tools/inspect_rml2016.py`, `src/data/rml2016.py` |
| Split | per (class, SNR) stratum: one permutation (seed 42), cut 700/150/150, giving train 154,000 / val 33,000 / test 33,000. Disjoint and exhaustive (checked automatically) | `tools/prepare_rml2016_splits.py`, `splits_seed42.npz` |
| Label budgets | first 7/35/70/175/700 train examples of each stratum's permutation = 1/5/10/25/100%; nested; exact (no rounding); every stratum represented | same |
| Amplitude | one global multiplier = 1 / mean per-example RMS over the RML2016 train split = 117.0698; applied to every example in both arms; no per-example normalisation | `protocol_seed42.json` |
| Model | `IQCNNClassifier(num_classes=11, feature_dim=128, patch_size=64, signal_length=128)`: 2 patches of 64 samples, mean-pooled (same patch length as pretraining) | `src/models/cnn.py`, `src/rml2016_transfer.py` |
| Transfer arm | all 23 `encoder.*` tensors from `cnn_mae_full_v1_best.pt` (RML2018 MAE, epoch 5) loaded strictly; 33 MAE-only tensors (pos. embeddings, mask token, decoder, reconstruction head) excluded; fresh 11-class head | `load_pretrained_encoder` |
| Scratch arm | same construction, no loading. The head init is identical to the transfer arm for a given seed | |
| Training (both arms) | AdamW lr 1e-3, weight decay 1e-4, cosine by epoch, batch 128, 10 epochs, AMP fp16, gradient clipping at 1.0, no early stopping; checkpoint = best val macro-F1 (full val split) | matches `experiments/train_classifier.py` |
| Seeds | split/subsets fixed (seed 42); training seeds 42, 43, 44 | |
| Test | the full test split, evaluated once per selected checkpoint after every training run has finished; test indices are not loaded during training | `experiments/evaluate_rml2016_transfer.py` |

## Known limitations

- The split is per example: RML2016.10a has no source or session IDs, so examples
  generated in the same session can land in different partitions.
- Amplitude statistics differ even after scaling. RML2018 RMS grows with SNR (median 1.00,
  about 1.8 at high SNR, and the 99th percentile is about 10), while RML2016 is roughly
  power-normalised (0.97–1.04 at every SNR). The two datasets also differ in their
  generators, channel models, sample rates, signal length (128 vs 1024) and class sets.
- The transfer arm also benefits from RML2018 pretraining compute and data. No MAE
  pretraining is done on RML2016.
- The pretrained encoder comes from a single pretraining run.
