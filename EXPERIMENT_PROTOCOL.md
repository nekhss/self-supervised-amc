# DL-Project: experiment protocol and state

Question: does CNN-MAE pretraining improve automatic modulation classification
on RadioML 2018.01A (24 classes, input [2, 1024]) when labelled data is limited,
compared with the same CNN trained from scratch?

## Fixed protocol (unchanged from the existing code)

| Item | Setting | Source |
|---|---|---|
| Split | 70/15/15, stratified by class x SNR, seed 42; train 1,789,132 / val 383,386 / test 383,386; disjoint (verified) | `create_splits.py`, `data_splits.npz` |
| Label budgets | 1, 5, 10, 25, 100% of train; nested; class-balanced; seed 42 | `create_label_subsets_v2.py`, `label_subsets_v2.npz` |
| Classifier validation | 12,480 val samples, 20 per class x SNR | `experiments/create_validation_subset.py`, `validation_subset_v1.npz` |
| Encoder | `IQCNNEncoder` over 16 patches of 64 samples, mean-pooled; same module in both arms | `src/models/cnn.py` |
| MAE pretraining | mask 50% of patches, MSE on masked patches only, transformer decoder (2 layers, dim 128); all 1,789,132 train signals, labels unused; AdamW lr 1e-4, wd 1e-4, cosine by epoch, batch 64, 5 epochs, seed 42; checkpoint chosen by val MSE (full val split) | `experiments/pretrain_cnn_mae_fixed.py`, `src/models/cnn_mae.py` |
| Fine-tuning (both arms) | full network trained; AdamW lr 1e-3, wd 1e-4, cosine, batch 128, 10 epochs, AMP; checkpoint = best val macro-F1 epoch | `experiments/train_classifier.py` |
| Seeds | 42, 43, 44 for 1/5/10%; 42 for 25/100% | `experiments/run_finetune_suite.py` |
| Test | full test split, evaluated once per selected checkpoint, after all training | `experiments/evaluate_test.py` |

The test split is never read during training, tuning or checkpoint selection.
No normalisation is fitted (raw IQ in both arms).

Known differences between the arms (report these):
- The MAE arm additionally sees all unlabelled training signals during
  pretraining (standard semi-supervised setting). It never sees val/test labels
  or test signals.
- The MAE arm uses extra compute for pretraining (~8 h on an RTX 4050).
- The label-fraction seed varies only the fine-tuning; the subsets themselves are
  fixed (seed 42), and the MAE encoder is from a single pretraining run.
- 25% and 100% have one seed only, so no variability estimate there.

## Speed fix (no protocol change)

Per-sample HDF5 reads run at ~385 samples/s, which makes fine-tuning at large
budgets take days. `tools/build_cache.py` copies the needed index sets once into
memory-mappable .npy files under `C:\dlruns\cache` (bit-identical values,
spot-checked against the HDF5). `experiments/finetune_cached.py` runs
`train_classifier.py` unchanged with only the dataset class swapped for
`src/data/cached_dataset.py`. MAE pretraining still uses the original loader.

## Environment

- `C:\venvs\dlproj` (outside OneDrive): torch 2.10.0+cu126, numpy 2.4.6,
  h5py 3.16.0, scikit-learn 1.9.1, matplotlib 3.11.2. GPU: RTX 4050 Laptop 6 GB.
- The original `.venv` is broken (mixed torch 2.14.1 CPU / 2.10.0+cu126 files,
  then a partial uninstall on 20 Oct). It is left untouched; do not use it.

## Run order

1. MAE pretraining (started 20 Oct, writes `C:\dlruns\cnn_mae_full_v1*.pt` + `.log`).
2. `tools\phase2_scratch.ps1`, which can run during step 1: builds caches,
   runs smoke tests, then the scratch grid -> `C:\dlruns\finetune\scratch_*`.
3. After step 1 finishes: `tools\phase3_mae.ps1`, which checks the MAE, runs the
   MAE grid, does the test evaluation and writes the tables/figures to `C:\dlruns\results`.

## History / earlier artifacts

- `checkpoints/` holds pilots only: CNN-MAE runs of 5–20 batches per epoch, and
  1% classifier runs initialised from `cnn_mae_resume3_test.pt` (about 20 MAE
  batches). Val macro-F1 at 1% was 0.2775 (MAE pilot) vs 0.2640 (scratch); this
  is a pipeline check only, not evidence.
- The first full pretraining run (`checkpoints\cnn_mae_full_v1.pt`, `.venv`)
  stopped before finishing epoch 1, so no checkpoint was written and nothing
  could be recovered. The cause is unknown.

## Interpreting results

Report mean ± std over seeds per budget, plus the paired MAE − scratch
difference (bootstrap 95% CI on macro-F1, exact McNemar test on accuracy,
`paired_comparison.csv`). These tests reflect test-set sampling noise only, not
training-seed variability; claim an improvement only if it holds across seeds
and the intervals exclude zero.

## Status (20 Oct)

- MAE pretraining `C:\dlruns\cnn_mae_full_v1` finished 5/5 epochs in 4.9 h. Val
  MSE: 0.4825, 0.4679, 0.4635, 0.4632, 0.4585, so the best checkpoint is epoch 5.
- Caches were built in 1.1 min, and all spot-checks passed. `scratch_1pct_seed42` reproduced
  the earlier HDF5-based pilot exactly (val macro-F1 0.2640), which shows the cached path
  gives the same results as the original loader.
- Scratch grid done (val macro-F1): 1% 0.2640/0.2535/0.2552; 5% 0.3119/0.3096/0.3334;
  10% 0.3848/0.3439/0.3483; 25% 0.3635; 100% 0.3760 (best epoch 4).
- Caveats seen so far:
  - Scratch performance levels off at about 0.37 macro-F1 from 10% upward, below
    typical 2018.01A CNN results. This is most likely the patch-mean-pooled encoder
    (no cross-patch order), not the data volume. The comparison remains controlled,
    since both arms use the same encoder, but the conclusions only cover this encoder.
  - On the validation subset, predicting each signal's visible-patch channel mean gives
    masked MSE about 0.50, versus about 2.42 for predicting zeros. The full MAE's 0.4585
    val MSE should be judged against the mean baseline: `tools/check_mae.py` reports
    both, broken down by SNR.
- Next: `tools\phase3_mae.ps1`.
