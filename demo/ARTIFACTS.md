# Artifacts used by the demo (read-only)

## Data

| Artifact | Provenance | Use |
|---|---|---|
| `DL-Project\data_splits.npz` → `test_idx` (383,386) | `create_splits.py`: 70/15/15 split stratified by class × SNR, seed 42 | defines the held-out test split; "test index i" = `test_idx[i]` |
| `C:\dlruns\cache\test\` (`x.npy`, `y.npy`, `snr.npy`, `indices.npy`) | `tools\build_cache.py`: bit-identical copy of the test rows of the HDF5 file, 300 spot checks | signals, labels and SNR, read through `src\data\cached_dataset.py` |
| `C:\RadioML2018\raw\GOLD_XYZ_OSC.0001_1024.hdf5` | original dataset | fallback reader (`src\data\radioml_dataset.py`) |
| `C:\dlruns\results\test_labels_snr.npz` | `experiments\evaluate_test.py` | labels and SNR in `test_idx` order; cross-check for every sample read |
| Class names | `class_names` in the primary result JSONs (written by `evaluate_test.py` from `classes-fixed.json`) | label index → modulation name |

## Models: primary grid (22 checkpoints)

`C:\dlruns\finetune\{scratch,mae}_{1,5,10}pct_seed{42,43,44}_v1.pt` and `{scratch,mae}_{25,100}pct_seed42_v1.pt`

- **Produced by:** `experiments\run_finetune_suite.py`, which runs the unmodified `experiments\train_classifier.py` through `finetune_cached.py`.
- **Architecture:** `IQCNNClassifier(num_classes=24, feature_dim=128, patch_size=64)`; 55,544 parameters.
- **Training:** AdamW with lr 1e-3 and weight decay 1e-4, cosine schedule, 10 epochs, batch 128.
- **Epoch selection:** best macro-F1 on `validation_subset_v1.npz` (12,480 signals).
- **MAE arm:** the encoder is initialised from `C:\dlruns\cnn_mae_full_v1_best.pt` (5-epoch MAE pretraining on all 1,789,132 training signals, without labels).
- **Validation when loaded:** the app loads every checkpoint with `torch.load(..., weights_only=True)`. It refuses the checkpoint unless its metadata (initialisation, label fraction, seed) matches the file name and the state dict loads strictly.

## Model: supplementary (clearly labelled in the app)

`C:\dlruns\classifier_mae_fullpretrain_1pct_seed42.pt`

- **What it is:** a complete checkpoint (epoch 9; MAE initialisation from `cnn_mae_full_v1_best.pt`; 1% labels; seed 42; lr 1e-3; weight decay 1e-4).
- **Why it is not primary:** its epoch was selected on the **full validation split (383,386 signals)**, not the 12,480-signal subset used by the primary protocol.
- **No stored test evaluation:** none exists for it, so the app shows its live predictions only and never mixes it into the results.

## Not used

| Artifact | Reason |
|---|---|
| `DL-Project\checkpoints\classifier_*` | earlier pilot conditions (e.g. a 20-batch MAE initialisation); not primary models |
| `C:\dlruns\cnn_mae_full_v1*.pt` used as classifiers | these are MAE (encoder + decoder) checkpoints, not trained classifiers |
| `C:\dlruns\tuning_*`, `matched_wd1e5`, `results_matched_wd1e5` | exploratory, run after the primary test results were known |
| `C:\dlruns\finetune_50ep_v1\*\best.pt` as Live Prediction models | the exploratory checkpoints are shown as results only; Live Prediction keeps the primary models |
| `C:\dlruns\continued\` | a 3-epoch resume of the supplementary checkpoint; no epoch improved, so no new checkpoint exists |
| `C:\dlruns\rml2016_transfer\...\test.json`, `summary.json`, `per_seed.csv` | a separate cross-dataset experiment (RadioML 2018 → RadioML 2016.10a). Not this research question, and kept out of the demo |
| `C:\dlruns\diagnostics` | BatchNorm diagnostics (validation only); mentioned in the limitations text only |

## Results

| Artifact | Producer | Use |
|---|---|---|
| `C:\dlruns\results\{arm}_{budget}pct_seed{seed}_v1.json` (22 files) | `evaluate_test.py`, one test evaluation per selected checkpoint | all test metrics, recomputed live into budget means and SDs; per-SNR accuracy |
| `C:\dlruns\results\{arm}_{budget}pct_seed{seed}_v1_pred.npy` | `evaluate_test.py` | stored test-set predictions; the app shows whether its live prediction matches |
| `C:\dlruns\results\paired_comparison.csv` | `summarize_results.py` | per-seed bootstrap CIs and McNemar tests (expandable table) |
| `DL-Project\report\tables\values.json` | `report\scripts\build_report_assets.py` | cross-check of the 5% paired t-test p (0.288) |

The app recomputes the budget means from the 22 JSON files and compares them with the reference values in `demo_config.py`. A discrepancy is shown as a warning, never silently corrected.

Verified on 21 October:

| Budget | Seeds | Scratch | CNN-MAE | Δ |
|---|---|---|---|---|
| 1% | 3 | 0.2577 | 0.2566 | −0.0011 |
| 5% | 3 | 0.3199 | 0.3320 | +0.0121 |
| 10% | 3 | 0.3583 | 0.3571 | −0.0012 |
| 25% | 1 | 0.3630 | 0.3629 | −0.0002 |
| 100% | 1 | 0.3821 | 0.3838 | +0.0017 |

The 5% paired t-test across seeds gives p = 0.288 (2 degrees of freedom).

## Exploratory 50-epoch fine-tuning (separate tab; never pooled with the primary grid)

These runs were designed **after** the primary test results were known, so they are labelled exploratory.

- **Setup:** 1% labels, seeds 42–44, CNN-MAE and scratch.
- **Same as the primary protocol:** label subset, the 12,480-signal validation subset, lr 1e-3, weight decay 1e-4, batch 128, and the 5-epoch MAE checkpoint `cnn_mae_full_v1_best.pt`.
- **Different:** **50 fine-tuning epochs**, with a cosine schedule over all 50.
- **Produced by:** `experiments\run_long_finetune.py`, which runs the unchanged `train_classifier.py` on cached data.
- **Tested by:** `experiments\evaluate_test.py`, once per selected checkpoint.

| Run | Training folder (manifest.json, per_epoch.csv, best.pt, train.log) | Test result |
|---|---|---|
| CNN-MAE seed 42 | `C:\dlruns\finetune_50ep_v1\mae_1pct_seed42\` | `C:\dlruns\results_50ep_v1_exploratory\mae_1pct_seed42\best.json` |
| Scratch seed 42 | `C:\dlruns\finetune_50ep_v1\scratch_1pct_seed42\` | `...\results_50ep_v1_exploratory\scratch_1pct_seed42\best.json` |
| CNN-MAE seed 43 | **`C:\dlruns\finetune_50ep_v1${a}_1pct_seed43\`** (misnamed sibling folder, see below) | `...\results_50ep_v1_exploratory\mae_1pct_seed43\best.json` |
| Scratch seed 43 | `C:\dlruns\finetune_50ep_v1\scratch_1pct_seed43\` | `...\results_50ep_v1_exploratory\scratch_1pct_seed43\best.json` |
| CNN-MAE seed 44 | `C:\dlruns\finetune_50ep_v1\mae_1pct_seed44\` | `...\results_50ep_v1_exploratory\mae_1pct_seed44\best.json` |
| Scratch seed 44 | `C:\dlruns\finetune_50ep_v1\scratch_1pct_seed44\` | `...\results_50ep_v1_exploratory\scratch_1pct_seed44\best.json` |

**How `exploratory_data.py` reads them (read-only):**
- **Discovery:** it finds runs from their manifests (seed argument, presence of an MAE init checkpoint, `status: complete`, 50/50 epochs), never from folder names.
- **Misnamed folder:** the MAE seed-43 run was written to a misnamed folder by a launch-script quoting error. Its manifest records that path, and the run is used in place.
- **Matching:** each `best.json` is matched to its run through the checkpoint path it stores. It must agree with the manifest on seed, initialisation, selected epoch and validation macro-F1.
- **Integrity:** `best.pt` must still have the manifest's SHA-256, and each `test_summary.csv` must hold exactly one evaluation.
- **Problems:** any problem is shown in the app, never silently corrected.

**What each number means:**
- **Validation macro-F1** is the manifest's exact value, the maximum over 50 epochs on the 12,480-signal subset, so it is optimistic.
- **Training loss** in the curves is the value printed by `train_classifier.py` (6 decimals), the only stored record. Validation metrics are the exact values returned by the evaluation function.
- **Comparability:** the supplementary checkpoint selected its epoch on the full validation split, so its validation score is not comparable with these.

Cross-check values (recorded in `demo_config.EXPLORATORY_REFERENCE`, verified 22 October):

| Seed | Val MAE / scratch | Test MAE / scratch | Test Δ |
|---|---|---|---|
| 42 | 0.3346 / 0.3261 | 0.3349 / 0.3255 | +0.0094 |
| 43 | 0.3278 / 0.3240 | 0.3291 / 0.3212 | +0.0079 |
| 44 | 0.3185 / 0.3185 | 0.3192 / 0.3213 | −0.0021 |
| Mean ± SD | 0.3270 ± 0.0081 / 0.3229 ± 0.0039 | 0.3277 ± 0.0079 / 0.3227 ± 0.0024 | +0.0051 ± 0.0062 |

## Exploratory 100-epoch follow-up (third tab; never pooled)

- **Same design as the 50-epoch study, with 100 fine-tuning epochs.** Cosine over 100 epochs; all six runs were trained anew from the verified initialisation. Exact continuation was impossible: the 50-epoch checkpoints hold only their selected-epoch state.
- **Training runs:** `C:\dlrunsinetune_100ep_v2uns\<arm>_1pct_seed<seed>\`, written by `run_all.py` through `experimentsun_long_finetune.py`.
- **Test results:** `C:\dlrunsinetune_100ep_v2	est_results\<run>est.json`, one evaluation per selected checkpoint.
- **Analysis:** `C:\dlrunsinetune_100ep_v2nalysis\`.
- **How the demo reads them:** `exploratory_data.load("100")`, with the same checks as the 50-epoch study.
- **Cross-check:** `demo_config.EXPLORATORY_100_REFERENCE` holds the reported test macro-F1 summary: MAE 0.3281 ± 0.0090, scratch 0.3299 ± 0.0100, paired Δ −0.0018 ± 0.0170. Two seeds favour MAE and one favours scratch.
