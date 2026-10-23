# Smoke test

## Automated (`demo\smoke_test.py`)

```powershell
cd C:\Users\admin\OneDrive\Desktop\DL-Project
C:\venvs\dlproj\Scripts\python.exe demo\smoke_test.py --quick   # 5% seed-42 pair + supplementary
C:\venvs\dlproj\Scripts\python.exe demo\smoke_test.py           # all 22 primary models + supplementary
```

| # | Check | Expected |
|---|---|---|
| 1 | Test split opens from `data_splits.npz → test_idx` via the cache | 383,386 signals; each sample is `[2,1024]` float32 and finite; its HDF5 row equals `test_idx[i]`; label and SNR equal `results\test_labels_snr.npz` |
| 2 | Invalid sample index | rejected with `IndexError` |
| 3 | Verified results | 22 result files; all five budget means match the reference values; 5% paired p = 0.288 |
| 4 | Every available checkpoint | loads with `weights_only=True` and passes the metadata check; predictions return a class in 0–23, probabilities sum to 1, latency > 0; **25 random test signals per model give exactly the stored test-set predictions** |
| 5 | Missing checkpoint | raises `ModelUnavailable` (the app shows a message instead of crashing) |
| 6 | Streamlit pages (AppTest, headless) | all four pages render without exceptions or error boxes; the results page has separate primary and exploratory tabs, the exploratory tab carries its label and design-timing note, and the training-curve seed selector re-renders; changing the test index re-renders |
| 7 | Exploratory 50-epoch artifacts | six runs, one per arm × seed, identified from their manifests. MAE seed 43 maps to the misnamed `finetune_50ep_v1${a}_1pct_seed43` folder, the others to `finetune_50ep_v1\<arm>_1pct_seed<seed>`. For each run: seed, initialisation, `complete`, 50/50 epochs and checkpoint SHA-256 match the manifest; its test result names that checkpoint (383,386 signals); 50 per-epoch rows; 12,480-signal validation subset. Means, SDs and paired differences are recomputed independently from the raw JSON files. Per-seed values equal the cross-check table. Each 10- vs 50-epoch pair uses the matching primary 1% file. Exploratory files are disjoint from the primary results, and Live Prediction uses only primary and supplementary checkpoints |
| 8 | No writes | every file in `results\`, `finetune\`, the exploratory run folders and `results_50ep_v1_exploratory\` has the same size and modification time after the run as before |

The run ends with `ALL CHECKS PASSED`, or lists each failed check.

Check 4 is the strongest correctness test. It confirms that input shape, dtype, preprocessing (none), class mapping and precision match `experiments\evaluate_test.py`. Predictions are made on GPU with FP16 autocast, as in the original evaluation. On a CPU-only machine (FP32) a near-tie can very rarely differ.

## Manual checklist (in the browser)

- [ ] `python -m streamlit run app.py` starts and the page shows the title and subtitle.
- [ ] Signal Explorer: true class, SNR, test index, HDF5 row and `2 × 1024 · float32`; the I/Q plot and constellation render.
- [ ] Changing the test index (or using Class & SNR) updates the signal.
- [ ] Live Prediction: both cards show a prediction, correct/incorrect against the true class, model confidence, latency, the top-5 chart and "Matches the stored test-set prediction".
- [ ] Changing the budget and seed loads other models. A missing checkpoint shows a yellow message.
- [ ] Ticking "supplementary" adds a third card marked *supplementary*, with its provenance note.
- [ ] Experimental Results → Primary tab: green "match the reference values" badge, both charts, the table with seed counts, and the interpretation with p = 0.29; the Accuracy toggle works; the SNR expander renders.
- [ ] Experimental Results → Exploratory tab: the yellow EXPLORATORY note; the "6 runs verified" and cross-check badges; the per-seed table (seed 44 validation Δ shown as ≈0); the all-metrics table labelled by data split; the paired-difference and 10-vs-50 charts; the interpretation; the training curves for each seed, with dashed selected-epoch lines; the source-artifact table.
- [ ] About the Experiment renders.
