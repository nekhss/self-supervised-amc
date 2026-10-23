# CNN-MAE for Automatic Modulation Classification: research demo

A Streamlit app for the DL-Project research question:

> Does self-supervised CNN masked-autoencoder (MAE) pretraining improve automatic modulation classification on RadioML 2018.01A when labelled data is limited, compared with training the same CNN from scratch?

It has four sections:

| Section | What it shows |
|---|---|
| Signal Explorer | A real signal from the **held-out test split**: true class, SNR, I/Q waveform, constellation |
| Live Prediction | Real inference with the primary-grid scratch and CNN-MAE classifiers on that signal: prediction, model confidence (softmax), measured latency, top-5 classes, and agreement with the stored test-set prediction |
| Experimental Results | Two tabs. **Primary experiments** (authoritative): test results recomputed from the stored result files, with macro-F1 and accuracy by label budget, per-seed differences, seed counts, a cautious interpretation, and per-SNR accuracy (descriptive). **Exploratory · 50-epoch** and **Exploratory · 100-epoch follow-ups** (separate tabs): the six later 1% runs of each study (CNN-MAE vs scratch, seeds 42–44), with per-seed validation and test macro-F1, mean ± SD, paired differences, the matched 10- vs 50-epoch comparison, training curves, and their limitations |
| About the Experiment | Task, data, models, the two training stages (5-epoch MAE pretraining vs classifier fine-tuning), the primary protocol, the exploratory follow-up, limitations and a cautious conclusion |

The app is read-only. It never trains, never writes into `C:\dlruns`, and never changes checkpoints, results or the report.

## Setup (Windows PowerShell, once)

The project environment `C:\venvs\dlproj` already has PyTorch 2.10.0+cu126, NumPy and h5py. The demo needs only **Streamlit**, which also brings in Altair (charts) and pandas:

```powershell
C:\venvs\dlproj\Scripts\python.exe -m pip install -r C:\Users\admin\OneDrive\Desktop\DL-Project\demo\requirements-demo.txt
```

Expected side effect: current Streamlit releases require Pillow below 13, so pip will replace Pillow 13.0.0 (installed with matplotlib) with a 12.x release. matplotlib (which needs Pillow 8 or newer) and PyTorch keep working. Streamlit also adds new packages (pandas, pyarrow, altair, tornado and others); pip lists every change, and Pillow is the only downgrade expected from the current package list.

## Launch

```powershell
cd C:\Users\admin\OneDrive\Desktop\DL-Project\demo
C:\venvs\dlproj\Scripts\python.exe -m streamlit run app.py
```

Then open the address it prints (normally http://localhost:8501).

Launching from the `demo` folder applies the theme in `demo\.streamlit\config.toml`. The app also works when started from another folder, because all paths are resolved from the location of `app.py`.

## Smoke test

```powershell
cd C:\Users\admin\OneDrive\Desktop\DL-Project
C:\venvs\dlproj\Scripts\python.exe demo\smoke_test.py --quick   # about 1 minute: 5% seed-42 models + supplementary
C:\venvs\dlproj\Scripts\python.exe demo\smoke_test.py           # all 22 primary models + supplementary
```

It ends with `ALL CHECKS PASSED` or lists the failed checks. See `SMOKE_TEST.md` for what each check covers.

## Configuration

Defaults suit this machine. To override a path, set an environment variable before launching, for example `$env:DLPROJ_RUNS = "D:\dlruns"`.

| Variable | Default | Used for |
|---|---|---|
| `DLPROJ_ROOT` | the parent folder of `demo\` | `data_splits.npz`, the project's `src\` code, `report\tables\values.json` |
| `DLPROJ_RUNS` | `C:\dlruns` | checkpoints (`finetune\`), result files (`results\`), exploratory runs (`finetune_50ep_v1\`, `results_50ep_v1_exploratory\`) |
| `RADIOML_CACHE` | `<runs>\cache` | memory-mapped test signals (`cache\test`), the fast path |
| `RADIOML_HDF5` | `C:\RadioML2018\raw\GOLD_XYZ_OSC.0001_1024.hdf5` | fallback reader if the cache is missing |

## Files

| File | Purpose |
|---|---|
| `app.py` | the Streamlit app |
| `demo_config.py` | paths, experiment plan, reference values, colours |
| `data_access.py` | held-out test split access (reuses `src\data\cached_dataset.py` / `radioml_dataset.py`) |
| `inference.py` | checkpoint discovery and validation, model loading (reuses `src\models\cnn.py`), prediction |
| `results_data.py` | verified primary results recomputed from `C:\dlruns\results` |
| `exploratory_data.py` | exploratory 50-epoch runs, discovered from their manifests and cross-checked against their test results; never pooled with the primary results |
| `charts.py` | Altair charts |
| `smoke_test.py`, `SMOKE_TEST.md` | automated checks and checklist |
| `ARTIFACTS.md` | the checkpoints and result files used, with provenance |
| `requirements-demo.txt` | the one extra dependency |
