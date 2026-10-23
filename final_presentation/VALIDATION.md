# Slide-by-slide validation checklist

Status: ✅ checked · ⚠️ checked with a caveat · ❌ not possible here

## Automated checks (last build)

- ✅ 11 data assertions in `build_deck.py` passed (`build_checks.json`):
  - primary macro-F1 at all 5 budgets;
  - 5% paired p = 0.29;
  - 25% and 100% have one seed each;
  - 50- and 100-epoch summaries, with MAE ahead in 2 of 3 seeds;
  - low-SNR band values;
  - every longer run beats its 10-epoch run.
- ✅ The source CSVs come from `final_artifacts\make_final_figures.py`, which passed 57 checks against the raw result files (`final_artifacts\VERIFICATION.md`).
- ✅ The skill's OOXML validator reports "All validations PASSED".
- ✅ python-pptx re-parses the deck: 14 slides, no shape extends past the slide edge.
- ✅ Slides 7 and 8 have 15 connectors, all glued to shapes at both ends and with arrowheads. Diagrams are native, editable shapes.
- ✅ The deck rendered in Microsoft PowerPoint (COM) to a PDF and 14 PNGs. Every slide was inspected visually; title wrapping, a diagonal arrow, a figure/table overlap and a box running into the footer were found and fixed.

## Per slide

| # | Checks |
|---|---|
| 1 | ✅ Title, all three names and roll numbers, course, group and programme match the mid-review. ✅ The waveform is real data: QPSK, +30 dB, test index 1,229. |
| 2 | ✅ Signals are real (QPSK at +18 dB, test index 1,419; at −10 dB, test index 1,539). ✅ The research question is worded as specified. |
| 3 | ✅ Status is shown honestly. The linear probe and the test of whether reconstruction error predicts accuracy were planned in the mid-review but not run. ✅ The transfer study's differences (−0.009 to −0.012) come from `C:\dlruns\rml2016_transfer\report_v1\audit_final.json`. |
| 4 | ✅ Studies and years match the mid-review reference list. ✅ Raw-IQ MAE is not claimed as new; RIS-MAE is named as the closest prior work. ⚠️ Bibliographic details were checked against the mid-review list and `report\references.bib` only, not against the original publishers. |
| 5 | ✅ No architecture novelty is claimed. ✅ 55,544 parameters (from `demo\ARTIFACTS.md` and the report). |
| 6 | ✅ 2,555,904 signals, 24 classes and 26 SNR levels (from the report's dataset audit). ✅ The split sizes and the five label-budget sizes come from the protocol and report. ⚠️ No per-class sample count is shown, because no saved class-distribution artifact was found to back it. |
| 7 | ✅ The split is stratified by class × SNR (`create_splits.py`). ✅ Label subsets are nested and class-balanced, but not balanced by SNR (`create_label_subsets_v2.py`). ✅ No normalisation or augmentation is applied (data loaders). ✅ The loader rejects non-finite samples and invalid labels. No duplicate removal is claimed. |
| 8 | ✅ The diagram matches `src\models\cnn.py` and `src\models\cnn_mae.py`: 16 × 64 patches, 50% masking, encoder on visible patches only, 2-layer Transformer decoder, MSE on masked patches, mean-pooled patch features into a linear head, full fine-tuning. |
| 9 | ✅ Settings match `train_classifier.py`, `pretrain_cnn_mae_fixed.py` and the run manifests. ✅ Environment (PyTorch 2.10.0+cu126, RTX 4050 Laptop) is as recorded. ⚠️ "Python 3.11" is the interpreter version of the venv. |
| 10 | ✅ The table and native chart are built from `primary_summary.csv` and match the specified values to 4 d.p. ✅ The one-seed budgets are stated. |
| 11 | ✅ The figure is the verified fig4 (all budgets, full SNR range). ✅ SNR-band means are computed from `primary_per_snr_accuracy.csv`, and the slide labels them descriptive. ⚠️ No precision or recall is shown: they are saved per class, but no aggregate was verified for slides. |
| 12 | ✅ The slide is labelled exploratory. ✅ Values come from `exploratory_{50,100}ep_summary.csv` and the per-seed CSVs, and match the specified summaries. ✅ The figure is the verified fig7. |
| 13 | ✅ Limitations include p = 0.29, the single-seed budgets, the exploratory status and the deployment gap. ✅ The conclusion makes no claim of a consistent advantage. |
| 14 | ✅ References are from the mid-review list, plus the DeepSig dataset page (read directly). ⚠️ DOIs are not given, because none were verified. |

## Not possible or not done

- ❌ Testing in presentation mode on a projector. The PNG renders were checked at 1600 × 900; body text is at least 12 pt, with captions at 11 pt.
- ⚠️ The deck uses the fonts Cambria and Calibri, which ship with Office.
