# Final presentation: Group 41

**Deck:** `Group41_CNN-MAE_Final_Presentation.pptx` (14 slides, 16:9, editable).
**Preview:** `Group41_CNN-MAE_Final_Presentation_preview.pdf` and `preview\slideNN.png`, rendered by the installed Microsoft PowerPoint.

| # | Slide | Main content |
|---|---|---|
| 1 | Title | Project title, team and roll numbers, course, group, programme; I/Q traces of a real test signal drawn as native lines |
| 2 | Problem and motivation | AMC, label cost, low SNR, unlabelled data; real QPSK signals at +18 dB and −10 dB; research question |
| 3 | Objectives | 7 objectives with status: completed / exploratory / separate study / not done |
| 4 | Literature and gap | 5-study table (from the mid-review); gap statement |
| 5 | Contributions | 6 experimental contributions; what is explicitly not claimed |
| 6 | Dataset | RadioML 2018.01A counts, 70/15/15 split, label-budget sizes and seeds; RadioML 2016.10a marked secondary |
| 7 | Preprocessing and leakage | Implemented pipeline; leakage controls; what was checked and what is not claimed |
| 8 | Architecture | Native shapes and glued connectors: stage 1 MAE pretraining, stage 2 paths A (scratch) and B (CNN-MAE), evaluation |
| 9 | Methodology and setup | Rationale; pretraining vs fine-tuning settings table; environment; primary vs exploratory tags |
| 10 | Primary results | Native line chart (macro-F1 vs budget) and results table, with a 5% p-value takeaway |
| 11 | Per-SNR | fig4 (all budgets, full SNR range) and a table of SNR-band differences |
| 12 | Longer fine-tuning (exploratory) | fig7 (10/50/100 epochs) and 50- and 100-epoch summary cards with per-seed Δ |
| 13 | Limitations, future work, conclusion | Three columns and a take-home message |
| 14 | References and questions | 9 references, including the DeepSig dataset page |

## Rebuild

```powershell
C:\venvs\pptx\Scripts\python.exe C:\Users\admin\OneDrive\Desktop\DL-Project\final_presentation\build_deck.py
powershell -ExecutionPolicy Bypass -File C:\Users\admin\OneDrive\Desktop\DL-Project\final_presentation\render_preview.ps1
```

- `C:\venvs\pptx` is a separate environment containing python-pptx 1.0.2 and pypdf; the project environment `C:\venvs\dlproj` was not changed.
- `build_deck.py` reads every result number from `final_artifacts\tables\*.csv`. It asserts the numbers against the verified values and stops if any check fails. Results of the last build are in `build_checks.json`.
- `assets\extract_signals.py` (run with `C:\venvs\dlproj`) produced `assets\signals.json`: three real QPSK test signals taken through the demo's verified test-split reader.
- Speaker notes are included on 8 slides.

See `VALIDATION.md` for the slide-by-slide checklist.
