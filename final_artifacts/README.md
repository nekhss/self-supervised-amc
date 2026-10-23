# Final figures and tables

Regenerate (read-only on all experiment outputs):

```powershell
C:\venvs\dlproj\Scripts\python.exe C:\Users\admin\OneDrive\Desktop\DL-Project\final_artifacts\make_final_figures.py
```

The script reads results through `demo\results_data.py` and `demo\exploratory_data.py`, which validate manifests, checkpoint hashes and the one-to-one mapping between checkpoints and test results. It then re-reads every plotted value from the raw files and asserts that the two agree. The outcome is written to `VERIFICATION.md`.

| Figure | Content | Stage |
|---|---|---|
| fig1 / fig2 | test macro-F1 / accuracy vs label budget; mean ± SD and individual seeds | primary |
| fig3 | per-seed MAE − scratch with test-set bootstrap CIs; 5% paired t-test p = 0.29 | primary |
| fig4 | per-SNR test accuracy, all budgets, full SNR range | primary |
| fig5 / fig6 | per-seed MAE vs scratch, validation and test macro-F1 | exploratory 50 / 100 epochs |
| fig7 | same arm and seed at 10 (primary), 50 and 100 (exploratory) epochs | both, separated |
| fig8 | training/validation loss, validation macro-F1 and accuracy per seed, with the selected epoch marked | exploratory 50 / 100 epochs |
| fig9a / fig9b | per-class F1 and row-normalised confusion matrices at 5% (mean of 3 seeds) | primary |

The tables in `tables\` hold the data behind each figure.

Notes:
- The exploratory studies were designed after the primary test results were known.
- Three seeds do not support claims of significance.
- `paired_comparison.csv` stores values rounded to 4 d.p., so its CIs are plotted at that precision.
- Training loss in fig8 is the value logged at 6 d.p.
