# Phase 3: run only after cnn_mae_full_v1 pretraining has finished all 5 epochs.
# Checks the MAE, fine-tunes it on every budget/seed, evaluates every
# validation-selected checkpoint on the test split once, and builds the tables/figures.
param([string]$MaeCheckpoint = "C:\dlruns\cnn_mae_full_v1_best.pt")
$ErrorActionPreference = "Continue"
$py = "C:\venvs\dlproj\Scripts\python.exe"
Set-Location (Split-Path -Parent $PSScriptRoot)
function Step($title, [scriptblock]$cmd) {
  Write-Host "`n=== $title ===" -ForegroundColor Cyan
  & $cmd
  if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $title (exit $LASTEXITCODE)" -ForegroundColor Red; exit 1 }
}
if (Get-Process python -ErrorAction SilentlyContinue | Where-Object { (Get-CimInstance Win32_Process -Filter "ProcessId=$($_.Id)").CommandLine -match 'pretrain_cnn_mae' }) {
  Write-Host "MAE pretraining is still running. Wait for it to finish." -ForegroundColor Red; exit 1 }
Step "MAE reconstruction and transfer check" { & $py -u tools\check_mae.py $MaeCheckpoint }
Step "MAE fine-tuning grid" { & $py -u experiments\run_finetune_suite.py --arm mae --mae-checkpoint $MaeCheckpoint }
Step "Test evaluation (all validation-selected checkpoints)" { & $py -u experiments\evaluate_test.py "C:\dlruns\finetune\*.pt" --out-dir C:\dlruns\results }
Step "Tables and figures" { & $py -u experiments\summarize_results.py --results-dir C:\dlruns\results }
Write-Host "`nPhase 3 complete. Results in C:\dlruns\results" -ForegroundColor Green
