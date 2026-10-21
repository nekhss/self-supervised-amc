# Phase 2 (safe to run while MAE pretraining is still going).
# 1) one sequential pass over the HDF5 to build caches outside OneDrive
# 2) 1-minute smoke test of the cached fine-tuning, test evaluation and MAE check
# 3) the full scratch-baseline grid (does not depend on the MAE checkpoint)
# Stops at the first failure. Writes only under C:\dlruns.
$ErrorActionPreference = "Continue"
$py = "C:\venvs\dlproj\Scripts\python.exe"
Set-Location (Split-Path -Parent $PSScriptRoot)
function Step($title, [scriptblock]$cmd) {
  Write-Host "`n=== $title ===" -ForegroundColor Cyan
  & $cmd
  if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $title (exit $LASTEXITCODE)" -ForegroundColor Red; exit 1 }
}
Step "Build caches" { & $py -u tools\build_cache.py --sets valsub test train_25pct train_100pct }
Step "Smoke: cached fine-tuning (20 batches)" {
  if (Test-Path C:\dlruns\smoke) { Remove-Item C:\dlruns\smoke -Recurse -Force }
  & $py -u experiments\finetune_cached.py --label-fraction 1 --epochs 1 --batch-size 128 --max-train-batches 20 --validation-subset validation_subset_v1.npz --seed 42 --output C:\dlruns\smoke\smoke_scratch.pt }
Step "Smoke: test evaluation" { & $py -u experiments\evaluate_test.py C:\dlruns\smoke\smoke_scratch.pt --out-dir C:\dlruns\smoke }
Step "Smoke: MAE check on pilot checkpoint" { & $py -u tools\check_mae.py checkpoints\cnn_mae_schedule_resume_test_best.pt }
Step "Scratch baseline grid" { & $py -u experiments\run_finetune_suite.py --arm scratch }
Write-Host "`nPhase 2 complete." -ForegroundColor Green
