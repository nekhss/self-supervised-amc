# Read-only diagnostics for the DL-Project training state. Does not modify or stop anything.
param([int]$TargetPid = 21864)
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Write-Output "=== Time: $(Get-Date -Format o)"
Write-Output "=== Python processes"
Get-CimInstance Win32_Process -Filter "Name like 'python%'" |
  Select-Object ProcessId, ParentProcessId, CreationDate,
    @{n='WS_MB';e={[math]::Round($_.WorkingSetSize/1MB,1)}}, CommandLine |
  Format-List
$p = Get-Process -Id $TargetPid -ErrorAction SilentlyContinue
if ($p) {
  $c1 = $p.CPU; $io1 = (Get-CimInstance Win32_Process -Filter "ProcessId=$TargetPid").ReadTransferCount
  Start-Sleep -Seconds 20
  $p.Refresh(); $c2 = $p.CPU; $io2 = (Get-CimInstance Win32_Process -Filter "ProcessId=$TargetPid").ReadTransferCount
  Write-Output ("=== PID {0}: started {1}; responding={2}; threads={3}" -f $TargetPid, $p.StartTime, $p.Responding, $p.Threads.Count)
  Write-Output ("CPU s: {0:N1} -> {1:N1} over 20s (= {2:N2} cores)" -f $c1, $c2, (($c2-$c1)/20))
  Write-Output ("Read bytes over 20s: {0:N1} MB" -f (($io2-$io1)/1MB))
  Write-Output ("WS MB: {0:N1}; Private MB: {1:N1}" -f ($p.WorkingSet64/1MB), ($p.PrivateMemorySize64/1MB))
  $par = (Get-CimInstance Win32_Process -Filter "ProcessId=$TargetPid").ParentProcessId
  Get-CimInstance Win32_Process -Filter "ProcessId=$par" | Select-Object ProcessId, Name, CommandLine | Format-List
} else { Write-Output "=== PID $TargetPid NOT RUNNING" }
Write-Output "=== GPU"
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv } else { Write-Output "nvidia-smi not found" }
Write-Output "=== Recent files under project (excluding .venv), last 6h"
Get-ChildItem $root -Recurse -File -ErrorAction SilentlyContinue |
  Where-Object { $_.FullName -notmatch '\\.venv\\' -and $_.LastWriteTime -gt (Get-Date).AddHours(-6) } |
  Sort-Object LastWriteTime -Descending | Select-Object LastWriteTime, Length, FullName | Format-Table -AutoSize
Write-Output "=== Python env"
& "$root\.venv\Scripts\python.exe" -c "import torch,numpy,h5py,sklearn,sys;print(sys.version.split()[0],'torch',torch.__version__,'cuda',torch.cuda.is_available(),'numpy',numpy.__version__,'h5py',h5py.__version__,'sklearn',sklearn.__version__)"
& "$root\.venv\Scripts\python.exe" "$PSScriptRoot\inspect_checkpoints.py"
