# Runs one campaign block end to end: pre-checks, dry run, the block, verify, backup.
# Usage (from any PowerShell window):
#   powershell -ExecutionPolicy Bypass -File C:\Users\USER\llm-6dof-arm\thesis3\bench\block.ps1 -Block A
# Blocks in order: A, B, C1, C2, D. Re-running the same block resumes it.
param(
    [Parameter(Mandatory = $true)][ValidateSet('A', 'B', 'C1', 'C2', 'D')][string]$Block,
    [switch]$SkipDry
)
$ErrorActionPreference = 'Continue'
$Repo = 'C:\Users\USER\llm-6dof-arm'
$Py = 'D:\condaenvs\thesis3meter\python.exe'
$Out = 'thesis3\out\c1'
$Backup = 'D:\thesis3_backup\c1'
Set-Location $Repo
$log = "thesis3\out\block_${Block}_$(Get-Date -Format yyyyMMdd_HHmmss).log"
New-Item -ItemType Directory -Force thesis3\out | Out-Null
Start-Transcript -Path $log | Out-Null

function Stop-Here($msg) {
    Write-Host ""
    Write-Host "STOPPED: $msg" -ForegroundColor Red
    Write-Host "Paste this window's last lines (or $log) to Claude."
    Stop-Transcript | Out-Null
    exit 1
}

Write-Host "== 1/6 pre-checks ==" -ForegroundColor Cyan
try { Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:8085/data.json' -TimeoutSec 5 | Out-Null }
catch { Stop-Here 'LibreHardwareMonitor web server not reachable: start it as administrator, Options > Remote Web Server > Run.' }
Stop-Process -Name chrome, iCUE, 'Camera Hub', RadeonSoftware, AMDRSSrcExt, vgtray, SystemSettings, logioptionsplus_agent -ErrorAction SilentlyContinue
powercfg /setactive SCHEME_MIN
& $Py thesis3\bench\telemetry.py --probe | Select-Object -Last 2
if ($LASTEXITCODE -ne 0) { Stop-Here 'telemetry probe failed' }

if (-not $SkipDry) {
    Write-Host "== 2/6 dry run of block $Block (3 commands per system, nothing saved) ==" -ForegroundColor Cyan
    & $Py thesis3\bench\run_bench.py --dry $Block
    if ($LASTEXITCODE -ne 0) { Stop-Here "dry run of block $Block failed" }
}

Write-Host "== 3/6 block $Block (switch the monitor off now; do not use the PC) ==" -ForegroundColor Cyan
& $Py thesis3\bench\run_bench.py --campaign c1 --block $Block
$runExit = $LASTEXITCODE

Write-Host "== 4/6 verify ==" -ForegroundColor Cyan
& $Py thesis3\bench\verify_campaign.py $Out --partial
$verifyExit = $LASTEXITCODE

Write-Host "== 5/6 backup to $Backup ==" -ForegroundColor Cyan
robocopy $Out $Backup /MIR /NFL /NDL /NJH /NP | Out-Host
if ($LASTEXITCODE -ge 8) { Stop-Here "robocopy failed with code $LASTEXITCODE" }

Write-Host "== 6/6 check the backup ==" -ForegroundColor Cyan
& $Py thesis3\bench\verify_campaign.py $Backup --check-manifest
$manExit = $LASTEXITCODE

Write-Host ""
if ($runExit -ne 0) { Stop-Here "run_bench stopped with exit code $runExit (a gate fired); data so far is verified and backed up" }
if ($verifyExit -ne 0) { Stop-Here 'verify found a FAIL (see the FAIL lines above)' }
if ($manExit -ne 0) { Stop-Here 'backup copy does not match the manifest' }
Write-Host "BLOCK $Block COMPLETE: verified and backed up. Copy $Backup off the machine too." -ForegroundColor Green
Stop-Transcript | Out-Null
