# Runs the whole campaign unattended: blocks A, B, C1, C2, D back to back, then finish.ps1.
#   powershell -ExecutionPolicy Bypass -File C:\Users\USER\llm-6dof-arm\thesis3\bench\campaign.ps1
# - A block that finished green is marked thesis3\out\block_<X>.done and skipped on a restart.
# - A block that stops red is retried once automatically (it resumes, finished runs are skipped).
#   If it stops again, the chain stops there and nothing after it runs.
# - Restart after any stop or crash: run the same command again.
$ErrorActionPreference = 'Continue'
$Repo = 'C:\Users\USER\llm-6dof-arm'
Set-Location $Repo
New-Item -ItemType Directory -Force thesis3\out | Out-Null
$Status = 'thesis3\out\CAMPAIGN_STATUS.txt'
function Say($msg, $color = 'White') {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $msg"
    Write-Host $line -ForegroundColor $color
    Add-Content -Path $Status -Encoding ascii -Value $line
}

# no sleep or hibernation while the chain runs (the sampler also keeps the PC awake during blocks)
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /setactive SCHEME_MIN

Say 'campaign start'
foreach ($b in 'A', 'B', 'C1', 'C2', 'D') {
    $done = "thesis3\out\block_$b.done"
    if (Test-Path $done) { Say "block $b already complete, skipped"; continue }
    Say "block $b start" 'Cyan'
    & powershell -ExecutionPolicy Bypass -File "$Repo\thesis3\bench\block.ps1" -Block $b
    if ($LASTEXITCODE -ne 0) {
        Say "block $b stopped, retrying once (resumes, finished runs are skipped)" 'Yellow'
        & powershell -ExecutionPolicy Bypass -File "$Repo\thesis3\bench\block.ps1" -Block $b -SkipDry
    }
    if ($LASTEXITCODE -ne 0) {
        Say "CAMPAIGN STOPPED in block $b. Paste the red STOPPED line and the lines above it to Claude." 'Red'
        exit 1
    }
    New-Item -ItemType File -Force $done | Out-Null
    Say "block $b complete" 'Green'
}

Say 'all blocks complete, running finish.ps1' 'Cyan'
& powershell -ExecutionPolicy Bypass -File "$Repo\thesis3\bench\finish.ps1"
Say 'campaign finished: send Claude thesis3\out\thesis3_results.zip; copy D:\thesis3_backup\c1 off the machine' 'Green'
