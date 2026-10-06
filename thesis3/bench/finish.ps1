# After block D: analysis, final completeness gate, backup, and one zip of the results to send.
#   powershell -ExecutionPolicy Bypass -File C:\Users\USER\llm-6dof-arm\thesis3\bench\finish.ps1
$ErrorActionPreference = 'Continue'
Set-Location 'C:\Users\USER\llm-6dof-arm'
$Py = 'D:\condaenvs\thesis3meter\python.exe'
$Out = 'thesis3\out\c1'

& $Py thesis3\bench\analyze.py $Out
if ($LASTEXITCODE -ne 0) { Write-Host 'STOPPED: analyze.py failed' -ForegroundColor Red; exit 1 }
& $Py thesis3\bench\verify_campaign.py $Out --after-analysis
$v = $LASTEXITCODE
robocopy $Out 'D:\thesis3_backup\c1' /MIR /NFL /NDL /NJH /NP | Out-Host
& $Py thesis3\bench\verify_campaign.py 'D:\thesis3_backup\c1' --check-manifest

$files = 'summary.csv', 'report.md', 'quality.csv', 'compare.csv', 'per_loc_type.csv', 'per_verb.csv',
         'per_pass.csv', 'confusion.csv', 'verify_report.txt', 'repro_check.csv', 'runs.csv', 'machine.json'
$paths = $files | ForEach-Object { Join-Path $Out $_ } | Where-Object { Test-Path $_ }
Compress-Archive -Path ($paths + "$Out\figures") -DestinationPath 'thesis3\out\thesis3_results.zip' -Force
Write-Host ''
if ($v -eq 0) { Write-Host 'FINAL VERDICT PASS. Send Claude: thesis3\out\thesis3_results.zip' -ForegroundColor Green }
else { Write-Host 'FINAL VERDICT FAIL: send Claude the FAIL lines and thesis3\out\thesis3_results.zip' -ForegroundColor Red }
