# autorun.ps1 - runs the remaining final-run steps in order, stops at the first failure.
# Phase A : finish held_out_v3. Phase B : held_out_v4c, energy, metrics, significance, report.
# If it stops: paste the last lines to the advisor, fix, run the SAME command again.

param(
    [Parameter(Mandatory = $true)][ValidateSet("A", "B")][string]$Phase,
    [switch]$RerunVerified,
    [switch]$V3Reviewed
)

$ErrorActionPreference = "Continue"
$Repo = "C:\Users\USER\llm-6dof-arm"
$GPU  = "D:\condaenvs\thesis2gpu\python.exe"
$OLL  = "C:\Users\USER\.conda\envs\thesis\python.exe"
$Out  = Join-Path $Repo "thesis2\out\final"
$env:OLLAMA_MODELS = "D:\LLM"
$env:PYTHONIOENCODING = "utf-8"
Set-Location $Repo

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$StepLog = Join-Path $Out ("logs\autorun_phase{0}_{1}.log" -f $Phase, $stamp)
$script:LastIdle = $null

function Log([string]$msg) {
    $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Write-Host $line
    Add-Content -Path $StepLog -Value $line -Encoding ascii
}

function Stop-Run([string]$why) {
    Log ("AUTORUN STOPPED: " + $why)
    Log "Paste the lines above to your advisor. Fix, then run the SAME command again."
    exit 1
}

function Run-Step([string]$Label, [string]$Exe, [string[]]$ArgList) {
    Write-Host ""
    Write-Host ("=" * 90)
    Log ("START  " + $Label)
    Write-Host ("=" * 90)
    $t0 = Get-Date
    & $Exe @ArgList
    $code = $LASTEXITCODE
    $mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    if ($code -ne 0) { Stop-Run ("{0}  (exit code {1}, after {2} min)" -f $Label, $code, $mins) }
    Log ("DONE   {0}  ({1} min)" -f $Label, $mins)
}

function Has-Pred([string]$arm, [string]$split) {
    return (Test-Path (Join-Path $Out ("pred_{0}_{1}.csv" -f $arm, $split)))
}

function Check-RowCount([string]$split) {
    $f = Join-Path $Out ("results_v5cells_{0}.csv" -f $split)
    if (-not (Test-Path $f)) { Stop-Run ("missing " + $f) }
    $n = @(Import-Csv $f).Count
    if ($n -ne 12152) { Stop-Run ("{0} has {1} rows, expected 12152" -f $f, $n) }
    Log ("CHECK  {0}: 12152 rows" -f $split)
}

function Has-Energy([string]$name, [string]$split) {
    $f = Join-Path $Out "unified_efficiency.csv"
    if (-not (Test-Path $f)) { return $false }
    $rows = @(Import-Csv $f | Where-Object { $_.arm -eq $name -and $_.split -eq $split })
    return ($rows.Count -gt 0)
}

function Ensure-Idle {
    $stale = ($script:LastIdle -eq $null) -or (((Get-Date) - $script:LastIdle).TotalMinutes -ge 100)
    if ($stale) {
        Run-Step "S3-1 idle baseline" $GPU @("thesis2\final\s3_energy.py", "--idle")
        $script:LastIdle = Get-Date
    }
}

function Print-Accuracy([string]$split) {
    Log ("SUMMARY action match per arm on " + $split)
    Get-ChildItem (Join-Path $Out ("pred_*_{0}.csv" -f $split)) | Sort-Object Name | ForEach-Object {
        $rows = @(Import-Csv $_.FullName)
        $ok = @($rows | Where-Object { @("1", "true", "1.0") -contains ([string]$_.correct).ToLower() }).Count
        $arm = $_.Name.Replace("pred_", "").Replace(("_{0}.csv" -f $split), "")
        Log ("   {0,-40} {1:N4}" -f $arm, ($ok / [math]::Max($rows.Count, 1)))
    }
}

$LocalArms = @("flat_gpu","flat_cpu","factorized_fp16","factorized_int4","tinybert","minilm_probe",
    "knn_minilm","rules_open","rules_firewalled","tfidf_logreg","bilstm","textcnn","mlpbow","flant5",
    "lora_greedy","salsa_full_qwen2.5-3b-lora","salsa_full_Qwen2.5-3B-Instruct",
    "salsa_full_qwen2.5-0.5b-lora","salsa_full_Qwen2.5-0.5B-Instruct","grpo_base","grpo_sft",
    "cascade_local_tauA","pipeline_always_local","cascade_oracle")

$Spec = @(
    @{ arm = "a8_speculative";  script = "thesis2\scripts\a8_speculative.py" },
    @{ arm = "a10_speculative"; script = "thesis2\scripts\a10_prefix_cache.py" },
    @{ arm = "a12_smalllm";     script = "thesis2\scripts\a12_smalllm_drafter.py" }
)

$V5Cells = @("ollama:qwen2.5-3b-lora/few_shot","ollama:qwen2.5-3b-lora/cot","ollama:qwen2.5-3b-lora/role_based",
    "ollama:qwen2.5-3b-lora/rag","ollama:qwen2.5:3b/zero_shot","ollama:qwen2.5:3b/few_shot","ollama:qwen2.5:3b/cot",
    "ollama:qwen2.5:3b/role_based","ollama:qwen2.5:3b/rag","ollama:qwen2.5-0.5b-lora/zero_shot",
    "ollama:qwen2.5-0.5b-lora/few_shot","ollama:qwen2.5-0.5b-lora/cot","ollama:qwen2.5-0.5b-lora/role_based",
    "ollama:qwen2.5-0.5b-lora/rag","ollama:qwen2.5:0.5b/zero_shot","ollama:qwen2.5:0.5b/few_shot",
    "ollama:qwen2.5:0.5b/cot","ollama:qwen2.5:0.5b/role_based","ollama:qwen2.5:0.5b/rag",
    "ollama:llama3.2:3b/zero_shot","ollama:llama3.2:3b/few_shot","ollama:llama3.2:3b/cot",
    "ollama:llama3.2:3b/role_based","ollama:llama3.2:3b/rag","ollama:deepseek-r1:7b/zero_shot",
    "ollama:deepseek-r1:7b/few_shot","ollama:deepseek-r1:7b/cot","ollama:deepseek-r1:7b/role_based",
    "ollama:deepseek-r1:7b/rag","classifier:distilbert/classifier")

$EnergyArms = $LocalArms[0..13] + @("lora_greedy","a8_speculative","a10_speculative","a12_smalllm") + $LocalArms[15..23]

if ($Phase -eq "A") {
    if (-not $RerunVerified) { Stop-Run "Phase A needs -RerunVerified: the 12-cell re-run must have passed and been shown to the advisor." }
    Log "PHASE A - finish held_out_v3"
    Run-Step "S1-4 V5 cloud cells, held_out_v3 (5,425 billed calls)" $OLL @("thesis2\final\s1_run_v5_cells.py","--split","held_out_v3","--kind","cloud")
    Check-RowCount "held_out_v3"
    foreach ($a in @("cascade_cloud_tauA","cascade_cloud_always","ensemble")) {
        Run-Step ("S1-5 {0} held_out_v3" -f $a) $GPU @("thesis2\final\s1_run_arm.py","--arm",$a,"--split","held_out_v3")
    }
    Run-Step "S1-5b reproducibility gate held_out_v3" $GPU @("thesis2\final\s1_repro_check.py","--split","held_out_v3")
    Print-Accuracy "held_out_v3"
    Log "PHASE A COMPLETE. Paste this log to the advisor. Do NOT start Phase B before review."
    Log ("step log: " + $StepLog)
    exit 0
}

if ($Phase -eq "B") {
    if (-not $V3Reviewed) { Stop-Run "Phase B needs -V3Reviewed: the advisor must review Phase A first. v4c runs only once." }
    if (-not (Has-Pred "a8_speculative" "held_out_v3")) { Stop-Run "pred_a8_speculative_held_out_v3.csv missing: the speculative pred-file fix is not in place yet." }
    Log "PHASE B - held_out_v4c, energy, metrics, significance, report. Keep the machine untouched."

    foreach ($a in $LocalArms) {
        if (Has-Pred $a "held_out_v4c") { Log ("SKIP   {0} held_out_v4c (exists)" -f $a); continue }
        Run-Step ("S1-6 {0} held_out_v4c" -f $a) $GPU @("thesis2\final\s1_run_arm.py","--arm",$a,"--split","held_out_v4c")
    }
    foreach ($s in $Spec) {
        if (Has-Pred $s.arm "held_out_v4c") { Log ("SKIP   {0} held_out_v4c (exists)" -f $s.arm); continue }
        Run-Step ("S1-6 {0} held_out_v4c" -f $s.arm) $GPU @($s.script,"--split","held_out_v4c")
    }
    Run-Step "S1-6 V5 local cells, held_out_v4c" $OLL @("thesis2\final\s1_run_v5_cells.py","--split","held_out_v4c","--kind","local")
    Run-Step "S1-6 V5 cloud cells, held_out_v4c" $OLL @("thesis2\final\s1_run_v5_cells.py","--split","held_out_v4c","--kind","cloud")
    Check-RowCount "held_out_v4c"
    foreach ($a in @("cascade_cloud_tauA","cascade_cloud_always","ensemble")) {
        if (Has-Pred $a "held_out_v4c") { Log ("SKIP   {0} held_out_v4c (exists)" -f $a); continue }
        Run-Step ("S1-6 {0} held_out_v4c" -f $a) $GPU @("thesis2\final\s1_run_arm.py","--arm",$a,"--split","held_out_v4c")
    }
    Print-Accuracy "held_out_v4c"

    foreach ($sp in @("held_out_v3","held_out_v4c")) {
        foreach ($c in @("ollama:qwen2.5-3b-lora/zero_shot") + $V5Cells) {
            if (Has-Energy $c $sp) { Log ("SKIP   energy {0} {1}" -f $c, $sp); continue }
            Ensure-Idle
            Run-Step ("S3-2 energy {0} {1}" -f $c, $sp) $OLL @("thesis2\final\s3_energy.py","--v5-cell",$c,"--split",$sp)
        }
    }
    foreach ($sp in @("held_out_v3","held_out_v4c")) {
        foreach ($a in $EnergyArms) {
            if (Has-Energy $a $sp) { Log ("SKIP   energy {0} {1}" -f $a, $sp); continue }
            Ensure-Idle
            Run-Step ("S3-2 energy {0} {1}" -f $a, $sp) $GPU @("thesis2\final\s3_energy.py","--arm",$a,"--split",$sp)
        }
    }

    Run-Step "S2 metrics" $GPU @("thesis2\final\s2_metrics.py")
    Run-Step "S4-1 significance" $GPU @("thesis2\final\s4_significance.py")
    Run-Step "S4-2 final table" $GPU @("thesis2\final\s4_metrics_final.py")
    Run-Step "S4-3 figures" $GPU @("thesis2\final\s4_figures.py")
    Run-Step "S6 report" $GPU @("thesis2\final\s6_report.py")

    $sig = Join-Path $Out "final_significance.csv"
    if (Test-Path $sig) {
        Log "PRIMARY FAMILY"
        Import-Csv $sig | Where-Object { $_.family -eq "primary" } | Format-Table -AutoSize | Out-String | ForEach-Object { Log $_ }
    }
    Log "PHASE B COMPLETE. Commit, then paste this log to the advisor."
    Log ("step log: " + $StepLog)
    exit 0
}
