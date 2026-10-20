# Start the Stage 4 mixtape training run - or, if it has a checkpoint
# already, resume it. Double-click tools\train_mixtape.cmd to run this in a
# window of its own (closing that window stops training; double-click again
# to resume from the last checkpoint, saved about every hour).
#
# Everything the run prints is also appended to
#   %USERPROFILE%\tlm-data\mixtape-train.log
# so samples and losses can be pasted into JOURNAL.md later.
#
# If training CRASHES (train.py exits with code 3 - e.g. the GPU ran out of
# memory because another program took some), this waits 2 minutes and
# resumes from the last checkpoint by itself, up to 10 times.

$Host.UI.RawUI.WindowTitle = "TLM mixtape training - closing this window stops it"
Set-Location (Split-Path $PSScriptRoot -Parent)          # the tlm/ folder

# The venv recommended in the README; set TLM_PYTHON to use another.
$py = if ($env:TLM_PYTHON) { $env:TLM_PYTHON } else {
    Join-Path $env:USERPROFILE "tlm-venv\Scripts\python.exe" }
$ckpt = if ($env:TLM_CKPT_DIR) { $env:TLM_CKPT_DIR } else {
    Join-Path $env:USERPROFILE "tlm-checkpoints" }
$data = if ($env:TLM_DATA_DIR) { $env:TLM_DATA_DIR } else {
    Join-Path $env:USERPROFILE "tlm-data" }
$log = Join-Path $data "mixtape-train.log"
$wait = if ($env:TLM_RESUME_WAIT) { [int]$env:TLM_RESUME_WAIT } else { 120 }
$maxResumes = 10

$trainArgs = @("-u", "-m", "tlm.train", "--config", "mixtape")
if (Test-Path (Join-Path $ckpt "mixtape\latest.pt")) { $trainArgs += "--resume" }
$trainArgs += $args          # extra options, e.g. train_mixtape.cmd --rewarm 500

for ($resumes = 0; ; $resumes++) {
    "=== $(Get-Date -Format s)  python $($trainArgs -join ' ')" |
        Add-Content -LiteralPath $log -Encoding UTF8
    # Show each line here AND append it to the log (a "tee").
    & $py @trainArgs 2>&1 | ForEach-Object {
        "$_"
        Add-Content -LiteralPath $log -Value "$_" -Encoding UTF8
    }
    $code = $LASTEXITCODE
    if ($code -ne 3 -or $resumes -ge $maxResumes) { break }
    $msg = "training crashed (exit code 3); resuming from the last checkpoint " +
           "in $wait s (auto-resume $($resumes + 1) of $maxResumes)"
    Write-Host ""
    Write-Host $msg -ForegroundColor Yellow
    "=== $(Get-Date -Format s)  $msg" | Add-Content -LiteralPath $log -Encoding UTF8
    Start-Sleep -Seconds $wait
    # resume plainly: one-off options such as --rewarm applied to the first launch
    $trainArgs = @("-u", "-m", "tlm.train", "--config", "mixtape", "--resume")
}
Write-Host ""
Write-Host "Training stopped (exit code $code). If it didn't finish,"
Write-Host "double-click tools\train_mixtape.cmd to resume from the last checkpoint."
