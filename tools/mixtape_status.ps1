# One-shot health check of the Stage 4 mixtape training run (read-only).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\mixtape_status.ps1
#
# Checks that training is alive and advancing, that throughput and losses
# look normal, the GPU's temperature and slowdown flags (from the log kept
# by tools\gpu_health.ps1, plus one live reading) and the free disk space.
# Prints a report whose FIRST line is one of
#   STATUS: OK
#   STATUS: PROBLEM - <what is wrong>
#   STATUS: FINISHED - <final result>
# and appends the same report to %USERPROFILE%\tlm-data\mixtape-checkins.md.
# It never starts, stops or changes anything else - which makes it the one
# command the scheduled check-in needs permission to run.

$ErrorActionPreference = "Continue"
$data = if ($env:TLM_DATA_DIR) { $env:TLM_DATA_DIR } else {
    Join-Path $env:USERPROFILE "tlm-data" }
$ckpt = if ($env:TLM_CKPT_DIR) { $env:TLM_CKPT_DIR } else {
    Join-Path $env:USERPROFILE "tlm-checkpoints" }
$evalCsv = Join-Path $ckpt "mixtape\log.csv"
$trainLog = Join-Path $data "mixtape-train.log"
$healthCsv = Join-Path $data "gpu-health.csv"
$report = Join-Path $data "mixtape-checkins.md"
$maxSteps, $tokensPerStep = 76000, 32768
$now = Get-Date
$problems = New-Object System.Collections.Generic.List[string]
$lines = New-Object System.Collections.Generic.List[string]

# --- finished?
$done = $null
if (Test-Path $trainLog) {
    $done = Select-String -LiteralPath $trainLog -Pattern "^done\. best val loss" |
        Select-Object -Last 1
}

# --- training process
$proc = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like "*tlm.train --config mixtape*" }
if (-not $proc -and -not $done) {
    $problems.Add("training process not running (double-click tools\train_mixtape.cmd to resume)")
}

# --- eval log
$rows = @()
if (Test-Path $evalCsv) { $rows = @(Import-Csv -LiteralPath $evalCsv) }
if ($rows.Count -lt 2) {
    $problems.Add("no eval rows yet in $evalCsv")
} else {
    $last = $rows[-1]
    $step = [int]$last.step
    $vals = $rows | ForEach-Object { [double]$_.val_loss }
    $best = ($vals | Measure-Object -Minimum).Minimum
    $age = ($now - (Get-Item -LiteralPath $evalCsv).LastWriteTime).TotalMinutes
    $recent = @($rows | Select-Object -Last 4 | Where-Object { [int]$_.step -gt 0 } |
        ForEach-Object { [double]$_.tok_per_sec })
    $tps = ($recent | Measure-Object -Average).Average
    $lines.Add(("- Step {0:N0} of {1:N0} ({2:N1}%), eval log written {3:N0} min ago" -f
                $step, $maxSteps, (100 * $step / $maxSteps), $age))
    $lines.Add(("- Val loss {0} (best {1:N4}): book {2} | recipe {3} | review {4} | story {5} | wiki {6}" -f
                $last.val_loss, $best, $last.val_book, $last.val_recipe,
                $last.val_review, $last.val_story, $last.val_wiki))
    $lines.Add(("- Throughput {0:N0} tok/s (average of the last evals)" -f $tps))
    if (-not $done) {
        if ($age -gt 60) { $problems.Add(("no new eval in {0:N0} min (expected every ~28)" -f $age)) }
        if ($tps -lt 8000) { $problems.Add(("throughput low: {0:N0} tok/s (normal ~9,700)" -f $tps)) }
        $eta = $now.AddSeconds(($maxSteps - $step) * $tokensPerStep / [math]::Max($tps, 1))
        $lines.Add("- Projected finish: $($eta.ToString('ddd yyyy-MM-dd HH:mm'))")
    }
    if ($vals | Where-Object { [double]::IsNaN($_) }) { $problems.Add("NaN in the val loss") }
    if ([double]$last.val_loss -gt $best + 0.3) {
        $problems.Add(("val loss {0} is more than 0.3 above its best {1:N4}" -f $last.val_loss, $best))
    }
}

# --- GPU health log (written every 5 min by tools\gpu_health.ps1)
if (Test-Path $healthCsv) {
    $health = @(Import-Csv -LiteralPath $healthCsv | Where-Object { $_.temp_c -ne "" })
    $since = $now.AddHours(-6)
    $window = @($health | Where-Object { [datetime]$_.time -ge $since })
    if ($window.Count) {
        $maxTemp = ($window | ForEach-Object { [double]$_.temp_c } | Measure-Object -Maximum).Maximum
        $flagged = @($window | Where-Object {
            $_.hw_slowdown -eq "1" -or $_.sw_thermal -eq "1" -or $_.hw_thermal -eq "1" })
        $lines.Add(("- GPU over the last 6 h: max {0} C, {1} readings, {2} with a slowdown flag" -f
                    $maxTemp, $window.Count, $flagged.Count))
        if ($maxTemp -ge 83) { $problems.Add("GPU reached $maxTemp C (target 83, max operating 92)") }
        if ($flagged.Count) { $problems.Add("GPU heat/hardware slowdown flagged $($flagged.Count) times") }
        $tail = @($health | Select-Object -Last 3)
        if (-not $done -and $proc -and $tail.Count -eq 3 -and
                -not ($tail | Where-Object { [double]$_.util_pct -ge 20 })) {
            $problems.Add("GPU utilization under 20% for the last 15 min")
        }
    }
    $lastHealth = ($health | Select-Object -Last 1)
    if ($lastHealth -and -not $done) {
        $hAge = ($now - [datetime]$lastHealth.time).TotalMinutes
        if ($hAge -gt 20) {
            $problems.Add(("GPU health log not updated for {0:N0} min (double-click tools\gpu_health.cmd)" -f $hAge))
        }
    }
} elseif (-not $done) {
    $problems.Add("no GPU health log (double-click tools\gpu_health.cmd)")
}

# --- one live reading
$live = & nvidia-smi "--query-gpu=temperature.gpu,fan.speed,power.draw,utilization.gpu,memory.used" "--format=csv,noheader,nounits" 2>$null
if ($live) {
    $v = ($live -split ",") | ForEach-Object { $_.Trim() }
    $lines.Add("- Live: $($v[0]) C, fan $($v[1])%, $($v[2]) W, util $($v[3])%, $($v[4]) MiB")
} else {
    $problems.Add("nvidia-smi gave no reading")
}

# --- disk
$freeGB = (Get-PSDrive C).Free / 1GB
$lines.Add(("- Free on C: {0:N1} GB" -f $freeGB))
if ($freeGB -lt 10) { $problems.Add(("only {0:N1} GB free on C:" -f $freeGB)) }

# --- verdict
if ($done) {
    $status = "FINISHED - $($done.Line.Trim())"
} elseif ($problems.Count) {
    $status = "PROBLEM - $($problems -join '; ')"
} else {
    $status = "OK"
}
$out = @("STATUS: $status") + $lines
$out | ForEach-Object { Write-Output $_ }

if (-not (Test-Path $report)) {
    "# TLM mixtape check-ins" | Set-Content -LiteralPath $report -Encoding UTF8
}
$entry = @("", "## $($now.ToString('yyyy-MM-dd HH:mm')) - $status") + $lines
Add-Content -LiteralPath $report -Value $entry -Encoding UTF8
