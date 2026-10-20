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
# The extension (from step 79,000) re-warms the LR to 2e-4 by step 80,000,
# which RAISES the loss for a while by design. From then on, alarms compare
# against the best since 80,000; the step-79,000 baseline is shown apart.
$settleStep, $baseline = 80000, 2.5860
$maxSteps, $tokensPerStep = 117000, 32768   # 76,000, then 79,000, until the extension
$now = Get-Date
$problems = New-Object System.Collections.Generic.List[string]
$lines = New-Object System.Collections.Generic.List[string]

# --- finished?
$done = $null
if (Test-Path $trainLog) {
    # Only the current launch counts: an extended run's log already holds
    # an earlier "done." line.
    $launch = Select-String -LiteralPath $trainLog -Pattern "python -u -m tlm\.train" |
        Select-Object -Last 1
    $done = Select-String -LiteralPath $trainLog -Pattern "^done\. best val loss" |
        Where-Object { $launch -and $_.LineNumber -gt $launch.LineNumber } |
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
    # Judge by the AVERAGE of the five genre losses: the mixed val_loss
    # swings +/-0.2 with whichever genres its few sequences happen to draw.
    $genres = "book", "recipe", "review", "story", "wiki"
    $vals = $rows | ForEach-Object {
        $r = $_; ($genres | ForEach-Object { [double]$r."val_$_" } |
            Measure-Object -Average).Average }
    $mean = $vals[-1]
    $since = @(for ($i = 0; $i -lt $rows.Count; $i++) {
        if ([int]$rows[$i].step -ge $settleStep) { $vals[$i] } })
    $best = (($(if ($since.Count) { $since } else { $vals })) | Measure-Object -Minimum).Minimum
    $age = ($now - (Get-Item -LiteralPath $evalCsv).LastWriteTime).TotalMinutes
    # The MEDIAN of the last 4 evals: the first eval after a (re)start
    # includes startup time and would drag an average down.
    $recent = @($rows | Select-Object -Last 4 | Where-Object { [int]$_.step -gt 0 } |
        ForEach-Object { [double]$_.tok_per_sec } | Sort-Object)
    $mid = [int][math]::Floor($recent.Count / 2)
    $tps = if ($recent.Count % 2) { $recent[$mid] } else { ($recent[$mid - 1] + $recent[$mid]) / 2 }
    $lines.Add(("- Step {0:N0} of {1:N0} ({2:N1}%), eval log written {3:N0} min ago" -f
                $step, $maxSteps, (100 * $step / $maxSteps), $age))
    $lines.Add(("- Training-log score (fresh random passages each eval, +/-0.03 luck): genre average {0:N4} (best {1:N4}): book {2} | recipe {3} | review {4} | story {5} | wiki {6}" -f
                $mean, $best, $last.val_book, $last.val_recipe,
                $last.val_review, $last.val_story, $last.val_wiki))
    # Health columns, logged since the QK-norm restart (2026-10-18)
    if ($last.PSObject.Properties.Name -contains "attn_score_max" -and $last.attn_score_max) {
        $lines.Add(("- Health: grad norm {0} (max {1}), largest attention score {2}" -f
                    $last.grad_norm, $last.grad_norm_max, $last.attn_score_max))
        if ([double]$last.attn_score_max -gt 1000) {
            $problems.Add("largest attention score $($last.attn_score_max) - runaway attention again?")
        }
        if ([double]$last.grad_norm -gt 8) {
            $problems.Add("gradient norm $($last.grad_norm) (healthy is about 1-2)")
        }
    }
    $lines.Add(("- Throughput {0:N0} tok/s (median of the last 4 evals)" -f $tps))
    if (-not $done) {
        if ($age -gt 60) { $problems.Add(("no new eval in {0:N0} min (expected every ~28)" -f $age)) }
        if ($tps -lt 8000) { $problems.Add(("throughput low: {0:N0} tok/s (normal ~9,700)" -f $tps)) }
        $eta = $now.AddSeconds(($maxSteps - $step) * $tokensPerStep / [math]::Max($tps, 1))
        $lines.Add("- Projected finish: $($eta.ToString('ddd yyyy-MM-dd HH:mm'))")
    }
    if ($vals | Where-Object { [double]::IsNaN($_) }) { $problems.Add("NaN in the val loss") }
    if ($mean -gt $best + 0.15) {
        $problems.Add(("genre-average val loss {0:N4} is more than 0.15 above its best {1:N4}" -f $mean, $best))
    }
}

# --- fixed-passage scores (tools\track_fixed_eval.py): every checkpoint
# scored on the SAME 60 val passages, so a change is the model, not luck
$fixedCsv = Join-Path $ckpt "mixtape\fixed-eval.csv"
if (Test-Path $fixedCsv) {
    $trend = @(Import-Csv -LiteralPath $fixedCsv | Where-Object {
        $_.note -notlike "first run*" -and $_.note -notlike "trial start*" })
    if ($trend.Count) {
        $lastF = $trend[-1]
        $judged = @($trend | Where-Object { [int]$_.step -ge $settleStep })
        if (-not $judged.Count) { $judged = $trend }
        $bestF = ($judged | ForEach-Object { [double]$_.mean } | Measure-Object -Minimum).Minimum
        $steps = ($trend | Select-Object -Last 6 | ForEach-Object {
            "{0:N0}: {1}" -f [int]$_.step, $_.mean }) -join " -> "
        $lines.Add(("- Fixed-passage score (same 60 passages every checkpoint): {0} at step {1:N0} (best {2:N4})" -f
                    $lastF.mean, [int]$lastF.step, $bestF))
        $lines.Add("  - trend: $steps")
        if ([int]$lastF.step -ge $settleStep) {
            $lines.Add(("  - vs the step-79,000 baseline {0:N4}: {1:+0.0000;-0.0000} (the extension must end below it)" -f
                        $baseline, ([double]$lastF.mean - $baseline)))
        }
        if ([double]$lastF.mean -gt $bestF + 0.05) {
            $problems.Add(("fixed-passage score {0} is more than 0.05 above its best {1:N4}" -f $lastF.mean, $bestF))
        }
        if ($rows.Count -and -not $done -and ([int]$rows[-1].step - [int]$lastF.step) -gt 2500) {
            $problems.Add("fixed-passage scores are behind (last scored step $($lastF.step))")
        }
    }
    $tracker = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like "*track_fixed_eval*" }
    if (-not $tracker -and -not $done) {
        $problems.Add("fixed-passage tracker not running (double-click tools\track_fixed_eval.cmd)")
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
