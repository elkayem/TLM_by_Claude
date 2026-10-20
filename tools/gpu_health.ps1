# GPU health logger for long training runs. Double-click
# tools\gpu_health.cmd to run it in a window of its own.
#
# Every 5 minutes it records one line to %USERPROFILE%\tlm-data\gpu-health.csv:
# temperature, fan, power, clock, utilization, memory, NVIDIA's own
# "why is the clock lower" flags, and the training step reached so far.
# It prints each reading here too, and a red WARNING when something needs
# a look:
#   - temperature at or above 83 C (NVIDIA's target for this card)
#   - a heat or hardware slowdown flag (the card protecting itself)
#   - GPU utilization under 20% three readings in a row (training stopped?)
# It exits by itself when the mixtape training log says "done.".
#
# (For scale, on the RTX 3050: max operating temperature 92 C, slowdown
# 94 C, shutdown 97 C. Being held at the 70 W power cap is normal.)

$Host.UI.RawUI.WindowTitle = "TLM GPU health logger"
$intervalSec = 300
$data = if ($env:TLM_DATA_DIR) { $env:TLM_DATA_DIR } else {
    Join-Path $env:USERPROFILE "tlm-data" }
$ckpt = if ($env:TLM_CKPT_DIR) { $env:TLM_CKPT_DIR } else {
    Join-Path $env:USERPROFILE "tlm-checkpoints" }
$out = Join-Path $data "gpu-health.csv"
$trainCsv = Join-Path $ckpt "mixtape\log.csv"
$trainLog = Join-Path $data "mixtape-train.log"

$fields = "temperature.gpu,fan.speed,power.draw,power.limit,clocks.sm," +
          "utilization.gpu,memory.used," +
          "clocks_event_reasons.sw_power_cap,clocks_event_reasons.hw_slowdown," +
          "clocks_event_reasons.sw_thermal_slowdown," +
          "clocks_event_reasons.hw_thermal_slowdown"
if (-not (Test-Path $out)) {
    "time,temp_c,fan_pct,power_w,power_limit_w,sm_clock_mhz,util_pct," +
    "mem_used_mib,power_cap,hw_slowdown,sw_thermal,hw_thermal,train_step" |
        Set-Content -LiteralPath $out -Encoding ASCII
}
Write-Host "Logging every $($intervalSec / 60) min to $out  (Ctrl+C or close to stop)"

$idle = 0
while ($true) {
    $time = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $raw = & nvidia-smi "--query-gpu=$fields" "--format=csv,noheader,nounits" 2>$null
    $step = ""
    if (Test-Path $trainCsv) {
        $step = ((Get-Content -LiteralPath $trainCsv -Tail 1) -split ",")[0]
    }
    if (-not $raw) {
        Write-Host "$time  WARNING: nvidia-smi gave no reading" -ForegroundColor Red
        "$time,,,,,,,,,,,,$step" | Add-Content -LiteralPath $out -Encoding ASCII
    } else {
        $v = ($raw -split ",") | ForEach-Object { $_.Trim() }
        $flags = $v[7..10] | ForEach-Object { if ($_ -eq "Active") { 1 } else { 0 } }
        "$time,$($v[0..6] -join ','),$($flags -join ','),$step" |
            Add-Content -LiteralPath $out -Encoding ASCII
        Write-Host ("{0}  {1} C  fan {2}%  {3}/{4} W  {5} MHz  util {6}%  mem {7} MiB  step {8}" -f
                    $time, $v[0], $v[1], $v[2], $v[3], $v[4], $v[5], $v[6], $step)
        $warn = @()
        if ([double]$v[0] -ge 83) { $warn += "temperature $($v[0]) C" }
        if ($flags[1] -or $flags[2] -or $flags[3]) { $warn += "slowdown flag (hw/sw thermal/hw)" }
        if ([double]$v[5] -lt 20) { $idle++ } else { $idle = 0 }
        if ($idle -ge 3) { $warn += "GPU nearly idle for 15+ min - is training still running?" }
        if ($warn) { Write-Host "$time  WARNING: $($warn -join '; ')" -ForegroundColor Red }
    }
    # Only the current launch counts: an extended run's log already holds
    # an earlier "done." line.
    $txt = if (Test-Path $trainLog) { Get-Content -LiteralPath $trainLog -Raw } else { "" }
    $launch = $txt.LastIndexOf("python -u -m tlm.train")
    if ($launch -ge 0 -and $txt.IndexOf("done. best val loss", $launch) -ge 0) {
        Write-Host "The mixtape training log says done. Logger stopping." -ForegroundColor Green
        break
    }
    Start-Sleep -Seconds $intervalSec
}
