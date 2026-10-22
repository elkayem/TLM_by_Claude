"""
Measure how fast Stages 1-3 train on the GPU, to replace the README's
estimates with measurements.

- Stage 1 (`shakespeare`) runs in full: it's short.
- Stages 2 and 3 (`stories`, `instruct`) run --steps steps; the full
  run's time is extrapolated from the sustained tokens/s (the average over
  the evals after the first, which already includes eval and sample
  overhead).

Checkpoints go to ~/tlm-checkpoints/gpu-validation (the CPU-trained
checkpoints are never touched). Results go to ~/tlm-data/gpu-benchmark.md,
progress to ~/tlm-data/gpu-benchmark.log.

    python tools\\benchmark_gpu.py [--steps 2000]
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from tlm.config import DATA_ROOT, PRESETS  # noqa: E402

CKPT = os.path.join(os.path.expanduser("~"), "tlm-checkpoints",
                    "gpu-validation")
LOG = os.path.join(DATA_ROOT, "gpu-benchmark.log")
OUT = os.path.join(DATA_ROOT, "gpu-benchmark.md")
CPU_HOURS = {"shakespeare": 2.0, "stories": 55.7, "instruct": 118.0}
CPU_LOSS = {"shakespeare": 1.609, "stories": 1.595, "instruct": 1.257}


def say(msg):
    line = f"{datetime.now():%H:%M:%S}  {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--steps", type=int, default=2000)
    args = ap.parse_args()
    env = dict(os.environ, TLM_CKPT_DIR=CKPT)
    rows = []
    for name, cap in (("shakespeare", None), ("stories", args.steps),
                      ("instruct", args.steps)):
        cfg = PRESETS[name]
        full_steps = cfg.max_steps
        tps_step = cfg.batch_size * cfg.block_size * cfg.grad_accum
        code = ("import sys; from tlm import train; "
                "from tlm.config import PRESETS; "
                f"PRESETS['{name}'].max_steps = {cap or full_steps}; "
                f"sys.argv = ['train', '--config', '{name}', "
                "'--device', 'cuda']; train.main()")
        say(f"{name}: {cap or full_steps} of {full_steps} steps")
        t0 = time.time()
        with open(LOG, "a", encoding="utf-8") as f:
            subprocess.run([sys.executable, "-u", "-c", code], cwd=REPO,
                           env=env, stdout=f, stderr=subprocess.STDOUT,
                           check=True)
        wall = time.time() - t0
        with open(os.path.join(CKPT, cfg.run_name, "log.csv")) as f:
            log = list(csv.DictReader(f))
        tps = [float(r["tok_per_sec"]) for r in log[2:]]   # after the 1st eval
        sustained = sum(tps) / len(tps)
        hours = full_steps * tps_step / sustained / 3600
        best = min(float(r["val_loss"]) for r in log)
        rows.append((name, cap or full_steps, full_steps, sustained, hours,
                     wall / 3600, best))
        say(f"{name}: {sustained:,.0f} tok/s sustained -> full run "
            f"{hours:.2f} h (this run {wall / 60:.1f} min, best val {best:.3f})")

    lines = [f"# GPU benchmark, {datetime.now():%Y-%m-%d}", "",
             "RTX 3050 (6 GB), bf16 autocast. Full-run time extrapolated from "
             "the sustained tok/s (Stage 1 ran in full).", "",
             "| stage | steps run | tok/s (GPU) | full run, GPU | full run, CPU "
             "| speed-up | best val (run) |", "|---|---|---|---|---|---|---|"]
    for name, ran, full, sus, hours, wall, best in rows:
        lines.append(f"| {name} | {ran:,} of {full:,} | {sus:,.0f} | "
                     f"{hours * 60:.0f} min | {CPU_HOURS[name]:g} h | "
                     f"{CPU_HOURS[name] / hours:.0f}x | {best:.3f} |")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    say(f"done: {OUT}")


if __name__ == "__main__":
    main()
