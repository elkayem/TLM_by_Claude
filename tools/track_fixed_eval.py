"""
Score each new checkpoint of a run on the SAME validation passages.

train.py's evals draw fresh random passages every time, so their numbers
move by about +/-0.03 on the luck of the draw - more than a model improves
in an hour this late in a run. Scoring every checkpoint on one fixed set of
passages cancels that luck: a change in this score is a change in the model.

It watches the run's latest.pt (saved every 1,000 steps), copies each new
one, scores the copy on the CPU (so the GPU keeps training at full speed;
about a minute per checkpoint), and appends a row to fixed-eval.csv in the
run's folder. It stops after scoring the final checkpoint.

    python tools\\track_fixed_eval.py [--run mixtape]
    (or double-click tools\\track_fixed_eval.cmd)

The passages: 12 per genre from val_<genre>.bin, 1,024 tokens each, drawn
with seed 0 - the set used to diagnose Stage 4's loss spike on 2026-10-18,
so the rows measured that day line up with these.
"""

import argparse
import csv
import os
import shutil
import sys
import time
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tlm.config import (DATA_ROOT, PRESETS, checkpoint_dir,  # noqa: E402
                        config_from_checkpoint, data_dir)
from tlm.model import TLM  # noqa: E402

GENRES = ["book", "recipe", "review", "story", "wiki"]
N_SEQ = 12


def training_finished(train_log):
    """Has the CURRENT launch finished? The log keeps every launch, each
    starting with a '=== <time>  python -u -m tlm.train ...' line, so only
    look after the last one: an extended run's log already has a 'done.'."""
    if not os.path.exists(train_log):
        return False
    text = open(train_log, encoding="utf-8", errors="ignore").read()
    start = text.rfind("python -u -m tlm.train")
    return start >= 0 and "done. best val loss" in text[start:]


def fixed_passages(dataset, block_size):
    rng = np.random.default_rng(0)
    out = {}
    for g in GENRES:
        data = np.memmap(os.path.join(data_dir(dataset), f"val_{g}.bin"),
                         dtype=np.uint16, mode="r")
        ix = rng.integers(0, len(data) - block_size - 1, N_SEQ)
        out[g] = torch.from_numpy(np.stack(
            [data[i:i + block_size + 1].astype(np.int64) for i in ix]))
    return out


@torch.no_grad()
def score(path, passages):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = TLM(config_from_checkpoint(ck["config"]))
    model.load_state_dict(ck["model"])
    model.eval()
    losses = {}
    for g in GENRES:
        total = 0.0
        for j in range(0, N_SEQ, 4):
            b = passages[g][j:j + 4]
            total += model(b[:, :-1], b[:, 1:])[1].item() * len(b)
        losses[g] = total / N_SEQ
    return ck["step"], losses


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default="mixtape")
    ap.add_argument("--poll", type=int, default=60, help="seconds")
    args = ap.parse_args()
    torch.set_num_threads(6)
    config = PRESETS[args.run]
    run_dir = checkpoint_dir(config.run_name)
    latest = os.path.join(run_dir, "latest.pt")
    out = os.path.join(run_dir, "fixed-eval.csv")
    train_log = os.path.join(DATA_ROOT, f"{args.run}-train.log")
    passages = fixed_passages(config.dataset, config.block_size)

    done_steps = set()
    if os.path.exists(out):
        done_steps = {int(r["step"]) for r in csv.DictReader(open(out))}
    else:
        with open(out, "w", newline="") as f:
            csv.writer(f).writerow(["step", "scored_at"] + GENRES
                                   + ["mean", "note"])
    print(f"watching {latest}; scores go to {out}")

    seen_mtime = 0.0
    while True:
        finished = training_finished(train_log)
        if os.path.exists(latest) and os.path.getmtime(latest) != seen_mtime:
            seen_mtime = os.path.getmtime(latest)
            copy = latest + ".scoring"
            shutil.copyfile(latest, copy)      # a few seconds; then the
            try:                               # trainer may replace latest.pt
                step, losses = score(copy, passages)
            finally:
                os.remove(copy)
            if step not in done_steps:
                done_steps.add(step)
                mean = sum(losses.values()) / len(losses)
                row = ([step, datetime.now().strftime("%Y-%m-%d %H:%M")]
                       + [f"{losses[g]:.4f}" for g in GENRES]
                       + [f"{mean:.4f}", ""])
                with open(out, "a", newline="") as f:
                    csv.writer(f).writerow(row)
                print(f"{row[1]}  step {step:6d}  "
                      + "  ".join(f"{g} {losses[g]:.4f}" for g in GENRES)
                      + f"  | mean {mean:.4f}", flush=True)
        if finished and os.path.getmtime(latest) == seen_mtime:
            print("training is done and its final checkpoint is scored.")
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
