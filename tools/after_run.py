"""
Wait for a training run to finish, then compare its final model with a
baseline checkpoint - so the comparison is ready when you wake up.

Written for Stage 4's extension (step 79,000 -> 117,000): did the extra
36 hours beat the model we already had? When the run's log says "done.",
this:

1. scores the baseline and the final latest.pt on the 60 fixed validation
   passages (tools/track_fixed_eval.py), on the CPU;
2. writes <out>/<name>-summary.md: both scores per genre, the difference,
   and a verdict (keep the new model if it's at least --bar better);
3. writes a BLIND A/B sample sheet of the two (tools/sample_sheet.py) to
   <out>/<name>-blind.md, with the key in <name>-blind.key.json;
4. redraws docs/mixtape-run.svg (tools/plot_mixtape_run.py).

    python tools\\after_run.py --baseline PATH [--run mixtape] [--name NAME]

Start it outside any app (see README, "Long runs"); it logs to
~/tlm-data/after-run.log.
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

TOOLS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TOOLS)
sys.path.insert(0, REPO)
sys.path.insert(0, TOOLS)
import track_fixed_eval as fixed  # noqa: E402
from tlm.config import DATA_ROOT, PRESETS, checkpoint_dir  # noqa: E402

LOG = os.path.join(DATA_ROOT, "after-run.log")


def say(msg):
    line = f"{datetime.now():%Y-%m-%d %H:%M}  {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--run", default="mixtape")
    ap.add_argument("--name", default="extension")
    ap.add_argument("--out", default=os.path.join(DATA_ROOT, "samples"))
    ap.add_argument("--bar", type=float, default=0.03,
                    help="how much better the new model must be to keep it")
    ap.add_argument("--poll", type=int, default=120, help="seconds")
    args = ap.parse_args()
    config = PRESETS[args.run]
    run_dir = checkpoint_dir(config.run_name)
    final = os.path.join(run_dir, "latest.pt")
    train_log = os.path.join(DATA_ROOT, f"{args.run}-train.log")
    os.makedirs(args.out, exist_ok=True)

    say(f"waiting for the {args.run} run to finish (checking every "
        f"{args.poll}s)")
    while not fixed.training_finished(train_log):
        time.sleep(args.poll)
    time.sleep(60)                    # let the final save and logs settle
    say("training finished; scoring both models on the fixed passages")

    passages = fixed.fixed_passages(config.dataset, config.block_size)
    b_step, b = fixed.score(args.baseline, passages)
    f_step, f = fixed.score(final, passages)
    b_mean = sum(b.values()) / len(b)
    f_mean = sum(f.values()) / len(f)
    diff = f_mean - b_mean
    if diff <= -args.bar:
        verdict = (f"KEEP the new model: {-diff:.4f} better than the "
                   f"baseline, past the {args.bar} bar.")
    elif diff < 0:
        verdict = (f"Slightly better ({-diff:.4f}), but under the {args.bar} "
                   "bar: let the blind test decide.")
    else:
        verdict = (f"RESTORE the baseline: the new model is {diff:.4f} worse.")
    rows = ["| model | " + " | ".join(fixed.GENRES) + " | **average** |",
            "|---" * (len(fixed.GENRES) + 2) + "|"]
    for label, step, s, m in (("baseline", b_step, b, b_mean),
                              ("final", f_step, f, f_mean)):
        rows.append(f"| {label}, step {step:,} | "
                    + " | ".join(f"{s[g]:.4f}" for g in fixed.GENRES)
                    + f" | **{m:.4f}** |")
    rows.append("| difference | " + " | ".join(
        f"{f[g] - b[g]:+.4f}" for g in fixed.GENRES) + f" | **{diff:+.4f}** |")
    summary = os.path.join(args.out, f"{args.name}-summary.md")
    with open(summary, "w", encoding="utf-8") as fh:
        fh.write(f"# {args.name}: final model vs baseline\n\n"
                 f"Written {datetime.now():%Y-%m-%d %H:%M} by tools/after_run.py. "
                 "Loss on the 60 fixed validation passages (lower is "
                 "better).\n\n" + "\n".join(rows) + f"\n\n**{verdict}**\n\n"
                 f"Blind samples: `{args.name}-blind.md` (key in "
                 f"`{args.name}-blind.key.json`; look after judging).\n")
    say(f"baseline {b_mean:.4f}, final {f_mean:.4f} ({diff:+.4f}): {verdict}")

    say("writing the blind A/B sample sheet (a few minutes on the CPU)")
    blind = os.path.join(args.out, f"{args.name}-blind.md")
    subprocess.run([sys.executable, "-u", os.path.join(TOOLS, "sample_sheet.py"),
                    args.baseline, "--compare", final, "--out", blind],
                   check=True)
    say("redrawing docs/mixtape-run.svg")
    subprocess.run([sys.executable, os.path.join(TOOLS, "plot_mixtape_run.py")],
                   check=True)
    say(f"all done: {summary} and {blind}")


if __name__ == "__main__":
    main()
