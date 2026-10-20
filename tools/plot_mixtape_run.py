"""
Chart the Stage 4 mixtape run into docs/mixtape-run.svg (standard library
only - the SVG is written by hand, like the icon in make_icon.py).

Two measures of the same thing, the validation loss averaged over the five
genres (lower is better), on one axis:
- the training-log score: each eval draws fresh random passages, so it
  zigzags by about +/-0.03 on luck alone
- the fixed-passage score: every checkpoint on the same 60 passages
  (tools/track_fixed_eval.py), so its movements are the model's

It marks the loss spike at step 50,000, the QK-norm restart at 51,000 and
the extension at 79,000. Re-run it as the run goes on:

    python tools\\plot_mixtape_run.py
"""

import csv
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.environ.get("TLM_CKPT_DIR",
                      os.path.join(os.path.expanduser("~"), "tlm-checkpoints"))
OUT = os.path.join(REPO, "docs", "mixtape-run.svg")
GENRES = ["book", "recipe", "review", "story", "wiki"]
X_FROM = 10_000

W, H = 900, 470
LEFT, RIGHT, TOP, BOTTOM = 64, 70, 104, 46        # plot-area margins

STYLE = """
  .surface { fill: #fcfcfb; }
  .title { fill: #0b0b0b; font: 600 17px system-ui, 'Segoe UI', sans-serif; }
  .sub, .note { fill: #52514e; font: 12.5px system-ui, 'Segoe UI', sans-serif; }
  .tick { fill: #898781; font: 12px system-ui, 'Segoe UI', sans-serif; }
  .value { fill: #0b0b0b; font: 600 12.5px system-ui, 'Segoe UI', sans-serif; }
  .grid { stroke: #e1e0d9; stroke-width: 1; }
  .event { stroke: #898781; stroke-width: 1; }
  .s1 { stroke: #2a78d6; } .s1f { fill: #2a78d6; }
  .s2 { stroke: #eb6834; } .s2f { fill: #eb6834; }
  .line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
  .ring { stroke: #fcfcfb; stroke-width: 2; }
  @media (prefers-color-scheme: dark) {
    .surface { fill: #1a1a19; }
    .title, .value { fill: #f0efec; }
    .sub, .note { fill: #c3c2b7; }
    .grid { stroke: #2c2c2a; }
    .s1 { stroke: #3987e5; } .s1f { fill: #3987e5; }
    .s2 { stroke: #d95926; } .s2f { fill: #d95926; }
    .ring { stroke: #1a1a19; }
  }
"""


def genre_mean(row):
    return sum(float(row["val_" + g]) for g in GENRES) / len(GENRES)


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    first = [(int(r["step"]), genre_mean(r))
             for r in read_rows(os.path.join(CKPT, "mixtape-v1", "log.csv"))
             if int(r["step"]) >= X_FROM]
    trial = [(int(r["step"]), genre_mean(r))
             for r in read_rows(os.path.join(CKPT, "mixtape", "log.csv"))
             if int(r["step"]) > 51_000]
    fixed_rows = read_rows(os.path.join(CKPT, "mixtape", "fixed-eval.csv"))
    fixed_first = [(int(r["step"]), float(r["mean"])) for r in fixed_rows
                   if r["note"].startswith("first run")]
    fixed = [(int(r["step"]), float(r["mean"])) for r in fixed_rows
             if not r["note"].startswith("first run")]

    x_to = max(117_000, trial[-1][0])
    ys = [y for _, y in first + trial + fixed + fixed_first]
    y_lo = int(min(ys) * 10) / 10
    y_hi = int(max(ys) * 10 + 1) / 10

    def px(step):
        return LEFT + (step - X_FROM) / (x_to - X_FROM) * (W - LEFT - RIGHT)

    def py(v):
        return TOP + (y_hi - v) / (y_hi - y_lo) * (H - TOP - BOTTOM)

    def path(points):
        return " ".join(f"{'M' if i == 0 else 'L'}{px(s):.1f},{py(v):.1f}"
                        for i, (s, v) in enumerate(points))

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
         f'width="{W}" height="{H}" role="img" aria-labelledby="t d">',
         f"<style>{STYLE}</style>",
         '<title id="t">Stage 4 mixtape run: validation loss</title>',
         '<desc id="d">Genre-average validation loss by training step: the '
         'noisy training-log score and the fixed-passage score, with the '
         'loss spike at 50,000, the QK-norm restart at 51,000 and the '
         'extension from 79,000.</desc>',
         f'<rect class="surface" width="{W}" height="{H}" rx="8"/>',
         f'<text class="title" x="{LEFT}" y="30">Stage 4 mixtape run: '
         'validation loss, averaged over the five genres</text>',
         f'<text class="sub" x="{LEFT}" y="50">Lower is better. Steps of '
         '32,768 tokens; the run started at step 0 (not shown, loss ~9.8).'
         '</text>']
    # legend: line keys + labels in text ink
    ly = 76
    o += [f'<line class="line s1" x1="{LEFT}" y1="{ly - 4}" x2="{LEFT + 22}" '
          f'y2="{ly - 4}"/>',
          f'<text class="sub" x="{LEFT + 30}" y="{ly}">Training-log score '
          '(fresh random passages each eval: zigzags on luck)</text>',
          f'<line class="line s2" x1="{LEFT + 430}" y1="{ly - 4}" '
          f'x2="{LEFT + 452}" y2="{ly - 4}"/>',
          f'<text class="sub" x="{LEFT + 460}" y="{ly}">Fixed-passage score '
          '(the same 60 passages every checkpoint)</text>']
    # gridlines and ticks
    v = y_lo
    while v <= y_hi + 1e-9:
        y = py(v)
        o += [f'<line class="grid" x1="{LEFT}" y1="{y:.1f}" x2="{W - RIGHT}" '
              f'y2="{y:.1f}"/>',
              f'<text class="tick" x="{LEFT - 10}" y="{y + 4:.1f}" '
              f'text-anchor="end">{v:.1f}</text>']
        v = round(v + 0.1, 2)
    for s in range(X_FROM, x_to + 1, 10_000):
        o.append(f'<text class="tick" x="{px(s):.1f}" y="{H - BOTTOM + 20}" '
                 f'text-anchor="middle">{s // 1000}K</text>')
    o.append(f'<text class="tick" x="{W - RIGHT}" y="{H - 8}" '
             'text-anchor="end">training step</text>')
    # events
    events = [(50_000, "spike", "end", -4), (51_000, "QK-norm restart",
                                             "start", 4),
              (79_000, "extension: LR rewarmed", "start", 4)]
    for s, label, anchor, dx in events:
        x = px(s)
        o += [f'<line class="event" x1="{x:.1f}" y1="{TOP - 6}" x2="{x:.1f}" '
              f'y2="{H - BOTTOM}"/>',
              f'<text class="note" x="{x + dx:.1f}" y="{TOP - 10}" '
              f'text-anchor="{anchor}">{label}</text>']
    # series
    o += [f'<path class="line s1" d="{path(first)}"/>',
          f'<path class="line s1" d="{path(trial)}"/>',
          f'<path class="line s2" d="{path(fixed)}"/>']
    for s, val in fixed_first:          # the first run's two fixed scores
        o.append(f'<circle class="s2f ring" cx="{px(s):.1f}" '
                 f'cy="{py(val):.1f}" r="5"/>')
    s27, v27 = fixed_first[0]           # label below the line, with a leader
    o += [f'<line class="event" x1="{px(s27):.1f}" y1="{py(v27) + 6:.1f}" '
          f'x2="{px(s27):.1f}" y2="{py(v27 - 0.075):.1f}"/>',
          f'<text class="note" x="{px(s27):.1f}" y="{py(v27 - 0.075) + 14:.1f}" '
          'text-anchor="middle">first run, fixed passages</text>']
    graft_s, graft_v = fixed[0]
    o.append(f'<text class="note" x="{px(graft_s) + 10:.1f}" '
             f'y="{py(graft_v) + 4:.1f}">just after the graft</text>')
    end_s, end_v = fixed[-1]
    o += [f'<circle class="s2f ring" cx="{px(end_s):.1f}" '
          f'cy="{py(end_v):.1f}" r="5"/>',
          f'<text class="value" x="{px(end_s) + 9:.1f}" '
          f'y="{py(end_v) + 18:.1f}">{end_v:.3f}</text>']
    o.append("</svg>")
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(o) + "\n")
    print(f"wrote {OUT}: fixed-passage {end_v:.4f} at step {end_s:,}")


if __name__ == "__main__":
    main()
