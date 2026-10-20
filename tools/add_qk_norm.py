"""
Graft QK-norm onto a checkpoint trained without it, so training can
continue from there instead of starting over.

Stage 4's first mixtape run hit "attention logit growth" in layer 1 (scores
in the millions, heads locked onto single tokens, training stalled and
spiked). This takes its step-51,000 checkpoint and adds the q_gain/k_gain
parameters of QK-norm (see CausalSelfAttention in tlm/model.py) - to the
layers named by --layers only: normalizing a trained layer changes what it
does (measured on that checkpoint: QK-norm everywhere raised the loss from
2.86 to 3.63; on layer 1 alone, to 3.13), so graft it where it's needed.

The trick is choosing the new gains so the model barely notices. QK-norm
scales each head's query (and key) vector to RMS 1 and multiplies by the
gain, so if a head's gain equals the RMS its vectors usually have, a
typical vector comes out unchanged. We measure those RMS values on some
training text and use them as the gains. Then every head can score at most
about gain_q * gain_k * sqrt(hs); the runaway heads, whose bound would be in
the millions, get their gains scaled down to --cap. Healthy heads are left
alone.

The optimizer state comes along too (re-indexed: the new gains shift every
parameter's position in AdamW's list), unless --fresh-optimizer drops it -
the right call when the old state is itself damaged (pair it with
train.py --rewarm so the fresh optimizer starts gently).

    python tools\\add_qk_norm.py SRC.pt DST.pt [--layers 1] [--cap 30]
        [--fresh-optimizer] [--best-val X]
"""

import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tlm.config import TLMConfig, config_from_checkpoint, data_dir  # noqa: E402
from tlm.model import TLM  # noqa: E402


def param_order(model):
    """AdamW's parameter order in train.py: decayed matrices first, then the
    rest (biases, norms, embeddings)."""
    named = list(model.named_parameters())
    decay = [n for n, p in named if p.dim() >= 2 and "emb" not in n]
    rest = [n for n, p in named if not (p.dim() >= 2 and "emb" not in n)]
    return decay, rest


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--layers", default="",
                    help="comma-separated layers to add QK-norm to "
                         "(default: all)")
    ap.add_argument("--fresh-optimizer", action="store_true",
                    help="drop AdamW's saved momentum/variance")
    ap.add_argument("--cap", type=float, default=30.0,
                    help="largest score a capped head can produce")
    ap.add_argument("--runaway", type=float, default=1000.0,
                    help="heads whose bound exceeds this get capped")
    ap.add_argument("--best-val", type=float, default=float("inf"),
                    help="best score so far, for the resumed run's best.pt")
    ap.add_argument("--seqs", type=int, default=8,
                    help="training sequences to measure the RMS values on")
    args = ap.parse_args()

    src = torch.load(args.src, map_location="cpu", weights_only=True)
    old_cfg = config_from_checkpoint(src["config"])
    assert not old_cfg.qk_norm, "this checkpoint already has QK-norm"
    layers = tuple(int(i) for i in args.layers.split(",") if i.strip())
    new_cfg = TLMConfig(**{**old_cfg.to_dict(), "qk_norm": True,
                           "qk_norm_layers": layers})
    layers = layers or tuple(range(old_cfg.n_layer))
    old, new = TLM(old_cfg), TLM(new_cfg)
    old.load_state_dict(src["model"])
    missing, unexpected = new.load_state_dict(src["model"], strict=False)
    assert not unexpected and all(k.endswith(("q_gain", "k_gain"))
                                  for k in missing), (missing, unexpected)

    # ---- measure each head's typical query/key RMS on training text
    H, hs = old_cfg.n_head, old_cfg.n_embd // old_cfg.n_head
    data = np.memmap(os.path.join(data_dir(old_cfg.dataset), "train.bin"),
                     dtype=np.uint16, mode="r")
    rng = np.random.default_rng(0)
    ix = rng.integers(0, len(data) - old_cfg.block_size, args.seqs)
    x = torch.from_numpy(np.stack(
        [data[i:i + old_cfg.block_size].astype(np.int64) for i in ix]))
    sums = {}

    def hook(layer):
        def fn(module, inputs, out):           # out: qkv, (B, T, 3C)
            q, k, _ = out.split(old_cfg.n_embd, dim=2)
            for name, t in (("q", q), ("k", k)):
                t = t.view(*t.shape[:2], H, hs).pow(2).mean((0, 1, 3))  # (H,)
                sums[(layer, name)] = sums.get((layer, name), 0) + t
        return fn

    hooks = [b.attn.qkv_proj.register_forward_hook(hook(i))
             for i, b in enumerate(old.blocks)]
    old.eval()
    with torch.no_grad():
        for j in range(0, len(x), 2):
            old(x[j:j + 2])
    for h in hooks:
        h.remove()
    n_batches = math.ceil(len(x) / 2)

    # ---- gains: the measured RMS, capped for the runaway heads
    state = new.state_dict()
    print("layer: largest head bound before -> after (heads capped)")
    for i in layers:
        rq = (sums[(i, "q")] / n_batches).sqrt()
        rk = (sums[(i, "k")] / n_batches).sqrt()
        bound = rq * rk * math.sqrt(hs)
        scale = torch.where(bound > args.runaway,
                            (args.cap / bound).sqrt(), torch.ones_like(bound))
        rq, rk = rq * scale, rk * scale
        state[f"blocks.{i}.attn.q_gain"] = rq.repeat_interleave(hs)
        state[f"blocks.{i}.attn.k_gain"] = rk.repeat_interleave(hs)
        after = rq * rk * math.sqrt(hs)
        print(f"  {i:2d}: {bound.max().item():12,.0f} -> "
              f"{after.max().item():8,.1f}  ({int((scale < 1).sum())} of {H})")

    # ---- optimizer state, re-indexed by parameter name
    old_dec, old_rest = param_order(old)
    new_dec, new_rest = param_order(new)
    old_idx = {n: i for i, n in enumerate(old_dec + old_rest)}
    opt = src["optimizer"]
    new_state = {}
    for j, n in enumerate(new_dec + new_rest):
        if (not args.fresh_optimizer and n in old_idx
                and old_idx[n] in opt["state"]):
            new_state[j] = opt["state"][old_idx[n]]
    groups = [dict(opt["param_groups"][0], params=list(range(len(new_dec)))),
              dict(opt["param_groups"][1],
                   params=list(range(len(new_dec),
                                     len(new_dec) + len(new_rest))))]

    out = dict(src, model=state, config=new_cfg.to_dict(),
               optimizer={"state": new_state, "param_groups": groups},
               best_val=args.best_val)
    tmp = args.dst + ".tmp"
    torch.save(out, tmp)
    os.replace(tmp, args.dst)
    print(f"wrote {args.dst} (step {src['step']}, best_val {args.best_val})")


if __name__ == "__main__":
    main()
