"""
The TLM training loop.

Usage (from the tlm/ project root, venv activated):

    python -m tlm.train --config shakespeare
    python -m tlm.train --config stories --resume   # continue a run

WHAT ONE TRAINING STEP IS
-------------------------
1. Grab a random batch of token sequences from the dataset.
2. Forward pass: model predicts a next-token distribution at every
   position; cross-entropy measures how wrong it was.
3. Backward pass: loss.backward() computes, for every one of the millions
   of weights, the gradient d(loss)/d(weight) - "which direction would
   nudging this weight move the loss?"
4. optimizer.step(): nudge every weight a tiny amount against its
   gradient.

That's all training is. Repeat tens of thousands of times.

The rest of this file is the practical scaffolding around that loop:
learning-rate scheduling, evaluation, checkpointing (so a multi-day run
survives a reboot), logging, and periodic sample generation so you can
watch the model learn. See docs/06-training.md.
"""

import argparse
import csv
import math
import os
import time

import numpy as np
import torch

from .config import PRESETS, checkpoint_dir
from .model import TLM
from .tokenizer import load_tokenizer

# Resolve paths relative to the project root (the directory containing
# this package), so the script works no matter where you invoke it from.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_batch(data, config, generator):
    """Sample a random batch of training examples from the token stream.

    `data` is a numpy memmap of the ENTIRE tokenized corpus as one long
    uint16 array. We don't chop the corpus into fixed documents; we just
    pick batch_size random offsets and slice block_size+1 tokens at each.

    Why +1? The input x and the target y are the SAME slice shifted by one:
        tokens:  [t0 t1 t2 t3 t4]
        x     =  [t0 t1 t2 t3]
        y     =  [t1 t2 t3 t4]
    So at position i the model sees t0..ti and must predict t(i+1): the
    causal mask in the model makes every position an honest prediction
    problem, and one batch of shape (B, T) yields B*T training signals.
    """
    ix = torch.randint(len(data) - config.block_size - 1,
                       (config.batch_size,), generator=generator)
    # .astype(int64): torch wants int64 indices; the memmap stores uint16
    # to keep the file small (2 bytes/token, fine for vocab < 65536).
    x = torch.stack([torch.from_numpy(
        data[i:i + config.block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(
        data[i + 1:i + 1 + config.block_size].astype(np.int64)) for i in ix])
    return x, y


def get_lr(step, config):
    """Learning-rate schedule: linear warmup, then cosine decay.

    - Warmup (first `warmup_steps`): LR ramps 0 -> learning_rate. Freshly
      initialized weights produce large, noisy gradients; full-size steps
      at that point can throw the model into a bad region it never
      recovers from. Warmup lets the optimizer's gradient statistics
      settle first.
    - Cosine decay: LR then glides smoothly down to min_lr. Big steps
      early (cover ground fast), small steps late (settle into a minimum
      instead of bouncing around it).
    """
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    # progress through the decay phase, 0 -> 1
    t = (step - config.warmup_steps) / max(
        1, config.max_steps - config.warmup_steps)
    # cosine shape: coeff goes 1 -> 0 smoothly as t goes 0 -> 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


@torch.no_grad()
def estimate_loss(model, splits, config, generator):
    """Average the loss over eval_iters random batches per split.

    A single batch's loss is noisy (it depends on which slices you drew);
    averaging ~50 batches gives a stable number. torch.no_grad() turns
    off gradient bookkeeping - we're measuring, not learning.

    We track VAL loss (data the model never trains on) as the real
    scorecard: train loss falling while val loss rises is the signature
    of overfitting - memorization instead of learning.
    """
    model.eval()   # switch off dropout for a fair measurement
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(config.eval_iters)
        for i in range(config.eval_iters):
            x, y = get_batch(data, config, generator)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[name] = losses.mean().item()
    model.train()  # dropout back on
    return out


def fmt_hms(seconds):
    """Format an elapsed duration as H:MM:SS (e.g. 2:07:34)."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def save_checkpoint(path, model, optimizer, step, config, best_val, elapsed):
    """Atomic checkpoint save: write to a temp file, then rename over the
    old one. If the machine dies mid-write we keep the previous good
    checkpoint instead of a corrupt half-file. (Also plays nicer with
    OneDrive sync than rewriting a file in place.)
    """
    tmp = path + ".tmp"
    torch.save({
        "model": model.state_dict(),          # all the learned weights
        "optimizer": optimizer.state_dict(),  # AdamW's momentum/variance
                                              # state - needed so a resumed
                                              # run continues smoothly
        "step": step,
        "best_val": best_val,
        "config": config.to_dict(),
        "elapsed": elapsed,   # cumulative training seconds across all
                              # sessions, so --resume continues the clock
    }, tmp)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description="Train the TLM")
    parser.add_argument("--config", default="shakespeare",
                        choices=sorted(PRESETS.keys()))
    parser.add_argument("--resume", action="store_true",
                        help="continue from the latest checkpoint")
    parser.add_argument("--init-from", default=None, metavar="RUN",
                        help="fine-tune: start from another run's best.pt "
                             "weights (fresh optimizer and step counter)")
    # Early stopping, OFF by default - deliberately. With a cosine LR
    # schedule the largest val improvements often arrive in the final
    # anneal (the stories run looked stalled from step 49k to 53k, then
    # set its best at 57k of 60k), and eval noise is around +/-0.01, so
    # short patience stops runs at exactly the wrong moment. If you use
    # this, be generous: at least 20 evals.
    parser.add_argument("--patience", type=int, default=None, metavar="N",
                        help="stop if val loss sets no new best for N "
                             "consecutive evals (default: run to the end)")
    args = parser.parse_args()
    config = PRESETS[args.config]

    torch.manual_seed(1337)
    # Dedicated RNG for batch sampling so resumed runs don't replay the
    # exact same batches from the start.
    generator = torch.Generator().manual_seed(int(time.time()))

    # ---- data: memory-mapped token files written by the prepare_* scripts.
    # np.memmap reads pages from disk on demand instead of loading the
    # whole corpus into RAM - essential for the ~1GB TinyStories bin on a
    # 16GB machine, free for the tiny Shakespeare one.
    data_dir = os.path.join(ROOT, "data", config.dataset)
    splits = {
        "train": np.memmap(os.path.join(data_dir, "train.bin"),
                           dtype=np.uint16, mode="r"),
        "val": np.memmap(os.path.join(data_dir, "val.bin"),
                         dtype=np.uint16, mode="r"),
    }
    tokenizer = load_tokenizer(os.path.join(data_dir, "tokenizer.json"))
    config.vocab_size = tokenizer.vocab_size

    # ---- model & optimizer
    model = TLM(config)
    print(f"config: {args.config} | params: {model.num_params()/1e6:.2f}M | "
          f"vocab: {config.vocab_size} | "
          f"tokens/step: {config.batch_size * config.block_size:,}")

    # FINE-TUNING (--init-from): instead of random weights, start from a
    # model trained on something else and continue training on THIS
    # dataset. Only the weights carry over - the optimizer state, step
    # counter and LR schedule start fresh. This is the cheap way to adapt
    # a trained model to a new task/format, and it only makes sense when
    # the architecture and tokenizer both match the source run exactly:
    # the embedding table is indexed BY the source tokenizer's ids, so a
    # different tokenizer would scramble every token's meaning.
    if args.init_from and not args.resume:
        src_path = os.path.join(checkpoint_dir(args.init_from), "best.pt")
        src = torch.load(src_path, weights_only=True)
        for key in ("vocab_size", "block_size", "n_layer", "n_head", "n_embd"):
            assert src["config"][key] == getattr(config, key), (
                f"can't fine-tune: {key} differs "
                f"({src['config'][key]} vs {getattr(config, key)})")
        model.load_state_dict(src["model"])
        print(f"initialized weights from '{args.init_from}' "
              f"(step {src['step']}, val {src['best_val']:.4f})")

    # AdamW: the default optimizer for transformers. Per-weight adaptive
    # step sizes from running averages of the gradient (momentum) and its
    # square (scale), plus DECOUPLED weight decay - the decay is applied
    # directly to weights rather than mixed into the gradient, which is
    # the "W" and works correctly with the adaptive scaling.
    # We only decay the 2D weight matrices: biases, LayerNorm params and
    # embeddings are standardly exempted (decaying them hurts).
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 and "emb" not in name else no_decay).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": config.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=config.learning_rate, betas=(0.9, 0.95))

    # ---- checkpointing / resume
    ckpt_dir = checkpoint_dir(config.run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"checkpoints -> {ckpt_dir}")
    ckpt_path = os.path.join(ckpt_dir, "latest.pt")
    best_path = os.path.join(ckpt_dir, "best.pt")
    log_path = os.path.join(ckpt_dir, "log.csv")

    start_step, best_val = 0, float("inf")
    elapsed_before = 0.0   # training time accumulated in previous sessions
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step, best_val = ckpt["step"] + 1, ckpt["best_val"]
        elapsed_before = ckpt.get("elapsed", 0.0)
        print(f"resumed from step {start_step} "
              f"(prior training time {fmt_hms(elapsed_before)})")
    elif not args.resume and os.path.exists(ckpt_path):
        raise SystemExit(
            f"checkpoint already exists at {ckpt_path}; "
            f"pass --resume to continue it, or delete the folder to restart")

    if start_step == 0:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["step", "lr", "train_loss", "val_loss", "tok_per_sec"])

    # ---- the loop
    model.train()
    # t0 measures throughput between eval prints (it resets each time);
    # session_start measures this process's wall time, which we add to
    # elapsed_before to get cumulative training time across resumes.
    t0, tokens_since = time.time(), 0
    session_start = time.time()
    evals_since_best = 0  # for --patience
    for step in range(start_step, config.max_steps):
        # set this step's learning rate on every param group
        lr = get_lr(step, config)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # (1) sample a batch
        x, y = get_batch(splits["train"], config, generator)
        # (2) forward: predictions + loss
        _, loss = model(x, y)
        # (3) backward: populate .grad on every parameter.
        # zero_grad first because backward() ACCUMULATES into .grad.
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Gradient clipping: if the overall gradient vector is longer than
        # grad_clip, rescale it to that length. Direction is kept; only
        # the size of the step is capped. Insurance against the occasional
        # weird batch producing a weight-wrecking gradient spike.
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        # (4) apply the update
        optimizer.step()

        tokens_since += config.batch_size * config.block_size

        # ---- periodic evaluation & logging
        if step % config.eval_interval == 0 or step == config.max_steps - 1:
            losses = estimate_loss(model, splits, config, generator)
            dt = time.time() - t0
            tps = tokens_since / dt if dt > 0 else 0.0
            elapsed = elapsed_before + (time.time() - session_start)
            print(f"step {step:6d} | lr {lr:.2e} | "
                  f"train {losses['train']:.4f} | val {losses['val']:.4f} | "
                  f"{tps:,.0f} tok/s | elapsed {fmt_hms(elapsed)}")
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [step, f"{lr:.2e}", f"{losses['train']:.4f}",
                     f"{losses['val']:.4f}", f"{tps:.0f}"])
            t0, tokens_since = time.time(), 0
            # keep a separate copy of the best-ever model by val loss
            if losses["val"] < best_val:
                best_val = losses["val"]
                evals_since_best = 0
                save_checkpoint(best_path, model, optimizer, step, config,
                                best_val, elapsed)
            else:
                evals_since_best += 1
                if (args.patience is not None
                        and evals_since_best >= args.patience):
                    print(f"early stop: no new best val in "
                          f"{evals_since_best} evals (best {best_val:.4f})")
                    save_checkpoint(ckpt_path, model, optimizer, step,
                                    config, best_val, elapsed)
                    return

        # ---- periodic sample: watch the model learn to write
        if step > 0 and step % config.sample_interval == 0:
            ctx = torch.tensor([tokenizer.encode("\n")], dtype=torch.long)
            out = model.generate(ctx, max_new_tokens=200, temperature=0.8,
                                 top_k=50)
            model.train()  # generate() switched to eval mode
            print("-" * 60)
            print(tokenizer.decode(out[0].tolist()))
            print("-" * 60)

        if step > 0 and step % config.checkpoint_interval == 0:
            elapsed = elapsed_before + (time.time() - session_start)
            save_checkpoint(ckpt_path, model, optimizer, step, config,
                            best_val, elapsed)

    elapsed = elapsed_before + (time.time() - session_start)
    save_checkpoint(ckpt_path, model, optimizer, config.max_steps - 1,
                    config, best_val, elapsed)
    print(f"done. best val loss {best_val:.4f}. "
          f"total training time {fmt_hms(elapsed)}. "
          f"checkpoints in {ckpt_dir}")


if __name__ == "__main__":
    main()
