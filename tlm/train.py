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
import contextlib
import csv
import math
import os
import time

import numpy as np
import torch

from .config import PRESETS, checkpoint_dir, data_dir
from .model import TLM
from .tokenizer import load_tokenizer


def pick_device(requested):
    """'auto' means the GPU when PyTorch can see one, else the CPU.

    Everything else in this file is device-agnostic: the model and each
    batch are moved to `device`, and PyTorch runs the same code on
    whichever processor holds the tensors.
    """
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda, but PyTorch can't see a GPU "
                         "(is the CUDA build of torch installed?)")
    return requested


def autocast_for(device):
    """MIXED PRECISION, GPU only.

    Inside this context, matrix multiplies run in bfloat16: 16-bit floats
    with the same exponent range as 32-bit ones but fewer digits of
    precision. A GPU's tensor cores do bf16 math several times faster
    than fp32 and the activations take half the memory. The weights
    themselves (and the optimizer's arithmetic) stay fp32, and precision-
    sensitive ops like softmax and the loss are automatically kept in fp32
    - that's the "mixed" part. Because bf16 has fp32's range, no loss
    scaling is needed (unlike the older float16 recipe).

    On the CPU we change nothing, so CPU runs behave exactly as before.
    """
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def get_batch(data, config, generator, device):
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
    if device == "cuda":
        # Batches are built in CPU memory, then copied to the GPU.
        # pin_memory() puts them in page-locked RAM the GPU can read
        # directly, and non_blocking=True lets the copy run while the CPU
        # moves on to queueing the next piece of work.
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
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
    - Extending a finished run (config.restart_step): a second, smaller
      warmup and cosine - "rewarm and re-anneal". With the LR already at
      its floor, more steps would barely move the model; a moderate
      rewarm lets it keep learning, and the new decay settles it again.
    """
    if config.restart_step and step >= config.restart_step:
        s = step - config.restart_step
        if s < config.restart_warmup:
            return config.min_lr + (config.restart_lr - config.min_lr) * (
                (s + 1) / config.restart_warmup)
        t = (s - config.restart_warmup) / max(
            1, config.max_steps - config.restart_step - config.restart_warmup)
        coeff = 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))
        return config.min_lr + coeff * (config.restart_lr - config.min_lr)
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    # progress through the decay phase, 0 -> 1 (ending at restart_step if
    # the run was later extended, so the first schedule stays as it was)
    end = config.restart_step or config.max_steps
    t = (step - config.warmup_steps) / max(1, end - config.warmup_steps)
    # cosine shape: coeff goes 1 -> 0 smoothly as t goes 0 -> 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


@torch.no_grad()
def estimate_loss(model, splits, config, generator, device, autocast):
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
            x, y = get_batch(data, config, generator, device)
            with autocast:
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
    # On Windows, replacing a file fails while another program has it open
    # (the Mixtape app loading best.pt, a backup tool, an antivirus scan).
    # Wait and retry for up to a minute rather than crash a multi-day run.
    for _ in range(60):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(1)
    os.replace(tmp, path)   # still locked after a minute: let the error show


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
    # After surgery on a checkpoint (tools/add_qk_norm.py) with a fresh
    # optimizer, ramp the learning rate up again over N steps, as at the
    # start of training: a fresh AdamW's first steps are erratic, and
    # full-size ones could undo the repair.
    parser.add_argument("--rewarm", type=int, default=0, metavar="N",
                        help="on resume, ramp the LR up from ~0 over the "
                             "first N steps of this session")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="where to train (default: the GPU if there "
                             "is one)")
    args = parser.parse_args()
    config = PRESETS[args.config]
    device = pick_device(args.device)
    autocast = autocast_for(device)

    torch.manual_seed(1337)
    # Dedicated RNG for batch sampling so resumed runs don't replay the
    # exact same batches from the start.
    generator = torch.Generator().manual_seed(int(time.time()))

    # ---- data: memory-mapped token files written by the prepare_* scripts.
    # np.memmap reads pages from disk on demand instead of loading the
    # whole corpus into RAM - essential for the ~1GB TinyStories bin on a
    # 16GB machine, free for the tiny Shakespeare one.
    ddir = data_dir(config.dataset)
    splits = {
        "train": np.memmap(os.path.join(ddir, "train.bin"),
                           dtype=np.uint16, mode="r"),
        "val": np.memmap(os.path.join(ddir, "val.bin"),
                         dtype=np.uint16, mode="r"),
    }
    # Mixed datasets (Stage 4) also hold out a validation set PER SOURCE,
    # val_<genre>.bin: one overall val number can hide that the model has
    # mastered recipes but still struggles with old novels.
    genre_vals = sorted(f[4:-4] for f in os.listdir(ddir)
                        if f.startswith("val_") and f.endswith(".bin"))
    for g in genre_vals:
        splits[f"val_{g}"] = np.memmap(os.path.join(ddir, f"val_{g}.bin"),
                                       dtype=np.uint16, mode="r")
    tokenizer = load_tokenizer(os.path.join(ddir, "tokenizer.json"))
    config.vocab_size = tokenizer.vocab_size

    # ---- model & optimizer
    # The weights are initialized on the CPU (under the seed above, so a
    # GPU run starts from exactly the same weights a CPU run would), then
    # moved to the training device in one go.
    model = TLM(config).to(device)
    tokens_per_step = config.batch_size * config.block_size * config.grad_accum
    print(f"config: {args.config} | params: {model.num_params()/1e6:.2f}M | "
          f"vocab: {config.vocab_size} | "
          f"tokens/step: {tokens_per_step:,}")
    if device == "cuda":
        print(f"device: {torch.cuda.get_device_name()} | "
              f"bf16 autocast: {torch.cuda.is_bf16_supported()}")
    else:
        print("device: cpu")

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
        # map_location="cpu": load onto the CPU whatever device saved it;
        # load_state_dict then copies the values into the model wherever
        # it lives. So checkpoints move freely between CPU and GPU runs.
        src = torch.load(src_path, weights_only=True, map_location="cpu")
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
    # fused=True (GPU only): update every weight in one kernel launch
    # instead of a loop of small ones - same math, less overhead.
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": config.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=config.learning_rate, betas=(0.9, 0.95),
        fused=(device == "cuda"))

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
        ckpt = torch.load(ckpt_path, weights_only=True, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        # the optimizer moves its saved state onto each weight's device
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step, best_val = ckpt["step"] + 1, ckpt["best_val"]
        elapsed_before = ckpt.get("elapsed", 0.0)
        print(f"resumed from step {start_step} "
              f"(prior training time {fmt_hms(elapsed_before)})")
    elif not args.resume and os.path.exists(ckpt_path):
        raise SystemExit(
            f"checkpoint already exists at {ckpt_path}; "
            f"pass --resume to continue it, or delete the folder to restart")

    # A fresh run starts a fresh log; a resumed run appends to its own (and
    # starts one if it has none - e.g. a run seeded with another run's
    # checkpoint).
    if start_step == 0 or not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["step", "lr", "train_loss", "val_loss", "tok_per_sec"]
                + [f"val_{g}" for g in genre_vals]
                + (["val_genre_mean"] if genre_vals else [])
                + ["grad_norm", "grad_norm_max", "attn_score_max"])

    # ---- the loop
    model.train()
    # t0 measures throughput between eval prints (it resets each time);
    # session_start measures this process's wall time, which we add to
    # elapsed_before to get cumulative training time across resumes.
    t0, tokens_since = time.time(), 0
    session_start = time.time()
    evals_since_best = 0  # for --patience
    gn_sum, gn_max, gn_n = 0.0, 0.0, 0   # gradient norms since the last eval
    for step in range(start_step, config.max_steps):
        # set this step's learning rate on every param group
        lr = get_lr(step, config)
        if step - start_step < args.rewarm:
            lr *= (step - start_step + 1) / args.rewarm
        for group in optimizer.param_groups:
            group["lr"] = lr

        # zero_grad first because backward() ACCUMULATES into .grad.
        optimizer.zero_grad(set_to_none=True)
        # Steps (1)-(3) run grad_accum times per optimizer step (once for
        # every preset before Stage 4). Because backward() accumulates,
        # the micro-batches' gradients add up in .grad; dividing each loss
        # by grad_accum makes the sum the gradient of the AVERAGE loss -
        # exactly what one big batch would have produced.
        for _ in range(config.grad_accum):
            # (1) sample a batch
            x, y = get_batch(splits["train"], config, generator, device)
            # (2) forward: predictions + loss
            with autocast:
                _, loss = model(x, y)
            # (3) backward: add this batch's share to .grad on every
            # parameter.
            (loss / config.grad_accum).backward()
        # Gradient clipping: if the overall gradient vector is longer than
        # grad_clip, rescale it to that length. Direction is kept; only
        # the size of the step is capped. Insurance against the occasional
        # weird batch producing a weight-wrecking gradient spike.
        # It returns the norm from BEFORE clipping, which we log: a gradient
        # norm that keeps climbing is an early warning (in Stage 4's first
        # run it went from ~1.4 to ~6 before the loss spiked).
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                            config.grad_clip).item()
        gn_sum, gn_max, gn_n = gn_sum + gn, max(gn_max, gn), gn_n + 1
        # (4) apply the update
        optimizer.step()

        tokens_since += tokens_per_step

        # ---- periodic evaluation & logging
        if step % config.eval_interval == 0 or step == config.max_steps - 1:
            losses = estimate_loss(model, splits, config, generator,
                                   device, autocast)
            dt = time.time() - t0
            tps = tokens_since / dt if dt > 0 else 0.0
            elapsed = elapsed_before + (time.time() - session_start)
            # peak GPU memory so far: how close we are to the 6 GB ceiling
            mem = (f" | mem {torch.cuda.max_memory_allocated() / 1e9:.2f}GB"
                   if device == "cuda" else "")
            print(f"step {step:6d} | lr {lr:.2e} | "
                  f"train {losses['train']:.4f} | val {losses['val']:.4f} | "
                  f"{tps:,.0f} tok/s | elapsed {fmt_hms(elapsed)}{mem}")
            # The number that picks best.pt. With per-genre val sets it's
            # their AVERAGE: the mixed val estimate draws only a few dozen
            # sequences, a different genre blend each time (recipes score
            # ~1.5, old novels ~3.4), so it swings by +/-0.2 on luck alone.
            score = (sum(losses["val_" + g] for g in genre_vals)
                     / len(genre_vals) if genre_vals else losses["val"])
            if genre_vals:
                print("             val by genre: " + " | ".join(
                    f"{g} {losses['val_' + g]:.4f}" for g in genre_vals)
                    + f" | mean {score:.4f}")
            # Health check: gradient norms since the last eval, and the
            # largest attention score on one val batch (see QK-norm in
            # model.py for why that one matters).
            model.eval()
            xb, _ = get_batch(splits["val"], config, generator, device)
            with autocast:
                attn_max = max(model.attention_score_max(xb))
            model.train()
            gn_mean = gn_sum / max(gn_n, 1)
            print(f"             health: grad norm {gn_mean:.2f} "
                  f"(max {gn_max:.2f}) | attention score max {attn_max:,.1f}")
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [step, f"{lr:.2e}", f"{losses['train']:.4f}",
                     f"{losses['val']:.4f}", f"{tps:.0f}"]
                    + [f"{losses['val_' + g]:.4f}" for g in genre_vals]
                    + ([f"{score:.4f}"] if genre_vals else [])
                    + [f"{gn_mean:.3f}", f"{gn_max:.3f}", f"{attn_max:.1f}"])
            t0, tokens_since = time.time(), 0
            gn_sum, gn_max, gn_n = 0.0, 0.0, 0
            # keep a separate copy of the best-ever model by that score
            if score < best_val:
                best_val = score
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
        # Samples take turns through config.sample_prompts (just "\n"
        # before Stage 4), and stop at <|endoftext|> if the model says so.
        if step > 0 and step % config.sample_interval == 0:
            prompts = config.sample_prompts
            prompt = prompts[(step // config.sample_interval - 1)
                             % len(prompts)]
            ctx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long,
                               device=device)
            with autocast:
                out = model.generate(ctx, max_new_tokens=200,
                                     temperature=0.8, top_k=50,
                                     stop_token=tokenizer.eot_id)
            model.train()  # generate() switched to eval mode
            print("-" * 60)
            print(tokenizer.decode(out[0].tolist()))
            print("-" * 60)

        if step > 0 and step % config.checkpoint_interval == 0:
            elapsed = elapsed_before + (time.time() - session_start)
            save_checkpoint(ckpt_path, model, optimizer, step, config,
                            best_val, elapsed)

        # SNAPSHOTS: latest.pt is overwritten every checkpoint_interval, so
        # by the time a problem shows in the log, the last good state may be
        # gone (Stage 4's first run had nothing between step 27,000 and the
        # spike). Keep a few older full copies to roll back to.
        if (config.snapshot_interval and step > 0
                and step % config.snapshot_interval == 0):
            snap_dir = os.path.join(ckpt_dir, "snapshots")
            os.makedirs(snap_dir, exist_ok=True)
            elapsed = elapsed_before + (time.time() - session_start)
            save_checkpoint(os.path.join(snap_dir, f"step-{step:06d}.pt"),
                            model, optimizer, step, config, best_val, elapsed)
            snaps = sorted(f for f in os.listdir(snap_dir)
                           if f.startswith("step-") and f.endswith(".pt"))
            for old in snaps[:-config.snapshots_kept]:
                os.remove(os.path.join(snap_dir, old))

    elapsed = elapsed_before + (time.time() - session_start)
    save_checkpoint(ckpt_path, model, optimizer, config.max_steps - 1,
                    config, best_val, elapsed)
    print(f"done. best val loss {best_val:.4f}. "
          f"total training time {fmt_hms(elapsed)}. "
          f"checkpoints in {ckpt_dir}")


if __name__ == "__main__":
    main()
