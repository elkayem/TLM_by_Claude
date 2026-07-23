# Chapter 6 — Training: how the weights learn

*Code: `tlm/train.py`*

Everything so far describes a *function* with ~7M adjustable knobs. Training
is the process of turning those knobs so the function predicts text well.
The core loop is four lines; the rest of `train.py` is the scaffolding that
makes a two-day run survivable and observable.

## The objective: cross-entropy

For each position the model outputs logits; the data says which token
actually came next. The loss is

> **−log P(the true next token)**

averaged over all B×T positions in the batch. Assign the truth probability
1.0 → loss 0. Probability 0.37 → loss 1.0. Uniform guessing → loss
ln(vocab_size). Minimizing it is literally maximizing the probability the
model assigns to the training text.

Intuition anchor: **e^−loss ≈ the probability given to the right answer.**
A Stage 1 loss of 1.5 means the model puts ~22% on the correct next
character — impressive against a 65-way choice. (Related jargon: e^loss is
the "perplexity", the effective number of choices the model is torn
between.)

## One step, anatomically

```python
x, y = get_batch(...)        # 1. sample data
_, loss = model(x, y)        # 2. forward: how wrong are we?
optimizer.zero_grad(); loss.backward()   # 3. backward: whose fault?
optimizer.step()             # 4. nudge every weight
```

- **get_batch** slices 32 random windows out of the corpus-as-one-long-array.
  The target `y` is the input `x` shifted one token right — see the comment
  block in the code; this offset *is* next-token prediction. No epochs, no
  shuffling infrastructure: random windows from a giant stream are
  statistically close enough and far simpler.
- **loss.backward()** is backpropagation: one sweep of the chain rule from
  the loss back through every operation this file and `model.py` performed,
  filling in `d(loss)/d(weight)` for all parameters. PyTorch recorded the
  computation graph during the forward pass; this replays it in reverse.
  You wrote no derivative code anywhere in this project — autograd is the
  entire reason libraries like PyTorch exist.
- **optimizer.step()** moves each weight a tiny distance against its
  gradient. Repeated ~60,000 times, this is the whole of learning.

## AdamW: gradient descent with shock absorbers

Plain SGD (`w -= lr * grad`) works but is miserable to tune. AdamW keeps
two running averages per weight — of the gradient (momentum: smooths out
batch-to-batch noise) and of its square (scale: gives each weight its own
effective step size, so rarely-updated weights take bigger steps). The "W"
is *decoupled weight decay*: a gentle constant pull of every weight toward
zero, applied directly rather than through the gradient machinery, as
regularization. Note the code exempts biases, LayerNorm gains, and
embeddings from decay — standard practice; decaying those hurts.

## The learning rate: the one knob that matters most

Too high → loss diverges or plateaus high (steps overshoot every valley).
Too low → glacial progress. And the best value *changes over the run*,
hence `get_lr`'s schedule:

- **Warmup** (first 200–1000 steps): ramp 0 → peak. A freshly initialized
  model produces violent, misleading gradients, and AdamW's running
  averages are still garbage; full-size steps here can wreck the run in
  ways it never recovers from.
- **Cosine decay** to 10% of peak: big steps early to cover ground, small
  steps late to settle *into* a minimum rather than orbiting it.

**Gradient clipping** (`clip_grad_norm_`) is the other safety rail: if a
freak batch produces a gradient vector longer than 1.0, rescale it (same
direction, capped length) so no single step can catapult the weights.

## Overfitting, and why val loss is the real score

The model's job is to learn *the language*, but a small dataset lets it
instead memorize *the text*. You can't tell the difference from training
loss — memorization drives it down beautifully. Hence the held-out **val
split** (text never trained on): `estimate_loss` averages 50 batches of it
(one batch is too noisy) with dropout off.

- Both falling → learning.
- Train falling, val flat/rising → memorization has begun; the gap is
  its size.

Stage 1's 1MB corpus **will** overfit if pushed — that's why its config has
dropout 0.2 and only 5000 steps, and why `best.pt` (lowest-val-loss
checkpoint) is kept separately from `latest.pt`. Stage 2's 500M tokens are
seen ~once each; memorizing is nearly impossible, so dropout drops to 0.05.
Watching the two runs behave differently *is* the lesson on overfitting.

## The scaffolding

Worth reading, unglamorous, and exactly what real training infra does at
1000× scale: **checkpoints** every 1000 steps carry model weights *and*
optimizer state (AdamW's averages — resuming without them causes a visible
loss bump), written atomically (temp file + rename) so a mid-write crash
can't corrupt the only copy. `--resume` picks up where it left off.
**log.csv** records step/lr/losses/speed for plotting. And the periodic
**sample generations** are qualitative eyes on the run: gibberish →
words → grammar is far more informative than a loss number.

## What to expect, concretely

- **Stage 1** (~2h): loss starts at 4.17. Within minutes: word-shaped
  gibberish. By the end: val ≈ 1.5, and samples with speaker names, iambic
  rhythm, archaic vocabulary — statistically Shakespeare-flavored noise.
- **Stage 2** (~48h): starts at 8.32, and the interesting band is ~1.0–1.5.
  Early samples are word salad; by the end you should see multi-sentence
  stories that mostly track their characters. Keep JOURNAL.md open;
  note the step number whenever a new ability appears.

## Experiments

- Set learning_rate to 3e-2 (100×) in the smoke config and watch the loss
  explode. Then 3e-6 and watch it crawl. Nothing teaches LR sensitivity
  like breaking it.
- Kill the training mid-run (Ctrl+C) and `--resume`: confirm the loss
  continues from where it stopped.
- Plot log.csv (train vs val) for Stage 1 and find where the curves part.
