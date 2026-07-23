# Chapter 3 — Attention: the heart of the transformer

*Code: `tlm/model.py`, class `CausalSelfAttention`*

Everything before this point processes each token in isolation. Attention
is the mechanism by which positions **communicate** — it is the only place
in the entire network where information moves between tokens. If you deeply
understand this one class, you understand transformers.

## The problem it solves

To predict the token after "…Lily picked up the ball and threw", the model
standing at "threw" needs to know about "ball" and "Lily" — information
sitting at *other positions*. Attention lets each position reach back,
decide which earlier positions are relevant, and pull in a summary of them.
Crucially, *which* positions are relevant is computed from the content
itself, fresh for every input — not fixed wiring.

## Query, Key, Value

Each position's vector is linearly projected three ways
(`qkv_proj` in the code):

- **query** — what this position is looking for
- **key** — how this position advertises what it contains
- **value** — what this position actually hands over if attended to

The library analogy: you (query) scan the spines on a shelf (keys), and
from the books that match you take the contents (values). Separating "how
I'm found" from "what you get from me" is what gives the mechanism its
flexibility — a token can be easy to find for one purpose and deliver
information for another.

## The equation, in four steps

The whole thing is one line of math, `softmax(qkᵀ/√d)·v`, and four lines of
code. For position *i*:

**1. Score every candidate.** `att = q @ k.transpose(-2,-1) / sqrt(hs)`
computes the dot product of each query with every key — one number per
(i, j) pair meaning "how relevant is token j to token i". This produces a
T×T matrix, and this matrix is why attention costs grow **quadratically**
with context length: doubling `block_size` quadruples this computation.

*Why the √hs?* A dot product of two 32-dimensional vectors with unit-ish
entries has standard deviation ~√32. Left unscaled, scores would arrive at
softmax large, softmax would saturate (all weight on one token), and the
gradients through it would vanish. Dividing by √hs keeps scores O(1) at any
head size. This tiny detail is in the title of the original paper ("scaled
dot-product attention") because without it, training is dramatically worse.

**2. Hide the future.** `masked_fill(mask == 0, -inf)` overwrites every
score where j > i with −∞. After softmax, e^{−∞} = 0: exactly zero weight
on future tokens. This **causal mask** is what makes next-token training
honest — the answer is sitting right there in the training batch, one
position over, and the mask is the only thing preventing the model from
just copying it. (Masking with −∞ *before* softmax, rather than zeroing
after, keeps the surviving weights summing to 1.)

**3. Scores → weights.** `softmax(att, dim=-1)` turns each row of scores
into a probability distribution over visible positions: non-negative, sums
to 1. Softmax rather than a hard argmax pick keeps everything
differentiable — "attend 60% here, 30% there" has a gradient; "pick the
best" doesn't.

**4. Weighted pickup.** `y = att @ v`: each position's output is the
weighted average of the value vectors it attends to. This is the actual
payload — information from wherever the weights point, delivered to
position i.

## Multi-head: several conversations at once

One softmax gives one blend — one "kind of lookup" per layer. Instead, the
code splits the channels into `n_head` groups (`view` + `transpose`
gymnastics in the code, annotated shape by shape), each with its own q/k/v
projections, each attending independently. Head 1 might track "the previous
token", head 2 "the subject of the sentence", head 3 "the last quotation
mark". The heads' outputs are concatenated and mixed by `out_proj`.

Note the economics: 8 heads of 32 dims cost the same FLOPs and parameters
as 1 head of 256 dims — the split is free. What multi-head buys is
*diversity* of attention patterns per layer, and inspecting trained heads
(the experiment below) shows they really do specialize.

## What attention is NOT

- It contains **no nonlinearity** in the value path — scoring is bilinear,
  pickup is a weighted average. The "thinking" happens in the MLP
  (chapter 4). A useful slogan: **attention moves information; the MLP
  transforms it.**
- It has **no notion of position** of its own — the T×T score matrix is
  computed pairwise from content. All order-awareness was injected by the
  position embeddings (chapter 2).

## Experiments

- Save the `att` tensor from a trained Stage 1 model, and plot a few
  heads' T×T matrices as heatmaps (after `softmax`). You'll find heads
  attending to the previous character, to the start of the current word,
  to matching earlier characters — discovered, not designed.
- Set `n_head = 1` (keeping n_embd fixed) and retrain: same parameter
  count, measurably worse loss.
- Remove the `/ math.sqrt(hs)` and watch training degrade.
