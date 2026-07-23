# Chapter 4 — The block: MLP, residuals, LayerNorm

*Code: `tlm/model.py`, classes `MLP` and `Block`*

A transformer is one block, stacked. The block is four lines:

```python
x = x + self.attn(self.ln1(x))   # communicate
x = x + self.mlp(self.ln2(x))    # compute
```

Attention (chapter 3) mixes information *across* positions; this chapter
covers the other three ingredients: the MLP, the `x = x + ...` residual
pattern, and the LayerNorms.

## The MLP: where the model thinks

After attention delivers information to a position, the MLP processes it —
applied to every position **independently and identically** (no
token-to-token interaction here at all). It's the plainest network
imaginable: expand from C to 4C channels, apply a GELU nonlinearity,
contract back to C.

Why does this matter when attention gets all the press? Three reasons:

- **The nonlinearity lives here.** Attention's value path is linear; stack
  pure attention and you get (nearly) one big linear map. GELU is what lets
  the network compute genuinely nonlinear functions. (GELU ≈ a smooth ReLU;
  smoothness gives better-behaved gradients near zero.)
- **Most parameters live here** — 2/3 of the block's weights (8C² vs
  attention's 4C²). Interpretability research suggests MLP layers act as
  key-value memories storing the model's factual/statistical knowledge.
- A useful mental model of the 4C hidden layer: **a bank of 4C learned
  feature detectors** ("this looks like the end of a quotation", "this word
  is a name"), whose findings the contraction re-summarizes into the stream.

Why 4×? Honest answer: the original paper used it, it works, and it stuck.
It's a knob (`experiments below`).

## Residual connections: the load-bearing trick

Notice that neither sub-layer ever *replaces* x. The block computes
`x + f(x)` — each sub-layer only **adds a correction** to what flows
through. This pattern (from ResNets, 2015) is arguably the single most
important trick in deep learning, for two reasons:

**Gradient flow.** Backprop through `x = x + f(x)` sends the gradient down
two paths, and the identity path passes it through *unchanged*. However
deep the stack, layer 1 receives a direct, undecayed error signal. Without
residuals, gradients shrink multiplicatively through every layer and deep
transformers simply fail to train (the "vanishing gradient" problem that
capped network depth for decades).

**The residual stream view.** Because everything is additive, think of x as
a **communication bus** of width n_embd running from the embeddings to the
output head. Every attention head and every MLP *reads* from the bus
(through its input projections) and *writes* a small update back (through
its output projection). Layers don't transform the world — they annotate
it. This picture (from Anthropic's interpretability work) also explains two
init-time details in `TLM.__init__`: writes onto the bus are scaled by
1/√(2·n_layer) so that stacking many writers doesn't blow up the variance,
and it motivates why n_embd is *the* central width of the whole model.

## LayerNorm: keeping the numbers sane

`nn.LayerNorm(n_embd)` takes each individual token vector, rescales it to
mean 0 / variance 1 across its C channels, then applies a learned
per-channel gain and bias. That's all. Its job is boring but vital: after
many additive writes, vector magnitudes drift; unnormalized, they compound
across layers into saturated softmaxes and exploding activations. LayerNorm
resets the scale at the door of every sub-layer.

Note it normalizes **each token separately** — no averaging across the
batch (BatchNorm) or across positions. Nothing leaks between tokens, so
causality is preserved.

**Pre-norm vs post-norm** — a detail that decided an era. The 2017 paper
normalized *after* the residual add (`norm(x + f(x))`), which puts
normalizations *on* the bus, disrupting the clean gradient path; those
models needed careful warmup to train at all. GPT-2 moved the norm *inside*
(`x + f(norm(x))`) — the bus stays pure additions end to end. TLM follows
GPT-2, as does virtually everything modern. When comparing the code to the
original paper's diagram, this is the deliberate difference.

## The stack as iterative refinement

Blocks don't have separate jobs assigned; each just refines the stream
further. A rough empirical picture from probing real models: early layers
resolve local structure (syntax, word identity), middle layers integrate
longer-range context, late layers sharpen the actual next-token prediction.
Depth (`n_layer`) buys *rounds* of communicate-then-compute: with 8 layers,
information can hop through 8 chains of intermediaries — enough for "the
pronoun refers to the girl mentioned two sentences ago" style reasoning.

## Experiments

- Change the MLP ratio from 4 to 1 or 8 (adjusting for parameter count) and
  compare loss.
- Remove the residuals (`x = self.attn(self.ln1(x))`) in a 6-layer model:
  watch training crawl or collapse. The single most instructive ablation in
  the project.
- Train 2-layer vs 8-layer at matched parameter count (widen the shallow
  one): depth wins at this scale.
