# Chapter 5 — Assembling the full model

*Code: `tlm/model.py`, class `TLM`*

All the pieces exist; this chapter is about the glue. `TLM.forward` is
seven lines of substance:

```python
pos = torch.arange(T)
x = self.tok_emb(idx) + self.pos_emb(pos)   # ch. 2: ids -> vectors
x = self.drop(x)
for block in self.blocks:                    # ch. 3-4: the stack
    x = block(x)
x = self.ln_f(x)                             # final LayerNorm
logits = self.lm_head(x)                     # vectors -> vocab scores
```

## The output head: vectors → predictions

After the last block, position i holds a C-dim vector summarizing
everything relevant to predicting token i+1. `lm_head` — a single linear
layer, no activation — maps it to `vocab_size` raw scores called
**logits**. Higher logit = model considers that token more likely.

Why raw scores instead of probabilities? Numerical honesty: softmax
involves exponentials, and computing `log(softmax(...))` naively overflows
or loses precision. So the model outputs logits, and whoever consumes them
applies softmax in a numerically careful way — `F.cross_entropy` during
training (chapter 6), explicit softmax during sampling (chapter 7).

Note that logits come out at **every position**, shape (B, T, vocab): one
next-token prediction problem per position. Training uses all T of them at
once (that's the efficiency superpower of the causal mask); generation only
looks at the last one.

## Weight tying: one matrix, two jobs

The embedding table maps token → vector; `lm_head` maps vector → token
scores. Both are (vocab × C) matrices, and the code sets them to be **the
same tensor**:

```python
self.lm_head.weight = self.tok_emb.weight
```

The intuition: with a tied matrix, the logit for token t is the dot product
of the final hidden state with *token t's own embedding* — "how much does
what I've computed resemble this token's vector?" The two jobs are two
directions of one relationship, and making them share parameters saves
vocab×C weights (a full 1M of our 7.4M in Stage 2!) while slightly
*improving* quality (Press & Wolf, 2016). GPT-2 does exactly this.

## Initialization: why the starting point matters

`_init_weights` sets every weight to a small gaussian (std 0.02), biases to
zero. Init is one of those things that looks like boilerplate but decides
whether training works:

- **Too large**: softmaxes and GELUs start saturated, gradients are tiny or
  chaotic, and the model spends thousands of steps recovering — or never
  does.
- **Too small**: every token's representation is nearly identical; the
  early gradient signal is mush.
- **Just right**: activations arrive everywhere at O(1) scale, and the very
  first loss should equal the uniform-guess value `ln(vocab_size)` — a
  check you can (and should) verify at step 0 of any run.

Plus the residual-stream refinement from chapter 4: the two projections
that *write onto* the stream (`attn.out_proj`, `mlp.proj`) get std
0.02/√(2·n_layer), because 2·n_layer independent writers add variance
linearly, and this keeps the stream's total variance depth-independent at
step 0.

## Counting parameters

`num_params()` reports 2.68M (shakespeare) / 7.37M (stories). It's worth
once tracing where they are — chapter 0 has the table. The striking part at
tiny scale is how *large a share* the (tied) embedding is: 1.05M of 7.4M
≈ 14% (GPT-2: ~30%!). As models scale, block weights grow with n_embd² but
the embedding only with n_embd, so frontier models spend a far smaller
fraction on vocabulary.

## What TLM leaves out (deliberately)

Real modern architectures differ in ways that are all *refinements, not
rethinks* — good next reads once this code feels obvious:

- **RoPE** instead of learned position embeddings (better long-context
  behavior)
- **RMSNorm** instead of LayerNorm (cheaper, no mean subtraction)
- **KV-caching** at inference: our `generate()` re-runs the full forward
  pass per token, recomputing keys/values it already computed — O(T²) per
  token where a cache makes it O(T). Correct, just wasteful; fine at toy
  scale, unthinkable in production.
- **FlashAttention** (`F.scaled_dot_product_attention`): computes exactly
  our four attention lines but never materializes the T×T matrix. We write
  the explicit version because *seeing* the matrix is the point.
- Mixture-of-experts, grouped-query attention, sliding-window attention:
  all efficiency plays on the same skeleton.

## Experiments

- Verify the step-0 loss equals ln(vocab_size) for both configs.
- Untie the weights (delete the tying line) and retrain Stage 1: ~equal or
  slightly worse val loss, with 12k extra parameters.
- Print `model` — PyTorch's module tree is a nice map of everything built
  so far.
