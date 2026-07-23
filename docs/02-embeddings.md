# Chapter 2 — Embeddings: integers → vectors

*Code: `tlm/model.py`, top of `TLM.__init__` and `TLM.forward`*

Token id 17 is not "more" than token id 4 — the ids are arbitrary labels.
Before the network can compute with tokens, each id must become a **vector
of continuous numbers** that gradient descent can shape. That conversion is
the embedding.

## It's just a lookup table

`nn.Embedding(vocab_size, n_embd)` is nothing more than a matrix with one
row per vocabulary token. "Embedding token 17" means "take row 17". No math,
no mystery — but the rows are **learned parameters**, adjusted by
backpropagation like every other weight in the model.

What's remarkable is what training does to this table. The model is pushed
to give similar predictions in similar contexts, and the cheapest way to do
that is to give interchangeable tokens similar rows. So `cat` and `dog`
drift close together; `happy` and `sad` end up related-but-opposed. Nobody
programs this in — geometric structure appears because it reduces loss.
After training Stage 2, try measuring cosine similarity between a few word
embeddings; even a 7M model shows clear neighborhoods.

The vector dimension `n_embd` (TLM: 192–256, GPT-2: 768, frontier models:
tens of thousands) is the **width** of the model, and this same width is
used all the way up the network — it's the width of the "residual stream"
(chapter 4) that everything reads from and writes to.

## The position problem

Here is a genuinely non-obvious fact about the transformer: **attention is
order-blind.** Chapter 3 will show that attention is a weighted average
over a *set* of positions — permute the input tokens and (without a fix)
the outputs permute identically, with nothing changed. "dog bites man" and
"man bites dog" would be the same sentence. RNNs, which transformers
replaced, got order for free by processing tokens one at a time;
transformers gave that up to gain parallelism, and must buy order back.

The fix (in `TLM.forward`):

```python
x = self.tok_emb(idx) + self.pos_emb(pos)
```

A second lookup table, indexed by **position** (0..block_size-1) instead of
token id. Every token's vector becomes *what it is* plus *where it sits*.
Downstream attention can then learn position-sensitive behavior ("attend to
the token right before me") because position information is physically
present in the vectors.

Why does *adding* two meanings into one vector work, rather than smearing
them together? High-dimensional space is roomy: in 256 dimensions the two
tables can occupy roughly independent subspaces, and later layers can learn
projections that read out one part or the other. This "just add things to
the vector" move is very transformer-idiomatic — the residual stream
(chapter 4) runs on the same principle.

TLM uses **learned** position embeddings (as GPT-2 did) because they're the
simplest to understand: positions are tokens too, in their own table. Real
modern models mostly use RoPE (rotating q/k vectors by a position-dependent
angle inside attention itself), which extrapolates better to long contexts —
a good thing to read about once this chapter feels obvious.

One consequence worth internalizing: the position table has exactly
`block_size` rows, which is *why* the model has a hard context limit. Feed
it position 257 and there is literally no row to look up. When you hear
about an LLM's "context window", this (generalized) is the constraint.

## Experiments

- After training, compute cosine similarities between `tok_emb` rows for
  related words (Stage 2) or letters (Stage 1 — vowels cluster!).
- Delete `+ self.pos_emb(pos)` and retrain Stage 1. Loss falls, then
  plateaus well above normal: the model becomes a bag-of-characters
  predictor. Its samples are anagram soup — a vivid proof of order-blindness.
