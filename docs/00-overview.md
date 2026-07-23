# Chapter 0 — The big picture

## What a language model actually is

Strip away everything, and a language model is one function:

> Given a sequence of tokens, output a probability for every possible
> **next** token.

That's it. TLM's entire forward pass — embeddings, attention, MLPs — exists
to compute `P(next token | tokens so far)` well. Everything impressive an
LLM does emerges from doing this one prediction extremely well, then calling
it in a loop (append the sampled token, predict again — see chapter 7).

Why does next-token prediction produce models that seem to *understand*?
Because predicting well **requires** understanding. To predict the next word
of "The capital of France is ___" you must know geography. To continue a
story about Lily's lost dog without contradicting yourself, you must track
who Lily is and what happened. Prediction is a training objective that
smuggles in everything else. TLM is far too small to learn much of the
world, but you will watch it climb the early rungs of exactly this ladder:
letter frequencies → spelling → grammar → short-range plot.

## The pipeline at a glance

```
 text ──tokenizer──► token ids ──embedding──► vectors
                                                 │
                              ┌──────────────────┘
                              ▼
                  ┌─► [ transformer block ]  ×  n_layer
                  │        attention   : positions share information
                  │        MLP         : each position computes on its own
                  └──── (residual stream carries it all upward)
                              │
                              ▼
                    logits: one score per vocab token, per position
                              │
              training ◄──────┴──────► generation
       compare to actual              softmax → sample → append → repeat
       next token (loss),
       adjust weights
```

Each chapter of these docs zooms into one box of that diagram.

## Where the parameters live

For the `stories` config (n_layer=8, n_head=8, n_embd=256, vocab 4096):

| Component | Formula | Count |
|---|---|---:|
| Token embedding (tied with output head) | vocab × n_embd | 1.05M |
| Position embedding | block_size × n_embd | 0.07M |
| Attention, all layers | ~4 × n_embd² × n_layer | 2.10M |
| MLPs, all layers | ~8 × n_embd² × n_layer | 4.20M |
| LayerNorms, biases | small | ~0.02M |
| **Total** | | **~7.4M** |

Two things worth noticing: the MLPs hold about twice as many parameters as
attention (a 2:1 ratio fixed by the standard 4× MLP expansion), and the
rule-of-thumb "12 × n_layer × n_embd²" for the block weights. When you hear
"a 7B model", it's these same tables scaled up: GPT-2 (124M) is this exact
architecture with n_layer=12, n_head=12, n_embd=768, vocab≈50k.

## Why a "decoder-only" transformer?

The original 2017 transformer ("Attention Is All You Need") had an encoder
(reads the input sentence) and a decoder (writes the translation). GPT-style
models keep only the decoder half: one stack that reads what exists so far
and predicts what comes next. Nearly every modern LLM — GPT-4, Claude,
Llama — is decoder-only. So this tiny model is architecturally a real LLM;
the differences from the frontier are almost entirely **scale** (parameters,
data, compute) plus post-training (instruction tuning, RLHF) that TLM
doesn't attempt.

## How to read this project

Suggested order:
1. Skim all eight chapters once, fast, tolerating confusion.
2. Read `model.py` top to bottom with chapters 2–5 open beside it.
3. Run the Stage 1 training and *watch* it — the samples every 500 steps
   are the concepts becoming visible.
4. Break things. The docs end with experiment suggestions; the fastest way
   to understand a component is to remove it and watch what dies.
