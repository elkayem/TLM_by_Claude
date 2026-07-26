# Chapter 8 — Stage 3: instruction following

*Code: `data/prepare_instruct.py`, the Stage 3 presets in `config.py`,
special tokens in `tokenizer.py`, `--init-from` in `train.py`*

The stories model completes text. Ask a modern LLM assistant something,
though, and it doesn't merely continue your words — it *does what you
asked*. Stage 3 crosses that gap with a claim worth stating up front,
because it demystifies more of modern AI than any other single idea in
this project:

> **Instruction following is not a new mechanism. It is the same
> next-token loss, on text formatted as instruction + response.**

## The data is the whole trick

TinyStories-Instruct prepends structured fields to each story:

```
Features: Dialogue
Summary: Tom and Anna fly on a big plane to a sunny place.
Story:
Tom and Anna are brother and sister. They like to play...
<|endoftext|>
```

Train an ordinary next-token predictor on millions of these and something
useful falls out for free: by the time the model is predicting story
tokens, the summary/features/words are *in its context* — and matching
them measurably reduces loss. A story that ignores its stated summary is
statistically wrong. So the model learns to condition on the preamble,
and at generation time you exploit it: write your own `Summary:` and
`Words:` fields, end with `Story:`, and sample. The model "follows
instructions" because in its universe, that's simply what text does.

Chat assistants are this same move at scale: conversations formatted with
role markers, plus preference tuning (RLHF) layered on top to shape *how*
it responds. The mechanism under it all is the one in this repo.

## New concept 1: special tokens

The corpus separates stories with `<|endoftext|>`. In Stage 2 that string
was just 13 characters that BPE compressed like any other text — good
enough statistically, but nothing could rely on it *exactly*. Stage 3
makes it a **special token** (see `tokenizer.py`): carved out of the text
before BPE ever runs, mapped to one atomic id, never split, never
produced by merges. Two things become possible:

- story boundaries in the training data are exact and unambiguous;
- `generate()` can **stop** the moment the model emits that id
  (`stop_token` in `model.py`) — the model itself decides when the story
  is over, instead of rambling until the token budget runs out.

Every production LLM does this; chat-format markers are special tokens.

## New concept 2: fine-tuning (`--init-from`)

Stage 3 trains its big model from scratch, but first we run a cheaper
experiment: take the *finished* 7.4M stories model and continue training
it on Instruct data (`--config instruct-ft --init-from stories`). Only
the weights carry over — fresh optimizer, fresh (much lower) learning
rate, short schedule. Watch what it teaches:

- **Transfer**: the model already knows English and story-shape; it only
  needs to learn the preamble format. Its loss drops to useful levels in
  hours, not days — this asymmetry (pretraining is expensive, adaptation
  is cheap) is the economic foundation of the entire modern LLM
  ecosystem.
- **The tokenizer is welded to the model.** The fine-tune data must be
  encoded with the *stories* tokenizer, not the new instruct one, because
  row 1287 of the embedding table means whatever token 1287 meant during
  pretraining. Swap tokenizers and every id points at the wrong vector.
  (This is why `prepare_instruct.py` writes two datasets.)
- **The low learning rate matters**: big steps on new data can bulldoze
  what the model already knows — *catastrophic forgetting*. Fine-tuning
  is a nudge, not a re-education.

## New concept 3: sizing a run (Chinchilla)

Stage 3's config isn't guesswork; it's the standard capacity math. The
stories run ended with train ≈ val loss — the signature of a model that
is **capacity-limited** (too small for its data), not overfit. The fix is
scale: n_embd 256 → 512 gives ~27M params (parameters grow with the
*square* of width). How much data does that want? The Chinchilla rule of
thumb: **~20 tokens per parameter** → ~500M tokens ≈ one epoch of this
corpus ≈ what one week of this CPU delivers at measured throughput. Model
size, dataset size, and compute budget all pointing at the same number is
what a well-posed run looks like. block_size also doubles to 512, since
the model must see preamble *and* story in one window to learn the
connection between them.

## What to expect, and how to play

The run prints samples as always; the milestone to watch for in your
journal is the moment generations start *respecting their own preambles*.
Once trained, prompt it with your own instructions (note `--tokens` is a
budget; with EOS it stops when the story ends):

```
python -m tlm.generate --run instruct --tokens 400 --prompt "Words: dragon, cake, brave
Summary: A brave mouse shares cake with a dragon.
Story:
"
```

Experiments:
- Contradict yourself (a `Summary:` about rain, `Words:` about sunshine)
  and watch which constraint wins.
- Ask for `Features: BadEnding` — the model this size does learn tone.
- Compare the fine-tuned 7.4M vs the from-scratch 27M on identical
  prompts: same data, ~4x the capacity, side by side.
- Prompt the *stories* model with an instruct preamble to see what
  formatting alone does to a model that never saw it.
