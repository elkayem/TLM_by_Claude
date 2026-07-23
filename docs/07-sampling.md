# Chapter 7 — Sampling: turning predictions into text

*Code: `tlm/model.py`, method `TLM.generate`; CLI in `tlm/generate.py`*

The trained model computes `P(next token | context)` — a distribution, not
text. Generation is a loop around it:

```
context → forward pass → distribution over next token → pick one
        → APPEND it to the context → repeat
```

Two things about this loop deserve a pause:

- **The model has no memory between iterations.** Nothing persists except
  the text itself; the growing sequence is the model's entire working
  state. This is equally true of production LLMs — a chat model "remembers"
  the conversation only because the whole transcript is re-fed each turn.
- **The model eats its own output.** Token 50 of a generation was sampled,
  appended, and is now indistinguishable-from-data context for token 51.
  One bad sample can derail everything after it — and conversely,
  restricting *which* tokens can be sampled (below) is high leverage.

## Why not just pick the most likely token?

Greedy decoding (always argmax) seems natural and is quietly terrible: text
becomes dull and repetitive, frequently collapsing into literal loops
("the dog. the dog. the dog.") — once a phrase is in context, its own
repetition often becomes the modal continuation, a self-feeding cycle.
Real text is *surprising*; matching its statistics requires actually
sampling from the distribution. The knobs below all navigate the space
between too-boring (greedy) and too-random (raw sampling with its long
tail of barely-plausible tokens).

## The knobs

**Temperature** — divide the logits by `t` before softmax.
- `t → 0`: sharpens toward greedy (deterministic, repetitive)
- `t = 1`: the model's honest distribution
- `t > 1`: flattens toward uniform (creative → incoherent)

Why dividing logits works: softmax exponentiates, so scaling logits by 1/t
raises every probability ratio to the power 1/t — gaps between favorites
and long-shots stretch (t<1) or compress (t>1) smoothly.

**Top-k** — keep only the k highest-probability tokens, renormalize, sample
among them. Blunt but effective: whatever lives in the tail past rank 50
(usually nonsense continuations) gets exactly zero chance — important
because over hundreds of sampled tokens, even a 1% tail is hit regularly.

**Top-p (nucleus)** — keep the smallest set of tokens whose probabilities
sum to p (e.g. 0.9). Adaptive where top-k is fixed: when the model is
certain (one token at 95%), the nucleus is 1–2 tokens; when it's genuinely
torn among 30 options, all 30 stay in play. The implementation (sort,
cumulative-sum, cut — see the code) is a nice little tensor-manipulation
étude.

These compose: temperature reshapes the distribution, then top-k/top-p
truncate it. `temperature 0.8, top_k 50` (the CLI defaults) is a reasonable
classic; `temperature 0.9, top_p 0.95` is a good alternative.

## Things to try

```
python -m tlm.generate --run shakespeare --prompt "ROMEO:" --tokens 400
```

- **A temperature ladder**: same prompt and seed at t = 0.2 / 0.6 / 0.8 /
  1.0 / 1.3. The march from robotic to fluent to unhinged makes the knob
  visceral in a way no explanation can.
- **Prompt sensitivity** (Stage 2): "Once upon a time" (in-distribution)
  vs "The quarterly earnings report" (very much not). A model this small
  degrades fast off-distribution — a miniature of why prompting matters
  with real LLMs.
- **Watch a derailment**: at high temperature, find the exact token where
  a story goes off the rails, and notice how everything after is
  consistent with the mistake — self-conditioning, visible.
- `--seed 42` twice: identical output. The "creativity" is exactly the
  pseudo-randomness of `torch.multinomial`, nothing more.

## Where to go next

You now have the complete mental toolkit for the architecture generation
side of LLMs. Natural next steps, roughly in order of effort:

1. **KV-caching** — make `generate()` not re-compute the past every token
   (chapter 5 sketched why; implementing it is a satisfying exercise).
2. **An attention visualizer** — dump `att` matrices during generation and
   heatmap them (chapter 3's experiment, applied to generation).
3. **Fine-tuning** — take your trained stories model and continue training
   on a tiny corpus of your own text; watch it adapt.
4. Read Karpathy's nanoGPT and llm.c, then the GPT-2 paper — you'll be
   surprised how readable they've become.
