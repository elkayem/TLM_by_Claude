# Chapter 1 — Tokenization: text → integers

*Code: `tlm/tokenizer.py`*

A neural network is a pile of matrix multiplications; it cannot eat text.
The tokenizer converts text to a sequence of integers (**token ids**) and
back. It is trained separately, before the model, and frozen; the model
never sees characters, only ids.

## The design decision: what should one token be?

This is a genuine trade-off, and TLM's two stages sit at two points on it:

**Characters (Stage 1).** Vocab of 65. Pros: dead simple, no unknown-word
problem, tiny embedding table. Cons: sequences are long — 256 context slots
hold only 256 characters (~50 words), and the model must burn capacity just
learning to spell.

**Words?** (Not used.) Vocab explodes (every name, typo, and plural is a new
token), and any word not seen in training is unrepresentable.

**Subwords via BPE (Stage 2).** The compromise used by essentially all real
LLMs: common words become single tokens, rare words split into pieces
(`"unbelievable"` → `"un" + "believ" + "able"`), and in the worst case
anything decomposes to characters. Vocab is a chosen hyperparameter (TLM:
4096; GPT-2: 50,257). At ~4 characters per token, our 256 slots now hold a
whole short story. **Same model, 4× more effective context, bought purely
by the tokenizer.**

## How BPE training works

`BPETokenizer.train()` implements it directly — the algorithm is charmingly
simple:

1. Start with the vocabulary = every single character.
2. Count every **adjacent pair** of tokens in the corpus.
3. Merge the most frequent pair into one new token (`('t','h')` → `'th'`),
   add it to the vocabulary, rewrite the corpus.
4. Repeat until the vocabulary hits the target size.

Run `data/prepare_stories.py` and watch the printed merges: early rounds
learn digraphs (`th`, `in`, `an`), middle rounds word fragments (`ing`,
`ed`), late rounds entire words. The final merges for TinyStories are
things like `' once'`, `' little'`, `' happy'` — a frequency-portrait of
the corpus. That's also the key caveat: a BPE vocab is **fit to its
corpus**. Ours, trained on children's stories, would tokenize legal or
medical text very inefficiently (many tiny pieces per word).

Encoding new text replays the merge list in the order learned (order
matters — `'the'` requires `'th'` to exist first). Decoding is trivial:
tokens are literal substrings, so it's string concatenation.

Two practical details worth noticing in the code:

- **Pre-tokenization** (`_WORD_RE`): text is first split into "words" and
  merges only happen inside them. This prevents useless cross-word tokens
  (`'e t'`) and enables a word→ids cache that makes encoding 2 GB feasible
  in pure Python.
- **Spaces belong to tokens**: `' the'` (leading space) and `'the'` are
  different tokens. That's how BPE round-trips spacing exactly without any
  special machinery — and it's why, in real LLM APIs, a stray trailing
  space in a prompt can change the output: it changes the tokens.

## Connection to the model

The tokenizer fixes `vocab_size`, which sets the size of two things in
`model.py`: the embedding table (chapter 2) and the output layer (chapter
5). It also sets the *baseline loss*: a model guessing uniformly scores
`ln(vocab_size)` — 4.17 for Stage 1, 8.32 for Stage 2. Your first training
step will land almost exactly there, which is a satisfying check that
everything is wired correctly.

## Experiments

- Print `tokenizer.encode("Once upon a time")` for both tokenizers and
  compare lengths.
- Retrain BPE with vocab 512 vs 4096 and compare tokens-per-character
  compression on a sample. Where do diminishing returns kick in?
- Find a word TinyStories BPE tokenizes badly (try technical jargon).
