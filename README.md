# TLM — Tiny Language Model

A complete GPT-style transformer built from scratch in PyTorch, small enough
to train on a CPU, written to be **read**. Every file is heavily commented,
and the `docs/` folder is a chapter-by-chapter walkthrough of how and why it
all works.

**Note from elkayem:** This project was created entirely by Claude Fable 5 based on my prompts. 
My goal was to create an LLM that was simple enough for me to train on my home PC, and 
commented well enough that I could understand the inner mechanics. -Larry

## The three models

| | Stage 1: `shakespeare` | Stage 2: `stories` | Stage 3: `instruct` |
|---|---|---|---|
| Dataset | Tiny Shakespeare (~1 MB) | TinyStories (~1.9 GB) | TinyStories-Instruct (~2.7 GB) |
| Tokenizer | character-level (65 tokens) | BPE, trained from scratch (4096) | new BPE (4096) + `<\|endoftext\|>` special token |
| Size | ~2.7M parameters | ~7.4M parameters | ~27M parameters |
| Context | 256 tokens | 256 tokens | 512 tokens |
| Training time (CPU) | 2 hours | 56 hours (the long weekend) | 118 hours (5 days) |
| Best val loss | 1.609 | 1.595 | 1.257* |
| What you get | fake Shakespeare | coherent little children's stories | stories to order: give it a summary, required words, or features |

\* Not comparable across columns: each model is scored under its own
tokenizer and dataset, which changes what "loss per token" means.

Stage 1 is the fast feedback loop for learning the architecture. Stage 2 is
the payoff: the TinyStories dataset was designed (Eldan & Li 2023) so that
even single-digit-million-parameter models learn grammar, plot, and
character consistency. Stage 3 crosses from *completing* text to *following
instructions* — same loss, differently formatted data (see
[docs/08-instruct.md](docs/08-instruct.md)) — and adds fine-tuning: the 7.4M
stories model gets adapted to the instruct format as a warm-up baseline.
The Stage 1/2 numbers are measured results from an Intel Core Ultra 7 265
(20 cores, no GPU) — a sample from the finished stories model:

> Once upon a time, there was a little girl named Lily. She loved to play
> with her toys and eat candy. One day, she went to the park to play with
> her friends. They were having fun, but then Lily saw a butterfly. She
> wanted to catch it, but the butterfly flew away.

## Quickstart

All commands run from this `tlm/` directory, using the project venv
(`..\.venv` — created during setup; activate it or call its python directly).

```
# Stage 1 --------------------------------------------------------
python data\prepare_shakespeare.py          # download + tokenize (seconds)
python -m tlm.train --config shakespeare    # ~2 hours; safe to Ctrl+C
python -m tlm.generate --run shakespeare --prompt "ROMEO:" --tokens 400

# Stage 2 --------------------------------------------------------
python data\prepare_stories.py              # ~2GB download + BPE (run overnight)
python -m tlm.train --config stories        # the weekend run
python -m tlm.train --config stories --resume    # continue after interruption
python -m tlm.generate --run stories --prompt "Once upon a time"

# Stage 3: instruction following (see docs/08-instruct.md + PLAN-stage3.md)
python data\prepare_instruct.py             # ~2.7GB download + encode (hours)
python -m tlm.train --config instruct-ft --init-from stories   # fine-tune warm-up
python -m tlm.train --config instruct-pilot                    # measure s/step
python -m tlm.train --config instruct                          # the week-long run
```

While training, a text sample is printed every 500–1000 steps — watching the
output evolve from noise → words → sentences is the best part; keep notes in
[JOURNAL.md](JOURNAL.md).

**Where checkpoints go:** `~/tlm-checkpoints/<run>/` by default — deliberately
outside the repo, since checkpoints are large and rewritten every 1000 steps
(inside a synced folder like OneDrive that means constant re-uploading).
Set the `TLM_CKPT_DIR` environment variable to put them elsewhere. Each run
folder holds `best.pt` (lowest val loss), `latest.pt` (most recent, used by
`--resume`), and `log.csv` (step, lr, losses, throughput — for plotting).

**Practical notes for the weekend run:** set your power settings so the PC
doesn't sleep; the run checkpoints every 1000 steps (~50 min), so an
interruption costs at most that, and `--resume` picks up where it stopped
(including the elapsed-time clock).

## The guided tour (docs/)

Start with the [architecture figure](docs/architecture.svg) — the whole
model on one page, from token ids down to the causal attention matrix.
Then read in order, next to the code file each chapter covers:

1. [00-overview.md](docs/00-overview.md) — the big picture: what a language model is
2. [01-tokenization.md](docs/01-tokenization.md) — text → integers (`tokenizer.py`)
3. [02-embeddings.md](docs/02-embeddings.md) — integers → vectors (`model.py: TLM.__init__`)
4. [03-attention.md](docs/03-attention.md) — the heart of it (`model.py: CausalSelfAttention`)
5. [04-transformer-block.md](docs/04-transformer-block.md) — MLP, residuals, LayerNorm (`model.py: Block`)
6. [05-full-model.md](docs/05-full-model.md) — assembling TLM (`model.py: TLM`)
7. [06-training.md](docs/06-training.md) — the training loop (`train.py`)
8. [07-sampling.md](docs/07-sampling.md) — generating text (`generate.py`, `TLM.generate`)
9. [08-instruct.md](docs/08-instruct.md) — Stage 3: instruction following, special tokens, fine-tuning

## Layout

```
tlm/
├── README.md            you are here
├── JOURNAL.md           training diary: samples as the models learned
├── PLAN-stage3.md       what's next: instruction-following (~25M params)
├── docs/                the textbook + architecture figure
├── tlm/
│   ├── config.py        all hyperparameters + presets
│   ├── tokenizer.py     char + from-scratch BPE
│   ├── model.py         the transformer (start reading here)
│   ├── train.py         training loop, checkpointing, logging
│   └── generate.py      sampling CLI
├── data/
│   ├── prepare_shakespeare.py
│   ├── prepare_stories.py
│   └── prepare_instruct.py    (downloaded corpora land here, gitignored)
└── weights/             archived weights-only exports (gitignored)

~/tlm-checkpoints/       full checkpoints, outside the repo (see above)
```

## References

All links verified. Each paper below is the origin of something you can
point to in this repo's code.

### Datasets

- **Tiny Shakespeare** — from Andrej Karpathy's
  [char-rnn](https://github.com/karpathy/char-rnn) repository (2015, MIT
  license), introduced alongside the blog post
  ["The Unreasonable Effectiveness of Recurrent Neural Networks"](https://karpathy.github.io/2015/05/21/rnn-effectiveness/).
- **TinyStories** and **TinyStories-Instruct** — Ronen Eldan & Yuanzhi Li,
  ["TinyStories: How Small Can Language Models Be and Still Speak Coherent
  English?"](https://arxiv.org/abs/2305.07759) (2023). Datasets on Hugging
  Face: [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories),
  [roneneldan/TinyStoriesInstruct](https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct)
  (CDLA-Sharing-1.0 license).

### Architecture

- Vaswani et al., ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)
  (2017) — the transformer: scaled dot-product attention, multi-head
  attention, the 4× MLP (`model.py` throughout).
- Radford et al., ["Language Models are Unsupervised Multitask
  Learners"](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
  (2019) — GPT-2, the decoder-only recipe TLM follows: pre-norm blocks,
  learned positions, the init scheme, BPE at scale.
- Sennrich, Haddow & Birch, ["Neural Machine Translation of Rare Words with
  Subword Units"](https://arxiv.org/abs/1508.07909) (2015) — byte-pair
  encoding for tokenization (`tokenizer.py: BPETokenizer`).
- He et al., ["Deep Residual Learning for Image
  Recognition"](https://arxiv.org/abs/1512.03385) (2015) — residual
  connections (`model.py: Block`, the `x = x + f(x)` pattern).
- Ba, Kiros & Hinton, ["Layer Normalization"](https://arxiv.org/abs/1607.06450)
  (2016) — LayerNorm (`ln1`/`ln2`/`ln_f`).
- Hendrycks & Gimpel, ["Gaussian Error Linear Units
  (GELUs)"](https://arxiv.org/abs/1606.08415) (2016) — the MLP's
  activation function.
- Press & Wolf, ["Using the Output Embedding to Improve Language
  Models"](https://arxiv.org/abs/1608.05859) (2016) — weight tying between
  the embedding and the output head.

### Training & sampling

- Kingma & Ba, ["Adam: A Method for Stochastic
  Optimization"](https://arxiv.org/abs/1412.6980) (2014) and Loshchilov &
  Hutter, ["Decoupled Weight Decay Regularization"](https://arxiv.org/abs/1711.05101)
  (2017) — the AdamW optimizer (`train.py`).
- Hoffmann et al., ["Training Compute-Optimal Large Language
  Models"](https://arxiv.org/abs/2203.15556) (2022) — "Chinchilla" scaling:
  the ~20-tokens-per-parameter rule used to size the Stage 3 run.
- Holtzman et al., ["The Curious Case of Neural Text
  Degeneration"](https://arxiv.org/abs/1904.09751) (2019) — nucleus (top-p)
  sampling and why greedy decoding degenerates (`model.py: generate`).

The project's spiritual ancestors are Karpathy's
[char-rnn](https://github.com/karpathy/char-rnn) and
[nanoGPT](https://github.com/karpathy/nanoGPT) — TLM is an independent
implementation in the same teach-by-reading tradition.
