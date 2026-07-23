# TLM — Tiny Language Model

A complete GPT-style transformer built from scratch in PyTorch, small enough
to train on a CPU, written to be **read**. Every file is heavily commented,
and the `docs/` folder is a chapter-by-chapter walkthrough of how and why it
all works.

## The two models

| | Stage 1: `shakespeare` | Stage 2: `stories` |
|---|---|---|
| Dataset | Tiny Shakespeare (~1 MB) | TinyStories (~1.9 GB) |
| Tokenizer | character-level (65 tokens) | BPE, trained from scratch (4096 tokens) |
| Size | ~2.7M parameters | ~7.4M parameters |
| Training time (CPU) | ~2 hours | ~48 hours (the long weekend) |
| What you get | fake Shakespeare | coherent little children's stories |

Stage 1 is the fast feedback loop for learning the architecture. Stage 2 is
the payoff: the TinyStories dataset was designed (Eldan & Li 2023) so that
even single-digit-million-parameter models learn grammar, plot, and
character consistency.

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
```

While training, a text sample is printed every 500–1000 steps — watching the
output evolve from noise → words → sentences is the best part; keep notes in
[JOURNAL.md](JOURNAL.md). Progress is logged to `checkpoints/<run>/log.csv`.

**Practical notes for the weekend run:** set Windows power settings so the
PC doesn't sleep; the run checkpoints every 1000 steps (~50 min), so a
reboot costs at most that. `latest.pt` is the newest checkpoint, `best.pt`
the best validation loss so far.

## The guided tour (docs/)

Read in order, next to the code file each chapter covers:

1. [00-overview.md](docs/00-overview.md) — the big picture: what a language model is
2. [01-tokenization.md](docs/01-tokenization.md) — text → integers (`tokenizer.py`)
3. [02-embeddings.md](docs/02-embeddings.md) — integers → vectors (`model.py: TLM.__init__`)
4. [03-attention.md](docs/03-attention.md) — the heart of it (`model.py: CausalSelfAttention`)
5. [04-transformer-block.md](docs/04-transformer-block.md) — MLP, residuals, LayerNorm (`model.py: Block`)
6. [05-full-model.md](docs/05-full-model.md) — assembling TLM (`model.py: TLM`)
7. [06-training.md](docs/06-training.md) — the training loop (`train.py`)
8. [07-sampling.md](docs/07-sampling.md) — generating text (`generate.py`, `TLM.generate`)

## Layout

```
tlm/
├── README.md            you are here
├── JOURNAL.md           your training diary
├── docs/                the textbook
├── tlm/
│   ├── config.py        all hyperparameters + presets
│   ├── tokenizer.py     char + from-scratch BPE
│   ├── model.py         the transformer (start reading here)
│   ├── train.py         training loop, checkpointing, logging
│   └── generate.py      sampling CLI
├── data/
│   ├── prepare_shakespeare.py
│   └── prepare_stories.py
└── checkpoints/         created during training
```
