"""
Configuration for the Tiny Language Model (TLM).

Everything that defines a training run lives in one dataclass. Rather than
YAML files, we use plain Python: it's one fewer dependency, and you can see
the defaults and their documentation in a single place.

Three presets are defined at the bottom:

  smoke       - absurdly tiny, runs in seconds; only used to verify the
                pipeline works end to end.
  shakespeare - Stage 1: character-level model, ~1.6M params, trains to
                something entertaining in about an hour on CPU.
  stories     - Stage 2: BPE-tokenized model for TinyStories, ~9M params,
                the long-weekend run.
"""

from dataclasses import dataclass, asdict


@dataclass
class TLMConfig:
    # ---------------------------------------------------------------- model
    # These four numbers ARE the transformer. Parameter count is roughly
    #   12 * n_layer * n_embd^2   (attention + MLP weights dominate)
    # plus vocab_size * n_embd for the embedding table.
    vocab_size: int = 0       # set by the tokenizer during data prep
    block_size: int = 256     # context length: how many tokens the model can
                              # "see" at once. Attention cost grows with the
                              # SQUARE of this, so it's expensive to raise.
    n_layer: int = 8          # number of stacked transformer blocks
    n_head: int = 8           # attention heads per block (must divide n_embd)
    n_embd: int = 256         # width of the residual stream: every token is
                              # represented by a vector of this many numbers
    dropout: float = 0.1      # regularization: randomly zero activations
                              # during training so the model can't over-rely
                              # on any single pathway. 0.0 = disabled.

    # ------------------------------------------------------------- training
    batch_size: int = 32      # sequences per optimizer step
    max_steps: int = 5000     # total optimizer steps to train for
    learning_rate: float = 3e-4   # peak LR after warmup
    min_lr: float = 3e-5          # LR floor at the end of cosine decay
    warmup_steps: int = 200       # linear LR ramp-up from 0 (stabilizes the
                                  # chaotic first steps of training)
    weight_decay: float = 0.1     # AdamW L2-style penalty on weights
    grad_clip: float = 1.0        # cap on gradient norm (prevents rare huge
                                  # gradients from wrecking the weights)

    # ------------------------------------------------------------- logging
    eval_interval: int = 250  # measure val loss every N steps
    eval_iters: int = 50      # batches to average for the val loss estimate
    sample_interval: int = 1000   # generate a text sample every N steps so
                                  # you can literally watch it learn
    checkpoint_interval: int = 1000  # save model to disk every N steps

    # ---------------------------------------------------------------- paths
    dataset: str = "shakespeare"  # subdirectory of data/ holding the bins
    run_name: str = "tlm"         # checkpoints go to checkpoints/<run_name>/

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Presets. Start from the defaults above and override what differs.
# ---------------------------------------------------------------------------

PRESETS = {
    # Sanity check only: verifies data -> model -> loss -> checkpoint works.
    "smoke": TLMConfig(
        block_size=64, n_layer=2, n_head=2, n_embd=64,
        batch_size=8, max_steps=50, warmup_steps=10,
        eval_interval=25, eval_iters=5, sample_interval=50,
        checkpoint_interval=50, dropout=0.0,
        dataset="shakespeare", run_name="smoke",
    ),

    # Stage 1: char-level Shakespeare. ~1.6M parameters.
    # The dataset is tiny (~1MB) so we keep the model small and the dropout
    # meaningful: with this little data, a big model would just memorize.
    "shakespeare": TLMConfig(
        block_size=256, n_layer=6, n_head=6, n_embd=192,
        batch_size=32, max_steps=5000, warmup_steps=200,
        eval_interval=250, eval_iters=50, sample_interval=500,
        checkpoint_interval=500, dropout=0.2,
        dataset="shakespeare", run_name="shakespeare",
    ),

    # Stage 2: BPE TinyStories. ~9M parameters, the weekend run.
    # More data (~500M tokens) means less dropout needed and many more steps.
    # At batch 32 x block 256 = 8,192 tokens per step, 60k steps sees about
    # 500M tokens: roughly one pass over the dataset.
    "stories": TLMConfig(
        block_size=256, n_layer=8, n_head=8, n_embd=256,
        batch_size=32, max_steps=60000, warmup_steps=1000,
        eval_interval=500, eval_iters=50, sample_interval=1000,
        checkpoint_interval=1000, dropout=0.05,
        dataset="stories", run_name="stories",
    ),
}
