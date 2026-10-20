"""
Configuration for the Tiny Language Model (TLM).

Everything that defines a training run lives in one dataclass. Rather than
YAML files, we use plain Python: it's one fewer dependency, and you can see
the defaults and their documentation in a single place.

Presets are defined at the bottom:

  smoke       - absurdly tiny, runs in seconds; only used to verify the
                pipeline works end to end.
  shakespeare - Stage 1: character-level model, ~1.6M params, trains to
                something entertaining in about an hour on CPU.
  stories     - Stage 2: BPE-tokenized model for TinyStories, ~9M params,
                the long-weekend run.
  instruct*   - Stage 3: instruction following on TinyStories-Instruct
                (a fine-tune warm-up, a pilot, and the ~27M main run).
  mixtape*    - Stage 4: ~126M params on five tagged genres of real
                English, trained on the GPU (plus a pilot and a smoke test).
"""

import os
from dataclasses import dataclass, asdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Checkpoints default to a folder OUTSIDE the repo — and, importantly,
# outside OneDrive: checkpoint files are large (hundreds of MB at Stage 3
# size) and rewritten every 1000 steps, which would make OneDrive
# re-upload them continuously for the whole run. Override with the
# TLM_CKPT_DIR environment variable to put them somewhere else.
CKPT_ROOT = os.environ.get(
    "TLM_CKPT_DIR",
    os.path.join(os.path.expanduser("~"), "tlm-checkpoints"))


# Datasets live outside the repo for the same reason: the Stage 4 mixtape
# needs tens of GB of downloads and a ~5GB token file. Override with the
# TLM_DATA_DIR environment variable.
DATA_ROOT = os.environ.get(
    "TLM_DATA_DIR",
    os.path.join(os.path.expanduser("~"), "tlm-data"))


def data_dir(dataset):
    """Resolve the folder holding a dataset (train.bin, val.bin,
    tokenizer.json).

    Datasets prepared before this setting existed live in the repo's
    data/ folder and are found there when they don't exist under
    DATA_ROOT, so Stages 1-3 keep working unmoved. New datasets go to
    DATA_ROOT.
    """
    new = os.path.join(DATA_ROOT, dataset)
    old = os.path.join(REPO_ROOT, "data", dataset)
    if not os.path.isdir(new) and os.path.isdir(old):
        return old
    return new


def checkpoint_dir(run_name):
    """Resolve the checkpoint folder for a run.

    Runs created before checkpoints moved out of the repo are found in
    the old location (checkpoints/ inside the repo) when they don't
    exist under CKPT_ROOT, so old runs keep working unmoved.
    """
    new = os.path.join(CKPT_ROOT, run_name)
    old = os.path.join(REPO_ROOT, "checkpoints", run_name)
    if not os.path.isdir(new) and os.path.isdir(old):
        return old
    return new


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
    fused_attention: bool = True  # use PyTorch's one-kernel attention
                              # (same math, far less memory on a GPU).
                              # False = the explicit step-by-step version in
                              # model.py, slower but identical in result.
    qk_norm: bool = True      # QK-NORM: normalize each head's query and key
                              # vectors before their dot product, so attention
                              # scores can't grow without bound. Stage 4's
                              # first run died of exactly that (see
                              # CausalSelfAttention in model.py). On for new
                              # models; the Stage 1-3 presets and every
                              # checkpoint saved before it existed are off.
    qk_norm_layers: tuple = ()    # ... only in these layers (empty = all).
                                  # For grafting QK-norm onto a model trained
                                  # without it: normalizing a trained layer
                                  # changes what it does, so touch only the
                                  # layers that need it.

    # ------------------------------------------------------------- training
    batch_size: int = 32      # sequences per forward/backward pass
    grad_accum: int = 1       # GRADIENT ACCUMULATION: forward/backward
                              # this many batches, summing their gradients,
                              # before each optimizer step. The step then
                              # sees batch_size * grad_accum sequences - a
                              # big batch that never has to fit in GPU
                              # memory all at once.
    max_steps: int = 5000     # total optimizer steps to train for
    learning_rate: float = 3e-4   # peak LR after warmup
    min_lr: float = 3e-5          # LR floor at the end of cosine decay
    warmup_steps: int = 200       # linear LR ramp-up from 0 (stabilizes the
                                  # chaotic first steps of training)
    weight_decay: float = 0.1     # AdamW L2-style penalty on weights
    grad_clip: float = 1.0        # cap on gradient norm (prevents rare huge
                                  # gradients from wrecking the weights)
    # EXTENDING a finished run ("rewarm and re-anneal"): from restart_step
    # on, the LR climbs from min_lr back up to restart_lr over
    # restart_warmup steps, then glides down to min_lr again at max_steps.
    # The first schedule keeps ending at restart_step. 0 = no restart.
    restart_step: int = 0
    restart_lr: float = 0.0
    restart_warmup: int = 1000

    # ------------------------------------------------------------- logging
    eval_interval: int = 250  # measure val loss every N steps
    eval_iters: int = 50      # batches to average for the val loss estimate
    fixed_eval: bool = True   # evaluate on the SAME batches every time, so
                              # the log shows real trends instead of the
                              # luck of each draw (see train.py)
    sample_interval: int = 1000   # generate a text sample every N steps so
                                  # you can literally watch it learn
    sample_prompts: tuple = ("\n",)  # what each sample starts from; with
                                  # several, successive samples take turns
    checkpoint_interval: int = 1000  # save model to disk every N steps
    snapshot_interval: int = 0    # also keep a full copy every N steps in
                                  # snapshots/ (0 = off), so a long run always
                                  # has something to roll back to
    snapshots_kept: int = 3       # ... keeping only the newest few

    # ---------------------------------------------------------------- paths
    dataset: str = "shakespeare"  # dataset folder (see data_dir() above)
    run_name: str = "tlm"         # checkpoints go to checkpoints/<run_name>/

    def to_dict(self):
        return asdict(self)


def config_from_checkpoint(saved):
    """The TLMConfig a checkpoint was trained with, from the dict saved in it.

    A field added since the checkpoint was saved takes its default - except
    qk_norm: checkpoints that don't mention it predate QK-norm and were
    trained without it, so it has to stay off for them to load. A field
    this code doesn't know (a checkpoint newer than the code reading it)
    is ignored: a program started before a setting was added must still
    load checkpoints that mention it.
    """
    known = TLMConfig.__dataclass_fields__
    return TLMConfig(**{"qk_norm": False,
                        **{k: v for k, v in saved.items() if k in known}})


# Training samples for the mixtape presets take turns through these, so the
# log shows every genre (and a few deliberate genre collisions) evolving.
MIXTAPE_PROMPTS = (
    "<|wiki|> Title: The Great Spoon Shortage of 1874\n",
    "<|book|> Title: The Adventures of a Reluctant Toaster\n",
    "<|review|> Category: Pet Supplies\nProduct: Self-Heating Socks for "
    "Cats\nRating: 1/5\n",
    "<|recipe|> Title: Grandma's Emergency Lasagna\n",
    "<|story|> Prompt: Your cat has been secretly running a small "
    "country.\n",
    "<|recipe|> Title: The French Revolution\n",
    "<|review|> Category: Toys and Games\nProduct: The Moon\nRating: 2/5\n",
    "<|wiki|> Title: Grandma's Emergency Lasagna\n",
)


# ---------------------------------------------------------------------------
# Presets. Start from the defaults above and override what differs.
# ---------------------------------------------------------------------------

PRESETS = {
    # The Stage 1-3 presets predate QK-norm: qk_norm=False keeps them exactly
    # as trained and documented (and their checkpoints loadable).

    # Sanity check only: verifies data -> model -> loss -> checkpoint works.
    "smoke": TLMConfig(
        qk_norm=False,
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
        qk_norm=False,
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
        qk_norm=False,
        block_size=256, n_layer=8, n_head=8, n_embd=256,
        batch_size=32, max_steps=60000, warmup_steps=1000,
        eval_interval=500, eval_iters=50, sample_interval=1000,
        checkpoint_interval=1000, dropout=0.05,
        dataset="stories", run_name="stories",
    ),

    # ------------------------------------------------------ Stage 3 presets

    # Warm-up: FINE-TUNE the finished 7.4M stories model on the Instruct
    # data (train with:  --config instruct-ft --init-from stories).
    # Architecture must EXACTLY match the stories preset - we're loading
    # its weights - and the data must use the stories tokenizer (a model's
    # embedding table is welded to the tokenizer it was trained with).
    # Lower peak LR (10x down) and short schedule: fine-tuning nudges an
    # already-good model toward a new format; big steps would wreck what
    # it knows ("catastrophic forgetting").
    "instruct-ft": TLMConfig(
        qk_norm=False,
        block_size=256, n_layer=8, n_head=8, n_embd=256,
        batch_size=32, max_steps=4000, warmup_steps=100,
        learning_rate=3e-5, min_lr=3e-6,
        eval_interval=250, eval_iters=50, sample_interval=500,
        checkpoint_interval=500, dropout=0.05,
        dataset="instruct-ft", run_name="instruct-ft",
    ),

    # Pilot for the big run: same architecture, ~1k steps (~2-3h expected),
    # to measure the true seconds/step before committing a week. If it
    # comes in slower than ~15 s/step, shrink n_embd to 448 in `instruct`.
    "instruct-pilot": TLMConfig(
        qk_norm=False,
        block_size=512, n_layer=8, n_head=8, n_embd=512,
        batch_size=16, max_steps=1000, warmup_steps=100,
        eval_interval=100, eval_iters=20, sample_interval=500,
        checkpoint_interval=500, dropout=0.05,
        dataset="instruct", run_name="instruct-pilot",
    ),

    # Stage 3: ~27M params on TinyStories-Instruct, the week-long run.
    # block_size doubles to 512 so a whole example (Summary/Words/Features
    # preamble + story, ~300-400 tokens) fits in one window - the model
    # can't learn to FOLLOW the preamble if it rarely sees preamble and
    # story together. batch_size halves to keep tokens/step at 8,192.
    # 65k steps x 8,192 = ~533M tokens ~= 20 tokens/parameter, the
    # Chinchilla-optimal budget for this size; the pilot measured
    # 1,220 tok/s on the dev machine, making this ~5 days wall-clock.
    "instruct": TLMConfig(
        qk_norm=False,
        block_size=512, n_layer=8, n_head=8, n_embd=512,
        batch_size=16, max_steps=65000, warmup_steps=1000,
        eval_interval=500, eval_iters=50, sample_interval=1000,
        checkpoint_interval=1000, dropout=0.05,
        dataset="instruct", run_name="instruct",
    ),

    # ------------------------------------------------------ Stage 4 presets
    # TLM-Mixtape: five genres of real English, each document tagged with
    # a genre token (see data/prepare_mixtape.py and PLAN-stage4.md).

    # ~126M params: GPT-2-small's width (768) with 16 layers instead of
    # 12. 76k steps x 32,768 tokens = ~2.5B tokens = 20 tokens/param.
    # Each optimizer step is 16 micro-batches of 2 x 1,024 tokens
    # (grad_accum): the full 32k-token batch would never fit in the RTX
    # 3050's 6 GB at once. No dropout: every source is seen about once,
    # so there's nothing to memorize.
    # QK-norm (layer 1 only) and snapshots were added at step 51,000
    # (2026-10-18), after the first run's layer 1 attention scores ran away
    # and training stalled and spiked; the run continued from that step's
    # checkpoint with QK-norm grafted onto layer 1 (tools/add_qk_norm.py).
    # A run started from scratch would use QK-norm in every layer.
    # max_steps went from 76,000 to 79,000 at step 68,000 (2026-10-19), so
    # the run ends when Larry gets home instead of idling for 3 hours. The
    # cosine just stretches: the LR at step 68,001 is 8.7e-5 instead of
    # 7.5e-5, and it still glides to min_lr at the (new) last step.
    # EXTENDED after it finished (2026-10-19): 38,000 more steps (+50%,
    # a second pass over the data) to 117,000, rewarming to 2e-4 over
    # 1,000 steps and re-annealing to min_lr. The step-79,000 model is
    # kept as the baseline the extension has to beat.
    "mixtape": TLMConfig(
        block_size=1024, n_layer=16, n_head=12, n_embd=768, dropout=0.0,
        qk_norm=True, qk_norm_layers=(1,),
        batch_size=2, grad_accum=16, max_steps=117000, warmup_steps=2000,
        restart_step=79000, restart_lr=2e-4, restart_warmup=1000,
        fixed_eval=False,   # it ran with random eval batches throughout
        learning_rate=6e-4, min_lr=6e-5,
        eval_interval=500, eval_iters=20, sample_interval=1000,
        checkpoint_interval=1000, snapshot_interval=5000,
        sample_prompts=MIXTAPE_PROMPTS,
        dataset="mixtape", run_name="mixtape",
    ),

    # Pilot: the same model for 500 steps, to measure tok/s and peak GPU
    # memory before committing ~4 days (the gate is >= 5,000 tok/s).
    "mixtape-pilot": TLMConfig(
        block_size=1024, n_layer=16, n_head=12, n_embd=768, dropout=0.0,
        batch_size=2, grad_accum=16, max_steps=500, warmup_steps=100,
        learning_rate=6e-4, min_lr=6e-5,
        eval_interval=100, eval_iters=10, sample_interval=250,
        checkpoint_interval=250,
        sample_prompts=MIXTAPE_PROMPTS,
        dataset="mixtape", run_name="mixtape-pilot",
    ),

    # Pipeline check on the ~5M-token test dataset
    # (python data\prepare_mixtape.py --small): a toy model, a minute.
    "mixtape-smoke": TLMConfig(
        block_size=256, n_layer=2, n_head=2, n_embd=64, dropout=0.0,
        batch_size=4, grad_accum=2, max_steps=60, warmup_steps=10,
        eval_interval=20, eval_iters=5, sample_interval=30,
        checkpoint_interval=30,
        sample_prompts=MIXTAPE_PROMPTS,
        dataset="mixtape-small", run_name="mixtape-smoke",
    ),
}
