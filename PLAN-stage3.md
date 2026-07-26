# Stage 3 plan — TLM-Instruct (agreed 2026-07-25)

Goal: train a ~25M-param model on **TinyStories-Instruct** (same HF repo:
`roneneldan/TinyStories`, the `TinyStories-Instruct-*.txt` files) so the
model learns *conditional* generation — given `Summary:`/`Words:`/
`Features:` fields, write a story that complies. The core lesson:
instruction-following is the same next-token loss on formatted text, not a
different mechanism. Reproduces the instruct experiment from the
TinyStories paper (Eldan & Li 2023).

## Agreed design

- **Model (`instruct` preset):** n_embd=512, n_layer=8, n_head=8 (head
  size 64), ~25M params in blocks + ~2M tied embedding.
- **block_size=512** (instruct examples are ~300-400 tokens: preamble +
  story must fit together or conditioning can't be learned). Consider
  batch_size 16 to compensate for the doubled tokens/step.
- **Retrain BPE on the Instruct corpus** (field labels like `Summary:`
  should tokenize cleanly). Add `<|endoftext|>` as a TRUE special token
  (atomic id, never split by BPE) so generation can stop at story end —
  small tokenizer + generate() change, worth a mini-doc.
- **Budget:** ~1 week wall-clock on the CPU. Measured 2,410 tok/s at
  7.37M params -> expect ~700-800 tok/s at 25M -> ~430-480M tokens in
  168h ~= 1 epoch ~= Chinchilla-optimal (20 tok/param). Well-posed run.

## Order of work

1. Prep script for TinyStories-Instruct (download, tokenizer retrain,
   encode) — model the existing data/prepare_stories.py.
2. Special-token support in tokenizer + stop-at-EOS in generation.
3. **Warm-up experiment: fine-tune the existing 7.4M stories model on
   Instruct for a few hours.** De-risks the data pipeline, teaches
   fine-tuning (new docs chapter), gives a baseline the 25M must beat.
4. **Pilot the 25M config ~1k steps (~2-3h)** to measure real s/step
   before committing the week. If slower than ~15 s/step at block 512,
   trim to ~18-20M (n_embd 448). Finishing the schedule beats size.
5. Launch the week-long run; journal as before.
6. New docs chapter(s): conditional generation / instruction following;
   fine-tuning.

## Logistics

- **DONE (2026-07-25): checkpoints relocated outside OneDrive.**
  Default is now `C:\Users\lkmcg\tlm-checkpoints` (override via
  TLM_CKPT_DIR env var; see checkpoint_dir() in tlm/config.py). Both
  finished runs live there now; the in-repo checkpoints/ folder is gone.
- Windows Update / sleep settings: same drill as Stage 2; --resume works
  and the elapsed clock carries across restarts.
- **The repo is published at https://github.com/elkayem/TLM_by_Claude**
  (remote `origin`, branch `master`). Push commits as work lands.

## State as of 2026-07-25

- Stage 1 (shakespeare): done, best val 1.6094, 2.7M params, ~2h.
- Stage 2 (stories): **COMPLETE** — all 60k steps in 55.7h, best val
  1.5949 @ step 57k. train~=val throughout (no overfitting; the model is
  capacity-limited, which is why Stage 3 scales up rather than trains
  longer). Produces coherent multi-sentence stories.
- Weights-only archives (~30MB/11MB) in tlm/weights/, gitignored but
  synced by OneDrive. Full checkpoints in ~/tlm-checkpoints/.
- Venv at ../.venv (torch 2.13 CPU, Python 3.14).
