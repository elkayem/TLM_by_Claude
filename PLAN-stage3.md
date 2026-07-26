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

1. ~~Prep script for TinyStories-Instruct~~ **DONE** —
   data/prepare_instruct.py (note: dataset lives in its own HF repo,
   roneneldan/TinyStoriesInstruct, 2.66GB). Writes data/instruct/ (new
   4096 BPE) and data/instruct-ft/ (stories tokenizer, 400MB slice).
2. ~~Special-token support + stop-at-EOS~~ **DONE** — atomic
   <|endoftext|>, eot_id, generate(stop_token=...), --ignore-eos flag.
3. **Warm-up experiment (NEXT):** fine-tune the 7.4M stories model:
   `python -m tlm.train --config instruct-ft --init-from stories`
   (--init-from implemented and smoke-verified end to end).
4. ~~Pilot~~ **DONE (2026-07-26):** 1,220 tok/s sustained = 6.7 s/step —
   passes the 15 s/step gate with 2.2x margin; no size trim needed.
   Consequence: max_steps raised 55k -> 65k (~533M tokens ~= 20
   tok/param, Chinchilla-optimal; ~5.1 days). Fine-tune warm-up also
   done: best val 1.5048 in 3.75h, model follows its own preambles.
5. Launch: `python -m tlm.train --config instruct`; journal as before.
6. ~~Docs~~ **DONE** — docs/08-instruct.md (instruction following as
   data formatting, special tokens, fine-tuning, Chinchilla sizing).

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
