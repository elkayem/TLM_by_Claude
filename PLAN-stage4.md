# Stage 4 plan — TLM-Mixtape (draft, 2026-10-15)

Goal: train a ~126M-param model on the new GPU, on a **blend of five
genres of real English**, each example tagged with a genre token
(`<|wiki|>`, `<|book|>`, `<|review|>`, `<|recipe|>`, `<|story|>`). The tag
tells the model which voice to write in. That's the "control code" idea
from Salesforce's CTRL (Keskar et al. 2019,
[arXiv 1909.05858](https://arxiv.org/abs/1909.05858)), shrunk to fit one
small GPU.

The point is fun, as with Stage 2. A model this size can't write real
jokes, but it can nail a familiar format while getting the specifics
confidently wrong. The tags let you set up the premise:

- straight: `<|wiki|> Title: The Great Spoon Shortage of 1874`
- collisions (where the laughs should be): `<|recipe|> Title: The French
  Revolution`, `<|review|> Product: The Moon`, `<|wiki|> Title: Grandma's
  Emergency Lasagna`

Whether the model blends genres cleverly or ignores a mismatched title is
unknown in advance — that's the experiment. Journal it either way.

The core lessons: running on a GPU (devices, mixed precision, fused
attention, gradient accumulation), data mixtures (how real LLMs weight
their sources), and control tokens.

The finished model gets a **desktop app** (see "The Mixtape app"
below): double-click an icon, pick a genre from a pulldown, choose one of
a few rotating humorous ideas or type your own, and watch it write.

## Hardware (checked 2026-10-15)

- NVIDIA RTX 3050 6 GB (70 W): ~13.5 TFLOPS bf16 peak (datasheet math),
  ~5.3 GB of memory free with the Windows desktop running (it also drives
  the display).
- Intel Core Ultra 7 265 (20 cores), 16 GB RAM, ~185 GB free on C:.
- The venv now has **torch 2.13.0+cu130** (the CUDA 13.0 build; it was
  2.13.0+cpu).
- **Measured (step 1):** the Stage 3 model trains at **32,700 tok/s**
  sustained, evals included, against the CPU's ~1,220–1,285. That's
  **~27×**, better than the 10–20× estimated. At ~189M FLOPs/token it's
  ~6.2 TFLOPS sustained, ~46% of peak. Stage 3's 118 h would now take
  ~4.5 h.

## Proposed design

**Model (`mixtape` preset) — decided 2026-10-15: the deeper option.**
n_layer=16, n_head=12 (head size 64), n_embd=768, block_size=1024,
vocab 16,384 → ~113M params in blocks + ~13M tied embedding (+ ~0.8M
positions, not counted) ≈ **126M**. It's GPT-2-small's width with four
more layers and a smaller vocab. dropout=0.0: every source is seen about
once, so there is nothing to memorize.

**Budget:** 20 tokens/param ≈ **2.5B tokens** (Chinchilla). At ~907M
FLOPs/token (6 × 126M + 151M for attention), that's ~2.3×10¹⁸ FLOPs.
If step 1's measured ~6.2 TFLOPS held at this size, that would be
~6,800 tok/s and ~4.2 days. **The pilot measured ~9,300 tok/s, so ~3.1
days** (step 5).

Why deeper rather than the original 12-layer, 98M plan: the GPU turned
out ~1.8× faster than estimated, so the 98M model on 2B tokens would
have finished in ~2.6 days. The extra layers use the remaining days, and
the 98M shape stays as the fallback.

**Batching:** micro-batch 2 × 1024 tokens, gradient accumulation × 16 =
32,768 tokens per optimizer step → **max_steps 76,000** (≈ 2.49B
tokens). Peak LR 6e-4 → 6e-5 cosine, 2,000 warmup steps, other
optimizer settings as before. Micro-batch 2 is what step 1's memory
measurement predicts will fit (see Risks). If the pilot shows room for 3
or 4, raise it and lower the accumulation count, keeping ~32k
tokens/step.

**Gate (same idea as Stage 3's):** the pilot must sustain **≥ 5,000
tok/s** (≤ ~5.8 days). Step 1's measurement predicts ~6,800, but the
small micro-batch may cost some efficiency, and the pilot will tell. If
it falls short, fall back to the 12-layer, 98M shape on 2B tokens (same
mixture shares; max_steps 61,000).

**Tokenizer:** new from-scratch BPE, vocab 16,384, trained on a balanced
sample of all five genres (~20 MB each; Stage 3 used 10 MB for 4,096 —
time a 20 MB run first). Special tokens: `<|endoftext|>` plus the five
genre tags. `BPETokenizer.train(..., special_tokens=[...])` already
takes a list, so no tokenizer changes are needed.

## The mixture (~2.5B tokens)

| Tag | Source | Share | Tokens | How much of the source |
|---|---|---|---|---|
| wiki | English Wikipedia | 30% | 750M | a random slice of articles |
| book | PG-19 (Gutenberg books published before 1919) | 25% | 625M | ~1/4 of the books |
| review | Amazon Reviews 2023, 3–4 categories | 20% | 500M | a few categories |
| recipe | RecipeNLG | 15% | 375M | ~1.25 passes (estimate) |
| story | WritingPrompts | 10% | 250M | about one pass |

**Why this split:** Wikipedia and books are the backbone (55%). The
collisions only work if the model knows about things — a recipe for "The
French Revolution" is only funny if it knows guillotines and baguettes.
The three funny formats get far more than their natural share of text.
The two small sources get one pass or a little more (repeating data up
to ~4 times is nearly as good as fresh data), so the model learns their
*format* without memorizing particular recipes or stories. Shares are a
starting proposal; adjust them after looking at the prep's token counts.

**How the mix is built:** prep writes ONE pre-mixed `train.bin`. It picks
documents from each source until that source hits its token target,
shuffles at the document level, and concatenates. `get_batch` stays as
it is (random windows ≈ one pass over the bin). At 16-bit ids that's
~5 GB on disk. Validation: ~1–2M held-out tokens per genre, written to
`val_<genre>.bin` plus a combined `val.bin`, so the log shows which
genres the model is learning fastest.

**Long documents get re-tagged.** Training reads random 1,024-token
windows, so a window from the middle of a book would usually have no tag
in view, and the model would never learn what `<|book|>` means. Books and
long articles are cut at paragraph boundaries into chunks of ≤ ~900
tokens, and every chunk starts with its tag and title.

## Formats and sources

```
<|wiki|> Title: <article title>
<article text, or one chunk of it>
<|endoftext|>
<|book|> Title: <book title>
<a ≤ ~900-token passage>
<|endoftext|>
<|review|> Category: Pet Supplies
Product: <product name, first ~15 words>
Rating: 2/5
<review headline>
<review text>
<|endoftext|>
<|recipe|> Title: <dish name>
Ingredients:
<one per line>
Directions:
<numbered steps>
<|endoftext|>
<|story|> Prompt: <premise>
<story, truncated so the whole example fits in one window>
<|endoftext|>
```

- **wiki** — HF `wikimedia/wikipedia`, config `20231101.en` (6.41M
  articles; CC BY-SA 3.0 / GFDL). Download only enough parquet shards for
  750M tokens after sampling. Probably skip "List of ..." pages (mostly
  tables) and very short stubs.
- **book** — PG-19 (Rae et al. 2019; Apache 2.0, books public domain):
  HF `deepmind/pg19` or the `deepmind-gutenberg` GCS bucket. 28,602 train
  books, ~2B words, ~11 GB. Gutenberg boilerplate is already stripped,
  and offensive slurs were replaced with placeholders. `metadata.csv` has
  `short_book_title` (no author field).
- **review** — HF `McAuley-Lab/Amazon-Reviews-2023`: per-category
  `raw/review_categories/<Category>.jsonl` plus
  `raw/meta_categories/meta_<Category>.jsonl` (uncompressed), joined on
  `parent_asin` to get the product name. Chosen: Pet Supplies, Toys and
  Games, Grocery and Gourmet Food (8.4 + 7.3 + 6.0 GB of reviews).
  Filters:
  - drop reviews under ~20 words
  - **rebalance star ratings** — real reviews skew heavily to 5 stars;
    roughly equal shares make the `Rating:` field mean something, and the
    1–2-star reviews are the funny ones
  - truncate product names to ~15 words (Amazon titles are keyword soup)
  - keep only category, product, rating, headline and text; user ids are
    dropped

  The dataset card states no license, so this data and anything trained
  on it stays local.
- **recipe** — RecipeNLG (Bień et al. 2020): 2,231,142 recipes. **Manual
  download** from recipenlg.cs.put.poznan.pl (accept their terms; HF
  `mbien/recipe_nlg` lists the license as unknown and also needs the
  manual file). Keep `source == Gathered` (~1.6M cleaner recipes).
- **story** — HF `euclaise/writingprompts` (tagged MIT; Reddit
  r/WritingPrompts via Fan et al. 2018; 303k stories, ~200M words). Strip
  the `[WP]` style tags, **drop NSFW-tagged prompts**, convert any
  `<newline>` markers back to real newlines, and truncate stories so
  prompt + story fit in one window (fairseq likewise kept only the first
  1,000 words).

## Order of work

1. ~~GPU port + calibration (Stage 4a)~~ **DONE (2026-10-15).**
   - torch 2.13.0+cu130 is installed.
   - `train.py`:
     - `--device auto|cuda|cpu` (auto = GPU if visible)
     - bf16 autocast (GPU only; the CPU path is unchanged)
     - pinned-memory batch copies
     - fused AdamW
     - `grad_accum` config field (default 1)
     - peak GPU memory in the eval line
     - `map_location="cpu"` loads, so checkpoints move between devices
   - `model.py`: `F.scaled_dot_product_attention` by default, with the
     explicit version kept as `explicit_attention()` behind
     `fused_attention`.
   - `generate.py`: `--device`.
   - Checks, on the Stage 3 best.pt with one fixed val batch:
     - every CPU/GPU × explicit/fused combination gives loss 1.21786
       exactly; bf16 gives 1.21799
     - grad accumulation (2×8) matches one batch of 16 to 1.5e-5
     - fused attention: 2× faster and half the memory of explicit
       (41k vs 22k tok/s; 2.4 vs 4.3 GB) in a micro-benchmark
     - a CPU checkpoint resumes on the GPU with fused AdamW
   - Calibration (`instruct-pilot`, 1,000 steps, checkpoints under
     `~/tlm-checkpoints/gpu-calibration/`):

     | step | CPU train / val | GPU train / val |
     |---|---|---|
     | 100 | 4.4319 / 4.4325 | 4.4391 / 4.4466 |
     | 200 | 3.9372 / 3.9255 | 3.9395 / 3.9350 |
     | 300 | 3.6468 / 3.6341 | 3.6442 / 3.6415 |
     | 400 | 3.4285 / 3.4506 | 3.4237 / 3.3787 |

     The curves track within eval noise (20-batch estimates). All 1,000
     steps took **4 min 27 s at 32,700 tok/s**, peaking at 2.13 GB of
     GPU memory. The CPU took 6.7 s/step, ~1 h 52 min for the same
     1,000 steps.
   - Moved to step 3, because they need the mixtape data: per-genre val
     splits, and samples drawn from the prompt bank.
2. ~~Data location~~ **DONE (2026-10-15).**
   - `tlm/config.py` has `DATA_ROOT` (`TLM_DATA_DIR`, default
     `~/tlm-data/`) and `data_dir(dataset)`. Like `checkpoint_dir()`, it
     falls back to the repo's `data/<dataset>/` when a dataset only
     exists there, so Stages 1–3 work unmoved.
   - `train.py`, `generate.py` and all three older prepare scripts use
     it, so fresh setups write everything to `~/tlm-data/`.
   - Also done: the venv moved out of OneDrive to
     `C:\Users\lkmcg\tlm-venv` (see README notes).
3. ~~`data/prepare_mixtape.py`~~ **WRITTEN AND TESTED (2026-10-15).**
   It has five resumable stages (download, format, tokenizer, encode,
   mix) plus `inspect`, and `--small` / `--genres` for test runs.
   - **Where things go:** raw files in `~/tlm-data/mixtape-raw/`
     (shared by small and full runs); everything else in
     `~/tlm-data/mixtape/`: `formatted/<genre>.txt`,
     `tokens/<genre>.bin` + `.lens.npy`, `tokenizer.json`, `train.bin`,
     `val.bin`, `val_<genre>.bin`, `stats.json`. Re-mixing with
     different shares only needs `train.bin` deleted and `mix` re-run.
   - **Changes from the original design:**
     - *Tokenizer speed:* `BPETokenizer.train()` recounts every pair on
       every merge, which would take hours on real English, so
       `tokenizer.py` gained `count_words()` and `train_fast()`.
       `train_fast()` keeps pair counts between merges and uses a heap.
       Checked on TinyStories: every merge it picks is the true most
       frequent pair; it differs from `train()` only in how ties break;
       it's 43× faster.
     - *Plain ASCII:* everything is reduced to ASCII. Curly quotes and
       dashes are mapped, accents stripped (NFKD), and documents more
       than 1–3% non-ASCII are dropped (other languages). There's also
       a cheap English check (" the / and / of / to" ≥ 3% of words).
       This keeps the base alphabet at ~95 characters instead of
       spending vocabulary on every Unicode symbol.
     - *Chunking:* long texts are cut at paragraph breaks into ≤ 3,500
       characters (~850 tokens), each piece repeating the tag + title (or
       prompt). This now applies to stories too: long WritingPrompts
       stories are chunked like books instead of truncated, which keeps
       their endings.
     - *Wikipedia:* 10 of the 41 shards, picked at random with a fixed
       seed. Skips list/index/outline/disambiguation pages, cuts
       articles at the first reference-type heading (See also,
       References, ...), and tidies what stripped non-Latin script
       leaves behind (empty parentheses).
     - *Books:* hard-wrapped lines are re-joined, `_italics_`
       underscores are dropped, and the first and last 2% of each book
       (front matter, indexes, publishers' ads) are skipped. Titles come
       from PG-19's metadata and include the author ("... by Mark
       Twain"), so the author can be named in a prompt.
     - *Reviews:* Pet Supplies, Toys and Games, and Grocery and Gourmet
       Food (~21.6 GB of reviews + ~5.6 GB of product data).
       - keep-rates by rating 1/1/1/0.6/0.2, which brings 5-star down
         from ~65% to ~30%
       - reviews need ≥ 20 words
       - product names are cut to 15 words
       - each file is sampled evenly from start to end, because the
         files are grouped by user
     - *WritingPrompts:* all splits are used (we make our own
       validation split). The `[ WP ]`-style tags are stripped, prompts
       are detokenized, and NSFW-tagged prompts are dropped.
   - **Test run (`--small`, four genres):** ~4.25M training tokens, at
     3.9–4.1 characters/token. The tags each encode to one id
     (16378–16383), and `mixtape-smoke` trains on it: per-genre val
     losses print and land in `log.csv`, and samples rotate through the
     prompt bank.
   - **Full run, download + format done 2026-10-16** (log:
     `~/tlm-data/mixtape-prep.log`). ~33 GB downloaded in ~30 min
     (~20 MB/s); four genres formatted in ~17 min into 9.6 GB:

     | genre | docs | formatted | note |
     |---|---|---|---|
     | wiki | 1,542,142 | 3,469 MB | 8 shards came up ~10% short; raised to 10 |
     | book | 1,048,278 | 3,289 MB | 9,280 books downloaded |
     | review | 5,270,548 | 2,525 MB | kept 37 / 49 / 56% of Pet / Toys / Grocery lines |
     | story | 412,625 | 955 MB | ~1 pass of the 250M share |

     **Finished 2026-10-16.** The official RecipeNLG site returned
     "Internal Server Error", so we used the exact copy on Hugging Face
     (`SandhyaKilari/RecipeNLG_dataset`, 2.29 GB CSV, same 2,231,142
     rows and columns). It needed no code changes.
     - **recipe format:** 1,631,468 "Gathered" recipes, 1,225 MB, in
       1.4 min
     - **tokenizer:** 5 × 10 MB sample, 203,400 unique words, the full
       16,384 vocab in 0.1 min (`train_fast`); special ids
       16378–16383
     - **encode:** under 3 min for all five genres on 16 processes. At
       3.82–4.02 chars/token: wiki 889.6M, book 842.4M, review 628.5M,
       recipe 321.0M, story 240.2M tokens.
     - **mix:** 8,605,776 documents shuffled into `train.bin` in 3.4 min

     | genre | train tokens | share | passes | val tokens | tokens/doc |
     |---|---|---|---|---|---|
     | wiki | 749,999,879 | 30% | 0.84 | 1,500,411 | 577 |
     | book | 624,999,801 | 25% | 0.74 | 1,500,118 | 804 |
     | review | 499,999,980 | 20% | 0.80 | 1,500,016 | 119 |
     | recipe | 374,999,931 | 15% | 1.17 | 1,500,053 | 197 |
     | story | 249,999,570 | 10% | 1.05 | 1,500,159 | 582 |
     | total | 2,499,999,161 | | | 7,500,757 | |

     Disk: raw downloads 35 GB (including the 2.2 GB recipe CSV). The
     dataset folder is 22 GB: formatted text 11 GB, tokens/ 5.5 GB,
     `train.bin` 5.0 GB, val files 30 MB. ~109 GB free afterwards.
   - **New presets:** `mixtape`, `mixtape-pilot`, and `mixtape-smoke`
     (the toy model for the `--small` data). `MIXTAPE_PROMPTS` in
     `config.py` is the training prompt bank.
4. ~~Inspect before training~~ **DONE (2026-10-16).**
   `python data\prepare_mixtape.py inspect` prints 3 random documents
   per genre, the special-token ids and the mix stats.
   - Recipes read cleanly (title, `- ingredient` lines, numbered
     steps). The source's own quirks remain, e.g. a step split
     mid-sentence ("cook onions in3 / 2. tablespoons oil").
   - `train.bin`: every id is < 16,384.
   - In the first 20M tokens, `<|endoftext|>` (69,012) matches the sum
     of the five tags (69,013; one document is cut off at the end), so
     document boundaries are intact.
5. ~~Pilot~~ **DONE (2026-10-16): gate passed.** 500 steps × 32,768
   tokens in 29 min 20 s (log in JOURNAL.md).
   - **Speed:** 9,213–9,492 tok/s sustained, ~9,300 on average,
     including evals, samples and checkpoint saves. That's ~8.5 TFLOPS
     (~63% of the bf16 peak), better than step 1's 6.2, because the
     bigger matrices keep the GPU busier. The gate was ≥ 5,000.
   - **Memory:** 3.73 GB peak allocated at micro-batch 2, plus ~0.4 GB of
     CUDA context. Micro-batch 4 would need ~1.7 GB more, which is too
     tight beside the desktop's share of the 6 GB, so **keep
     micro-batch 2 × grad_accum 16**. `max_steps` stays 76,000.
   - **Full run:** 2.5B tokens / ~9,300 tok/s ≈ **74 h ≈ 3.1 days**
     (the plan said ~4.2). A checkpoint every 1,000 steps ≈ every hour,
     ~1.6 GB each (best.pt + latest.pt ≈ 3.2 GB).
   - **Losses at step 499:** recipe 3.06 < review 4.34 < story 5.03 <
     book 5.28 < wiki 5.48. Formulaic text is learned first. The first
     sample (step 250) already stopped by itself at `<|endoftext|>`.
6. **LAUNCHED 2026-10-16 07:02** (~3.1 days, so expect it to finish
   around **2026-10-19 mid-morning**). Journal it with the prompt bank.
   - **Launcher:** `tools\train_mixtape.cmd`, double-clickable. It runs
     `tools\train_mixtape.ps1`, which starts `--config mixtape`, or adds
     `--resume` when `~/tlm-checkpoints/mixtape/latest.pt` exists, and
     tees all output to `~/tlm-data/mixtape-train.log`. After a restart
     or crash, double-click it again.
   - **Started through Explorer:** the process chain is explorer → cmd →
     powershell → python, not a child of a Claude session.
   - **Pre-flight:** AC sleep and hibernate are both "never" (a desktop,
     no battery).
   - **At launch:** step 0 val 9.83 (≈ ln 16,384); GPU 100%, 4.9 GB used
     including the desktop, 69 W, 55 °C.
   - **While it trains,** the desktop app falls back to the CPU (the GPU
     has < 1.5 GB free) and Reload picks up the newest `mixtape/best.pt`.
   - **Progress at 08:30 (step 1,500):** ~9,800 tok/s, faster than the
     pilot because the full run evaluates less often. Projected finish
     **2026-10-19 ~05:45**, 2.95 days in total.
   - **GPU health:** 68 °C (NVIDIA's limits for this card: target 83,
     max operating 92, slowdown 94, shutdown 97), fan 72%, held at the
     70 W power cap, no heat or hardware slowdown flags.
   - **Monitoring:**
     - `tools\gpu_health.cmd`/`.ps1` logs every 5 min to
       `~/tlm-data/gpu-health.csv` (temperature, fan, power, clock,
       utilization, memory, slowdown flags, training step). It prints a
       red warning at ≥ 83 °C, on a slowdown flag, or after 15 idle
       minutes, and stops itself when the training log says "done.". It
       was started through Explorer, like the run.
     - A Claude scheduled task, `tlm-mixtape-checkin`, runs at
       00:17 / 06:17 / 12:17 / 18:17 on Oct 16–19. It checks progress,
       losses, GPU health and disk space, appends to
       `~/tlm-data/mixtape-checkins.md`, and push-notifies only on a
       problem or at the finish. It runs only while the Claude app is
       open; a check missed while the app is closed runs at the next
       launch. Delete it after the run (cron has no year: it would wake
       again in October 2027, though it's told to just say so).
     - The task runs ONE fixed command, `tools\mixtape_status.ps1`, a
       read-only health check. Its first line is `STATUS: OK`,
       `PROBLEM - ...` or `FINISHED - ...`, and it appends its report to
       `~/tlm-data/mixtape-checkins.md`. It replaced a first version
       whose many multi-part PowerShell commands could only be approved
       "once" per run, so every scheduled run would have stopped to ask.
     - **Unattended:** the project's `.claude/settings.local.json` (not
       committed; covered by the global git ignore) allows exactly
       `PowerShell(& "C:\...\tools\mixtape_status.ps1")`, nothing
       broader. A first version of the rule, for
       `powershell -NoProfile -ExecutionPolicy Bypass -File ...`, still
       stopped for approval: launching a nested PowerShell with a policy
       bypass seems to need a prompt regardless. The PowerShell tool
       already runs with execution policy Bypass, so the direct `&` call
       works. Tested 2026-10-16 09:02: the scheduled run finished in 17 s
       with no prompt. Remove the rule along with the task after the run.
6b. **STOPPED at step 52,000 (2026-10-18 07:27): attention logit growth.**
   - **Symptom:** the per-genre val losses fell steadily to step 49,500
     (book 3.17, recipe 1.51, review 2.53, story 3.42, wiki 2.99), then
     every genre jumped at once at 50,000, nearly recovered at 50,500,
     and spiked harder at 51,000 and 51,500 (wiki 3.46). Train loss rose
     too. No restart, reboot, error or GPU warning.
   - **Diagnosis** (CPU, on copies of the checkpoints):
     - Layer 1's attention scores: max ~36,000 at step 27,000, **9.2
       million** at 51,000 (healthy layers: under ~250). 4 of its 12
       heads had locked onto single tokens (attention entropy ~0) at
       27,000; 11 of 12 by 51,000. Its query/key weights tripled.
     - The training gradient norm grew from ~1.4 to ~5–6, mostly from
       layers 0–1, so clipping at 1.0 cut every other layer's gradient
       to ~1/6.
     - AdamW's second moments collapsed: 31M of 127M weights had
       sqrt(v) < 1e-8 (0.85M at 27,000), including 94–97% of the
       last four layers' attention weights. Those layers had nearly
       stopped learning, and their next real gradients became oversized
       steps: the spikes.
     - On the same 60 val passages, step 51,000 scored about the same
       as step 27,000 (genre average 2.86 vs 2.84): the spike erased
       the progress since 27,000.
   - **Why no rollback to 49,500:** only `latest.pt` (overwritten every
     1,000 steps) and `best.pt` (chosen by the noisy mixed val loss, so
     stuck at 27,000) existed. A copy of `latest.pt` at 51,000 was saved
     to `~/tlm-checkpoints/mixtape-rescue/` before it was overwritten.
   - **The first run** is archived as `~/tlm-checkpoints/mixtape-v1`.
6c. **TRIAL: step 51,000 + QK-norm on layer 1 (launched 2026-10-18 07:42).**
   - **QK-norm** (`qk_norm`, `qk_norm_layers` in config; off by
     default): each head's query and key are scaled to RMS 1 times a
     learned gain, so a score can't exceed ~gain_q · gain_k · sqrt(hs).
   - **Grafted with `tools/add_qk_norm.py`:** gains = each head's
     measured query/key RMS, so a typical vector is unchanged; layer 1's
     11 runaway heads capped to a bound of 30. Measured cost of the
     graft on the 60 passages: layer 1 only, 2.86 → 3.13; all 16 layers,
     → 3.63 (healthy layers rely on query/key length, so they were left
     alone). Fresh AdamW state (the old one was the damage) and
     `--rewarm 500` (LR ramps up over 500 steps).
   - **Also new in train.py:**
     - each eval logs the gradient norm (mean and max since the last
       eval) and the largest attention score on one val batch
     - `best.pt` is chosen by the average of the five genre losses
     - full snapshots every 5,000 steps (newest 3 kept) in
       `mixtape/snapshots/`
     - the log gets a header when missing, for seeded runs
   - The new `mixtape/` holds the step-27,000 model as `best.pt`
     (genre average 2.8091, the bar to beat) and the first run's log
     history to step 51,000.
   - **Pass/fail:** by ~step 54,000 (~3 h) the genre average should be
     back near **2.75** (the step-49,500 level), with attention scores
     bounded and the gradient norm back near 1–2. If not, restart from
     scratch with QK-norm in every layer (~70 h).
   - `mixtape_status.ps1` now judges by the genre average (alarm at
     0.15 above its best) and alarms on an attention score > 1,000 or a
     gradient norm > 8.
   - **PASSED, within the first eval.** Genre average 2.706 at 51,500
     (a new best; `best.pt` replaced at 08:10), then 2.747, 2.707,
     2.713 through 53,000: at or below the pre-spike level (~2.72–2.74).
     Gradient norm 1.00 in the first 500 steps (max 12 just after the
     fresh start), then a steady ~0.52. (Not comparable with the ~1.4
     measured on the CPU at 27,000: that was one 2-sequence fp32 batch,
     this is a 32-sequence accumulated bf16 step, whose averaging alone
     shrinks the norm.)
     Largest attention score 388 → 342 and falling. ~9,700 tok/s.
     Projected finish **Mon 2026-10-19 ~07:10**.
   - **Fixed-passage tracking (Larry asked to see the trend):** the
     training-log evals draw fresh random passages (±0.03 luck), more
     than the model improves in an hour now. `tools/track_fixed_eval.py`
     (double-click `tools\track_fixed_eval.cmd`; started 09:47) scores
     each new `latest.pt` on the same 60 val passages, on the CPU, into
     `mixtape/fixed-eval.csv`, and stops after the final checkpoint.
     Seeded with the day's measurements: first run 27,000 = 2.8395,
     51,000 = 2.8592; graft start 3.1337; trial 51,500 = 2.7144,
     53,000 = 2.6990 (all five genres better than at 51,500).
     `mixtape_status.ps1` shows both scores, with the fixed-passage
     trend, and alarms if that score rises 0.05 above its best, falls
     behind, or the tracker stops.
   - **Killed by a Claude app update, 2026-10-18 23:25** (step ~67,700;
     the 00:19 and 06:19 check-ins flagged it). The trial and the tracker
     had been launched from inside a Claude session (Start-Process), so
     Windows counted them as part of the Claude app and stopped them when
     the app package was serviced. The Explorer-launched GPU logger
     survived. latest.pt (step 67,000) was intact; ~40 min of training
     lost, plus 7 idle hours. The nvlddmkm event 153 logged at that
     second (and at the 07:27 hand stop) is just the driver noting a GPU
     process killed mid-work.
   - **Resumed 2026-10-19 06:35 from step 67,001**, both launched through
     WMI (`Invoke-CimMethod Win32_Process -MethodName Create`): their
     parent is WmiPrvSE.exe, outside Claude, in the desktop session.
     (`explorer.exe file.cmd` from a Claude shell is NOT enough: the
     short-lived explorer.exe it starts is itself inside Claude.)
     Projected finish ~15:00.
   - **Stretched to 79,000 steps** at Larry's request, so it finishes
     when he's home (~17:50) instead of idling from 15:00: `max_steps`
     76,000 → 79,000, applied by a restart right after the step-68,000
     checkpoint. The cosine stretches (LR at 68,001: 8.7e-5 instead of
     7.5e-5) and still ends at min_lr. ~2.59B tokens in all.
   - **FINISHED 2026-10-19 17:47:** 79,000 steps, 73.7 h of training.
     Fixed-passage score **2.5860** (book 3.227, recipe 1.361, review
     2.343, story 3.134, wiki 2.866): 0.13 better than at the fix
     (51,500), 0.25 better than the first run's step 27,000. Training-
     log best 2.5590 at step 74,500; that `best.pt` was swapped for the
     final model (the fixed passages put 74,500 at ~2.595) and kept as
     `best-step74500.pt`.
6d. **EXTENSION (Larry's call, launched 2026-10-19 18:09):** 79,000 →
   117,000 steps (+50%, a second pass over the data), finishing
   ~Wed 2026-10-21 06:00.
   - **Schedule:** new config fields `restart_step` / `restart_lr` /
     `restart_warmup`: from 79,000 the LR climbs 6e-5 → 2e-4 over 1,000
     steps, then follows a cosine back to 6e-5 at 117,000. The first
     schedule is unchanged (it now ends at `restart_step`). Old presets
     unaffected.
   - **Baseline:** `mixtape/baseline-step79000.pt` (fixed-passage
     2.5860). Expect the fixed-passage score to RISE for a few thousand
     steps while the LR is high, then fall. Keep the extension only if
     it ends at least ~0.03 below the baseline; otherwise the baseline
     goes back in as `best.pt`.
   - **Helpers:** the tracker, the GPU logger and `mixtape_status.ps1`
     now only count a "done." line after the latest launch line (the
     log already held one from step 79,000). The status script's
     throughput is the median of the last 4 evals (the first eval after
     a restart includes startup). All three were launched through WMI,
     outside Claude. The check-in task now runs through Oct 21.
   - **Ready for Wednesday morning:** `tools/after_run.py` (launched
     through WMI on 2026-10-19, window "TLM after-run watcher") waits for
     the extension's "done.", then scores the baseline and the final
     model on the fixed passages, writes
     `~/tlm-data/samples/extension-summary.md` (per-genre table and a
     keep/restore verdict: keep if ≥ 0.03 better), a blind A/B sheet
     `extension-blind.md` (key in `extension-blind.key.json`), and
     redraws the chart. Log: `~/tlm-data/after-run.log`.
   - **Crashed 2026-10-19 19:44:56 at step ~80,800: "CUDA error: out of
     memory".** Training leaves only ~1 GB of the 6 GB card free (the card
     also drives the desktop); something else took GPU memory - likely
     Edge, running in the background since 16:59 (startup boost; its GPU
     process is a GPU client, and it started new renderers at 19:45:01).
     Worse, the process then HUNG on exit (one core spinning, GPU memory
     held), so the check-in saw a live process; found at 22:40 by a manual
     check ("no new eval in 185 min", GPU idle). Killed; ~3.5 h lost.
   - **Fixes (train.py, train_mixtape.ps1):** a step that fails with GPU
     out-of-memory is retried 3 times, 60 s apart (`train_step()`,
     `OOM_RETRIES`); any other crash prints the traceback and exits hard
     with code 3 (`os._exit`), so CUDA can't hang it; the launcher resumes
     from the last checkpoint after a code-3 exit, 2 minutes later, up to
     10 times (`TLM_RESUME_WAIT` overrides the wait). Tested on the CPU:
     a crash exits 3, argument errors exit 2, a simulated OOM is retried,
     and the launcher resumes after a simulated crash.
   - **Resumed 22:44 from step 80,001** (WMI); finish now ~Wed 09:30.
   - **Tracker crashed ~23:40** on the first checkpoint from the 22:44
     launch: started at 18:09, its code predated `fixed_eval`, which the
     new checkpoints record, and `TLMConfig(**saved)` rejected the unknown
     key. `config_from_checkpoint()` now ignores unknown keys. Restarted
     2026-10-20 05:35 (and the after-run watcher, on the same code);
     85,000 backfilled from its snapshot; 81-84,000 and 86,000 are lost.
     Training itself ran clean overnight. (The WER "BlueScreen" /
     "LiveKernelEvent" reports in the Application log are retries of
     reports from January, not new events.)
   - **Extension so far (fixed passages):** 79,000 2.586 → 80,000 2.654 →
     85,000 2.672 → 87,000 2.671: still rising/flat with the LR near its
     2e-4 peak; it has to come down as the LR decays.
   - **At equal learning rates** the extension was 0.02-0.03 better than the
     first run all the way down (e.g. LR 1.2e-4: step 100,000 2.623 vs
     step 60,000 2.650), though at step 100,000 it was still above the
     annealed baseline: a high LR adds a loss penalty that only goes away
     at the floor.
   - **FINISHED 2026-10-21 ~09:28:** 117,000 steps (3.83B tokens, ~1.5
     passes), 109.4 h of training in all. Fixed passages: **2.5645**
     (book 3.197, recipe 1.353, review 2.338, story 3.104, wiki 2.831),
     0.0215 better than the step-79,000 baseline, in every genre (~0.03 in
     wiki/book/story, < 0.01 in recipe/review). Under the 0.03 bar, so
     the watcher said "let the blind test decide"; Larry chose to keep the
     final model without comparing. `best.pt` = the final model. No
     crashes after the 2026-10-19 fixes.
   - **Wrap-up 2026-10-21:** check-in task deleted and its permission rule
     removed; README, docs/10 and JOURNAL finalized; idle helper windows
     closed. Kept for Larry to decide: `mixtape-v1` (3.0 GB),
     `mixtape-rescue` (3.0 GB), `mixtape/snapshots` (4.4 GB),
     `baseline-step79000.pt` and `best-step74500.pt` (1.5 GB each).
7. ~~The Mixtape app~~ **BUILT AND TESTED (2026-10-16)**, ahead of
   schedule, while waiting on RecipeNLG. Developed against the
   `mixtape-smoke` model; it loads the real one once trained.
   - **`model.py`:**
     - `generate_stream()`: a KV cache plus a generator that yields one
       token at a time. The cache restarts from the last 3/4 of the text
       when the window fills, because learned positions can't slide.
     - `sample_next()`: the temperature/top-k/top-p code, now shared
       with `generate()`, which stays the simple documented version.
     - Checked: with the same seed, both give the same tokens, on CPU
       and GPU, with fused or explicit attention. 300 tokens on the
       Stage 3 model: 4.5 → 1.5 s on the CPU, 1.5 → 0.75 s on the GPU.
   - **`tlm/ideas.json`:** 248 ideas (48–50 per genre, ~27% mix-ups).
   - **`tlm/gui.py`:**
     - the genre pulldown, a per-genre form, and 4 clickable ideas from
       a persistent shuffled deck; More ideas, Surprise me
     - Creativity and Length (150 / 300 / 600). Creativity is now a
       pulldown, "mellow 0.6 / groovy 0.8 / far out 1.0 / cosmic 1.2",
       which shows the temperature next to each name.
     - streaming output with Stop; generation also stops at
       `<|endoftext|>` or any genre tag
     - Show what the model sees; Save to favorites; Copy
     - a status line, a model line, and Reload model
     - the GPU if ≥ 1.5 GB is free, else the CPU; Windows DPI-aware
     - it loads the first of `mixtape`, `mixtape-pilot` and
       `mixtape-smoke` that has a best.pt (`--run` / `--device`
       override)
   - **`tools/make_icon.py`:** draws the cassette into
     `assets/mixtape.ico` (+ a `.png` preview) using the standard
     library only.
   - **70s redesign (2026-10-16, after Larry asked for more pizazz):**
     - the palette: chocolate brown, rust, burnt orange, mustard,
       avocado and cream
     - fonts: Cooper Black for titles and buttons, Bookman Old Style for
       labels, Courier New for the output text (each with fallbacks)
     - sunset stripes under the header
     - the ideas on a mustard-edged "J-card"
     - a header cassette, drawn on a canvas, whose reels spin and whose
       tape winds left to right as text arrives, plus a blinking REC
       light
     - the cassette label is handwritten (Ink Free) with the genre and
       track number, shrinking to fit. Full idea titles were too long
       for the label; the title still heads the output.
     - real cassette proportions (100.4 × 63.8 mm, hubs 42% of the
       width apart, a little above the middle), after Larry spotted the
       first draft's reels sitting too low and too close together
     - the tape is no longer drawn across the window between the packs
       (Larry's catch: a real tape never runs there), nor along the
       bottom edge
     - the layout redone from Larry's photo of a real cassette, every
       position measured from the photo's pixels:
       - a label with clipped top corners and the photo's stripes
         (golden, orange, teal, golden, rust) running behind the window
       - black plastic inside the label's cutout, so the tape packs show
         only through the small clear pane in the middle (Larry's catch);
         a pack too small to reach the pane is hidden
       - hubs seen through round holes: a grey rim, a white face and six
         short teeth around its inside
       - a lower trapezoid widening down to the bottom edge (Larry caught
         it upside down), with a centre screw, two round holes and two
         square holes
       - four corner screws, and the small tab on each side of the shell
         (Larry's catch)
     - the icon redrawn from the same geometry
   - **`tools/make_shortcut.py`:** creates the shortcut through
     PowerShell's WScript.Shell COM object. Created on this PC:
     `C:\Users\lkmcg\OneDrive\Desktop\TLM Mixtape.lnk` → the
     tlm-venv's `pythonw.exe -m tlm.gui`.
   - **Tested:**
     - an automated run: idea click, two genres, More ideas, Stop,
       Surprise me, favorites, deck persistence, plus screenshots
     - a real launch from the shortcut: the window opened and nothing
       was logged as an error
     - the gibberish smoke model sometimes emitted a genre tag
       mid-text, which is why generation now stops at any tag
8. **Docs + README — WRITTEN 2026-10-16, results pending.**
   - `docs/09-gpu.md` (Stage 4a: devices, bf16 mixed precision, fused
     attention, gradient accumulation, and the calibration that proved
     the port) and `docs/10-mixtape.md` (the five genres, building a
     mixture, per-genre validation, control tokens and collisions,
     `train_fast`, the KV cache and the app). `docs/00-overview.md` points
     at both.
   - README:
     - a fourth column in the models table
     - a new **Setup** section (venv outside OneDrive, the two PyTorch
       builds, the driver check, `--device`, where data and checkpoints
       live)
     - Stage 4 quickstart lines with data-prep notes (RecipeNLG and its
       mirror, disk and time, re-mixing)
     - long-run notes (the launcher)
     - a **Mixtape app** section
     - chapters 9–10 in the tour, an updated layout
     - 9 new references (Wikipedia, PG-19, Amazon Reviews 2023,
       RecipeNLG, WritingPrompts, FlashAttention, CTRL, mixed precision,
       Muennighoff et al.). Every link was fetched and its title checked.
   - Every "Ready" item from the README notes below is now folded in.
   - **Still to fill in after the run** (placeholders say "TBD"): the
     Stage 4 training time and best val loss in the README table, and
     the "Results" section of `docs/10-mixtape.md` (final losses per
     genre, prompt-bank samples, which collisions worked).

## Prompt bank (samples during training + journal)

Rotate through these at each `sample_interval`, so the journal shows
every genre evolving. They also go into the app's idea bank (below).

```
<|wiki|> Title: The Great Spoon Shortage of 1874
<|book|> Title: The Adventures of a Reluctant Toaster
<|review|> Category: Pet Supplies\nProduct: Self-Heating Socks for Cats\nRating: 1/5
<|recipe|> Title: Grandma's Emergency Lasagna
<|story|> Prompt: Your cat has been secretly running a small country.
<|recipe|> Title: The French Revolution                    (collision)
<|review|> Category: Toys and Games\nProduct: The Moon\nRating: 2/5               (collision)
<|wiki|> Title: Grandma's Emergency Lasagna                (collision)
```

## The Mixtape app (desktop GUI)

A small window for playing with the model, with no command line.
Double-click **TLM Mixtape** on the desktop, pick a genre, pick or type an
idea, press Generate, and watch the text appear word by word.

```
+- TLM Mixtape -----------------------------------------------+
| Genre: [ <wiki>  encyclopedia article        v ]            |
|                                                             |
| Ideas:  The Great Spoon Shortage of 1874                    |
|         Competitive Napping at the 1912 Olympics            |
|         The Treaty of Mild Annoyance                        |
|         Grandma's Emergency Lasagna          (mix-up)       |
|         [More ideas]                                        |
|                                                             |
| Title:  [ The Great Spoon Shortage of 1874            ]     |
|                                                             |
| Creativity [----o-----] 0.8      Length [ medium  v ]       |
| [Generate]  [Surprise me]  [Stop]                           |
| +---------------------------------------------------------+ |
| | The Great Spoon Shortage of 1874 was a period of        | |
| | (the model's writing streams in here, word by word)     | |
| +---------------------------------------------------------+ |
| [Save to favorites]  [Copy]   [ ] Show what the model sees  |
| Model: mixtape best.pt, step ..., val ...   [Reload]  CPU   |
+-------------------------------------------------------------+
```

**Launching**
- `tools/make_shortcut.py` (run once) puts a **TLM Mixtape** shortcut on
  the desktop. On this PC the desktop is redirected to
  `C:\Users\lkmcg\OneDrive\Desktop`, so the script asks Windows for the
  real Desktop folder instead of assuming `~/Desktop`.
- The shortcut runs `pythonw.exe -m tlm.gui` from the repo folder, using
  the Python of whichever venv ran the script (`sys.executable`'s
  folder; here `C:\Users\lkmcg\tlm-venv`). `pythonw` means no console
  window.
- It has a custom icon, `assets/mixtape.ico` (a little cassette tape),
  drawn by `tools/make_icon.py`. That script uses only the standard
  library: Pillow isn't installed, and ICO files can hold raw pixels.
- If the repo ever moves, re-run the shortcut script.

**Toolkit:** Tkinter. It ships with Python (Tk 8.6.15, confirmed in the
venv), so there are no new dependencies. The genre pulldown is a
`ttk.Combobox`. Tk 8.6 can't show emoji, so the window uses plain text
and ★.

**Genre pulldown.** It offers `<wiki>`, `<book>`, `<review>`, `<recipe>`
and `<story>`, each with a plain-English label. Picking a genre changes
the input fields to match that genre's format, and deals fresh ideas:

| Genre | Fields you fill in |
|---|---|
| `<wiki>` | article title |
| `<book>` | book title, optional opening line |
| `<review>` | product name, star rating (★ 1–5), category (pulldown of the trained categories) |
| `<recipe>` | dish name |
| `<story>` | premise |

The app builds the exact tagged prompt from those fields (e.g.
`<|review|> Category: ...\nProduct: ...\nRating: 1/5\n`), so nobody has
to type the format. A **Show what the model sees** checkbox reveals that
raw prompt, in keeping with the project's "written to be read" spirit.

**Idea bank.** `tlm/ideas.json` holds ~50 pre-written ideas per genre
(~250 in all). About a quarter are genre mix-ups, labeled "(mix-up)".
- The window shows 4 ideas at a time. Clicking one fills in the fields;
  **More ideas** deals the next 4.
- Ideas are dealt like cards from a shuffled deck: no repeats until all
  of a genre's ideas have been shown, then the deck reshuffles.
- The deck position is saved between launches, so reopening the app
  doesn't show the same four again.
- **Surprise me** picks a random genre and idea and generates right
  away.
- The training prompt bank's eight entries are in the pool too.

Sample ideas (the full pool gets written along with the app):

- `<wiki>`: The Great Spoon Shortage of 1874 · Competitive Napping at the
  1912 Olympics · The Treaty of Mild Annoyance · Grandma's Emergency
  Lasagna (mix-up)
- `<book>`: The Adventures of a Reluctant Toaster · Lady Agatha and the
  Suspicious Teapot · A Gentleman's Guide to Arguing with Geese · The
  Phone Charger of Doom (mix-up: a modern object in a 1900s novel)
- `<review>`: Self-Heating Socks for Cats ★☆☆☆☆ · A Single Grape,
  Individually Wrapped ★★★★★ · Inflatable Dartboard ★☆☆☆☆ · The Moon
  ★★☆☆☆ (mix-up)
- `<recipe>`: Tuesday Casserole (Do Not Serve on Wednesday) · Microwave
  Soufflé for Impatient People · Grandma's Emergency Lasagna · The French
  Revolution (mix-up)
- `<story>`: Your cat has been secretly running a small country · A
  pigeon is elected mayor and takes the job very seriously · Your toaster
  starts giving you life advice, and it is always wrong · The world's
  least intimidating dragon applies for a bank loan

**Generating**
- Text streams in token by token. **Stop** cancels, and generation also
  stops by itself at `<|endoftext|>`.
- **Creativity** is the sampling temperature (0.5–1.2, default 0.8).
  **Length** is short, medium or long (~150 / 300 / 600 tokens).
- Generation runs on a background thread, so the window never freezes.
  Tk isn't thread-safe, so the worker hands tokens to the window through
  a queue that the window polls with `after()`.
- **Model:** loads `mixtape/best.pt` from the checkpoint folder, and the
  status line shows its step and val loss. **Reload** picks up a newer
  best.pt, so during the ~3-day run you can watch the model get funnier.
  If there's no model yet, the window says how to train one instead of
  crashing.
- **Device:** Auto uses the GPU only if it has ≥ 1.5 GB free
  (`torch.cuda.mem_get_info`), otherwise the CPU. That way the app never
  fights a training run for the GPU.
- **Save to favorites** appends the prompt, output, model step and
  settings to `favorites.md` next to JOURNAL.md, a laugh log you can
  commit. **Copy** puts the output on the clipboard.
- **Errors:** `pythonw` has no console, so errors go to a message box and
  to `~/tlm-data/gui.log`.

**Model changes the app needs**
- **KV cache** in `TLM.generate`. Today every new token re-runs the
  model over the whole context. Caching each layer's keys and values
  means a new token only computes its own position. My estimate for ~300
  tokens on the CPU: down from a minute or two to a few seconds. That's
  what makes CPU generation, and so using the app during training,
  practical. Worth a docs section.
- **Streaming:** a generator version of `generate` that yields each token
  as it's sampled.

**Files:** `tlm/gui.py` (the app), `tlm/ideas.json` (idea bank),
`tools/make_shortcut.py`, `tools/make_icon.py`, `assets/mixtape.ico`.

## Risks / open questions

- **Speed** is estimated from the datasheet; steps 1 and 5 measure it.
  `torch.compile` isn't planned (Windows support is uncertain); if it
  works, it's a bonus.
- **Memory:** 6 GB with ~0.8 GB taken by the desktop, so ~5.3 GB
  usable. Step 1 measured ~42 bytes of activations per layer × token ×
  channel: the Stage 3 pilot peaked at 2.13 GB, of which ~0.44 GB was
  weights and optimizer state and ~0.3 GB logits. Scaled to the 126M
  model:
  - weights + gradients + AdamW state ≈ 2.0 GB
  - activations: ~1.0 GB at micro-batch 2, ~2.1 GB at micro-batch 4
  - logits over a 16k vocab: ~0.3 GB at micro-batch 2, ~0.6 GB at 4
  - CUDA context ≈ 0.4 GB (not counted in `max_memory_allocated`)

  That's ~3.7 GB at micro-batch 2, which fits, and ~5.1 GB at 4, which
  is too tight. If even 2 won't fit, activation checkpointing (recompute
  instead of store, ~30% slower) is the fallback before shrinking the
  model.
- **Checkpoints** grow to ~1.5 GB each (fp32 weights + AdamW state), so
  best.pt + latest.pt ≈ 3 GB. That's fine outside OneDrive.
- **Licenses:** Amazon (unstated), RecipeNLG (unknown) and WritingPrompts
  (Reddit) → data and weights stay local, as now (gitignored). Wikipedia
  is CC BY-SA.
- **Content:** old books carry dated attitudes, and Reddit stories can be
  crude even after the NSFW filter. It's a personal project, but worth
  knowing before sharing samples.
- **Collisions may not collide** — the model might ignore a title that
  doesn't fit its tag. Ideas if so: put titles on more chunks, or mix in a
  small set of hand-written collision examples. Decide after seeing
  samples.

## Later / optional

- **DONE 2026-10-22: GPU benchmark** (`tools/benchmark_gpu.py`, results in
  `~/tlm-data/gpu-benchmark.md`): Stage 1 in full, 173,220 tok/s, 4 min,
  val 1.606 (CPU run 1.609); Stage 2, 2,000 steps, 89,258 tok/s → ~1.5 h
  full run; Stage 3, 2,000 steps, 37,186 tok/s → ~4 h. 30×, 36×, 30× the
  CPU; ~3.4 / 4.5 / 7.0 TFLOPS (Stage 4 ~9). Folded into the README.
- **DONE 2026-10-22: training data and spare checkpoints removed** from
  the dev machine (Larry: no plans to retrain; the data can be
  re-downloaded, except that RecipeNLG may need its Hugging Face copy).
  88.3 GB moved to `C:\TLM-cleanup` for Larry to delete: all of
  `mixtape-raw`, the mixtape formatted text, token files and train.bin,
  `mixtape-small`, the Stage 1-3 text and train.bin files, `instruct-ft`
  data and checkpoint, the pilot/test/validation/old-run checkpoints,
  every `latest.pt`, the mixtape snapshots and spare `.pt` files. Kept:
  each stage's `best.pt` and logs, every `tokenizer.json`, the val files,
  `stats.json`, `weights/`. Checked afterwards: all four stages generate
  with `tlm.generate`, and the Mixtape app runs (README screenshot).
  Retraining would need the prepare_* scripts re-run first.
- (was:) **Validate the GPU time estimates for Stages 1–3** (the README table's
  † entries, added 2026-10-16). Do it once the GPU is free, i.e. after
  the mixtape run.

  | stage | CPU (measured) | GPU estimate | basis |
  |---|---|---|---|
  | 1 `shakespeare` | 2 h (~5,700 tok/s) | ~10 min | 19.5 MFLOP/token × 41M tokens at ~1.5–2.5 TFLOPS, plus eval/sample overhead |
  | 2 `stories` | 55.7 h (~2,450 tok/s) | ~2–3 h | 50.3 MFLOP/token × 492M tokens at ~2.5–4 TFLOPS |
  | 3 `instruct` | 118 h (~1,250 tok/s) | ~4.5 h | 188.7 MFLOP/token × 533M tokens at the measured 6.2 TFLOPS (calibration pilot: 32,700 tok/s) |

  Method:
  - run each preset on the GPU with
    `TLM_CKPT_DIR=~/tlm-checkpoints/gpu-validation`, so the existing CPU
    checkpoints aren't touched (train.py refuses to overwrite them
    anyway)
  - Stage 1 in full (~10 min)
  - Stages 2 and 3 for ~2,000 steps each, extrapolating the sustained
    tok/s from the last eval lines; the tok/s column already includes
    eval and sample overhead
  - optionally a full Stage 2 run (~2–3 h), which also tests whether the
    GPU run reaches the CPU run's val loss of 1.595

  Then replace the "(est.)" entries in the README with the measured
  times, add a line to the journal, and record the measured TFLOPS per
  model size: they show how GPU efficiency falls as models get smaller.
- More tags: SimpleStories (`<|tiny|>`, with style/persona dials),
  FineWeb-Edu (`<|web|>`), r/Jokes (small and crude, fine-tune only).
- Bits-per-byte as the reported loss, which compares across tokenizers
  (fixes the README footnote).
- RL with a checkable reward on the Stage 3 model (the `Words:` field) —
  a separate 1–2 day project.

## Logistics

- Same drill as Stages 2–3: no sleep, `--resume` after any restart.
  Checkpoint every 1,000 steps (~1 hour at the measured ~9,300 tok/s).
- The GPU also drives the display: the desktop stays usable but may lag.
  Avoid games and video work during the run.
- Push commits to `origin/master` as work lands
  (https://github.com/elkayem/TLM_by_Claude).

## README notes: setup & configuration (running list)

Every setup or configuration detail a new user would need goes here as
soon as it exists, so the README update (step 8) is a copy-edit rather
than an archaeology dig. Add to this list whenever something changes;
move items from "Coming" to "Ready" when they're built and tested.

### Ready to document (built and tested) — all folded into the README on 2026-10-16

(Kept here as the record; new items go below this list, or under "Coming".)

- **Two PyTorch builds; pick one.** Both commands are in
  `requirements.txt`:
  - NVIDIA GPU: `pip install -r requirements.txt --extra-index-url
    https://download.pytorch.org/whl/cu130` (~1.9 GB download, ~2.3 GB
    installed)
  - CPU only: the same with `.../whl/cpu`. Everything still works,
    including Stages 1–3 exactly as they were trained.
- **GPU driver requirement:** the cu130 build needs a driver that
  supports CUDA 13.0. `nvidia-smi` shows the highest CUDA version the
  driver supports (this PC: driver 616.92, CUDA 13.4). With an older
  driver, pick an older build from the same index (e.g. `cu126`).
- **Switching an existing CPU venv to the GPU build:**
  `pip install "torch==2.13.0+cu130" --index-url
  https://download.pytorch.org/whl/cu130` (replaces `2.13.0+cpu`).
- **Check that the GPU is visible:** `python -c "import torch;
  print(torch.__version__, torch.cuda.is_available())"` should print
  `2.13.0+cu130 True`.
- **Choosing the device:** `--device auto|cuda|cpu` on both `tlm.train`
  and `tlm.generate`. `auto` (the default) uses the GPU when PyTorch can
  see one, else the CPU. The training banner prints which one it picked.
- **Moving between devices:** checkpoints work across CPU and GPU in both
  directions, including `--resume` and `--init-from`.
- **New config fields** (`tlm/config.py`), with defaults that keep old
  presets and checkpoints working unchanged:
  - `grad_accum` (default 1): micro-batches per optimizer step
  - `fused_attention` (default True): False switches to the explicit
    attention (same result, slower)
- **GPU-only behavior:** bf16 mixed precision and fused AdamW. The
  training log adds a `mem` column (peak GPU memory).
- **Speed reference for the README table:** the Stage 1–3 times are CPU
  times (Intel Core Ultra 7 265). On the RTX 3050 6 GB, the Stage 3 model
  trains at ~32,700 tok/s vs ~1,250 on the CPU (~27×). On the CPU, fused
  attention is ~10% faster than the explicit version.
- **Keep big things out of synced folders.** Checkpoints default to
  `~/tlm-checkpoints` (`TLM_CKPT_DIR`) and datasets to `~/tlm-data`
  (`TLM_DATA_DIR`). The venv should live outside synced folders too: with
  the CUDA build it's ~3 GB. Done 2026-10-15: the dev machine's venv
  moved from `..\.venv` (inside OneDrive) to `C:\Users\lkmcg\tlm-venv`,
  rebuilt from `pip freeze` plus `pyarrow`, because venvs can't simply be
  moved. Setup commands for the README:
  `python -m venv %USERPROFILE%\tlm-venv`, then
  `%USERPROFILE%\tlm-venv\Scripts\activate` and the
  `pip install -r requirements.txt ...` line above.

- **Where datasets live:** `TLM_DATA_DIR`, default `~/tlm-data/`
  (`%USERPROFILE%\tlm-data`). All prepare scripts write there, and
  training and generation read from there. A dataset that only exists in
  the repo's old `data/<name>/` folder is still found there, so older
  setups keep working without moving anything. The README's Layout
  section ("downloaded corpora land here") needs updating to match.
- **New dependency `pyarrow`** (parquet reader, for the mixtape
  sources). It's in `requirements.txt`.
- **Mixtape data prep:**
  - `python data\prepare_mixtape.py` runs all stages (download,
    format, tokenizer, encode, mix). They can also be named one at a
    time, every stage resumes, and `inspect` shows samples and stats.
  - `--small` builds a ~5M-token test dataset (`mixtape-small`) in a
    few minutes, and `python -m tlm.train --config mixtape-smoke`
    trains a toy model on it.
  - To change the mixture, edit `TRAIN_TOKENS` in the script, delete
    `train.bin`, and re-run `mix`.
  - **Disk and time (measured on the dev machine):** raw downloads
    35 GB, the dataset folder 22 GB, so have ~60 GB free. Downloads take
    ~30 min at ~20 MB/s; after that, format ~20 min, tokenizer
    seconds, encode ~3 min, mix ~3.5 min. After training, `formatted/`
    and `mixtape-raw/` can be deleted to get ~45 GB back. Keep
    `tokens/` if you might re-mix.
- **RecipeNLG manual download:** from
  https://recipenlg.cs.put.poznan.pl/, accept the terms, download
  `dataset.zip`, and put it (or the `full_dataset.csv` inside) in
  `~/tlm-data/mixtape-raw/recipenlg/`. Until then, the other genres
  download and format, and the tokenizer stage waits for it. **If the site
  fails** (it returned "Internal Server Error" on 2026-10-16), use the
  exact copy on Hugging Face, which is what the dev machine used:
  `SandhyaKilari/RecipeNLG_dataset` → `RecipeNLG_dataset.csv` (2.29 GB,
  the same 2,231,142 recipes and columns). It's unofficial, so
  RecipeNLG's terms still apply. The script's help text gives the URL.
- **Desktop app:**
  - `python tools\make_shortcut.py` (run with the venv's python) puts a
    "TLM Mixtape" shortcut on the desktop. It asks Windows where the
    desktop is, which matters when OneDrive has moved it. Re-run it if
    the repo or the venv moves.
  - Or start it with `python -m tlm.gui` (`--run NAME`,
    `--device cpu|cuda`).
  - Tkinter comes with the python.org Windows installer (the "tcl/tk
    and IDLE" option, on by default); there's nothing to pip install.
  - It loads the first of `mixtape`, `mixtape-pilot`, `mixtape-smoke`
    that has a `best.pt`; Reload picks up a newer one mid-training. It
    runs on the GPU if ≥ 1.5 GB is free, otherwise the CPU.
  - Saved favorites go to `favorites.md` in the repo (a laugh log worth
    committing). The deck state goes to `~/tlm-data/gui_state.json` and
    errors to `~/tlm-data/gui.log`.
  - The ideas are in `tlm/ideas.json`. Users can add their own, and the
    app picks them up on its next launch.
  - `python tools\make_icon.py` redraws `assets/mixtape.ico`.

### Added 2026-10-18/19 (the restart and extension) — folded into the README on 2026-10-19

Also new on 2026-10-19: `fixed_eval` (on by default; off for the running
mixtape preset), `tools/sample_sheet.py` (fixed-seed sheets and blind A/B
comparisons), `tools/plot_mixtape_run.py` (docs/mixtape-run.svg).

- **Start long runs by double-clicking** `tools\train_mixtape.cmd` (or
  any `.cmd`) in Explorer, not from inside an AI assistant's or IDE's
  terminal: a process started by an app dies with it, and the Claude
  desktop app auto-updates (that stopped the mixtape run overnight on
  2026-10-18).
- `tools\train_mixtape.cmd` passes extra options through to train.py,
  e.g. `tools\train_mixtape.cmd --rewarm 500`.
- `train.py --rewarm N`: on resume, ramp the LR up over N steps (after
  checkpoint surgery with a fresh optimizer).
- Extending a finished run: raise `max_steps` and set `restart_step`
  (the old last step), `restart_lr` (the rewarm peak) and
  `restart_warmup` in its preset, then resume (double-click the
  launcher). Keep a copy of the old final checkpoint as the baseline.
- `python tools\add_qk_norm.py SRC.pt DST.pt --layers 1
  --fresh-optimizer --best-val X`: graft QK-norm onto a checkpoint.
- Config: `qk_norm`, `qk_norm_layers`, `snapshot_interval`,
  `snapshots_kept`. The mixtape run keeps 3 full snapshots (~1.6 GB
  each) in `~/tlm-checkpoints/mixtape/snapshots/`.
- `log.csv` gained `val_genre_mean`, `grad_norm`, `grad_norm_max` and
  `attn_score_max` columns; `best.pt` now follows `val_genre_mean` when
  a dataset has per-genre val sets.
- **QK-norm is ON by default** for new models (Larry's call,
  2026-10-18). The Stage 1-3 presets set `qk_norm=False`, so they
  reproduce exactly as documented. Checkpoints saved before the setting
  existed load with it off (`config_from_checkpoint()` in config.py,
  used by generate.py, gui.py and the graft tool). Tested: Shakespeare,
  Stories, Instruct and both mixtape runs load and generate on the CPU;
  a fresh default model has QK-norm in every layer.

### Coming (document when built)

The launcher, the Stage 4 quickstart commands, the README table column
and the new references were also folded in on 2026-10-16. On
2026-10-19 the README table, the Results section and "The run" in
`docs/10-mixtape.md`, the run chart and a JOURNAL entry were written with
the step-79,000 results, and finalized with the extension's on
2026-10-21 (chart redrawn by `tools/after_run.py`; blind A/B sheet in
`~/tlm-data/samples/`; check-in task and permission rule removed). Still
open: the GPU validation of the Stage 1-2 estimates ("Later / optional"),
and deleting the old checkpoints listed above (Larry's call).
