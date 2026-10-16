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
If step 1's measured ~6.2 TFLOPS holds at this size, it's ~6,800 tok/s
and **~4.2 days**.

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
5. **Pilot (`mixtape-pilot`, ~500 steps)** — measure tok/s against the
   5,000 gate and peak memory (`torch.cuda.max_memory_allocated`). Pick
   the micro-batch size and set `max_steps`.
6. **Launch** `python -m tlm.train --config mixtape` (~4.2 days); journal
   with the prompt bank.
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
     - Creativity (temperature 0.5–1.2) and Length (150 / 300 / 600)
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
8. **Docs + README** — `docs/09-gpu.md` (devices, mixed precision, fused
   attention, gradient accumulation) and `docs/10-mixtape.md` (data
   mixtures, control tokens, collisions, plus the KV cache). Add a Stage 4
   README column, quickstart lines (including
   `python tools/make_shortcut.py`), and verified references (CTRL, PG-19,
   RecipeNLG, WritingPrompts, Amazon Reviews 2023, Wikipedia). Then fold
   in everything from "README notes" at the end of this plan.

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
  best.pt, so during the ~4-day run you can watch the model get funnier.
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

- More tags: SimpleStories (`<|tiny|>`, with style/persona dials),
  FineWeb-Edu (`<|web|>`), r/Jokes (small and crude, fine-tune only).
- Bits-per-byte as the reported loss, which compares across tokenizers
  (fixes the README footnote).
- RL with a checkable reward on the Stage 3 model (the `Words:` field) —
  a separate 1–2 day project.

## Logistics

- Same drill as Stages 2–3: no sleep, `--resume` after any restart.
  Checkpoint every 1,000 steps (~80 min at ~6,800 tok/s).
- The GPU also drives the display: the desktop stays usable but may lag.
  Avoid games and video work during the run.
- Push commits to `origin/master` as work lands
  (https://github.com/elkayem/TLM_by_Claude).

## README notes: setup & configuration (running list)

Every setup or configuration detail a new user would need goes here as
soon as it exists, so the README update (step 8) is a copy-edit rather
than an archaeology dig. Add to this list whenever something changes;
move items from "Coming" to "Ready" when they're built and tested.

### Ready to document (built and tested)

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

### Coming (document when built)

- Mixtape quickstart: pilot + train commands, ~4.2 days on the RTX 3050,
  peak memory, and the micro-batch the pilot settled on (steps 5–6).
- README table: a Stage 4 column, a device row (CPU vs GPU timings), and
  the new references (step 8).
