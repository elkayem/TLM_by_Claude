# Stage 4 plan — TLM-Mixtape (draft, 2026-10-15)

Goal: train a ~100M-param model on the new GPU, on a **blend of five
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
- The venv has **torch 2.13.0+cpu**, so the CUDA build must be installed.
- Throughput estimate: ~10–20× the CPU. Stage 3's 118 h would take
  ~6–12 h. **These are estimates, not measurements** — the calibration
  pilot (step 1) replaces them with real numbers.

## Proposed design

**Model (`mixtape` preset):** n_layer=12, n_head=12 (head size 64),
n_embd=768, block_size=1024, vocab 16,384 → ~85M params in blocks +
~13M tied embedding + ~0.8M positions ≈ **98M**. GPT-2-small shape with
a smaller vocab. dropout=0.0: every source is seen about once, so there
is nothing to memorize.

**Budget:** 20 tokens/param ≈ **2B tokens** (Chinchilla). At an estimated
~700M FLOPs/token, that's ~1.4×10¹⁸ FLOPs, or ~5 days at a sustained
~3.5 TFLOPS.

**Batching:** micro-batch 4 × 1024 tokens, gradient accumulation × 8 =
32,768 tokens per optimizer step → **max_steps 61,000** (≈ 2.0B tokens).
Peak LR 6e-4 → 6e-5 cosine, 2,000 warmup steps, other optimizer settings
as before. Micro-batch 4 is a guess at what fits in 6 GB (see Risks); the
pilot picks the real value, and the accumulation count adjusts to keep
32k tokens/step.

**Gate (same idea as Stage 3's):** the pilot must sustain **≥ 4,000
tok/s** (≤ ~5.8 days). If not, fall back to n_layer=10, n_embd=640 (~60M
params) and keep the ~5-day budget. That trains past 20 tokens/param,
which is fine; small models routinely are.

**Tokenizer:** new from-scratch BPE, vocab 16,384, trained on a balanced
sample of all five genres (~20 MB each; Stage 3 used 10 MB for 4,096 —
time a 20 MB run first). Special tokens: `<|endoftext|>` plus the five
genre tags. `BPETokenizer.train(..., special_tokens=[...])` already
takes a list, so no tokenizer changes are needed.

## The mixture (~2B tokens)

| Tag | Source | Share | Tokens | How much of the source |
|---|---|---|---|---|
| wiki | English Wikipedia | 30% | 600M | a random slice of articles |
| book | PG-19 (Gutenberg books published before 1919) | 25% | 500M | ~1/5 of the books |
| review | Amazon Reviews 2023, 3–4 categories | 20% | 400M | a few categories |
| recipe | RecipeNLG | 15% | 300M | ~one full pass (estimate) |
| story | WritingPrompts | 10% | 200M | most of one pass |

**Why this split:** Wikipedia and books are the backbone (55%). The
collisions only work if the model knows about things — a recipe for "The
French Revolution" is only funny if it knows guillotines and baguettes.
The three funny formats get far more than their natural share of text.
The two small sources get about one pass so the model learns their
*format* without memorizing particular recipes or stories. Shares are a
starting proposal; adjust them after looking at the prep's token counts.

**How the mix is built:** prep writes ONE pre-mixed `train.bin`. It picks
documents from each source until that source hits its token target,
shuffles at the document level, and concatenates. `get_batch` stays as
it is (random windows ≈ one pass over the bin). At 16-bit ids that's
~4 GB on disk. Validation: ~1–2M held-out tokens per genre, written to
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
  600M tokens after sampling. Probably skip "List of ..." pages (mostly
  tables) and very short stubs.
- **book** — PG-19 (Rae et al. 2019; Apache 2.0, books public domain):
  HF `deepmind/pg19` or the `deepmind-gutenberg` GCS bucket. 28,602 train
  books, ~2B words, ~11 GB. Gutenberg boilerplate is already stripped,
  and offensive slurs were replaced with placeholders. `metadata.csv` has
  `short_book_title` (no author field).
- **review** — HF `McAuley-Lab/Amazon-Reviews-2023`: per-category
  `review_categories/<Category>.jsonl.gz` plus
  `meta_categories/meta_<Category>.jsonl.gz`, joined on `parent_asin` to
  get the product name. Pick 3–4 mid-sized categories in prep (candidates:
  Pet Supplies, Toys and Games, Grocery and Gourmet Food, Tools and Home
  Improvement). Filters:
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

1. **GPU port + calibration (Stage 4a)** — useful on its own, and every
   later step depends on it.
   - Install the CUDA build of PyTorch into `../.venv` (multi-GB
     download). Update the install note in `requirements.txt`. The CPU
     path must keep working.
   - `train.py`:
     - pick the device (cuda if available, else cpu) and move the model
       and batches there (pinned memory, non_blocking)
     - bf16 autocast on the forward pass
     - gradient accumulation (new config field)
     - extra val splits for the per-genre losses
     - samples drawn from the prompt bank (below) instead of "\n"
   - `model.py`: train with `F.scaled_dot_product_attention` (fused
     kernel, never builds the T×T matrix). Keep the hand-written attention
     as the documented reference path behind a flag, with a check that
     both give the same output.
   - `generate.py`: device support.
   - Calibration: run `instruct-pilot` on the GPU for ~400 steps. Val loss
     at steps 100/200/300/400 should track the CPU pilot's log (4.43 /
     3.93 / 3.63 / 3.45) within noise. Record tok/s against the CPU's
     1,220. Optional: the full `instruct` config overnight should land
     near val 1.257.
2. **Data location** — raw downloads (tens of GB) and the ~4 GB bin must
   NOT live in the repo: the repo is in OneDrive, which would upload all
   of it. Add a `TLM_DATA_DIR` env var / default `~/tlm-data/`, mirroring
   `TLM_CKPT_DIR`, and resolve old datasets in-repo as `checkpoint_dir()`
   does. `train.py` and `generate.py` both read `data/<dataset>/` today.
3. **`data/prepare_mixtape.py`** — downloads (resumable, like
   `prepare_stories.download`), one formatter per genre, tokenizer
   training, encoding, mixing, and the per-genre val bins. Add
   `!data/prepare_mixtape.py` to `.gitignore`. New dependency:
   `pyarrow` (parquet). Measure encode speed on a sample first:
   Wikipedia and books have far more distinct words than TinyStories, so
   the tokenizer's word cache will help less. Parallelize across the 20
   cores if needed. Manual step: the RecipeNLG download.
4. **Inspect before training** — print ~5 formatted examples per genre,
   the token count per genre, and confirm each tag encodes to one id.
5. **Pilot (`mixtape-pilot`, ~500 steps)** — measure tok/s against the
   4,000 gate and peak memory (`torch.cuda.max_memory_allocated`). Pick
   the micro-batch size and set `max_steps`.
6. **Launch** `python -m tlm.train --config mixtape` (~5 days); journal
   with the prompt bank.
7. **The Mixtape app** — build it while the 5-day run trains. It only
   needs a checkpoint (the step-5 pilot's is enough to develop against),
   and `best.pt` keeps improving underneath it. Includes the KV cache and
   streaming generation (see the app section).
8. **Docs + README** — `docs/09-gpu.md` (devices, mixed precision, fused
   attention, gradient accumulation) and `docs/10-mixtape.md` (data
   mixtures, control tokens, collisions, plus the KV cache). Add a Stage 4
   README column, quickstart lines (including
   `python tools/make_shortcut.py`), and verified references (CTRL, PG-19,
   RecipeNLG, WritingPrompts, Amazon Reviews 2023, Wikipedia).

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
<|review|> Category: Tools and Home Improvement\nProduct: The Moon\nRating: 2/5   (collision)
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
- The shortcut runs `..\.venv\Scripts\pythonw.exe -m tlm.gui` from the
  repo folder. `pythonw` means no console window.
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
  best.pt, so during the 5-day run you can watch the model get funnier.
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
- **Memory:** 6 GB with ~0.8 GB taken by the desktop. Rough budget at
  micro-batch 4:
  - weights + gradients + AdamW state ≈ 1.6 GB
  - activations ≈ 1.0–1.3 GB
  - logits over a 16k vocab ≈ 0.5–0.8 GB (temporary)
  - CUDA context ≈ 0.4 GB

  That's ~4 GB total, so micro-batch 8 probably won't fit.
- **Checkpoints** grow to ~1.2 GB each (weights + AdamW state), so
  best.pt + latest.pt ≈ 2.4 GB. That's fine outside OneDrive.
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
  Checkpoint every 1,000 steps (~2 h at the estimated speed).
- The GPU also drives the display: the desktop stays usable but may lag.
  Avoid games and video work during the run.
- Push commits to `origin/master` as work lands
  (https://github.com/elkayem/TLM_by_Claude).
