# Chapter 10 — Stage 4: the Mixtape (data mixtures and control tokens)

*Code: `data/prepare_mixtape.py`; `BPETokenizer.train_fast` in
`tokenizer.py`; the `mixtape*` presets and `MIXTAPE_PROMPTS` in
`config.py`; per-genre validation in `train.py`; `TLM.generate_stream` in
`model.py`; the desktop app in `gui.py`*

Stages 2 and 3 learned from one tidy synthetic dataset. Stage 4 learns
from real English — five very different sources, ~2.5 billion tokens — with
a 126M-parameter model (16 layers × 768 wide, 1,024-token context), about
4.6× Stage 3. Two ideas carry the stage:

> **1. A model is its data, and real models are trained on *mixtures*:
> how much of each source goes in is a design decision.**
>
> **2. Tag every document with its source, and the tag becomes a dial
> you can turn at generation time.**

The second is the "control code" trick of CTRL (Keskar et al. 2019),
shrunk to fit one small GPU. And there's a third, less academic goal: the
Stage 2 stories were funny because a small model nails a familiar format
while getting the details confidently wrong. Stage 4 is built to aim that
at things that don't exist.

## The five genres

Every training document starts with one of five genre tags — special
tokens, atomic like `<|endoftext|>` in chapter 8 — then a short header,
then the text:

```
<|wiki|> Title: Piaohao
Piaohao (literally 'draft banks'), ... were a type of bank that existed in China ...<|endoftext|>
<|review|> Category: Pet Supplies
Product: Maggii 9 Inch Grunting Latex Rubber Dog Toy Polka Dot Pig Dog Toy, Red
Rating: 1/5
Not durable
Purchased as a Christmas gift for my 10 mo old Husky. She ripped it to shreds in under 10 minutes.<|endoftext|>
<|recipe|> Title: Creamy Potato Soup
Ingredients:
- 6 bacon slices, cooked
...
Directions:
1. In large saucepan, cook onions ...<|endoftext|>
```

| tag | source | what it brings |
|---|---|---|
| `wiki` | English Wikipedia (10 of 41 shards) | facts, a deadpan encyclopedia voice, names of real things |
| `book` | PG-19: 28,602 Gutenberg books published before 1919 | narrative, dialogue, an old-fashioned literary voice |
| `review` | Amazon Reviews 2023: Pet Supplies, Toys and Games, Grocery | a shopper's voice; star rating → opinion |
| `recipe` | RecipeNLG: 1.6M recipes | ingredient lists, numbered steps |
| `story` | WritingPrompts: 303k Reddit stories, each answering a prompt | turning a premise into a story |

Book headers are `Title: <title> by <author>` (from PG-19's metadata), so
an author's name is a style dial too. Story headers are
`Prompt: <premise>`.

## Building a mixture

`prepare_mixtape.py` runs five stages, each skipping work already done:

```
download -> format -> tokenizer -> encode -> mix
 35 GB       11 GB      16,384      5.5 GB    5 GB train.bin
 ~30 min    ~20 min    seconds     ~3 min    ~3.5 min
```

The interesting decisions are all in **format** and **mix**:

- **Plain ASCII.** Real text is full of curly quotes, accents, emoji and
  other alphabets. A from-scratch tokenizer spends vocabulary on every
  distinct character it sees, so everything is mapped to ASCII (curly
  quotes to straight ones, `é` to `e`), and documents that are mostly
  *not* Latin text are dropped, as is anything failing a cheap English
  check (" the", " and", " of", " to" should be ≥ 3% of the words).
- **Chunking with repeated headers.** Training samples random 1,024-token
  windows. A window from the middle of a 70,000-word novel would usually
  contain no tag at all, and the model would never learn what `<|book|>`
  means. So long texts are cut at paragraph breaks into pieces of ≤ 3,500
  characters (~850 tokens), and *every* piece starts with the tag and
  title again.
- **Rebalancing review ratings.** About 65% of real reviews are 5 stars.
  Train on that and `Rating:` barely matters; the model learns "reviews
  are positive." The format stage keeps every 1–3-star review, 60% of the
  4s and 20% of the 5s, which brings 5-star down to ~30%. (The 1-star
  reviews are where the comedy is, anyway.)
- **Shares.** Wikipedia and books are the backbone (55%) because the
  genre collisions are only funny if the model knows about the world; the
  three playful formats get far more than their natural share. A source
  smaller than its share is repeated — Muennighoff et al. (2023) found up
  to ~4 passes over the same data are nearly as good as fresh data.
- **Shuffle and hold out.** Mix shuffles all 8.6M chosen documents of all
  genres together into one `train.bin`, after setting aside 1.5M random
  tokens *per genre* as validation data the model never trains on.

The result:

| genre | train tokens | share | passes | val tokens | tokens/doc |
|---|---|---|---|---|---|
| wiki | 750M | 30% | 0.84 | 1.5M | 577 |
| book | 625M | 25% | 0.74 | 1.5M | 804 |
| review | 500M | 20% | 0.80 | 1.5M | 119 |
| recipe | 375M | 15% | 1.17 | 1.5M | 197 |
| story | 250M | 10% | 1.05 | 1.5M | 582 |

2.5B tokens is ~20 per parameter — chapter 8's Chinchilla rule again.
Changing the recipe is cheap: edit `TRAIN_TOKENS`, delete `train.bin`,
re-run `mix` (minutes, no re-download or re-encoding).

### One val number hides a lot

`train.py` finds every `val_<genre>.bin` next to `val.bin` and reports a
loss for each genre. After just 500 steps of the pilot:

```
val by genre: book 5.28 | recipe 3.06 | review 4.34 | story 5.03 | wiki 5.48
```

Recipes are learned first — they're formulaic (`- 1 c. sugar`), so each
next token is easy to guess. Encyclopedia text, full of names and dates,
is hardest. Note these losses are *not* a ranking of quality: like the
README's footnote, a loss measures how predictable the text is as much as
how good the model is. The useful signal is each genre's curve over time.

## Control tokens

Nothing new happens inside the model. The tag is one more token in the
context, and the model learns what chapter 8's model learned about
`Summary:` — that text after `<|recipe|>` looks like a recipe, so matching
it reduces loss. At generation time you write the beginning of a document
yourself, and the model writes the most plausible rest of it:

```
<|recipe|> Title: Grandma's Emergency Lasagna\n                 straight
<|review|> Category: Pet Supplies\nProduct: Self-Heating Socks for Cats\nRating: 1/5\n
<|recipe|> Title: The French Revolution\n                       a collision
<|wiki|> Title: Grandma's Emergency Lasagna\n                   a collision
```

The last two are **genre collisions**: a tag and a header that never
appeared together in training. The model has to compose — what does a
recipe look like when its title is a historical event? It might blend the
two (`2 cups guillotine, finely chopped`), or it might ignore the title
and write an ordinary recipe. Nothing guarantees which; that's the
experiment. (CTRL used the same idea at 1.6B parameters, with control
codes taken from where each text came from; today's chat models carry it
on with role markers and system prompts.)

## A tokenizer for real English

Stages 2–3 used 4,096 tokens. Five genres of adult English need more:
the mixtape tokenizer has **16,384**, which compresses this text to ~3.9
characters per token. It isn't free — the embedding table alone is
16,384 × 768 ≈ 12.6M parameters, 10% of the model.

Training it exposed a problem with chapter 1's algorithm. `train()`
recounts every adjacent pair in every unique word *on every merge* —
fine for TinyStories' few tens of thousands of words, hopeless for real
English. `train_fast()` does what production BPE trainers do: it keeps the
pair counts *between* merges, updates only the words that contain the
pair just merged (a tiny fraction), and finds the most frequent pair with
a heap instead of a scan. It was checked against `train()`: every merge it
picks is the true most frequent pair (the two differ only in how they
break ties), and it's ~43× faster. On the mixtape sample — 50 MB, 203,400
unique words — the whole tokenizer stage, counting included, takes about
six seconds: 96 base characters + 16,282 merges + 6 special tokens =
16,384.

## The KV cache and the app

`generate()` (chapter 7) re-runs the model over the whole context for
every new token. But a position's keys and values depend only on that
position's own input — once computed, they never change.
`generate_stream()` keeps them: the prompt runs once (the *prefill*),
every layer caches its `k` and `v`, and each new token then costs one
position's worth of work. It yields tokens one at a time, so text can
appear as it's written. With the same seed it produces exactly
`generate()`'s tokens; on the CPU it's ~3× faster at 300 tokens, and the
gap grows with model size and text length. (One wrinkle: TLM's positions
are learned embeddings 0..1,023, so the cache can't slide along the way
`generate()`'s window does. When the context fills, it restarts from the
last 3/4 of the text.)

That's what makes the **Mixtape app** practical (`python -m tlm.gui`, or
the desktop shortcut from `tools\make_shortcut.py`). Each genre gets a
small form — a title; a product, rating and category; a premise — which
the app turns into exactly the document headers above. Tick *Show what
the model sees* to watch it happen. Ideas come from `tlm/ideas.json`,
dealt four at a time like cards from a shuffled deck, about a quarter of
them deliberate collisions. While a training run owns the GPU, the app
quietly runs on the CPU, and *Reload model* picks up the newest
checkpoint — you can watch it get funnier day by day.

## Results

*To be filled in when the run finishes (expected ~2026-10-19): the final
loss overall and per genre, samples from the prompt bank, and the
collisions that worked — and the ones that didn't.*

## Experiments

- Collide on purpose: `<|recipe|> Title: The Battle of Hastings`,
  `<|review|> ... Product: Gravity ... Rating: 5/5`. Does the model blend
  the genres or ignore the header?
- Leave the tag off: prompt with just `Title: Toast\n`. Which genre does
  the model guess?
- Same book title, different author: `... by Jane Austen` vs `... by Mark
  Twain`.
- Same product at 1 star and at 5 stars.
- Plot each genre's validation loss from `log.csv` (the `val_<genre>`
  columns). Which genre is still improving at the end?
- Re-mix (e.g. double the recipes), fine-tune the finished model on the
  new mix with `--init-from mixtape`, and compare.
