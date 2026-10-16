"""
Stage 4 data prep: TLM-Mixtape - five genres of real English, each tagged.

Every training document starts with a GENRE TAG (a special token) that
tells the model which voice to write in, and ends with <|endoftext|>:

    <|wiki|> Title: Anarchism
    Anarchism is a political philosophy and movement that ...<|endoftext|>
    <|review|> Category: Pet Supplies
    Product: Hurtta Pet Collection 14-Inch Padded Y-Harness, Pink
    Rating: 2/5
    Runs small
    The harness looked great but ...<|endoftext|>

That's the whole idea behind "control codes" (CTRL, Keskar et al. 2019):
the tag is just one more token the model learns to condition on, the
same way the Stage 3 model learned to condition on "Summary:". At
generation time you can hand it any tag with any title, including ones
that don't belong together - a recipe for "The French Revolution".

THE PIPELINE (each stage skips work that's already done, so re-running
after an interruption picks up where it stopped)

  download   fetch the raw sources into ~/tlm-data/mixtape-raw/ (tens of
             GB). RecipeNLG must be downloaded by hand; the script says
             where to put it.
  format     clean each source and write tagged documents to
             <dataset>/formatted/<genre>.txt. Long texts are cut into
             chunks of <= ~850 tokens, each repeating the tag and title,
             so a 1,024-token training window always contains a tag.
  tokenizer  train a 16,384-token BPE on a sample of all five genres,
             with <|endoftext|> and the five tags as special tokens.
  encode     tokenize every document (in parallel, one process per core)
             into <dataset>/tokens/<genre>.bin plus document lengths.
  mix        hold out a validation set per genre, take each genre's
             share of the ~2.5B training tokens, shuffle every document
             together, and write train.bin, val.bin and val_<genre>.bin.
  inspect    print a few random documents per genre and the dataset
             stats (not part of the default run).

Run from the tlm/ project root:

    python data\\prepare_mixtape.py                  # all stages
    python data\\prepare_mixtape.py format encode    # just these stages
    python data\\prepare_mixtape.py --small          # ~5M-token test run
    python data\\prepare_mixtape.py inspect          # look at the result

<dataset> is ~/tlm-data/mixtape (or mixtape-small with --small); set
TLM_DATA_DIR to put everything somewhere else. Changing the mixture is
cheap: edit TRAIN_TOKENS, delete train.bin, and re-run "mix".
"""

import argparse
import csv
import io
import itertools
import json
import os
import random
import re
import sys
import time
import unicodedata
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import numpy as np
import requests

# Use the Windows certificate store for TLS (see prepare_stories.py).
import truststore
truststore.inject_into_ssl()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tlm.config import DATA_ROOT, data_dir                # noqa: E402
from tlm.tokenizer import BPETokenizer, load_tokenizer    # noqa: E402
from prepare_stories import download                      # noqa: E402

# ------------------------------------------------------------- the recipe
GENRES = ("wiki", "book", "review", "recipe", "story")
EOT = "<|endoftext|>"
TAG = {g: f"<|{g}|>" for g in GENRES}
SPECIAL_TOKENS = [EOT] + [TAG[g] for g in GENRES]
VOCAB_SIZE = 16384

# The mixture: training tokens per genre, ~2.5B in all. Wikipedia and
# books are the backbone (the model needs to know about things for the
# genre collisions to be funny); the three playful formats get far more
# than their natural share. A source smaller than its share is repeated -
# up to ~4 passes is nearly as good as fresh data. See PLAN-stage4.md.
TRAIN_TOKENS = {"wiki": 750e6, "book": 625e6, "review": 500e6,
                "recipe": 375e6, "story": 250e6}
VAL_TOKENS = 1.5e6          # held out per genre, never trained on
SMALL_SCALE = 1 / 500       # --small: the same pipeline on 1/500th

CHARS_PER_TOKEN = 4.2       # rough; only used to size the format stage
FORMAT_MARGIN = 1.25        # (mix counts real tokens, so over-format)
MAX_CHUNK_CHARS = 3500      # ~850 tokens: chunk + header fits 1,024
BPE_SAMPLE_CHARS = 10_000_000   # tokenizer training text per genre
BPE_MIN_COUNT = 2
WORKERS = max(1, min(16, (os.cpu_count() or 4) - 2))
SEED = 1337

RAW = os.path.join(DATA_ROOT, "mixtape-raw")   # shared by full & --small

# ------------------------------------------------------------- the sources
HF = "https://huggingface.co/datasets/"
WIKI_URL = (HF + "wikimedia/wikipedia/resolve/main/20231101.en/"
            "train-{:05d}-of-00041.parquet")
WIKI_N_SHARDS = 41
WIKI_SHARDS = 10            # ~25% of English Wikipedia: ~3.4GB of text
PG19 = "https://storage.googleapis.com/deepmind-gutenberg/"
PG19_LIST = HF + "deepmind/pg19/resolve/main/data/train_files.txt"
AMAZON = HF + "McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/"
REVIEW_CATEGORIES = ("Pet_Supplies", "Toys_and_Games",
                     "Grocery_and_Gourmet_Food")
REVIEW_CATEGORIES_SMALL = ("All_Beauty",)
WP_API = "https://huggingface.co/api/datasets/euclaise/writingprompts/tree/main/data"
WP_URL = HF + "euclaise/writingprompts/resolve/main/"
RECIPE_DIR = os.path.join(RAW, "recipenlg")
RECIPE_HELP = f"""
RecipeNLG has to be downloaded by hand (its site asks you to accept the
terms of use first):
  1. go to https://recipenlg.cs.put.poznan.pl/
  2. accept the terms and download the dataset (dataset.zip, ~2 GB)
  3. put the zip (or the full_dataset.csv inside it) in
       {RECIPE_DIR}
Then re-run this script; the other genres don't wait for it until the
tokenizer stage, which needs all five.
"""


# ------------------------------------------------------------ text cleanup
# Everything is reduced to plain ASCII. Real-world text is full of curly
# quotes, dashes, accented letters, emoji and whole other alphabets, and
# a from-scratch tokenizer would spend part of its 16k vocabulary on every
# distinct character it sees. Mapping the common ones to ASCII equivalents
# keeps the base alphabet at ~95 characters; documents that are mostly
# NOT Latin text (another language) are dropped instead.
_ASCII_MAP = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "«": '"',
    "»": '"', "–": "-", "—": " - ", "―": " - ",
    "−": "-", "…": "...", " ": " ", " ": " ",
    "​": "", "­": "", "°": " degrees ", "½": "1/2",
    "¼": "1/4", "¾": "3/4", "⅓": "1/3", "⅔": "2/3",
    "×": "x", "•": "-", "·": "-", "\t": " ", "\r": "",
})
_CONTROL = re.compile(r"[\x00-\x09\x0b-\x1f\x7f]")


def clean_text(s, max_foreign=0.02):
    """Normalize to tidy ASCII; None if the text looks non-English."""
    s = s.translate(_ASCII_MAP)
    # Characters still outside ASCII after the map above (accents count
    # too, which is fine: a few per page is normal English).
    foreign = len(s) - len(s.encode("ascii", "ignore"))
    if foreign > max_foreign * max(len(s), 1):
        return None
    # NFKD splits accented letters into letter + accent ("e" + "'"); the
    # ASCII encode then drops the accent: cafe, naive, Bronte.
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore")
    s = _CONTROL.sub("", s.decode("ascii"))
    s = s.replace("<|", "<")             # never forge a special token
    s = re.sub(r"[^\S\n]+", " ", s)      # runs of spaces -> one
    s = re.sub(r" ?\n ?", "\n", s)       # no spaces around newlines
    s = re.sub(r"\n{3,}", "\n\n", s)     # at most one blank line
    return s.strip()


def looks_english(s):
    """Cheap language check: common English words should be frequent.
    ("the", "and", "of" and "to" make up ~13% of English running text.)
    """
    n_words = s.count(" ") + 1
    if n_words < 40:
        return True                      # too short to judge
    hits = (s.count(" the ") + s.count(" and ") + s.count(" of ")
            + s.count(" to "))
    return hits / n_words > 0.03


def one_line(s):
    return " ".join(s.split())


def chunk_text(text, max_chars=MAX_CHUNK_CHARS):
    """Cut text into pieces of at most max_chars, at paragraph breaks
    where possible (a giant paragraph is cut at a sentence end)."""
    pieces = []
    for para in text.split("\n"):
        while len(para) > max_chars:
            cut = para.rfind(". ", max_chars // 2, max_chars)
            if cut == -1:
                cut = para.rfind(" ", max_chars // 2, max_chars)
            if cut == -1:
                cut = max_chars - 1
            pieces.append(para[:cut + 1].strip())
            para = para[cut + 1:].strip()
        pieces.append(para)
    out, cur, size = [], [], 0
    for p in pieces:
        if cur and size + len(p) + 1 > max_chars:
            out.append("\n".join(cur).strip())
            cur, size = [], 0
        cur.append(p)
        size += len(p) + 1
    out.append("\n".join(cur).strip())
    return [c for c in out if c]


# ------------------------------------------------------------- downloading
def fetch(url, path):
    """Download url to path unless it's already there. Resumable: the
    bytes go to path.part (continued after a dropped connection, see
    prepare_stories.download) and only a finished file gets the real
    name, so a half-downloaded file is never mistaken for a whole one."""
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"downloading {os.path.basename(path)} ...", flush=True)
    download(url, path + ".part")
    os.replace(path + ".part", path)


def fetch_small(url, path, tries=5):
    """One small file (a book) in a single request, with retries."""
    if os.path.exists(path):
        return os.path.getsize(path)
    for attempt in range(tries):
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            with open(path + ".part", "wb") as f:
                f.write(r.content)
            os.replace(path + ".part", path)
            return len(r.content)
        except requests.exceptions.RequestException:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"download failed: {url}")


def wiki_shards(small):
    """Which Wikipedia shards to use: a fixed random pick, so --small's
    single shard is also the full run's first one."""
    order = np.random.default_rng(SEED).permutation(WIKI_N_SHARDS)
    return [int(i) for i in order[:1 if small else WIKI_SHARDS]]


def wiki_path(i):
    return os.path.join(RAW, "wikipedia", f"train-{i:05d}.parquet")


def pg19_book_order():
    """All PG-19 training books in a fixed random order."""
    path = os.path.join(RAW, "pg19", "train_files.txt")
    fetch(PG19_LIST, path)
    with open(path) as f:
        ids = [line.strip()[len("train/"):-len(".txt")] for line in f
               if line.strip()]
    rng = random.Random(SEED)
    rng.shuffle(ids)
    return ids


def pg19_titles():
    path = os.path.join(RAW, "pg19", "metadata.csv")
    fetch(PG19 + "metadata.csv", path)
    with open(path, encoding="utf-8", newline="") as f:
        return {row[0]: row[1] for row in csv.reader(f) if len(row) >= 2}


def book_path(book_id):
    return os.path.join(RAW, "pg19", "books", f"{book_id}.txt")


def wp_files(small):
    """WritingPrompts parquet files. The full run uses all splits (we make
    our own validation split); --small uses only the small one."""
    path = os.path.join(RAW, "writingprompts", "files.json")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        r = requests.get(WP_API, timeout=60)
        r.raise_for_status()
        with open(path, "w") as f:
            json.dump([i["path"] for i in r.json()
                       if i["path"].endswith(".parquet")], f)
    with open(path) as f:
        files = json.load(f)
    if small:
        files = [p for p in files if "validation" in p]
    return files


def wp_path(remote):
    return os.path.join(RAW, "writingprompts", os.path.basename(remote))


def review_path(category, meta=False):
    name = f"meta_{category}.jsonl" if meta else f"{category}.jsonl"
    return os.path.join(RAW, "amazon", name)


def find_recipe_file():
    if os.path.isdir(RECIPE_DIR):
        for name in sorted(os.listdir(RECIPE_DIR)):
            if name.endswith((".csv", ".zip")):
                return os.path.join(RECIPE_DIR, name)
    return None


def stage_download(genres, small, scale):
    budget = genre_budgets(scale)
    if "wiki" in genres:
        for i in wiki_shards(small):
            fetch(WIKI_URL.format(i), wiki_path(i))
    if "book" in genres:
        # Books are small files; fetch them in random order, 8 at a time,
        # until there's comfortably more raw text than the book budget.
        need = budget["book"] * 1.15
        os.makedirs(os.path.dirname(book_path("x")), exist_ok=True)
        pg19_titles()
        order, have, n = pg19_book_order(), 0, 0
        print(f"downloading PG-19 books until ~{need/1e6:,.0f}MB of text "
              f"...", flush=True)
        with ThreadPoolExecutor(8) as ex:
            for start in range(0, len(order), 64):
                ids = order[start:start + 64]
                sizes = ex.map(lambda b: fetch_small(
                    f"{PG19}train/{b}.txt", book_path(b)), ids)
                have += sum(sizes)
                n += len(ids)
                if start % 1024 == 0:
                    print(f"  {n:,} books, {have/1e6:,.0f}MB", flush=True)
                if have >= need:
                    break
        print(f"  {n:,} books, {have/1e6:,.0f}MB", flush=True)
    if "review" in genres:
        cats = REVIEW_CATEGORIES_SMALL if small else REVIEW_CATEGORIES
        for c in cats:
            fetch(AMAZON + f"review_categories/{c}.jsonl", review_path(c))
            fetch(AMAZON + f"meta_categories/meta_{c}.jsonl",
                  review_path(c, meta=True))
    if "story" in genres:
        for remote in wp_files(small):
            fetch(WP_URL + remote, wp_path(remote))
    if "recipe" in genres and not find_recipe_file():
        print(RECIPE_HELP)


# --------------------------------------------------------------- formatting
# One generator per genre, each yielding finished documents:
#   "<|tag|> Header: ...\n<body><|endoftext|>"

WIKI_SKIP_TITLES = ("List of", "Lists of", "Index of", "Outline of",
                    "Glossary of", "Timeline of")
# Sections that are lists of links and citations, not prose. An article
# is cut at the first of these headings (they come at the end).
WIKI_STOP = {"See also", "References", "Notes", "External links",
             "Further reading", "Bibliography", "Sources", "Citations",
             "Footnotes", "Works cited", "Notes and references",
             "References and notes", "Primary sources", "Secondary sources",
             "Explanatory notes", "Discography", "Filmography"}


def tidy_wiki(s):
    """Patch up what stripping non-Latin script leaves in Wikipedia
    openings: 'Piaohao (Traditional Chinese: , literally ...), also known
    as (), ()' -> 'Piaohao (literally ...), also known as'."""
    s = re.sub(r"\b[A-Z][A-Za-z ]{0,30}: ?(?=[,;)])", "", s)  # "Chinese: ,"
    s = re.sub(r"\(\s*(?:[,;]\s*)*\)", "", s)                 # "()"
    s = re.sub(r"\(\s*[,;]\s*", "(", s)                       # "(, x"
    s = re.sub(r"\s*[,;]\s*\)", ")", s)                       # "x ,)"
    s = re.sub(r"(?:,\s*)+(?=,)", "", s)                      # ", , ,"
    s = re.sub(r" +([,;.])", r"\1", s)                        # "x ,"
    return re.sub(r" {2,}", " ", s)


def wiki_docs(small):
    import pyarrow.parquet as pq
    for i in wiki_shards(small):
        for batch in pq.ParquetFile(wiki_path(i)).iter_batches(
                batch_size=2000, columns=["title", "text"]):
            for title, text in zip(batch.column(0).to_pylist(),
                                   batch.column(1).to_pylist()):
                if (title.startswith(WIKI_SKIP_TITLES)
                        or "(disambiguation)" in title
                        or "may refer to" in text[:300]):
                    continue
                lines = text.split("\n")
                for k, line in enumerate(lines):
                    if line.strip() in WIKI_STOP:
                        lines = lines[:k]
                        break
                body = clean_text("\n".join(lines))
                title = clean_text(title)
                if (not body or not title or len(body) < 500
                        or not looks_english(body)):
                    continue
                body = tidy_wiki(body)
                for chunk in chunk_text(body):
                    yield f"{TAG['wiki']} Title: {one_line(title)}\n{chunk}{EOT}"


def book_docs():
    titles = pg19_titles()
    for book_id in pg19_book_order():
        path = book_path(book_id)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read().replace("\r\n", "\n")
        # Gutenberg texts are hard-wrapped at ~70 columns: re-join the
        # lines of each paragraph (blank lines separate paragraphs), and
        # drop the _underscores_ that mark italics.
        text = re.sub(r"(?<=\S)\n(?=\S)", " ", text)
        text = re.sub(r"_([^_\n]+)_", r"\1", text)
        body = clean_text(text, max_foreign=0.01)
        if not body or not looks_english(body):
            continue
        # Skip the front matter (title page, dedication, contents) and the
        # back matter (index, publishers' advertisements): keep from the
        # first paragraph break after the first 2% of the book to the last
        # one before the final 2%.
        n = len(body)
        start = body.find("\n\n", n // 50)
        end = body.rfind("\n\n", 0, n - n // 50)
        if 0 <= start < end:
            body = body[start:end].strip()
        title = one_line(clean_text(titles.get(book_id, "")) or "Untitled")
        for chunk in chunk_text(body):
            yield f"{TAG['book']} Title: {title}\n{chunk}{EOT}"


# Real reviews are ~65% five-star. Keeping every review would teach the
# model that "Rating:" barely matters; these keep-rates even it out
# (roughly 23/12/16/18/30% for 1..5 stars) - and 1- and 2-star reviews are
# where the comedy is.
REVIEW_KEEP = {1: 1.0, 2: 1.0, 3: 1.0, 4: 0.6, 5: 0.2}


def review_titles(category):
    """parent_asin -> product name (first 15 words: Amazon titles are
    keyword soup like 'Dog Bed, Orthopedic, Washable, Large, Grey, ...')."""
    titles = {}
    with open(review_path(category, meta=True), encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            t = clean_text(d.get("title") or "")
            if t and d.get("parent_asin"):
                titles[d["parent_asin"]] = " ".join(t.split()[:15])
    return titles


def review_doc(d, category, titles):
    try:
        rating = int(d.get("rating") or 0)
    except (TypeError, ValueError):
        return None
    product = titles.get(d.get("parent_asin"))
    text = d.get("text") or ""
    if rating not in REVIEW_KEEP or not product or len(text.split()) < 20:
        return None
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = clean_text(text, max_foreign=0.03)
    if not text or not looks_english(text):
        return None
    headline = one_line(clean_text(d.get("title") or "", 0.03) or "")
    return (f"{TAG['review']} Category: {category.replace('_', ' ')}\n"
            f"Product: {product}\nRating: {rating}/5\n{headline}\n"
            f"{text}{EOT}")


def review_docs(small, budget_chars):
    cats = REVIEW_CATEGORIES_SMALL if small else REVIEW_CATEGORIES
    per_cat = budget_chars / len(cats)
    rng = random.Random(SEED)
    for cat in cats:
        titles = review_titles(cat)
        path = review_path(cat)
        # The review files are grouped by user, so stopping once we have
        # enough would sample only the first users in the file. Instead,
        # estimate from the first 20,000 lines how much usable text the
        # whole file holds, and keep each line with the probability that
        # spreads this category's share evenly across all of it.
        got, used = 0.0, 0
        with open(path, encoding="utf-8") as f:
            for line in itertools.islice(f, 20000):
                used += len(line.encode("utf-8"))
                try:
                    d = json.loads(line)
                    doc = review_doc(d, cat, titles)
                except ValueError:
                    continue
                if doc:
                    got += len(doc) * REVIEW_KEEP[int(d["rating"])]
        expected = got / max(used, 1) * os.path.getsize(path)
        keep = min(1.0, per_cat / max(expected, 1))
        print(f"  {cat}: ~{expected/1e6:,.0f}MB usable, keeping "
              f"{keep:.0%} of lines", flush=True)
        n_chars = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                if rng.random() >= keep:
                    continue
                try:
                    d = json.loads(line)
                    doc = review_doc(d, cat, titles)
                except ValueError:
                    continue
                if doc and rng.random() < REVIEW_KEEP[int(d["rating"])]:
                    yield doc
                    n_chars += len(doc)
                    if n_chars >= per_cat:
                        break


def recipe_rows(path):
    csv.field_size_limit(2**31 - 1)
    if path.endswith(".zip"):
        z = zipfile.ZipFile(path)
        name = next(n for n in z.namelist() if n.endswith(".csv"))
        f = io.TextIOWrapper(z.open(name), encoding="utf-8", newline="")
    else:
        f = open(path, encoding="utf-8", newline="")
    with f:
        yield from csv.DictReader(f)


def recipe_docs():
    path = find_recipe_file()
    if not path:
        raise SystemExit(RECIPE_HELP)
    for row in recipe_rows(path):
        # "Gathered" = the ~1.6M recipes RecipeNLG's authors scraped and
        # cleaned themselves; the rest came from an older, messier set.
        if row.get("source") != "Gathered":
            continue
        try:
            ingredients = json.loads(row["ingredients"])
            directions = json.loads(row["directions"])
        except (ValueError, KeyError):
            continue
        title = one_line(clean_text(row.get("title") or "") or "")
        if not title or not ingredients or not directions:
            continue
        body = clean_text(
            "Ingredients:\n" + "\n".join(f"- {one_line(i)}" for i in ingredients)
            + "\nDirections:\n" + "\n".join(
                f"{n}. {one_line(d)}" for n, d in enumerate(directions, 1)),
            max_foreign=0.03)
        if body and len(body) <= MAX_CHUNK_CHARS:
            yield f"{TAG['recipe']} Title: {title}\n{body}{EOT}"


# WritingPrompts prompts start with tags like "[ WP ]" (writing prompt) or
# "[ EU ]" (established universe), and were tokenized with spaces around
# punctuation ("You 've finally ... , and"). Undo both.
_WP_TAGS = re.compile(r"^\s*(?:[\[(]\s*[A-Za-z]{2,4}\s*[\])]\s*)+")


def detokenize(s):
    s = s.replace("``", '"').replace("''", '"')
    s = re.sub(r" (n't|'s|'re|'ve|'ll|'d|'m)\b", r"\1", s)
    s = re.sub(r" ([.,!?;:%)])", r"\1", s)
    s = re.sub(r"([($]) ", r"\1", s)
    s = re.sub(r'" (.*?) "', r'"\1"', s)
    return s


def story_docs(small):
    import pyarrow.parquet as pq
    for remote in wp_files(small):
        table = pq.read_table(wp_path(remote), columns=["prompt", "story"])
        for prompt, story in zip(table.column(0).to_pylist(),
                                 table.column(1).to_pylist()):
            if not prompt or not story or "nsfw" in prompt.lower():
                continue
            prompt = clean_text(detokenize(_WP_TAGS.sub("", prompt)))
            story = clean_text(story.replace("<newline>", "\n"))
            if (not prompt or not story or len(story) < 200
                    or not looks_english(story)):
                continue
            # Long stories are chunked like books, each piece repeating
            # the prompt, so every training window still sees it.
            for chunk in chunk_text(story):
                yield f"{TAG['story']} Prompt: {one_line(prompt)}\n{chunk}{EOT}"


def genre_budgets(scale):
    """How many characters of each genre to format (with a margin)."""
    return {g: (TRAIN_TOKENS[g] + VAL_TOKENS) * scale * CHARS_PER_TOKEN
            * FORMAT_MARGIN for g in GENRES}


def formatted_path(out, genre):
    return os.path.join(out, "formatted", f"{genre}.txt")


def stage_format(out, genres, small, scale):
    budget = genre_budgets(scale)
    for g in genres:
        path = formatted_path(out, g)
        if os.path.exists(path + ".done"):
            print(f"format {g}: already done")
            continue
        if g == "recipe" and not find_recipe_file():
            print(f"format recipe: skipped, file not found\n{RECIPE_HELP}")
            continue
        print(f"format {g} (up to {budget[g]/1e6:,.0f}MB) ...", flush=True)
        # Recipes and stories are smaller than their shares, so the full
        # run formats all of them; reviews stop per category themselves.
        docs = {"wiki": lambda: wiki_docs(small),
                "book": book_docs,
                "review": lambda: review_docs(small, budget["review"]),
                "recipe": recipe_docs,
                "story": lambda: story_docs(small)}[g]()
        limit = budget[g] if g in ("wiki", "book") or small else None
        os.makedirs(os.path.dirname(path), exist_ok=True)
        n_docs = n_chars = 0
        t0 = last = time.time()
        # newline="\n": no Windows CRLF translation - the bytes on disk
        # are exactly the documents.
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for doc in docs:
                f.write(doc)
                n_docs += 1
                n_chars += len(doc)
                if time.time() - last > 30:
                    print(f"  {n_docs:,} docs, {n_chars/1e6:,.0f}MB, "
                          f"{(time.time()-t0)/60:.1f} min", flush=True)
                    last = time.time()
                if limit and n_chars >= limit:
                    break
        with open(path + ".done", "w") as f:
            json.dump({"docs": n_docs, "chars": n_chars}, f)
        print(f"  {g}: {n_docs:,} docs, {n_chars/1e6:,.0f}MB in "
              f"{(time.time()-t0)/60:.1f} min", flush=True)


# ---------------------------------------------------------------- tokenizer
def _count_words(text):
    return BPETokenizer.count_words(text, SPECIAL_TOKENS)


def sample_text(path, n_chars, pieces=50):
    """n_chars of a big file, read as `pieces` slices spread evenly
    through it (the first 10MB of the books file would be ~25 books)."""
    size = os.path.getsize(path)
    per = max(1, n_chars // pieces)
    out = []
    with open(path, "rb") as f:
        for k in range(pieces):
            f.seek(int(k * max(size - per, 0) / max(pieces - 1, 1)))
            out.append(f.read(per).decode("utf-8", errors="ignore"))
    return out


def stage_tokenizer(out, genres, small):
    path = os.path.join(out, "tokenizer.json")
    if os.path.exists(path):
        print("tokenizer: already trained")
        return
    for g in genres:
        if not os.path.exists(formatted_path(out, g) + ".done"):
            raise SystemExit(f"tokenizer: format '{g}' first - the "
                             f"tokenizer must be trained on every genre")
    n = BPE_SAMPLE_CHARS // (10 if small else 1)
    texts = [t for g in genres for t in sample_text(formatted_path(out, g), n)]
    print(f"tokenizer: counting words in {len(genres)} x {n/1e6:.0f}MB ...",
          flush=True)
    t0 = time.time()
    counts = Counter()
    with Pool(WORKERS) as pool:
        for c in pool.imap_unordered(_count_words, texts):
            counts.update(c)
    print(f"  {len(counts):,} unique words; training {VOCAB_SIZE:,}-token "
          f"BPE (words seen < {BPE_MIN_COUNT}x ignored) ...", flush=True)
    tok = BPETokenizer.train_fast(counts, VOCAB_SIZE, SPECIAL_TOKENS,
                                  min_count=BPE_MIN_COUNT)
    for t in SPECIAL_TOKENS:     # each tag must be ONE token, never split
        assert tok.encode(t) == [tok.vocab[t]], t
    tok.save(path)
    print(f"  done in {(time.time()-t0)/60:.1f} min: vocab "
          f"{tok.vocab_size:,}, special ids "
          f"{[tok.vocab[t] for t in SPECIAL_TOKENS]}", flush=True)


# ----------------------------------------------------------------- encoding
_TOK = None


def _init_worker(tok_path):
    global _TOK
    _TOK = load_tokenizer(tok_path)


def _encode_docs(docs):
    ids, lens = [], []
    for d in docs:
        t = _TOK.encode(d)
        ids.extend(t)
        lens.append(len(t))
    # The word cache makes encoding fast, but real English has a long tail
    # of rare words (names, numbers): cap it so memory stays bounded.
    if len(_TOK._word_cache) > 200_000:
        _TOK._word_cache.clear()
    return np.array(ids, dtype=np.uint16), np.array(lens, dtype=np.int32)


def doc_batches(path, batch_chars=2_000_000):
    """Read a formatted file in blocks and yield lists of documents."""
    rest, batch, size = "", [], 0
    with open(path, encoding="utf-8", newline="\n") as f:
        while True:
            block = f.read(8_000_000)
            if not block:
                break
            parts = (rest + block).split(EOT)
            rest = parts.pop()           # an incomplete last document
            for p in parts:
                batch.append(p + EOT)
                size += len(p)
                if size >= batch_chars:
                    yield batch
                    batch, size = [], 0
    if batch:
        yield batch


def tokens_paths(out, genre):
    base = os.path.join(out, "tokens", genre)
    return base + ".bin", base + ".lens.npy"


def stage_encode(out, genres):
    tok_path = os.path.join(out, "tokenizer.json")
    if not os.path.exists(tok_path):
        raise SystemExit("encode: train the tokenizer first")
    for g in genres:
        bin_path, lens_path = tokens_paths(out, g)
        if os.path.exists(lens_path):
            print(f"encode {g}: already done")
            continue
        with open(formatted_path(out, g) + ".done") as f:
            total_chars = json.load(f)["chars"]
        os.makedirs(os.path.dirname(bin_path), exist_ok=True)
        print(f"encode {g} ({WORKERS} processes) ...", flush=True)
        t0 = last = time.time()
        n_tok, n_chars, all_lens = 0, 0, []
        batches = doc_batches(formatted_path(out, g))
        with Pool(WORKERS, _init_worker, (tok_path,)) as pool, \
                open(bin_path + ".part", "wb") as f:
            # Hand out a few batches per process at a time (Pool.imap
            # would read the whole multi-GB file into memory up front).
            while True:
                window = list(itertools.islice(batches, WORKERS * 2))
                if not window:
                    break
                n_chars += sum(len(d) for b in window for d in b)
                for ids, lens in pool.map(_encode_docs, window):
                    ids.tofile(f)
                    all_lens.append(lens)
                    n_tok += len(ids)
                if time.time() - last > 30:
                    print(f"  {n_chars/1e6:,.0f}/{total_chars/1e6:,.0f}MB, "
                          f"{n_tok/1e6:,.1f}M tokens, "
                          f"{(time.time()-t0)/60:.1f} min", flush=True)
                    last = time.time()
        os.replace(bin_path + ".part", bin_path)
        np.save(lens_path, np.concatenate(all_lens))   # last = "done" mark
        print(f"  {g}: {n_tok/1e6:,.1f}M tokens "
              f"({n_chars/max(n_tok, 1):.2f} chars/token) in "
              f"{(time.time()-t0)/60:.1f} min", flush=True)


# ------------------------------------------------------------------- mixing
def stage_mix(out, genres, scale):
    train_path = os.path.join(out, "train.bin")
    if os.path.exists(train_path):
        print("mix: train.bin exists (delete it to re-mix)")
        return
    rng = np.random.default_rng(SEED)
    sources, picks, stats = [], [], {}
    for gi, g in enumerate(genres):
        bin_path, lens_path = tokens_paths(out, g)
        if not os.path.exists(lens_path):
            raise SystemExit(f"mix: encode '{g}' first")
        ids = np.memmap(bin_path, dtype=np.uint16, mode="r")
        lens = np.load(lens_path).astype(np.int64)
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
        sources.append((ids, starts, lens))

        # Validation: random whole documents, held out completely.
        order = rng.permutation(len(lens))
        n_val = int(np.searchsorted(np.cumsum(lens[order]),
                                    max(VAL_TOKENS * scale, 20000))) + 1
        val_docs, pool = order[:n_val], order[n_val:]
        with open(os.path.join(out, f"val_{g}.bin"), "wb") as f:
            for d in val_docs:
                ids[starts[d]:starts[d] + lens[d]].tofile(f)

        # Training: this genre's share, in whole documents. If the source
        # is smaller than its share, every document repeats (passes > 1).
        target = TRAIN_TOKENS[g] * scale
        avail = int(lens[pool].sum())
        full_passes = int(target // avail)
        rest = target - full_passes * avail
        k = int(np.searchsorted(np.cumsum(lens[pool]), rest))
        chosen = np.concatenate([pool] * full_passes + [pool[:k]])
        picks.append(np.stack([np.full(len(chosen), gi), chosen], axis=1))
        stats[g] = {"docs": int(len(lens)),
                    "train_tokens": int(lens[chosen].sum()),
                    "val_tokens": int(lens[val_docs].sum()),
                    "passes": round(target / avail, 2),
                    "avg_doc_tokens": round(float(lens.mean()), 1)}

    # Shuffle every chosen document of every genre together, then stream
    # them into train.bin in that order.
    picks = np.concatenate(picks)
    picks = picks[rng.permutation(len(picks))]
    print(f"mix: writing {len(picks):,} documents ...", flush=True)
    t0 = time.time()
    with open(train_path + ".part", "wb") as f:
        buf, n_buf = [], 0
        for gi, d in picks:
            ids, starts, lens = sources[gi]
            buf.append(ids[starts[d]:starts[d] + lens[d]])
            n_buf += lens[d]
            if n_buf > 16_000_000:
                np.concatenate(buf).tofile(f)
                buf, n_buf = [], 0
        if buf:
            np.concatenate(buf).tofile(f)
    os.replace(train_path + ".part", train_path)
    with open(os.path.join(out, "val.bin"), "wb") as f:
        for g in genres:
            np.fromfile(os.path.join(out, f"val_{g}.bin"),
                        dtype=np.uint16).tofile(f)
    with open(os.path.join(out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  done in {(time.time()-t0)/60:.1f} min")
    print_stats(stats)


def print_stats(stats):
    total = sum(s["train_tokens"] for s in stats.values())
    print(f"\n  {'genre':8s} {'docs':>11s} {'train tokens':>14s} {'share':>6s}"
          f" {'passes':>7s} {'val tokens':>11s} {'tok/doc':>8s}")
    for g, s in stats.items():
        print(f"  {g:8s} {s['docs']:11,d} {s['train_tokens']:14,d} "
              f"{s['train_tokens']/total:6.0%} {s['passes']:7.2f} "
              f"{s['val_tokens']:11,d} {s['avg_doc_tokens']:8.0f}")
    print(f"  {'total':8s} {'':11s} {total:14,d}\n")


# ------------------------------------------------------------------ inspect
def stage_inspect(out, genres):
    rng = random.Random()
    tok_path = os.path.join(out, "tokenizer.json")
    if os.path.exists(tok_path):
        tok = load_tokenizer(tok_path)
        print("special tokens:", {t: tok.encode(t) for t in SPECIAL_TOKENS})
    for g in genres:
        path = formatted_path(out, g)
        if not os.path.exists(path):
            continue
        docs = [d for b in itertools.islice(doc_batches(path), 5) for d in b]
        print(f"\n{'=' * 25} {g}: 3 random documents (of the first "
              f"{len(docs):,}) {'=' * 25}")
        for d in rng.sample(docs, min(3, len(docs))):
            print(d[:700] + (" [...]" if len(d) > 700 else ""))
            print("-" * 60)
    stats_path = os.path.join(out, "stats.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            print_stats(json.load(f))


# --------------------------------------------------------------------- main
STAGES = ("download", "format", "tokenizer", "encode", "mix")


def main():
    parser = argparse.ArgumentParser(description="Prepare the mixtape data")
    parser.add_argument("stages", nargs="*",
                        help=f"any of {', '.join(STAGES)}, inspect "
                             f"(default: all except inspect)")
    parser.add_argument("--small", action="store_true",
                        help="~5M-token test run into mixtape-small")
    parser.add_argument("--genres", default=",".join(GENRES),
                        help="comma-separated subset, for testing")
    args = parser.parse_args()
    stages = args.stages or list(STAGES)
    for s in stages:
        if s not in STAGES + ("inspect",):
            parser.error(f"unknown stage {s!r}")
    genres = [g for g in args.genres.split(",") if g]
    for g in genres:
        if g not in GENRES:
            parser.error(f"unknown genre {g!r}")
    scale = SMALL_SCALE if args.small else 1.0
    out = data_dir("mixtape-small" if args.small else "mixtape")
    os.makedirs(out, exist_ok=True)
    print(f"dataset -> {out}\nraw downloads -> {RAW}\n")

    if "download" in stages:
        stage_download(genres, args.small, scale)
    if "format" in stages:
        stage_format(out, genres, args.small, scale)
    if "tokenizer" in stages:
        stage_tokenizer(out, genres, args.small)
    if "encode" in stages:
        stage_encode(out, genres)
    if "mix" in stages:
        stage_mix(out, genres, scale)
    if "inspect" in stages:
        stage_inspect(out, genres)


if __name__ == "__main__":
    main()
