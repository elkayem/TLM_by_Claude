"""
Stage 3 data prep: TinyStories-Instruct.

The Instruct variant of TinyStories (HF repo: roneneldan/TinyStoriesInstruct)
prepends structured fields to every story, e.g.:

    Features: Dialogue
    Summary: Tom and Anna fly on a big plane to a sunny place.
    Story:
    Tom and Anna are brother and sister. ...
    <|endoftext|>

(Field set and order vary per example: Summary, Features, Words, Random
sentence.) Trained on this, a model learns CONDITIONAL generation: prompt
it with fields of your choosing and it writes a story that complies.
Same loss, same architecture - instruction following comes entirely from
the data format.

This script produces TWO datasets:

  data/instruct/     - for the ~27M from-scratch run (and its pilot).
                       Fresh 4096-token BPE trained on this corpus, with
                       <|endoftext|> as a true SPECIAL TOKEN (one atomic
                       id - see tokenizer.py) so story boundaries are
                       exact and generation can stop itself.

  data/instruct-ft/  - for fine-tuning the existing 7.4M stories model
                       (--config instruct-ft --init-from stories).
                       Encoded with the ORIGINAL stories tokenizer,
                       because a model's embeddings only make sense under
                       the tokenizer it was trained with. Only a slice of
                       the corpus (fine-tuning needs ~35M tokens, not
                       600M), so this encode is quick.

Run from the tlm/ project root:  python data/prepare_instruct.py
(Download is ~2.7GB with auto-retry/resume; the full encode takes hours -
start it and walk away.)
"""

import os
import shutil
import sys
import time

import numpy as np

# Use the Windows certificate store for TLS (see prepare_stories.py).
import truststore
truststore.inject_into_ssl()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tlm.tokenizer import BPETokenizer, load_tokenizer   # noqa: E402
from prepare_stories import download                     # noqa: E402

BASE = ("https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct/"
        "resolve/main/")
FILES = {"train": "TinyStories-Instruct-train.txt",
         "val": "TinyStories-Instruct-valid.txt"}
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(DATA_DIR, "instruct")
FT_DIR = os.path.join(DATA_DIR, "instruct-ft")

VOCAB_SIZE = 4096
EOT = "<|endoftext|>"
BPE_SAMPLE_BYTES = 10 * 1024 * 1024   # train BPE on the first 10MB
CHUNK_CHARS = 4 * 1024 * 1024         # encode in 4M-char chunks
# Fine-tune slice: 4000 steps x 8192 tokens/step = ~33M tokens needed;
# 400MB of text at ~4 chars/token gives ~100M - a comfortable 3x margin,
# at a fraction of the full corpus's encode time.
FT_TRAIN_CHARS = 400 * 1024 * 1024


def encode_file(tokenizer, txt_path, bin_path, limit_chars=None):
    """Stream-encode a text file to a uint16 .bin (see prepare_stories.py),
    optionally stopping after limit_chars characters of input."""
    n_tokens, read = 0, 0
    t0 = time.time()
    size = os.path.getsize(txt_path)
    if limit_chars:
        size = min(size, limit_chars)
    with open(txt_path, encoding="utf-8") as fin, open(bin_path, "wb") as fout:
        while True:
            take = CHUNK_CHARS
            if limit_chars is not None:
                take = min(take, limit_chars - read)
                if take <= 0:
                    break
            chunk = fin.read(take)
            if not chunk:
                break
            read += len(chunk)
            ids = np.array(tokenizer.encode(chunk), dtype=np.uint16)
            ids.tofile(fout)
            n_tokens += len(ids)
            print(f"  {read/1e6:,.0f}/{size/1e6:,.0f}MB, "
                  f"{n_tokens/1e6:,.1f}M tokens, "
                  f"{(time.time()-t0)/60:.1f} min", flush=True)
    return n_tokens


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FT_DIR, exist_ok=True)

    # -- 1: download ---------------------------------------------------------
    paths = {}
    for split, fname in FILES.items():
        paths[split] = os.path.join(OUT_DIR, fname)
        print(f"downloading {fname} ...")
        download(BASE + fname, paths[split])

    # -- 2: train the Stage 3 BPE tokenizer ----------------------------------
    tok_path = os.path.join(OUT_DIR, "tokenizer.json")
    if os.path.exists(tok_path):
        tokenizer = BPETokenizer.load(tok_path)
        print(f"instruct tokenizer already trained "
              f"({tokenizer.vocab_size} tokens)")
    else:
        with open(paths["train"], encoding="utf-8") as f:
            sample = f.read(BPE_SAMPLE_BYTES)
        print(f"training BPE (vocab {VOCAB_SIZE}, special {EOT!r}) ...")
        t0 = time.time()
        tokenizer = BPETokenizer.train(sample, VOCAB_SIZE,
                                       special_tokens=[EOT])
        tokenizer.save(tok_path)
        print(f"BPE trained in {(time.time()-t0)/60:.1f} min; "
              f"eot_id = {tokenizer.eot_id}")

    # -- 3: encode for the from-scratch run ----------------------------------
    for split in ("val", "train"):     # val first: quick sanity check
        bin_path = os.path.join(OUT_DIR, f"{split}.bin")
        if os.path.exists(bin_path):
            print(f"instruct/{split}.bin already exists, skipping")
            continue
        print(f"encoding instruct/{split} ...")
        n = encode_file(tokenizer, paths[split], bin_path)
        print(f"instruct/{split}.bin: {n/1e6:,.1f}M tokens")

    # -- 4: encode the fine-tune slice with the STORIES tokenizer ------------
    stories_tok_path = os.path.join(DATA_DIR, "stories", "tokenizer.json")
    ft_tok_path = os.path.join(FT_DIR, "tokenizer.json")
    if not os.path.exists(ft_tok_path):
        shutil.copyfile(stories_tok_path, ft_tok_path)
    stories_tok = load_tokenizer(ft_tok_path)
    for split, limit in (("val", None), ("train", FT_TRAIN_CHARS)):
        bin_path = os.path.join(FT_DIR, f"{split}.bin")
        if os.path.exists(bin_path):
            print(f"instruct-ft/{split}.bin already exists, skipping")
            continue
        print(f"encoding instruct-ft/{split} (stories tokenizer) ...")
        n = encode_file(stories_tok, paths[split], bin_path,
                        limit_chars=limit)
        print(f"instruct-ft/{split}.bin: {n/1e6:,.1f}M tokens")

    print("done. next steps (see PLAN-stage3.md):")
    print("  1. python -m tlm.train --config instruct-ft --init-from stories")
    print("  2. python -m tlm.train --config instruct-pilot")
    print("  3. python -m tlm.train --config instruct")


if __name__ == "__main__":
    main()
