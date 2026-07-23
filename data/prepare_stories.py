"""
Stage 2 data prep: TinyStories, BPE-tokenized.

TinyStories (Eldan & Li, 2023, "How Small Can Language Models Be and
Still Speak Coherent English?") is a corpus of ~2GB of simple children's
stories, released under the open CDLA-Sharing-1.0 license precisely so
people can train tiny models on it. Its restricted vocabulary and simple
grammar are what make a ~9M-parameter model able to produce coherent
stories - the dataset is doing half the magic.

This script:
  1. Downloads TinyStories-train.txt (~1.9GB) and TinyStories-valid.txt
     (~19MB) from HuggingFace, with resume support for flaky connections.
  2. Trains a 4096-token BPETokenizer on a 10MB sample of the text.
     (BPE statistics converge quickly; training on all 1.9GB would take
     hours in pure Python for nearly identical merges.)
  3. Encodes both files to train.bin / val.bin (uint16), streaming in
     chunks so we never hold the whole corpus in RAM.

Expect step 3 to take a while in pure Python (an hour or two) - run it
before the weekend. The word cache in BPETokenizer is what makes it
feasible at all: TinyStories reuses a small vocabulary of words millions
of times, and each unique word is only BPE-merged once.

Run from the tlm/ project root:  python data/prepare_stories.py
"""

import os
import sys
import time

import numpy as np
import requests

# Use the Windows certificate store for TLS instead of Python's bundled
# certificates (fixes CERTIFICATE_VERIFY_FAILED on machines where
# antivirus or a proxy inspects HTTPS traffic).
import truststore
truststore.inject_into_ssl()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tlm.tokenizer import BPETokenizer  # noqa: E402

BASE = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
FILES = {"train": "TinyStories-train.txt", "val": "TinyStories-valid.txt"}
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stories")
VOCAB_SIZE = 4096
BPE_SAMPLE_BYTES = 10 * 1024 * 1024   # train BPE on the first 10MB
CHUNK_CHARS = 4 * 1024 * 1024         # encode in 4M-char chunks


def _download_attempt(url, path):
    """One download attempt with resume: if a partial file exists, ask the
    server for only the remaining bytes (HTTP Range header)."""
    done = os.path.getsize(path) if os.path.exists(path) else 0
    headers = {"Range": f"bytes={done}-"} if done else {}
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        if r.status_code == 416:   # "range not satisfiable" = already done
            return
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) + done
        # 206 = server honored the Range request, so append; anything else
        # means it sent the whole file again, so start over.
        mode = "ab" if done and r.status_code == 206 else "wb"
        t0 = time.time()
        with open(path, mode) as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                done += len(chunk)
                if time.time() - t0 > 5:
                    pct = 100 * done / total if total else 0
                    print(f"  {done/1e6:,.0f}MB / {total/1e6:,.0f}MB "
                          f"({pct:.0f}%)", flush=True)
                    t0 = time.time()


def download(url, path, max_retries=30):
    """Download with automatic retry. A dropped connection mid-download
    (common on a ~2GB file) just triggers another attempt, and because
    each attempt resumes from the bytes already on disk, no progress is
    ever lost."""
    for attempt in range(1, max_retries + 1):
        try:
            _download_attempt(url, path)
            return
        except requests.exceptions.RequestException as e:
            print(f"  connection dropped ({type(e).__name__}); "
                  f"resuming in 10s (attempt {attempt}/{max_retries})",
                  flush=True)
            time.sleep(10)
    raise RuntimeError(f"download failed after {max_retries} attempts: {url}")


def encode_file(tokenizer, txt_path, bin_path):
    """Stream-encode a text file to a uint16 .bin, chunk by chunk.

    We append each encoded chunk straight to the output file, so peak
    memory is one chunk of text plus its tokens - never the whole corpus.
    Chunk boundaries can split a word in two (costing a slightly
    suboptimal tokenization for that one word); at 4M chars/chunk that's
    ~500 words out of ~2 billion characters - noise.
    """
    n_tokens = 0
    t0 = time.time()
    size = os.path.getsize(txt_path)
    with open(txt_path, encoding="utf-8") as fin, open(bin_path, "wb") as fout:
        while True:
            chunk = fin.read(CHUNK_CHARS)
            if not chunk:
                break
            ids = np.array(tokenizer.encode(chunk), dtype=np.uint16)
            ids.tofile(fout)
            n_tokens += len(ids)
            print(f"  {fin.tell()/1e6:,.0f}/{size/1e6:,.0f}MB, "
                  f"{n_tokens/1e6:,.1f}M tokens, "
                  f"{(time.time()-t0)/60:.1f} min", flush=True)
    return n_tokens


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # -- 1: download ---------------------------------------------------------
    paths = {}
    for split, fname in FILES.items():
        paths[split] = os.path.join(OUT_DIR, fname)
        print(f"downloading {fname} ...")
        download(BASE + fname, paths[split])

    # -- 2: train the BPE tokenizer on a sample ------------------------------
    tok_path = os.path.join(OUT_DIR, "tokenizer.json")
    if os.path.exists(tok_path):
        tokenizer = BPETokenizer.load(tok_path)
        print(f"tokenizer already trained ({tokenizer.vocab_size} tokens)")
    else:
        with open(paths["train"], encoding="utf-8") as f:
            sample = f.read(BPE_SAMPLE_BYTES)
        print(f"training BPE (vocab {VOCAB_SIZE}) on "
              f"{len(sample)/1e6:.0f}MB sample ...")
        t0 = time.time()
        tokenizer = BPETokenizer.train(sample, VOCAB_SIZE)
        tokenizer.save(tok_path)
        print(f"BPE trained in {(time.time()-t0)/60:.1f} min")
        # Show off what it learned: the last-learned (most 'advanced')
        # merges are usually whole common words.
        tail = [a + b for a, b in tokenizer.merges[-10:]]
        print(f"final merges learned: {tail}")

    # -- 3: encode both splits to .bin ---------------------------------------
    for split in ("val", "train"):     # val first: quick sanity check
        bin_path = os.path.join(OUT_DIR, f"{split}.bin")
        if os.path.exists(bin_path):
            print(f"{split}.bin already exists, skipping")
            continue
        print(f"encoding {split} ...")
        n = encode_file(tokenizer, paths[split], bin_path)
        print(f"{split}.bin: {n/1e6:,.1f}M tokens")

    print("done - now run:  python -m tlm.train --config stories")


if __name__ == "__main__":
    main()
