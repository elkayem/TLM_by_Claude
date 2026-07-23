"""
Stage 1 data prep: Tiny Shakespeare, character-level.

Downloads the ~1.1MB Tiny Shakespeare corpus (public-domain text, hosted
in Andrej Karpathy's char-rnn repo), trains a CharTokenizer on it, and
writes three files to data/shakespeare/:

    tokenizer.json  the character vocabulary
    train.bin       first 90% of the corpus, tokenized, as uint16
    val.bin         last 10%, held out to measure generalization

THE .bin FORMAT
---------------
Just the token ids, back to back, as raw 16-bit unsigned integers - no
header, no structure. Training never needs documents or shuffled files;
it slices random windows out of one long token stream (see get_batch in
train.py). uint16 is enough for any vocab under 65,536, and keeps files
half the size of int32.

Run from the tlm/ project root:  python data/prepare_shakespeare.py
"""

import os
import sys

import numpy as np
import requests

# Use the Windows certificate store for TLS instead of Python's bundled
# certificates (fixes CERTIFICATE_VERIFY_FAILED on machines where
# antivirus or a proxy inspects HTTPS traffic).
import truststore
truststore.inject_into_ssl()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tlm.tokenizer import CharTokenizer  # noqa: E402

URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/"
       "data/tinyshakespeare/input.txt")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "shakespeare")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    raw_path = os.path.join(OUT_DIR, "input.txt")
    if not os.path.exists(raw_path):
        print(f"downloading {URL} ...")
        text = requests.get(URL, timeout=60).text
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        with open(raw_path, encoding="utf-8") as f:
            text = f.read()
    print(f"corpus: {len(text):,} characters")

    # 'Train' the tokenizer = collect the unique characters.
    tokenizer = CharTokenizer.train(text)
    tokenizer.save(os.path.join(OUT_DIR, "tokenizer.json"))
    print(f"vocab: {tokenizer.vocab_size} chars: "
          f"{''.join(tokenizer.chars)!r}")

    # 90/10 train/val split. We split by POSITION in the text (first 90% /
    # last 10%) rather than randomly, so no val text ever appears in
    # training - a random character-level split would leak heavily.
    ids = np.array(tokenizer.encode(text), dtype=np.uint16)
    n = int(0.9 * len(ids))
    ids[:n].tofile(os.path.join(OUT_DIR, "train.bin"))
    ids[n:].tofile(os.path.join(OUT_DIR, "val.bin"))
    print(f"train.bin: {n:,} tokens | val.bin: {len(ids) - n:,} tokens")
    print("done - now run:  python -m tlm.train --config shakespeare")


if __name__ == "__main__":
    main()
