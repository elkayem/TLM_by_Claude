"""
Tokenizers for TLM.

A language model never sees text. It sees INTEGERS. The tokenizer is the
translation layer between the two, and its design decides what "atoms" the
model reasons over.

Two implementations live here, used by the two stages of the project:

  CharTokenizer - Stage 1. One token per character. Trivially simple, which
                  is the point: while you're learning the transformer itself,
                  the tokenizer should not be part of the mystery.

  BPETokenizer  - Stage 2. Byte-Pair Encoding, the same family of algorithm
                  used by GPT-2/3/4. Learns frequent chunks ("the", "ing",
                  "once upon") as single tokens, so each of the model's 256
                  context slots carries ~4x more text.

Why does this matter? The model's context is a fixed number of SLOTS
(block_size). With a char tokenizer, 256 slots = 256 characters, barely a
paragraph. With BPE at ~4 chars/token, the same 256 slots cover a whole
short story. Same model, 4x more effective memory - bought purely with a
smarter tokenizer.

See docs/01-tokenization.md for the full walkthrough.
"""

import json
import re
from collections import Counter


class CharTokenizer:
    """Maps each distinct character in the corpus to an integer id.

    The 'vocabulary' is just the sorted list of unique characters. For the
    Shakespeare corpus that's 65 symbols: letters, digits, punctuation,
    space, newline.
    """

    def __init__(self, chars=None):
        self.chars = chars or []
        # stoi = "string to int", itos = "int to string" (Karpathy's naming,
        # now traditional in tiny-GPT codebases)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @classmethod
    def train(cls, text):
        """'Training' a char tokenizer is just collecting the unique chars."""
        return cls(sorted(set(text)))

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text):
        # Unknown characters (not seen during training) are silently dropped.
        # Crude, but fine for a model that only ever sees its own corpus.
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    @property
    def eot_id(self):
        # Char tokenizers have no end-of-text token; see BPETokenizer.
        return None

    # -- persistence --------------------------------------------------------
    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": "char", "chars": self.chars}, f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f)["chars"])


class BPETokenizer:
    """Byte-Pair Encoding tokenizer, trained from scratch on our own corpus.

    THE CORE IDEA
    -------------
    Start with a vocabulary of single characters. Then, repeatedly:

      1. Count how often each ADJACENT PAIR of tokens appears in the corpus.
      2. Take the most frequent pair, e.g. ('t', 'h').
      3. Merge it into a brand-new token 'th' and add it to the vocabulary.
      4. Repeat until the vocabulary reaches the target size.

    Early merges discover common digraphs ('th', 'in', 'er'); later merges
    build whole words ('the', 'and', 'once'). The final vocabulary is a
    frequency-adapted compression dictionary FOR THIS SPECIFIC CORPUS.

    To encode new text we replay the merge list in the order it was learned -
    order matters, because later merges build on the results of earlier ones.

    PRE-TOKENIZATION
    ----------------
    Like GPT-2, we first split text into "words" with a regex, and only merge
    WITHIN words. This stops the algorithm learning silly cross-word tokens
    like 'e t' and, crucially, lets us cache: the word "the" is encoded once,
    then every later occurrence is a dictionary lookup. TinyStories repeats
    its small vocabulary of words endlessly, so this cache makes encoding the
    ~2GB corpus feasible in pure Python.
    """

    # Simplified GPT-2-style pattern. Reading it left to right, a "word" is:
    #   ' hello'  - optional leading space + letters (space travels WITH the
    #               word, so ' the' and 'the' are different tokens - this is
    #               how BPE handles spaces without a special symbol)
    #   ' 42'     - optional leading space + digits
    #   ' ...'    - optional leading space + a run of punctuation/symbols
    #   '\n\n'    - runs of whitespace (newlines etc.) that remain
    _WORD_RE = re.compile(r" ?[A-Za-z]+| ?[0-9]+| ?[^A-Za-z0-9\s]+|\s+")

    def __init__(self, merges=None, vocab=None, special_tokens=None):
        # merges: list of pairs in the ORDER they were learned, e.g.
        #   [('t','h'), ('th','e'), ...]
        self.merges = merges or []
        # rank of each merge = its position in that list. Lower rank =
        # learned earlier = higher priority when encoding.
        self.merge_ranks = {tuple(p): i for i, p in enumerate(self.merges)}
        # vocab: token string -> id
        self.vocab = vocab or {}
        self.inv_vocab = {i: t for t, i in self.vocab.items()}
        self._word_cache = {}  # word -> [ids], the cache described above

        # SPECIAL TOKENS, e.g. "<|endoftext|>". A special token is ATOMIC:
        # it always maps to exactly one id, and BPE never sees inside it.
        # Why this matters: without it, "<|endoftext|>" is just 13
        # characters that BPE happens to compress into a few ordinary
        # tokens - the model can learn it *statistically*, but nothing can
        # rely on it exactly. With a real special id, the data pipeline
        # can mark story boundaries unambiguously and generate() can STOP
        # the instant the model emits it. Real LLMs work the same way
        # (chat markers like <|im_start|> are special tokens too).
        self.special_tokens = special_tokens or []
        self._special_re = (
            re.compile("(" + "|".join(re.escape(t)
                                      for t in self.special_tokens) + ")")
            if self.special_tokens else None)

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def eot_id(self):
        """Id of the end-of-text special token, or None if there isn't one."""
        return self.vocab.get("<|endoftext|>")

    # -- training -----------------------------------------------------------
    @classmethod
    def train(cls, text, vocab_size, special_tokens=None, verbose=True):
        """Learn the merge list from a training corpus.

        We never store the corpus token-by-token. Instead we exploit
        pre-tokenization: the corpus is just a bag of (word, count) pairs,
        and all pair-counting is done over unique words weighted by their
        counts. TinyStories has millions of words but only tens of
        thousands of UNIQUE words, so this is thousands of times faster
        than operating on the raw token stream.
        """
        special_tokens = special_tokens or []
        # Cut special tokens out of the training text so merge statistics
        # never see them: they must stay atomic, not become merge fodder.
        for tok in special_tokens:
            text = text.replace(tok, "\n")

        # Corpus as {word: count}, each word stored as a tuple of 1-char
        # tokens, e.g. ' the' -> (' ', 't', 'h', 'e')  [note leading space]
        word_counts = Counter(cls._WORD_RE.findall(text))
        words = {tuple(w): c for w, c in word_counts.items()}

        # Base vocabulary: every distinct character in the corpus.
        vocab = {ch for w in words for ch in w}
        merges = []

        # Reserve room at the end of the vocab for the special tokens.
        while len(vocab) + len(merges) + len(special_tokens) < vocab_size:
            # Step 1: count adjacent pairs across all unique words,
            # weighted by how often each word occurs.
            pairs = Counter()
            for w, count in words.items():
                for a, b in zip(w, w[1:]):
                    pairs[(a, b)] += count
            if not pairs:
                break  # nothing left to merge (tiny corpus)

            # Step 2: the single most frequent pair wins this round.
            (a, b), freq = pairs.most_common(1)[0]

            # Step 3: rewrite every word, replacing (a, b) with the merged
            # token a+b. E.g. with merge ('t','h'):
            #   (' ', 't', 'h', 'e')  ->  (' ', 'th', 'e')
            merged = a + b
            new_words = {}
            for w, count in words.items():
                out, i = [], 0
                while i < len(w):
                    if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                        out.append(merged)
                        i += 2  # consumed both halves of the pair
                    else:
                        out.append(w[i])
                        i += 1
                new_words[tuple(out)] = new_words.get(tuple(out), 0) + count
            words = new_words
            merges.append((a, b))

            if verbose and len(merges) % 200 == 0:
                print(f"  merge {len(merges)}: {(a, b)!r} -> {merged!r} "
                      f"(freq {freq})")

        # Final vocabulary: base characters first (sorted, for determinism),
        # then merged tokens in creation order, then special tokens last.
        tokens = sorted(vocab) + [a + b for a, b in merges] + special_tokens
        return cls(merges=merges,
                   vocab={t: i for i, t in enumerate(tokens)},
                   special_tokens=special_tokens)

    # -- encoding -----------------------------------------------------------
    def _encode_word(self, word):
        """Apply the learned merges to one word.

        Greedy strategy, same as GPT-2: repeatedly find the pair present in
        this word with the LOWEST merge rank (i.e. the merge learned
        earliest) and apply it, until no learned pair remains.
        """
        if word in self._word_cache:
            return self._word_cache[word]

        parts = list(word)  # start from single characters
        while len(parts) > 1:
            # Find which adjacent pair (if any) has the best (lowest) rank.
            best_pair, best_rank = None, None
            for pair in zip(parts, parts[1:]):
                rank = self.merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_pair, best_rank = pair, rank
            if best_pair is None:
                break  # no learned merge applies; word is fully tokenized

            # Apply that one merge everywhere it occurs in this word.
            a, b = best_pair
            out, i = [], 0
            while i < len(parts):
                if i < len(parts) - 1 and parts[i] == a and parts[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(parts[i])
                    i += 1
            parts = out

        # Map final token strings to ids; skip chars unseen in training.
        ids = [self.vocab[p] for p in parts if p in self.vocab]
        self._word_cache[word] = ids
        return ids

    def encode(self, text):
        # First split on special tokens (if any), so each special maps
        # straight to its id and BPE only runs on the text between them.
        segments = (self._special_re.split(text) if self._special_re
                    else [text])
        ids = []
        for seg in segments:
            if seg in self.vocab and seg in self.special_tokens:
                ids.append(self.vocab[seg])
                continue
            for word in self._WORD_RE.findall(seg):
                ids.extend(self._encode_word(word))
        return ids

    def decode(self, ids):
        # Because tokens are literal substrings of the text (spaces
        # included), decoding is pure concatenation.
        return "".join(self.inv_vocab[i] for i in ids)

    # -- persistence --------------------------------------------------------
    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": "bpe",
                       "merges": [list(p) for p in self.merges],
                       "vocab": self.vocab,
                       "special_tokens": self.special_tokens}, f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        # .get(): tokenizers saved before special-token support load fine.
        return cls(merges=[tuple(p) for p in d["merges"]], vocab=d["vocab"],
                   special_tokens=d.get("special_tokens", []))


def load_tokenizer(path):
    """Load whichever tokenizer type was saved at `path`."""
    with open(path, encoding="utf-8") as f:
        kind = json.load(f)["type"]
    return {"char": CharTokenizer, "bpe": BPETokenizer}[kind].load(path)
