"""
Write a sheet of samples from one checkpoint, or a BLIND comparison of two,
on fixed prompts and fixed random seeds.

- One checkpoint: a plain sheet, e.g. for picking journal samples.
- Two (--compare): both models write from the same prompt with the same
  seed. Which one appears as "A" is shuffled per item and the key goes to
  a separate file, so the reader can judge before knowing which is which.
  Loss numbers say which model predicts text better; only reading says
  which one is funnier.

Prompts: the training prompt bank (MIXTAPE_PROMPTS in config.py) plus the
first --ideas ideas per genre from tlm/ideas.json, turned into documents
the way the Mixtape app does. Sampling also matches the app: temperature
0.8 ("groovy"), top-k 50, stopping at <|endoftext|> or a new genre tag.
Runs on the CPU, so it can run alongside training.

    python tools\\sample_sheet.py CKPT --out sheet.md
    python tools\\sample_sheet.py CKPT --compare CKPT2 --out blind.md
"""

import argparse
import json
import os
import random
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from tlm.config import (MIXTAPE_PROMPTS, config_from_checkpoint,  # noqa: E402
                        data_dir)
from tlm.gui import build_prompt, to_ascii  # noqa: E402
from tlm.model import TLM  # noqa: E402
from tlm.tokenizer import load_tokenizer  # noqa: E402

GENRES = ["wiki", "book", "review", "recipe", "story"]


def prompts(n_ideas):
    out = list(MIXTAPE_PROMPTS)
    ideas = json.load(open(os.path.join(REPO, "tlm", "ideas.json"),
                           encoding="utf-8"))
    for g in GENRES:
        for idea in [i for i in ideas[g]
                     if not any(i["text"] in p for p in out)][:n_ideas]:
            text = to_ascii(idea["text"])
            f = {"title": text, "prompt": text, "product": text,
                 "category": idea.get("category", ""),
                 "rating": idea.get("rating", 3)}
            out.append(build_prompt(g, f))
    return out


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = TLM(config_from_checkpoint(ck["config"]))
    model.load_state_dict(ck["model"])
    return model.eval(), ck["step"]


def write(model, tok, prompt, seed, max_new):
    torch.manual_seed(seed)
    ctx = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    specials = {tok.vocab[s] for s in tok.special_tokens}
    out = []
    for t in model.generate_stream(ctx, max_new, temperature=0.8, top_k=50,
                                   stop_token=tok.eot_id):
        if int(t) in specials:
            break
        out.append(int(t))
    return tok.decode(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("checkpoint")
    ap.add_argument("--compare", metavar="CKPT2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ideas", type=int, default=3,
                    help="ideas per genre from tlm/ideas.json")
    ap.add_argument("--seeds", type=int, default=2, help="samples per prompt")
    ap.add_argument("--tokens", type=int, default=250)
    args = ap.parse_args()
    torch.set_num_threads(8)
    tok = load_tokenizer(os.path.join(data_dir("mixtape"), "tokenizer.json"))
    paths = [args.checkpoint] + ([args.compare] if args.compare else [])
    models = [load(p) for p in paths]
    names = [f"{os.path.basename(p)} (step {step:,})"
             for p, (_, step) in zip(paths, models)]
    items = [(p, s) for p in prompts(args.ideas) for s in range(args.seeds)]
    shuffle = random.Random(1234)
    key, lines, t0 = {}, [], time.time()

    title = ("Blind comparison" if args.compare else f"Samples: {names[0]}")
    lines += [f"# {title}", "",
              f"Temperature 0.8, top-k 50, up to {args.tokens} tokens; "
              f"{len(items)} items, fixed seeds.", ""]
    if args.compare:
        lines += ["For each item, mark which version you prefer (funnier, "
                  "more fun to read): **A**, **B** or **tie**. The key is "
                  "in the `.key.json` file next to this one; look after.", ""]
    for n, (prompt, s) in enumerate(items, 1):
        seed = 1000 * n + s
        first = prompt.strip().replace("\n", " | ")
        lines += [f"## {n}. {first}", ""]
        texts = [write(m, tok, prompt, seed, args.tokens) for m, _ in models]
        if args.compare:
            order = [0, 1]
            shuffle.shuffle(order)
            key[n] = {"A": names[order[0]], "B": names[order[1]]}
            for label, i in zip("AB", order):
                lines += [f"**{label}**", "", "```text", prompt + texts[i],
                          "```", ""]
            lines += ["Prefer: A / B / tie", ""]
        else:
            lines += ["```text", prompt + texts[0], "```", ""]
        print(f"{n}/{len(items)} done ({time.time() - t0:.0f}s)", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    if args.compare:
        with open(os.path.splitext(args.out)[0] + ".key.json", "w") as f:
            json.dump(key, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
