"""
Generate text from a trained TLM checkpoint.

Usage examples (from the tlm/ project root):

    python -m tlm.generate --run shakespeare
    python -m tlm.generate --run shakespeare --prompt "ROMEO:" --tokens 400
    python -m tlm.generate --run stories --prompt "Once upon a time" \
        --temperature 0.9 --top-p 0.95

This is deliberately thin: all the interesting machinery (the sampling
loop, temperature, top-k, top-p) lives in TLM.generate() in model.py,
where it is documented in detail. Play with the knobs:

  --temperature 0.2   near-deterministic, repetitive, "safe"
  --temperature 1.0   sampling from the model's raw distribution
  --temperature 1.4   creative to the point of chaos
"""

import argparse
import os

import torch

from .config import TLMConfig, checkpoint_dir
from .model import TLM
from .tokenizer import load_tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Sample from a trained TLM")
    parser.add_argument("--run", default="shakespeare",
                        help="run name (folder under checkpoints/)")
    parser.add_argument("--checkpoint", default="best.pt",
                        help="best.pt or latest.pt")
    parser.add_argument("--prompt", default="\n",
                        help="text to continue from")
    parser.add_argument("--tokens", type=int, default=300,
                        help="number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help="set for reproducible output")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Rebuild the exact model that was trained: the checkpoint stores its
    # config, so architecture and weights always match.
    ckpt_path = os.path.join(checkpoint_dir(args.run), args.checkpoint)
    ckpt = torch.load(ckpt_path, weights_only=True)
    config = TLMConfig(**ckpt["config"])
    model = TLM(config)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tokenizer = load_tokenizer(
        os.path.join(ROOT, "data", config.dataset, "tokenizer.json"))

    print(f"[{args.run}/{args.checkpoint} @ step {ckpt['step']}, "
          f"{model.num_params()/1e6:.2f}M params, "
          f"temp={args.temperature}, top_k={args.top_k}, "
          f"top_p={args.top_p}]\n")

    ids = tokenizer.encode(args.prompt)
    if not ids:  # empty/unknown prompt: fall back to a newline seed
        ids = tokenizer.encode("\n")
    ctx = torch.tensor([ids], dtype=torch.long)   # shape (1, T): batch of 1

    out = model.generate(ctx, max_new_tokens=args.tokens,
                         temperature=args.temperature,
                         top_k=args.top_k, top_p=args.top_p)
    # out still includes the prompt tokens at the front; print it all so
    # the continuation reads naturally.
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
