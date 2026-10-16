"""
TLM Mixtape - a small desktop app for playing with the Stage 4 model.

Pick a genre, click one of the suggested ideas (or type your own), press
Generate, and watch the model write, word by word. Launch it from the
desktop shortcut (python tools\\make_shortcut.py creates one) or with:

    python -m tlm.gui                        # the newest mixtape model
    python -m tlm.gui --run mixtape-pilot    # a particular run
    python -m tlm.gui --device cpu

HOW IT TALKS TO THE MODEL
-------------------------
Every genre has a small form (a title; a product and a star rating; a
premise...). The app turns the form into exactly the text the model was
trained on - for a review,

    <|review|> Category: Pet Supplies
    Product: Self-Heating Socks for Cats
    Rating: 1/5

- and the model simply continues it. That's all "conditioning" is: the
genre tag and the header are the start of a document, and the model
writes the most plausible rest of it. Tick "Show what the model sees" to
see the raw text, tag and all.

THE PARTS
---------
  - The idea bank (tlm/ideas.json) is dealt 4 at a time like a shuffled
    deck: no idea repeats until the whole genre has been shown, and the
    deck's position is saved between launches, so it stays fresh.
  - Generation runs on a background thread (TLM.generate_stream, which
    uses a KV cache) and hands each new token to the window through a
    queue - Tkinter may only be touched from its own thread.
  - The model runs on the GPU when it has room (>= 1.5 GB free),
    otherwise on the CPU - so the app never fights a training run.
  - Errors go to a message box and to ~/tlm-data/gui.log (the desktop
    shortcut runs pythonw, which has no console to print to).

Everything here is the Python standard library's Tkinter, plus the model.
"""

import argparse
import json
import logging
import os
import queue
import random
import sys
import threading
import time
import traceback
import unicodedata
import tkinter as tk
from tkinter import messagebox, ttk

import torch

from .config import DATA_ROOT, TLMConfig, checkpoint_dir, data_dir
from .model import TLM
from .tokenizer import load_tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
IDEAS_PATH = os.path.join(HERE, "ideas.json")
ICON_PATH = os.path.join(REPO, "assets", "mixtape.ico")
FAVORITES_PATH = os.path.join(REPO, "favorites.md")
STATE_PATH = os.path.join(DATA_ROOT, "gui_state.json")
LOG_PATH = os.path.join(DATA_ROOT, "gui.log")

# Which trained model to load: the first of these runs that has a best.pt.
RUNS = ("mixtape", "mixtape-pilot", "mixtape-smoke")
LENGTHS = {"short": 150, "medium": 300, "long": 600}   # max new tokens
IDEAS_SHOWN = 4
CATEGORIES = ("Pet Supplies", "Toys and Games", "Grocery and Gourmet Food")

GENRES = {                  # tag -> what the pulldown shows
    "wiki": "<wiki>   encyclopedia article",
    "book": "<book>   old-fashioned novel",
    "review": "<review>   product review",
    "recipe": "<recipe>   recipe",
    "story": "<story>   story from a premise",
}
# Each genre's form: (field, label, kind). The first field is the main one
# - the one an idea fills in.
FIELDS = {
    "wiki": [("title", "Title", "entry")],
    "book": [("title", "Title", "entry"),
             ("opening", "Opening words (optional)", "entry")],
    "review": [("product", "Product", "entry"),
               ("rating", "Rating", "rating"),
               ("category", "Category", "category")],
    "recipe": [("title", "Dish", "entry")],
    "story": [("prompt", "Premise", "entry")],
}


def stars(n):
    return "★" * n + "☆" * (5 - n)          # filled / empty stars


RATINGS = [f"{n}   {stars(n)}" for n in range(1, 6)]


def build_prompt(genre, f):
    """The form, as the start of a training document (see
    data/prepare_mixtape.py for the formats)."""
    if genre == "review":
        return (f"<|review|> Category: {f['category'] or CATEGORIES[0]}\n"
                f"Product: {f['product']}\nRating: {f['rating']}/5\n")
    if genre == "story":
        return f"<|story|> Prompt: {f['prompt']}\n"
    prompt = f"<|{genre}|> Title: {f['title']}\n"
    if genre == "book" and f.get("opening"):
        prompt += f["opening"]       # no newline: the model continues it
    return prompt


def to_ascii(s):
    """The model only knows plain ASCII (its training text was reduced to
    it), so map curly quotes and accents the same way."""
    s = (s.replace("‘", "'").replace("’", "'")
         .replace("“", '"').replace("”", '"')
         .replace("–", "-").replace("—", " - "))
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore")
    return " ".join(s.decode("ascii").split())


# ------------------------------------------------------------- idea deck
class IdeaDeck:
    """Deals ideas like cards from a shuffled deck, one deck per genre.
    The deck order and position live in `state` (saved between runs)."""

    def __init__(self, ideas, state, save):
        self.ideas, self.state, self.save = ideas, state, save

    def deal(self, genre, n):
        pile = self.ideas[genre]
        deck = self.state.setdefault("decks", {}).setdefault(genre, {})
        order, pos = deck.get("order"), deck.get("pos", 0)
        if not order or sorted(order) != list(range(len(pile))):
            order, pos = random.sample(range(len(pile)), len(pile)), 0
        hand = []
        for _ in range(min(n, len(pile))):
            if pos >= len(order):           # deck used up: reshuffle
                order, pos = random.sample(range(len(pile)), len(pile)), 0
            hand.append(pile[order[pos]])
            pos += 1
        deck["order"], deck["pos"] = order, pos
        self.save()
        return hand


# ----------------------------------------------------------- model loading
def find_run(requested):
    for run in ([requested] if requested else RUNS):
        path = os.path.join(checkpoint_dir(run), "best.pt")
        if os.path.exists(path):
            return run, path
    return None, None


def choose_device(requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        if free >= 1.5e9:
            return "cuda"
    return "cpu"


def load_model(requested_run, requested_device):
    """Returns (model, tokenizer, device, description). Runs on a thread."""
    run, path = find_run(requested_run)
    if not run:
        raise FileNotFoundError(
            "No trained mixtape model yet.\n\nTrain one with\n"
            "    python -m tlm.train --config mixtape\n"
            "(or mixtape-pilot for a quick one), then press Reload.")
    ckpt = torch.load(path, weights_only=True, map_location="cpu")
    config = TLMConfig(**ckpt["config"])
    tok = load_tokenizer(os.path.join(data_dir(config.dataset),
                                      "tokenizer.json"))
    if "<|wiki|>" not in tok.vocab:
        raise ValueError(f"'{run}' wasn't trained on the mixtape genres")
    device = choose_device(requested_device)
    model = TLM(config)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    desc = (f"{run} - step {ckpt['step']:,}, val loss {ckpt['best_val']:.3f}"
            f" - {model.num_params()/1e6:.0f}M params on {device.upper()}")
    return model, tok, device, desc


# -------------------------------------------------------------------- app
class MixtapeApp:
    def __init__(self, root, run=None, device="auto",
                 favorites_path=FAVORITES_PATH, state_path=STATE_PATH):
        self.root = root
        self.run, self.device_request = run, device
        self.favorites_path, self.state_path = favorites_path, state_path
        self.model = self.tok = None
        self.model_desc = ""
        self.q = queue.Queue()             # worker thread -> window
        self.stop_event = threading.Event()
        self.busy = False
        with open(IDEAS_PATH, encoding="utf-8") as f:
            self.ideas = {g: v for g, v in json.load(f).items()
                          if g in GENRES}
        self.state = self.load_state()
        self.deck = IdeaDeck(self.ideas, self.state, self.save_state)
        self.vars = {g: {k: tk.StringVar() for k, _, _ in FIELDS[g]}
                     for g in GENRES}
        self.vars["review"]["rating"].set(RATINGS[0])
        self.vars["review"]["category"].set(CATEGORIES[0])
        self.genre = None
        self.build_ui()
        self.set_genre(self.state.get("genre", "wiki"))
        self.reload_model()
        self.root.after(30, self.poll)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    # ---------------------------------------------------------- state
    def load_state(self):
        try:
            with open(self.state_path) as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def save_state(self):
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            with open(self.state_path, "w") as f:
                json.dump(self.state, f)
        except OSError:
            logging.exception("could not save app state")

    # ------------------------------------------------------------- UI
    def build_ui(self):
        root = self.root
        root.title("TLM Mixtape")
        root.geometry("820x720")
        root.minsize(660, 560)
        if os.path.exists(ICON_PATH):
            try:
                root.iconbitmap(ICON_PATH)
            except tk.TclError:
                pass
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure(".", font=("Segoe UI", 10))
        style.configure("Big.TButton", font=("Segoe UI", 10, "bold"))

        main = ttk.Frame(root, padding=14)
        main.pack(fill="both", expand=True)
        main.columnconfigure(1, weight=1)

        ttk.Label(main, text="Genre:").grid(row=0, column=0, sticky="w")
        self.genre_box = ttk.Combobox(main, state="readonly", width=34,
                                      values=list(GENRES.values()))
        self.genre_box.grid(row=0, column=1, sticky="w")
        self.genre_box.bind("<<ComboboxSelected>>", lambda e: self.set_genre(
            list(GENRES)[self.genre_box.current()]))

        ttk.Label(main, text="Ideas:").grid(row=1, column=0, sticky="nw",
                                           pady=(12, 0))
        ideas = ttk.Frame(main)
        ideas.grid(row=1, column=1, sticky="w", pady=(12, 0))
        self.idea_labels = []
        for _ in range(IDEAS_SHOWN):
            lab = tk.Label(ideas, anchor="w", cursor="hand2", fg="#1a5fb4",
                           font=("Segoe UI", 10))
            lab.pack(anchor="w", pady=1)
            lab.bind("<Enter>", lambda e, l=lab: l.config(
                font=("Segoe UI", 10, "underline")))
            lab.bind("<Leave>", lambda e, l=lab: l.config(
                font=("Segoe UI", 10)))
            self.idea_labels.append(lab)
        ttk.Button(ideas, text="More ideas", command=self.deal_ideas
                   ).pack(anchor="w", pady=(6, 0))

        self.fields_frame = ttk.Frame(main)
        self.fields_frame.grid(row=2, column=0, columnspan=2, sticky="ew",
                               pady=(14, 0))
        self.fields_frame.columnconfigure(1, weight=1)

        knobs = ttk.Frame(main)
        knobs.grid(row=3, column=0, columnspan=2, sticky="w", pady=(12, 0))
        ttk.Label(knobs, text="Creativity").pack(side="left")
        self.temperature = tk.DoubleVar(value=0.8)
        ttk.Scale(knobs, from_=0.5, to=1.2, variable=self.temperature,
                  length=170, command=lambda v: self.temp_label.config(
                      text=f"{float(v):.2f}")).pack(side="left", padx=6)
        self.temp_label = ttk.Label(knobs, text="0.80", width=5)
        self.temp_label.pack(side="left")
        ttk.Label(knobs, text="Length").pack(side="left", padx=(18, 6))
        self.length = ttk.Combobox(knobs, state="readonly", width=8,
                                   values=list(LENGTHS))
        self.length.set("medium")
        self.length.pack(side="left")

        buttons = ttk.Frame(main)
        buttons.grid(row=4, column=0, columnspan=2, sticky="w", pady=(12, 0))
        self.gen_btn = ttk.Button(buttons, text="Generate",
                                  style="Big.TButton", command=self.generate)
        self.gen_btn.pack(side="left")
        self.surprise_btn = ttk.Button(buttons, text="Surprise me",
                                       command=self.surprise)
        self.surprise_btn.pack(side="left", padx=8)
        self.stop_btn = ttk.Button(buttons, text="Stop", command=self.stop,
                                   state="disabled")
        self.stop_btn.pack(side="left")

        out = ttk.Frame(main)
        out.grid(row=5, column=0, columnspan=2, sticky="nsew", pady=(12, 0))
        main.rowconfigure(5, weight=1)
        out.columnconfigure(0, weight=1)
        out.rowconfigure(0, weight=1)
        self.out = tk.Text(out, wrap="word", font=("Georgia", 11), padx=12,
                           pady=10, relief="solid", borderwidth=1,
                           state="disabled")
        self.out.grid(row=0, column=0, sticky="nsew")
        self.out.tag_configure("header", foreground="#555555")
        scroll = ttk.Scrollbar(out, command=self.out.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.out.config(yscrollcommand=scroll.set)

        below = ttk.Frame(main)
        below.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(below, text="Save to favorites",
                   command=self.save_favorite).pack(side="left")
        ttk.Button(below, text="Copy", command=self.copy).pack(
            side="left", padx=8)
        self.show_raw = tk.BooleanVar(value=False)
        ttk.Checkbutton(below, text="Show what the model sees",
                        variable=self.show_raw).pack(side="right")

        status = ttk.Frame(main)
        status.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        status.columnconfigure(0, weight=1)
        self.status = ttk.Label(status, text="")
        self.status.grid(row=0, column=0, sticky="w")
        self.model_label = ttk.Label(status, text="", foreground="#666666")
        self.model_label.grid(row=1, column=0, sticky="w")
        self.reload_btn = ttk.Button(status, text="Reload model",
                                     command=self.reload_model)
        self.reload_btn.grid(row=0, column=1, rowspan=2, sticky="e")

    def set_genre(self, genre):
        if genre not in GENRES:
            genre = "wiki"
        self.genre = genre
        self.genre_box.current(list(GENRES).index(genre))
        for w in self.fields_frame.winfo_children():
            w.destroy()
        first = None
        for row, (key, label, kind) in enumerate(FIELDS[genre]):
            ttk.Label(self.fields_frame, text=label + ":").grid(
                row=row, column=0, sticky="w", pady=3, padx=(0, 10))
            var = self.vars[genre][key]
            if kind == "rating":
                w = ttk.Combobox(self.fields_frame, textvariable=var,
                                 values=RATINGS, state="readonly", width=12)
                w.grid(row=row, column=1, sticky="w")
            elif kind == "category":
                w = ttk.Combobox(self.fields_frame, textvariable=var,
                                 values=CATEGORIES, width=30)
                w.grid(row=row, column=1, sticky="w")
            else:
                w = ttk.Entry(self.fields_frame, textvariable=var)
                w.grid(row=row, column=1, sticky="ew")
                w.bind("<Return>", lambda e: self.generate())
            first = first or w
        first.focus_set()
        self.state["genre"] = genre
        self.deal_ideas()

    def deal_ideas(self):
        hand = self.deck.deal(self.genre, IDEAS_SHOWN)
        for lab, idea in zip(self.idea_labels, hand + [None] * IDEAS_SHOWN):
            if idea is None:
                lab.config(text="")
                continue
            text = idea["text"]
            if self.genre == "review":
                text += f"   {stars(idea['rating'])}"
            if idea.get("mixup"):
                text += "      (mix-up)"
            lab.config(text=text)
            lab.bind("<Button-1>", lambda e, i=idea: self.use_idea(i))

    def use_idea(self, idea):
        fields = self.vars[self.genre]
        fields[FIELDS[self.genre][0][0]].set(idea["text"])
        if self.genre == "review":
            fields["rating"].set(RATINGS[idea["rating"] - 1])
            fields["category"].set(idea.get("category", CATEGORIES[0]))
        if self.genre == "book":
            fields["opening"].set("")

    # -------------------------------------------------------- output pane
    def write(self, text, tag=None):
        self.out.config(state="normal")
        self.out.insert("end", text, tag)
        self.out.see("end")
        self.out.config(state="disabled")

    def clear(self):
        self.out.config(state="normal")
        self.out.delete("1.0", "end")
        self.out.config(state="disabled")

    def set_status(self, text):
        self.status.config(text=text)

    # ------------------------------------------------------------ model
    def reload_model(self):
        if self.busy:
            return
        self.model = None
        self.reload_btn.config(state="disabled")
        self.set_status("Loading the model ...")

        def work():
            try:
                self.q.put(("model", load_model(self.run, self.device_request)))
            except Exception as e:          # noqa: BLE001 - shown to user
                logging.exception("model load failed")
                self.q.put(("model_error", str(e)))
        threading.Thread(target=work, daemon=True).start()

    # --------------------------------------------------------- generate
    def generate(self):
        if self.busy:
            return
        if self.model is None:
            messagebox.showinfo("TLM Mixtape", "The model isn't loaded yet."
                                "\n\n" + self.model_label.cget("text"))
            return
        fields = {k: to_ascii(v.get()) for k, v in self.vars[self.genre].items()}
        if self.genre == "review":
            fields["rating"] = int(self.vars["review"]["rating"].get()[0])
        key, label, _ = FIELDS[self.genre][0]
        if not fields[key]:
            messagebox.showinfo("TLM Mixtape",
                                f"Type a {label.lower()} or click an idea.")
            return
        prompt = build_prompt(self.genre, fields)
        shown = prompt if self.show_raw.get() else prompt.split(" ", 1)[1]
        self.clear()
        self.write(shown, "header")
        self.last = {"genre": self.genre, "main": fields[key],
                     "temperature": self.temperature.get()}
        self.busy = True
        self.stop_event.clear()
        for b in (self.gen_btn, self.surprise_btn, self.reload_btn):
            b.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.set_status("Writing ...")
        ids = self.tok.encode(prompt)
        max_new = LENGTHS[self.length.get()]
        temperature = self.temperature.get()
        threading.Thread(target=self.worker, daemon=True,
                         args=(ids, max_new, temperature)).start()

    def worker(self, ids, max_new, temperature):
        """Background thread: sample tokens, hand each one to the window."""
        try:
            ctx = torch.tensor([ids], dtype=torch.long,
                               device=next(self.model.parameters()).device)
            # A genre tag means the model has started a NEW document:
            # treat it like <|endoftext|> and stop there.
            specials = {self.tok.vocab[s] for s in self.tok.special_tokens}
            n, t0 = 0, time.time()
            for t in self.model.generate_stream(
                    ctx, max_new, temperature=temperature, top_k=50,
                    stop_token=self.tok.eot_id):
                if self.stop_event.is_set():
                    self.q.put(("done", "Stopped."))
                    return
                if int(t) in specials:
                    break
                self.q.put(("text", self.tok.decode([int(t)])))
                n += 1
            ended = ("the model ended the text" if n < max_new
                     else "stopped at the length limit")
            self.q.put(("done", f"Done: {n} tokens in "
                                f"{time.time() - t0:.1f}s, {ended}."))
        except Exception:                   # noqa: BLE001 - shown to user
            logging.exception("generation failed")
            self.q.put(("error", traceback.format_exc()))

    def stop(self):
        self.stop_event.set()

    def surprise(self):
        if self.busy:
            return
        self.set_genre(random.choice(list(GENRES)))
        self.use_idea(self.deck.deal(self.genre, 1)[0])
        self.generate()

    def finish(self, message):
        self.busy = False
        for b in (self.gen_btn, self.surprise_btn, self.reload_btn):
            b.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.set_status(message)

    def poll(self):
        """Runs every 30 ms on the Tk thread: drain the worker's queue."""
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "text":
                    self.write(payload)
                elif kind == "done":
                    self.finish(payload)
                elif kind == "error":
                    self.finish("Something went wrong (details in gui.log).")
                    messagebox.showerror("TLM Mixtape", payload[-1500:])
                elif kind == "model":
                    self.model, self.tok, _, self.model_desc = payload
                    self.reload_btn.config(state="normal")
                    self.model_label.config(text=f"Model: {self.model_desc}")
                    self.set_status("Ready: pick an idea, then Generate.")
                elif kind == "model_error":
                    self.reload_btn.config(state="normal")
                    self.model_label.config(text="No model loaded.")
                    self.set_status("")
                    messagebox.showinfo("TLM Mixtape", payload)
        except queue.Empty:
            pass
        self.root.after(30, self.poll)

    # ------------------------------------------------ favorites / copy
    def output_text(self):
        return self.out.get("1.0", "end").strip()

    def save_favorite(self):
        text = self.output_text()
        if not text or self.busy:
            return
        new = not os.path.exists(self.favorites_path)
        with open(self.favorites_path, "a", encoding="utf-8") as f:
            if new:
                f.write("# Mixtape favorites\n\nThe best things the TLM "
                        "Mixtape model wrote, saved from the desktop app "
                        "(tlm/gui.py).\n")
            f.write(f"\n## {self.last['genre']}: {self.last['main']}\n\n"
                    f"*{time.strftime('%Y-%m-%d %H:%M')} - "
                    f"{self.model_desc} - creativity "
                    f"{self.last['temperature']:.2f}*\n\n")
            for line in text.split("\n"):
                f.write(f"> {line}\n" if line.strip() else ">\n")
        self.set_status(f"Saved to {os.path.basename(self.favorites_path)}.")

    def copy(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.output_text())
        self.set_status("Copied to the clipboard.")

    def close(self):
        self.stop_event.set()
        self.save_state()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="The TLM Mixtape app")
    parser.add_argument("--run", default=None,
                        help=f"checkpoint run to load (default: the first "
                             f"of {', '.join(RUNS)} that exists)")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    os.makedirs(DATA_ROOT, exist_ok=True)
    logging.basicConfig(filename=LOG_PATH, level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    # pythonw (the desktop shortcut) has no console: anything printed or
    # any warning would go nowhere, so send it to the log instead.
    if sys.stdout is None or sys.stderr is None:
        log = open(LOG_PATH, "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stdout or log
        sys.stderr = sys.stderr or log

    # Tell Windows we draw at the real screen resolution; otherwise it
    # stretches the window on high-DPI screens and the text looks blurry.
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass                              # not Windows, or an old one
    root = tk.Tk()

    def report(exc, val, tb):
        logging.error("".join(traceback.format_exception(exc, val, tb)))
        messagebox.showerror("TLM Mixtape", f"{exc.__name__}: {val}")
    root.report_callback_exception = report
    try:
        MixtapeApp(root, run=args.run, device=args.device)
        root.mainloop()
    except Exception:
        logging.exception("app crashed")
        raise


if __name__ == "__main__":
    main()
