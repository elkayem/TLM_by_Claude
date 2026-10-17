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
import math
import os
import queue
import random
import sys
import threading
import time
import traceback
import unicodedata
import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox

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
# Creativity = the sampling temperature (docs/07-sampling.md), in 70s terms.
CREATIVITY = {"mellow  0.6": 0.6, "groovy  0.8": 0.8, "far out  1.0": 1.0,
              "cosmic  1.2": 1.2}
IDEAS_SHOWN = 4
CATEGORIES = ("Pet Supplies", "Toys and Games", "Grocery and Gourmet Food")

# What's "handwritten" on the cassette's label for each genre.
TAPE_NAMES = {"wiki": "Encyclopedia", "book": "Old Books",
              "review": "Reviews", "recipe": "Recipes", "story": "Stories"}
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
             ("opening", "Opening (optional)", "entry")],
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


# --------------------------------------------------------------- the look
# 1970s: chocolate brown, rust, burnt orange, harvest-gold mustard and
# cream, the sunset stripes of every 70s tape label, and Cooper Black.
BROWN, RUST, ORANGE, MUSTARD = "#3B2314", "#B8461B", "#E07B26", "#F2B33D"
AVOCADO, CREAM, PAPER = "#6F7F2A", "#F6E7C8", "#FFF6E3"
INK = "#2F4C8C"                         # blue ballpoint, for the tape label
TEXT = "#2B1A10"
STRIPES = ("#5A2E1A", RUST, ORANGE, MUSTARD)
SHELL, WINDOW, TAPE = "#2B2622", "#1C1815", "#5A3824"
TEAL = "#2F8A86"                    # one more label stripe, as on real 70s tapes
# Fonts, best first: the first one installed is used.
FONTS = {"title": ("Cooper Black", "Bookman Old Style", "Georgia"),
         "body": ("Bookman Old Style", "Georgia", "Segoe UI"),
         "hand": ("Ink Free", "Segoe Print", "Comic Sans MS"),
         "type": ("Courier New", "Consolas", "Courier")}


class Cassette:
    """The cassette in the window's header, drawn with canvas shapes so it
    can move: while the model writes, the hubs turn and the tape winds from
    the left pack onto the right one (how far is how much of the length
    limit is used). The label is "handwritten" with the genre and track.

    Laid out from a photo of a real compact cassette (100.4 x 63.8 mm),
    in cassette coordinates: u across and v down, each 0..1.
    """

    REELS = (0.289, 0.711)                  # hub centres: 42% apart...
    REEL_V = 0.442                          # ... a little above the middle
    WIN_BOX = (0.208, 0.31, 0.792, 0.571)   # the label's cutout (u0 v0 u1 v1)
    PANE = (0.387, 0.342, 0.611, 0.547)     # the clear window in its middle
    HUB, EMPTY, FULL = 0.092, 0.095, 0.36   # radii, in shell heights
    # label stripes (v from, v to), behind the window like the real thing
    BANDS = ((.319, .365, MUSTARD), (.369, .478, ORANGE), (.55, .599, TEAL),
             (.614, .662, MUSTARD), (.678, .727, RUST))

    def __init__(self, canvas, x, y, height, fonts):
        self.c, self.x, self.y, self.h = canvas, x, y, height
        self.w = 100.4 / 63.8 * height
        self.fonts = fonts
        self.angle, self.progress, self.running = 0.0, 0.3, False
        self.draw()

    def at(self, u, v):
        """Cassette coordinates (0..1 across, 0..1 down) -> canvas pixels."""
        return self.x + u * self.w, self.y + v * self.h

    def rounded(self, u0, v0, u1, v1, r, **kw):
        """A rounded rectangle: a smoothed polygon whose doubled points keep
        the edges straight and round off only the corners."""
        (x0, y0), (x1, y1), r = self.at(u0, v0), self.at(u1, v1), r * self.h
        pts = [x0 + r, y0, x0 + r, y0, x1 - r, y0, x1 - r, y0, x1, y0,
               x1, y0 + r, x1, y0 + r, x1, y1 - r, x1, y1 - r, x1, y1,
               x1 - r, y1, x1 - r, y1, x0 + r, y1, x0 + r, y1, x0, y1,
               x0, y1 - r, x0, y1 - r, x0, y0 + r, x0, y0 + r, x0, y0]
        return self.c.create_polygon(pts, smooth=True, **kw)

    def circle(self, cx, cy, r, **kw):
        return self.c.create_oval(cx - r, cy - r, cx + r, cy + r, **kw)

    def draw(self):
        c, h = self.c, self.h
        hole = "#0E0C0B"
        # the shell with its shadow and the small tab on each side (they
        # stick out .013 of the height), its raised rim, and the four
        # corner screws
        t = .013 * h / self.w
        self.rounded(0.01, 0.025, 1.01, 1.025, 0.045, fill="#1E120A")
        for u0, u1 in ((-t, .05), (.95, 1 + t)):
            self.rounded(u0 + .01, .68, u1 + .01, .93, .008, fill="#1E120A")
            self.rounded(u0, .655, u1, .905, .008, fill=SHELL,
                         outline="#14110F")
        self.rounded(0, 0, 1, 1, 0.045, fill=SHELL, outline="#14110F")
        self.rounded(.014, .022, .986, .978, .035, fill="", outline="#3D3733")
        for u, v in ((.033, .044), (.967, .044), (.033, .956), (.967, .956)):
            self.screw(u, v)
        # the label, with its top corners clipped
        c.create_polygon(*self.at(.099, .083), *self.at(.901, .083),
                         *self.at(.951, .16), *self.at(.951, .727),
                         *self.at(.049, .727), *self.at(.049, .16),
                         fill=CREAM, outline="")
        for v in (.205, .30):                           # writing lines
            c.create_line(*self.at(.10, v), *self.at(.90, v), fill="#CDB892")
        c.create_text(*self.at(.10, .155), text="TLM  MIXTAPE", anchor="w",
                      fill=BROWN, font=(self.fonts["title"], -int(.08 * h)))
        c.create_text(*self.at(.90, .155), text="SIDE A", anchor="e",
                      fill=RUST, font=(self.fonts["title"], -int(.06 * h)))
        self.hand = c.create_text(*self.at(.5, .255), text="", fill=INK,
                                  font=(self.fonts["hand"], -int(.085 * h),
                                        "bold"))
        for v0, v1, colour in self.BANDS:
            (x0, y0), (x1, y1) = self.at(.049, v0), self.at(.951, v1)
            c.create_rectangle(x0, y0, x1, y1, fill=colour, outline="")
        # The label's cutout shows the black plastic shell. The tape packs
        # are seen only through the clear pane in the middle; the hubs
        # through two round holes.
        self.rounded(*self.WIN_BOX, .02, fill=SHELL, outline="#4A3F37",
                     width=2)
        self.rounded(*self.PANE, .02, fill=WINDOW, outline="")
        self.packs = [c.create_polygon(0, 0, 0, 0, 0, 0, fill=TAPE,
                                       outline="")
                      for _ in self.REELS]
        self.rounded(*self.PANE, .02, fill="", outline="#6B625B", width=2)
        # each hub: a grey rim, a white face, six teeth around its inside
        for u in self.REELS:
            cx, cy = self.at(u, self.REEL_V)
            self.circle(cx, cy, .104 * h, fill=hole, outline="#14110F")
            self.circle(cx, cy, self.HUB * h, fill="#C9C2B6",
                        outline="#8C857B")
            self.circle(cx, cy, .075 * h, fill="#F1EDE6", outline="")
        self.teeth = [[c.create_line(0, 0, 0, 0, fill="#A8A196",
                                     width=max(2, int(h / 50)))
                       for _ in range(6)] for _ in self.REELS]
        # the lower part: a trapezoid widening down to the bottom edge, with
        # a centre screw and the two round and two square openings
        c.create_polygon(*self.at(.172, .766), *self.at(.828, .766),
                         *self.at(.858, 1.0), *self.at(.142, 1.0),
                         fill="#262120", outline="#3D3733")
        for u in (.25, .75):
            self.circle(*self.at(u, .934), .024 * h, fill=hole, outline="")
        for u in (.356, .644):
            x, y = self.at(u, .911)
            c.create_rectangle(x - .022 * h, y - .018 * h, x + .022 * h,
                               y + .018 * h, fill=hole, outline="")
        self.screw(.5, .831)
        self.update()

    def screw(self, u, v):
        cx, cy = self.at(u, v)
        r = .022 * self.h
        self.circle(cx, cy, r, fill="#5E5650", outline="#14110F")
        self.c.create_line(cx - r * .6, cy, cx + r * .6, cy, fill="#14110F")
        self.c.create_line(cx, cy - r * .6, cx, cy + r * .6, fill="#14110F")

    def update(self):
        h, p = self.h, self.progress
        empty, full = self.EMPTY * h, self.FULL * h
        # Tape moves from the left pack to the right one; each pack's AREA
        # changes in proportion, so the radii go as square roots. Only the
        # part of each pack behind the clear pane shows, so clip it there
        # (a pack too small to reach the pane is hidden).
        radii = (math.sqrt(empty ** 2 + (1 - p) * (full ** 2 - empty ** 2)),
                 math.sqrt(empty ** 2 + p * (full ** 2 - empty ** 2)))
        (wx0, wy0), (wx1, wy1) = (self.at(*self.PANE[:2]),
                                  self.at(*self.PANE[2:]))
        for u, r, pack, teeth in zip(self.REELS, radii, self.packs,
                                     self.teeth):
            cx, cy = self.at(u, self.REEL_V)
            pts = []
            for k in range(72):
                a = 2 * math.pi * k / 72
                pts += [min(max(cx + r * math.cos(a), wx0 + 2), wx1 - 2),
                        min(max(cy + r * math.sin(a), wy0 + 2), wy1 - 2)]
            self.c.coords(pack, *pts)
            self.c.itemconfig(pack, state="normal" if cx - r < wx1 - 2 and
                              cx + r > wx0 + 2 else "hidden")
            for k, tooth in enumerate(teeth):        # inward-pointing teeth
                a = self.angle + k * math.pi / 3
                self.c.coords(tooth, cx + .054 * h * math.cos(a),
                              cy + .054 * h * math.sin(a),
                              cx + .075 * h * math.cos(a),
                              cy + .075 * h * math.sin(a))

    def label(self, text):
        """Write on the label, shrinking the handwriting until it fits."""
        room = .76 * self.w
        for size in range(int(.085 * self.h), int(.055 * self.h) - 1, -1):
            font = (self.fonts["hand"], -size, "bold")
            if tkfont.Font(font=font).measure(text) <= room:
                break
        while tkfont.Font(font=font).measure(text) > room and len(text) > 4:
            text = text[:-4] + "..."
        self.c.itemconfig(self.hand, text=text, font=font)

    def start(self):
        self.progress = 0.0
        if not self.running:
            self.running = True
            self.spin()

    def stop(self):
        self.running = False

    def spin(self):
        if self.running:
            self.angle += 0.22
            self.update()
            self.c.after(40, self.spin)


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


def wait_until_safe(run, status):
    """Don't read a checkpoint while its training run is about to replace it.

    A training run replaces best.pt right after each evaluation (about
    every 28 minutes for the mixtape run). On Windows, replacing a file
    fails while another program has it open - so if the app were reading
    best.pt at that moment, the TRAINING RUN would crash. While a run is
    live (its log.csv changed in the last hour), we only read during the
    quiet stretch after an evaluation, waiting for the next one if needed.
    """
    log = os.path.join(checkpoint_dir(run), "log.csv")
    if not os.path.exists(log):
        return

    def minutes_since_eval():
        return (time.time() - os.path.getmtime(log)) / 60

    age = minutes_since_eval()
    if age > 60 or 1.5 <= age <= 20:
        return          # not training, or just saved: the next save is far off
    # Either an evaluation happened moments ago (its saves may still be in
    # progress) or the next one is due soon. Wait for a fresh evaluation,
    # plus 1.5 minutes for its checkpoint saves to finish.
    before = os.path.getmtime(log) if age > 20 else None
    t0 = time.time()
    while True:
        age = minutes_since_eval()
        fresh = before is None or os.path.getmtime(log) != before
        if age > 60 or (fresh and age >= 1.5):
            return
        status(f"Training is running: waiting for it to save its next "
               f"checkpoint before loading ({(time.time() - t0) / 60:.0f} "
               f"min so far, usually under 30).")
        time.sleep(10)


def choose_device(requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        if free >= 1.5e9:
            return "cuda"
    return "cpu"


def load_model(requested_run, requested_device, status=lambda s: None):
    """Returns (model, tokenizer, device, description). Runs on a thread."""
    run, path = find_run(requested_run)
    if not run:
        raise FileNotFoundError(
            "No trained mixtape model yet.\n\nTrain one with\n"
            "    python -m tlm.train --config mixtape\n"
            "(or mixtape-pilot for a quick one), then press Reload.")
    wait_until_safe(run, status)
    status("Loading the model ...")
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
        self.busy = self.recording = False
        self.max_new, self.n_written, self.tracks = 1, 0, 0
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
    def font(self, kind, size, *style):
        """A font tuple: kind is a FONTS key, size in pixels."""
        return (self.fonts[kind], -size, *style)

    def button(self, parent, text, command, bg, fg, size=15, **kw):
        return tk.Button(parent, text=text, command=command, bg=bg, fg=fg,
                         activebackground=ORANGE if bg != ORANGE else MUSTARD,
                         activeforeground=CREAM, disabledforeground="#C9B48F",
                         relief="flat", bd=0, highlightthickness=0,
                         cursor="hand2", font=self.font("title", size),
                         padx=14, pady=4, **kw)

    def menu(self, parent, var, values, width, command=None):
        """A 70s pulldown: an orange OptionMenu with a cream menu."""
        om = tk.OptionMenu(parent, var, *values, command=command)
        om.config(bg=ORANGE, fg=CREAM, activebackground=MUSTARD,
                  activeforeground=BROWN, relief="flat", bd=0,
                  highlightthickness=0, width=width, anchor="w", padx=10,
                  pady=3, cursor="hand2", font=self.font("body", 16, "bold"))
        om["menu"].config(bg=PAPER, fg=TEXT, activebackground=ORANGE,
                          activeforeground=CREAM, bd=0,
                          font=self.font("body", 16))
        return om

    def build_ui(self):
        root = self.root
        root.title("TLM Mixtape")
        root.geometry("960x920")
        root.minsize(880, 820)
        root.configure(bg=CREAM)
        if os.path.exists(ICON_PATH):
            try:
                root.iconbitmap(ICON_PATH)
            except tk.TclError:
                pass
        installed = set(tkfont.families(root))
        self.fonts = {k: next((f for f in names if f in installed),
                              "TkDefaultFont")
                      for k, names in FONTS.items()}

        # ---- header: the cassette, the title, a REC light, the stripes
        header = tk.Canvas(root, height=204, bg=BROWN, highlightthickness=0)
        header.pack(fill="x")
        self.cassette = Cassette(header, 28, 14, 174, self.fonts)
        tx = 28 + self.cassette.w + 44
        header.create_text(tx, 66, text="TLM Mixtape", anchor="w",
                           fill=CREAM, font=self.font("title", 58))
        header.create_text(tx + 3, 118, anchor="w", fill=MUSTARD,
                           text="five genres · one tiny language model",
                           font=self.font("body", 19, "italic"))
        self.rec_dot = header.create_oval(tx + 4, 150, tx + 20, 166,
                                          fill="#5A2E1A", outline="")
        self.rec_text = header.create_text(tx + 28, 158, text="READY",
                                           anchor="w", fill="#B79A72",
                                           font=self.font("title", 16))
        self.header = header
        for colour in STRIPES:
            tk.Frame(root, height=7, bg=colour).pack(fill="x")

        body = tk.Frame(root, bg=CREAM, padx=24, pady=14)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)

        # ---- left: genre, its form, the knobs
        left = tk.Frame(body, bg=CREAM)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 18))
        left.columnconfigure(0, weight=1)
        tk.Label(left, text="Pick a genre", bg=CREAM, fg=BROWN,
                 font=self.font("title", 20)).grid(row=0, column=0,
                                                    sticky="w")
        self.genre_var = tk.StringVar()
        self.menu(left, self.genre_var, list(GENRES.values()), 30,
                  command=lambda label: self.set_genre(
                      next(g for g, l in GENRES.items() if l == label))
                  ).grid(row=1, column=0, sticky="w", pady=(4, 8))
        self.fields_frame = tk.Frame(left, bg=CREAM)
        self.fields_frame.grid(row=2, column=0, sticky="ew")
        self.fields_frame.columnconfigure(1, weight=1)
        knobs = tk.Frame(left, bg=CREAM)
        knobs.grid(row=3, column=0, sticky="w", pady=(8, 0))
        tk.Label(knobs, text="Creativity", bg=CREAM, fg=BROWN,
                 font=self.font("body", 15)).pack(side="left")
        self.creativity = tk.StringVar(value="groovy  0.8")
        self.menu(knobs, self.creativity, list(CREATIVITY), 11).pack(
            side="left", padx=(8, 18))
        tk.Label(knobs, text="Length", bg=CREAM, fg=BROWN,
                 font=self.font("body", 15)).pack(side="left")
        self.length_var = tk.StringVar(value="medium")
        self.menu(knobs, self.length_var, list(LENGTHS), 7).pack(
            side="left", padx=(8, 0))

        # ---- right: the ideas, on a "J-card"
        card = tk.Frame(body, bg=PAPER, padx=16, pady=10,
                        highlightthickness=3, highlightbackground=MUSTARD)
        card.grid(row=0, column=1, sticky="nsew")
        tk.Label(card, text="Ideas", bg=PAPER, fg=RUST,
                 font=self.font("title", 20)).pack(anchor="w")
        self.idea_rows = []
        for _ in range(IDEAS_SHOWN):
            row = tk.Frame(card, bg=PAPER)
            row.pack(anchor="w", fill="x", pady=2)
            tk.Label(row, text="♫", bg=PAPER, fg=ORANGE,
                     font=self.font("body", 15)).pack(side="left")
            idea = tk.Label(row, bg=PAPER, fg=RUST, cursor="hand2",
                            anchor="w", justify="left", wraplength=290,
                            font=self.font("body", 15))
            idea.pack(side="left", padx=(6, 0))
            badge = tk.Label(row, bg=PAPER, fg=AVOCADO,
                             font=self.font("body", 12, "italic"))
            badge.pack(side="left", padx=(8, 0))
            idea.bind("<Enter>", lambda e, l=idea: l.config(
                fg=ORANGE, font=self.font("body", 15, "underline")))
            idea.bind("<Leave>", lambda e, l=idea: l.config(
                fg=RUST, font=self.font("body", 15)))
            self.idea_rows.append((idea, badge))
        self.button(card, "More ideas ♫", self.deal_ideas, MUSTARD,
                    BROWN, size=14).pack(anchor="w", pady=(8, 0))

        # ---- the transport buttons
        buttons = tk.Frame(body, bg=CREAM)
        buttons.grid(row=1, column=0, columnspan=2, sticky="w", pady=(14, 0))
        self.gen_btn = self.button(buttons, "▶  Generate", self.generate,
                                   RUST, CREAM, size=21)
        self.gen_btn.config(padx=24, pady=6)
        self.gen_btn.pack(side="left")
        self.surprise_btn = self.button(buttons, "Surprise me!",
                                        self.surprise, MUSTARD, BROWN,
                                        size=18)
        self.surprise_btn.config(pady=6)
        self.surprise_btn.pack(side="left", padx=12)
        self.stop_btn = self.button(buttons, "■  Stop", self.stop,
                                    BROWN, CREAM, size=18, state="disabled")
        self.stop_btn.config(pady=6)
        self.stop_btn.pack(side="left")

        # ---- the output, typed on paper
        out = tk.Frame(body, bg=BROWN, padx=3, pady=3)
        out.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(14, 0))
        body.rowconfigure(2, weight=1)
        out.columnconfigure(0, weight=1)
        out.rowconfigure(0, weight=1)
        self.out = tk.Text(out, wrap="word", bg=PAPER, fg=TEXT, padx=16,
                           pady=12, relief="flat", bd=0,
                           insertbackground=BROWN, selectbackground=MUSTARD,
                           font=self.font("type", 16), state="disabled")
        self.out.grid(row=0, column=0, sticky="nsew")
        self.out.tag_configure("header", foreground=RUST,
                               font=self.font("body", 16, "italic"))
        scroll = tk.Scrollbar(out, command=self.out.yview, bg=MUSTARD,
                              troughcolor=CREAM, bd=0)
        scroll.grid(row=0, column=1, sticky="ns")
        self.out.config(yscrollcommand=scroll.set)

        below = tk.Frame(body, bg=CREAM)
        below.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        self.button(below, "★  Save to favorites", self.save_favorite,
                    AVOCADO, CREAM, size=14).pack(side="left")
        self.button(below, "Copy", self.copy, ORANGE, CREAM,
                    size=14).pack(side="left", padx=10)
        self.show_raw = tk.BooleanVar(value=False)
        tk.Checkbutton(below, text="Show what the model sees",
                       variable=self.show_raw, bg=CREAM, fg=BROWN,
                       activebackground=CREAM, activeforeground=RUST,
                       selectcolor=PAPER, highlightthickness=0,
                       font=self.font("body", 14)).pack(side="right")

        status = tk.Frame(body, bg=CREAM)
        status.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        status.columnconfigure(0, weight=1)
        self.status = tk.Label(status, text="", bg=CREAM, fg=BROWN,
                               anchor="w", font=self.font("body", 14))
        self.status.grid(row=0, column=0, sticky="w")
        self.model_label = tk.Label(status, text="", bg=CREAM, fg="#8A6A48",
                                    anchor="w", font=self.font("body", 13))
        self.model_label.grid(row=1, column=0, sticky="w")
        self.reload_btn = self.button(status, "Reload model",
                                      self.reload_model, BROWN, CREAM,
                                      size=13)
        self.reload_btn.grid(row=0, column=1, rowspan=2, sticky="e")

    def set_rec(self, on):
        """The REC light in the header: blinking red while the model writes."""
        self.recording = on
        if on:
            self.header.itemconfig(self.rec_text, text="REC", fill="#FF6B57")
            self.blink(True)
        else:
            self.header.itemconfig(self.rec_dot, fill="#5A2E1A")
            self.header.itemconfig(self.rec_text, text="READY",
                                   fill="#B79A72")

    def blink(self, lit):
        if self.recording:
            self.header.itemconfig(self.rec_dot,
                                   fill="#E0322B" if lit else "#5A2E1A")
            self.root.after(500, self.blink, not lit)

    def set_genre(self, genre):
        if genre not in GENRES:
            genre = "wiki"
        self.genre = genre
        self.genre_var.set(GENRES[genre])
        for w in self.fields_frame.winfo_children():
            w.destroy()
        first = None
        for row, (key, label, kind) in enumerate(FIELDS[genre]):
            tk.Label(self.fields_frame, text=label, bg=CREAM, fg=BROWN,
                     font=self.font("body", 15)).grid(
                row=row, column=0, sticky="w", pady=4, padx=(0, 12))
            var = self.vars[genre][key]
            if kind == "rating":
                w = self.menu(self.fields_frame, var, RATINGS, 12)
                w.grid(row=row, column=1, sticky="w", pady=4)
            elif kind == "category":
                w = self.menu(self.fields_frame, var, CATEGORIES, 26)
                w.grid(row=row, column=1, sticky="w", pady=4)
            else:
                w = tk.Entry(self.fields_frame, textvariable=var, bg=PAPER,
                             fg=TEXT, insertbackground=BROWN, relief="flat",
                             highlightthickness=2,
                             highlightbackground=MUSTARD,
                             highlightcolor=ORANGE,
                             font=self.font("body", 16))
                w.grid(row=row, column=1, sticky="ew", pady=4, ipady=3)
                w.bind("<Return>", lambda e: self.generate())
            first = first or w
        first.focus_set()
        self.state["genre"] = genre
        self.cassette.label(TAPE_NAMES[genre])
        self.deal_ideas()

    def deal_ideas(self):
        hand = self.deck.deal(self.genre, IDEAS_SHOWN)
        for (idea_label, badge), idea in zip(self.idea_rows,
                                             hand + [None] * IDEAS_SHOWN):
            if idea is None:
                idea_label.config(text="")
                badge.config(text="")
                continue
            text = idea["text"]
            if self.genre == "review":
                text += f"  {stars(idea['rating'])}"
            idea_label.config(text=text)
            badge.config(text="mix-up" if idea.get("mixup") else "")
            idea_label.bind("<Button-1>", lambda e, i=idea: self.use_idea(i))

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
                self.q.put(("model", load_model(
                    self.run, self.device_request,
                    status=lambda s: self.q.put(("status", s)))))
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
                     "temperature": CREATIVITY[self.creativity.get()]}
        self.busy = True
        self.stop_event.clear()
        for b in (self.gen_btn, self.surprise_btn, self.reload_btn):
            b.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.set_status("Recording ...")
        ids = self.tok.encode(prompt)
        max_new = LENGTHS[self.length_var.get()]
        temperature = CREATIVITY[self.creativity.get()]
        self.max_new, self.n_written = max_new, 0
        self.tracks += 1
        self.cassette.label(f"{TAPE_NAMES[self.genre]} · track "
                            f"{self.tracks}")
        self.cassette.start()
        self.set_rec(True)
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
        self.cassette.stop()
        self.set_rec(False)
        self.set_status(message)

    def poll(self):
        """Runs every 30 ms on the Tk thread: drain the worker's queue."""
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "text":
                    self.write(payload)
                    self.n_written += 1      # wind the tape along
                    self.cassette.progress = min(1.0, self.n_written
                                                 / self.max_new)
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
                elif kind == "status":
                    self.set_status(payload)
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
