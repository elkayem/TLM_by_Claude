"""
Draw the TLM Mixtape icon - a little cassette tape - into assets/mixtape.ico.

Standard library only (no Pillow): the shapes are simple enough to
"rasterize" by hand. Each pixel is sampled 4x4 times; at every sample we
ask "which shape is on top here?", and the pixel's colour is the average
of its samples, which gives smooth (anti-aliased) edges.

An .ico file is just a small directory of images at several sizes, so
Windows can pick the sharpest one for the desktop, taskbar or title bar:
16, 32 and 48 px are stored as classic bitmaps, 256 px as a PNG.

Run from the tlm/ project root:  python tools\\make_icon.py
"""

import math
import os
import struct
import zlib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "assets", "mixtape.ico")
SIZES = (16, 32, 48, 256)

# colours (R, G, B) - the same 1970s palette as the app (tlm/gui.py)
BODY = (43, 38, 34)        # charcoal shell
EDGE = (38, 33, 32)        # the lower trapezoid
LABEL = (246, 231, 200)    # cream label, with stripes (v from, v to, colour)
BANDS = ((.319, .365, (242, 179, 61)), (.369, .478, (224, 123, 38)),
         (.55, .599, (47, 138, 134)), (.614, .662, (242, 179, 61)),
         (.678, .727, (184, 70, 27)))
WINDOW = (28, 24, 21)      # the clear pane, dark behind the tape
TAPE = (90, 56, 36)        # tape packs, seen through the pane
HUB = (241, 237, 230)      # the hubs: white face, grey rim and teeth
RIM = (201, 194, 182)
TOOTH = (168, 161, 150)
HOLE = (14, 12, 11)
SCREW = (94, 86, 80)


def in_round_rect(x, y, x0, y0, x1, y1, r):
    if not (x0 <= x <= x1 and y0 <= y <= y1):
        return False
    cx = min(max(x, x0 + r), x1 - r)       # nearest point of the inner
    cy = min(max(y, y0 + r), y1 - r)       # (corner-free) rectangle
    return (x - cx) ** 2 + (y - cy) ** 2 <= r * r


def in_circle(x, y, cx, cy, r):
    return (x - cx) ** 2 + (y - cy) ** 2 <= r * r


# The cassette is laid out in its own coordinates, u across and v down
# (0..1 each), exactly like the app's header cassette (tlm/gui.py), which
# was measured from a photo of a real one (100.4 x 63.8 mm).
W = 0.92                                   # shell width in the icon square
H = W * 63.8 / 100.4                       # ... and height
X0, Y0 = 0.04, 0.5 - H / 2
A = W / H                                  # u units -> shell heights
# tape packs, wound 30% of the way from the left reel to the right one
EMPTY, FULL = 0.095, 0.36
PACKS = ((0.289, (EMPTY ** 2 + 0.7 * (FULL ** 2 - EMPTY ** 2)) ** 0.5),
         (0.711, (EMPTY ** 2 + 0.3 * (FULL ** 2 - EMPTY ** 2)) ** 0.5))


def shade(x, y):
    """Colour (with alpha) at a point of the unit square, top shape first."""
    u, v = (x - X0) / W, (y - Y0) / H      # into cassette coordinates
    for cu, _ in PACKS:     # the hubs, in round holes
        du, dv = (u - cu) * A, v - 0.442
        d = math.hypot(du, dv)
        if d <= 0.104:
            if d > 0.092:
                return HOLE + (255,)
            if d > 0.075:
                return RIM + (255,)
            a = math.atan2(dv, du) % (math.pi / 3)      # six teeth
            if d > 0.054 and min(a, math.pi / 3 - a) * d < 0.009:
                return TOOTH + (255,)
            return HUB + (255,)
    if 0.387 <= u <= 0.611 and 0.342 <= v <= 0.547:  # the clear pane: the
        for cu, r in PACKS:                          # only place tape shows
            if math.hypot((u - cu) * A, v - 0.442) <= r:
                return TAPE + (255,)
        return WINDOW + (255,)
    if 0.208 <= u <= 0.792 and 0.31 <= v <= 0.571:   # the label's cutout:
        return BODY + (255,)                         # black plastic
    clipped = ((u - 0.049) / 0.05 + (v - 0.083) / 0.077 < 1 or
               (0.951 - u) / 0.05 + (v - 0.083) / 0.077 < 1)
    if 0.049 <= u <= 0.951 and 0.083 <= v <= 0.727 and not clipped:
        for v0, v1, colour in BANDS:
            if v0 <= v < v1:
                return colour + (255,)
        return LABEL + (255,)
    for su, sv in ((.033, .044), (.967, .044), (.033, .956), (.967, .956),
                   (.5, .831)):
        if math.hypot((u - su) * A, v - sv) <= 0.022:
            return SCREW + (255,)
    spread = (v - 0.766) / 0.234 * 0.03    # trapezoid widens downward
    if 0.766 <= v <= 1.0 and 0.172 - spread <= u <= 0.828 + spread:
        for hu in (0.25, 0.75):
            if math.hypot((u - hu) * A, v - 0.934) <= 0.024:
                return HOLE + (255,)
        for hu in (0.356, 0.644):
            if abs((u - hu) * A) <= 0.022 and abs(v - 0.911) <= 0.018:
                return HOLE + (255,)
        return EDGE + (255,)
    if (in_round_rect(u * A, v, 0, 0, A, 1, 0.045) or      # the shell, and
            in_round_rect(u * A, v, -0.013, 0.655, A + 0.013, 0.905,
                          0.008)):                        # its side tabs
        return BODY + (255,)
    return (0, 0, 0, 0)                    # transparent


def render(n, ss=4):
    """n x n pixels as rows of (r, g, b, a), top row first."""
    rows = []
    for py in range(n):
        row = []
        for px in range(n):
            acc = [0, 0, 0, 0]
            for sy in range(ss):
                for sx in range(ss):
                    r, g, b, a = shade((px + (sx + 0.5) / ss) / n,
                                       (py + (sy + 0.5) / ss) / n)
                    acc[0] += r * a
                    acc[1] += g * a
                    acc[2] += b * a
                    acc[3] += a
            a = acc[3]
            row.append((0, 0, 0, 0) if a == 0 else
                       (round(acc[0] / a), round(acc[1] / a),
                        round(acc[2] / a), round(a / (ss * ss))))
        rows.append(row)
    return rows


def png(rows):
    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))
    n = len(rows)
    raw = b"".join(b"\x00" + bytes(c for px in row for c in px)
                   for row in rows)            # filter byte 0 per row
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", n, n, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def bmp(rows):
    """The classic icon bitmap: a BITMAPINFOHEADER, BGRA pixels bottom row
    first, then a 1-bit AND mask (all zero - the alpha channel rules)."""
    n = len(rows)
    header = struct.pack("<IiiHHIIiiII", 40, n, 2 * n, 1, 32, 0, 0, 0, 0,
                         0, 0)
    pixels = b"".join(bytes((b, g, r, a)) for row in reversed(rows)
                      for r, g, b, a in row)
    mask = b"\x00" * (((n + 31) // 32) * 4 * n)
    return header + pixels + mask


def main():
    images = [(n, png(render(n)) if n >= 256 else bmp(render(n)))
              for n in SIZES]
    offset = 6 + 16 * len(images)
    out = struct.pack("<HHH", 0, 1, len(images))
    for n, data in images:
        out += struct.pack("<BBBBHHII", n % 256, n % 256, 0, 0, 1, 32,
                           len(data), offset)
        offset += len(data)
    out += b"".join(data for _, data in images)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "wb") as f:
        f.write(out)
    with open(OUT[:-4] + ".png", "wb") as f:     # a preview, same drawing
        f.write(png(render(256)))
    print(f"wrote {OUT} ({len(out):,} bytes; sizes {SIZES})")


if __name__ == "__main__":
    main()
