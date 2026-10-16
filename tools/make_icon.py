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

import os
import struct
import zlib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "assets", "mixtape.ico")
SIZES = (16, 32, 48, 256)

# colours (R, G, B)
BODY = (46, 52, 64)        # dark slate shell
EDGE = (30, 34, 42)
LABEL = (246, 193, 119)    # warm orange label
STRIPE = (235, 111, 146)   # pink stripe across the label
WINDOW = (36, 40, 50)      # the clear window, dark
TAPE = (120, 82, 60)       # brown tape wound on the reels
REEL = (236, 239, 244)     # white reel hubs
SCREW = (110, 118, 135)


def in_round_rect(x, y, x0, y0, x1, y1, r):
    if not (x0 <= x <= x1 and y0 <= y <= y1):
        return False
    cx = min(max(x, x0 + r), x1 - r)       # nearest point of the inner
    cy = min(max(y, y0 + r), y1 - r)       # (corner-free) rectangle
    return (x - cx) ** 2 + (y - cy) ** 2 <= r * r


def in_circle(x, y, cx, cy, r):
    return (x - cx) ** 2 + (y - cy) ** 2 <= r * r


def shade(x, y):
    """Colour (with alpha) at a point of the unit square, top shape first."""
    for cx in (0.335, 0.665):              # the two reels
        if in_circle(x, y, cx, 0.505, 0.035):
            return WINDOW + (255,)         # hub hole
        if in_circle(x, y, cx, 0.505, 0.075):
            return REEL + (255,)
        if in_circle(x, y, cx, 0.505, 0.115):
            return TAPE + (255,)
    if in_round_rect(x, y, 0.25, 0.42, 0.75, 0.59, 0.07):
        return WINDOW + (255,)
    if in_round_rect(x, y, 0.15, 0.30, 0.85, 0.36, 0.0):
        return STRIPE + (255,)
    if in_round_rect(x, y, 0.15, 0.26, 0.85, 0.66, 0.05):
        return LABEL + (255,)
    for sx, sy in ((0.10, 0.25), (0.90, 0.25), (0.10, 0.74), (0.90, 0.74)):
        if in_circle(x, y, sx, sy, 0.025):
            return SCREW + (255,)
    if 0.66 <= y <= 0.80 and abs(x - 0.5) <= 0.22 + (0.80 - y) * 0.5:
        return EDGE + (255,)               # the trapezoid at the bottom
    if in_round_rect(x, y, 0.04, 0.17, 0.96, 0.83, 0.07):
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
