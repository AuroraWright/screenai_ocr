# SPDX-License-Identifier: MPL-2.0
"""Renders vertical text columns (one glyph per cell, top to bottom, columns right to left) with Pillow.
usage: vcol.py out.png size col1 col2 ...  (font: Hiragino Sans GB)"""
import sys
from PIL import Image, ImageDraw, ImageFont
out, size, cols = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
font = ImageFont.truetype('/System/Library/Fonts/Hiragino Sans GB.ttc', size)
h = max(len(c) for c in cols) * int(size * 1.15) + 2 * size
w = len(cols) * size * 2 + 2 * size
img = Image.new('RGB', (w, h), 'white'); d = ImageDraw.Draw(img)
for i, c in enumerate(cols):
    x = w - size * 2 - i * size * 2
    for j, ch in enumerate(c):
        d.text((x, size + j * int(size * 1.15)), ch, font=font, fill='black')
img.save(out)
