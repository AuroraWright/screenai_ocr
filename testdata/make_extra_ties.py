"""Tie-heavy detector test images (testdata/extra_ties/): identical, pixel-aligned words and symmetric layouts, so
equal link costs, equal projections and equal paragraph ids reach the detector's sorts and its R-tree queries.
Pages are 1280 px wide so the detector's main pyramid level is the page itself (scale 1), and the pitches are
multiples of 64 px so identical cells land on the same positions relative to the network's strides."""
import os, math
from PIL import Image, ImageDraw, ImageFont
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'extra_ties')
ARIAL = '/System/Library/Fonts/Supplemental/Arial.ttf'
HIRA = '/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc'

def page(name, size, items):
    img = Image.new('RGB', size, 'white'); d = ImageDraw.Draw(img)
    for x, y, text, font in items: d.text((x, y), text, font=font, fill='black')
    img.save(os.path.join(OUT, name + '.png'))
    return img

def on_path(img, text, font, path):
    s = 0.0
    for ch in text:
        w = font.getlength(ch); x, y, a = path(s + w / 2)
        g = Image.new('RGBA', (int(w) + 20, font.size * 2), (0, 0, 0, 0))
        ImageDraw.Draw(g).text((10, font.size // 2), ch, font=font, fill='black')
        r = g.rotate(-math.degrees(a), expand=True, resample=Image.BICUBIC)
        img.paste(r, (int(x - r.width / 2), int(y - r.height / 2)), r)
        s += w

f28, f32, f24 = (ImageFont.truetype(ARIAL, s) for s in (28, 32, 24))
# Grid of identical numeric cells: every row and column is a translated copy.
page('grid_8888', (1280, 960), [(128 + 256 * c, 128 + 64 * r, '8888', f28) for r in range(11) for c in range(4)])
# One word repeated along each line at a fixed pitch: equal gaps between consecutive words.
page('repeat_word', (1280, 960), [(128 + 64 * c, 128 + 64 * r, 'tes', f32) for r in range(11) for c in range(16)])
# Two identical columns: the same lines at the same y on both sides.
page('mirror_cols', (1280, 960), [(x, 128 + 64 * r, 'Same line of text', f28) for r in range(11) for x in (128, 768)])
# Repeated single glyphs and punctuation: many identical tiny boxes on one line.
page('glyph_rows', (1280, 960), [(128, 128 + 128 * r, t, f32) for r, t in
                                 enumerate(['O O O O O O O O O O O O O O O', 'x x x x x x x x x x x x x x x',
                                            '1 1 1 1 1 1 1 1 1 1 1 1 1 1 1', 'MMMM MMMM MMMM MMMM MMMM'])])
# Words exactly abutting at a pitch equal to their advance (centroid projections equally spaced).
page('abut_words', (1280, 960), [(128 + 128 * c, 128 + 128 * r, 'abcd', f32) for r in range(6) for c in range(8)])
# Table of identical cells with identical row heights.
page('table_cells', (1280, 960), [(128 + 128 * c, 128 + 64 * r, '100.00', f24) for r in range(11) for c in range(8)])
# Vertical Japanese columns of one repeated character.
img = Image.new('RGB', (960, 1280), 'white'); d = ImageDraw.Draw(img); fj = ImageFont.truetype(HIRA, 40)
for c in range(6):
    for r in range(16): d.text((128 + 128 * c, 128 + 64 * r), '口', font=fj, fill='black')
img.save(os.path.join(OUT, 'ja_vertical_same.png'))
# Arc symmetric about the vertical centre line, of a single repeated glyph, plus its mirror image below.
img = Image.new('RGB', (1280, 960), 'white')
R = 420; a0 = -11 * f32.getlength('O') / R
on_path(img, 'O' * 22, f32, lambda s: (640 + R * math.sin(a0 + s / R), 640 - R * math.cos(a0 + s / R), a0 + s / R))
on_path(img, 'O' * 22, f32, lambda s: (640 + R * math.sin(a0 + s / R), 160 + R * math.cos(a0 + s / R), -a0 - s / R))
img.save(os.path.join(OUT, 'sym_arc.png'))
print(sorted(os.listdir(OUT)))
