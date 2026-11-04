# SPDX-License-Identifier: MPL-2.0
"""Detector branch-coverage images (testdata/extra_det/): inputs that reach rarely taken GroupRpnDetectorV2 paths.
  drop_cap*      a line whose first word is much taller than the rest (ConstructLines splits the leading box off)
  tiny_*         very small pages (the pyramid upscales them ~30-100x)
  strip_*        extreme aspect ratios (strip_wide/_tall: the 1/8 pyramid level rounds to 0 px and is clamped to 1)
  huge_page      a page far above the 2048 px reference size (box-filter downscale by ~3x)
  big_text_*     lines tall enough that CropDetections crops them from a downscaled pyramid image with a scale that is
                 not a power of two (detection_image_scale, box scaled there and back with truncation)
  big_text_huge_2048  big_text_huge resized to 2048 px as owocr does (LANCZOS): its arc becomes a curved line that is
                 cropped from a pyramid level (curved pyramid crop)
The stage dumps in testdata/stages_extra_det/ were made from images rendered by Pillow 12.1.1 / FreeType 2.14.1 (the
system python3, not .venv's Pillow 13): other versions rasterize the text differently."""
import math
import os
from PIL import Image, ImageDraw, ImageFont
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'extra_det')
ARIAL = '/System/Library/Fonts/Supplemental/Arial.ttf'
F = lambda s: ImageFont.truetype(ARIAL, s)

def staircase(d, x, base, parts):
    """Draws (text, size) parts left to right on a common baseline."""
    for text, size in parts:
        f = F(size); d.text((x, base), text, font=f, fill='black', anchor='ls'); x += f.getlength(text + ' ')

img = Image.new('RGB', (1280, 720), 'white'); d = ImageDraw.Draw(img)
staircase(d, 60, 200, [('W', 150), ('ide', 90), ('open', 54), ('spaces and a long tail of small text', 32)])
staircase(d, 60, 420, [('BIG', 120), ('then', 70), ('smaller', 44), ('and the rest of the sentence', 28)])
staircase(d, 60, 620, [('Normal line of text for reference', 32)])
img.save(os.path.join(OUT, 'drop_cap.png'))
img = Image.new('RGB', (1280, 720), 'white'); d = ImageDraw.Draw(img)
staircase(d, 80, 240, [('A', 200), ('nd', 110), ('so', 66), ('it', 40), ('begins with a giant initial letter here', 28)])
staircase(d, 80, 300, [('second line under it in small type', 28)])
img.save(os.path.join(OUT, 'drop_cap2.png'))

for name, (w, h), text, size in [('tiny_word', (48, 16), 'Hi', 12), ('tiny_line', (160, 20), 'small text', 13),
                                 ('tiny_blank', (8, 8), '', 0)]:
    img = Image.new('RGB', (w, h), 'white')
    if text: ImageDraw.Draw(img).text((2, 1), text, font=F(size), fill='black')
    img.save(os.path.join(OUT, name + '.png'))

img = Image.new('RGB', (8000, 28), 'white')
ImageDraw.Draw(img).text((20, 2), 'A very long strip of text repeated ' * 12, font=F(20), fill='black')
img.save(os.path.join(OUT, 'strip_wide.png'))
img = Image.new('RGB', (3000, 40), 'white')
ImageDraw.Draw(img).text((10, 4), 'A long thin strip with readable text in it, ' * 5, font=F(28), fill='black')
img.save(os.path.join(OUT, 'strip_text.png'))
img = Image.new('RGB', (28, 8000), 'white'); d = ImageDraw.Draw(img)
for k, ch in enumerate('VERTICALSTRIPOFLETTERS' * 6): d.text((5, 20 + 56 * k), ch, font=F(20), fill='black')
img.save(os.path.join(OUT, 'strip_tall.png'))

img = Image.new('RGB', (6400, 4000), 'white'); d = ImageDraw.Draw(img)
for r in range(10):
    d.text((300, 300 + 340 * r), 'Huge page line %d with large text for the downscaled pyramid' % r, font=F(120 - 6 * r), fill='black')
img.save(os.path.join(OUT, 'huge_page.png'))
img = Image.new('RGB', (1800, 1200), 'white'); d = ImageDraw.Draw(img)
d.text((60, 80), 'Big', font=F(300), fill='black'); d.text((900, 140), 'Text', font=F(220), fill='black')
d.text((60, 700), 'normal sized line under the big words', font=F(40), fill='black')
img.save(os.path.join(OUT, 'big_text_mid.png'))
img = Image.new('RGB', (6400, 4000), 'white'); d = ImageDraw.Draw(img)
d.text((300, 300), 'HUGE', font=F(700), fill='black')
d.text((300, 1500), 'Large heading line', font=F(300), fill='black')
R = 2600; a0 = -0.45
for k, ch in enumerate('CURVED'):
    a = a0 + k * 0.18; x, y = 3600 + R * math.sin(a), 5600 - R * math.cos(a)
    g = Image.new('RGBA', (420, 520), (0, 0, 0, 0)); ImageDraw.Draw(g).text((20, 20), ch, font=F(380), fill='black')
    r = g.rotate(-math.degrees(a), expand=True, resample=Image.BICUBIC); img.paste(r, (int(x - r.width / 2), int(y - r.height / 2)), r)
img.save(os.path.join(OUT, 'big_text_huge.png'))
f = 2048 / max(img.size)  # owocr's ChromeScreenAI downscale to GetMaxImageDimension()
img.resize((int(img.width * f), int(img.height * f)), Image.Resampling.LANCZOS).save(os.path.join(OUT, 'big_text_huge_2048.png'))
print(sorted(os.listdir(OUT)))
