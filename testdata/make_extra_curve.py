"""Curved-text test images (testdata/extra_curve/): characters placed along arcs/waves so the detector takes its
curved-line path (CreateCurveRepresentation)."""
import os, math
from PIL import Image, ImageDraw, ImageFont
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'extra_curve')
ARIAL = '/System/Library/Fonts/Supplemental/Arial.ttf'
HIRA = '/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc'

def draw_on_path(img, text, font, path, fill='black'):
    """path(s) -> (x, y, tangent_angle_rad) for arc length s; draws each glyph rotated to the tangent."""
    s = 0.0
    for ch in text:
        w = font.getlength(ch)
        x, y, a = path(s + w / 2)
        g = Image.new('RGBA', (int(w) + 20, font.size * 2), (0, 0, 0, 0))
        ImageDraw.Draw(g).text((10, font.size // 2), ch, font=font, fill=fill)
        r = g.rotate(-math.degrees(a), expand=True, resample=Image.BICUBIC)
        img.paste(r, (int(x - r.width / 2), int(y - r.height / 2)), r)
        s += w

def arc(cx, cy, R, start, sign=1):
    return lambda s: (cx + R * math.sin(start + sign * s / R), cy - sign * R * math.cos(start + sign * s / R), start + sign * s / R)

def wave(x0, y0, amp, period):
    def p(s):  # approximate arc length by x
        x = x0 + s; y = y0 + amp * math.sin(2 * math.pi * s / period)
        return x, y, math.atan(amp * 2 * math.pi / period * math.cos(2 * math.pi * s / period))
    return p

def page(name, items, size=(1200, 800)):
    img = Image.new('RGB', size, 'white')
    for text, font, path, *fill in items: draw_on_path(img, text, font, path, *fill)
    img.save(os.path.join(OUT, name + '.png'))

page('arc_top', [('CURVED TEXT ALONG AN ARC', ImageFont.truetype(ARIAL, 40), arc(600, 700, 500, -0.55)),
                 ('Normal straight line below', ImageFont.truetype(ARIAL, 32), lambda s: (300 + s, 720, 0.0))])
page('arc_gentle', [('This line bends gently like a page curl near the spine', ImageFont.truetype(ARIAL, 30), arc(600, 2400, 2100, -0.24)),
                    ('Second gently curved line of body text for testing', ImageFont.truetype(ARIAL, 30), arc(600, 2460, 2100, -0.24))])
page('wave', [('Wavy baseline text that goes up and down', ImageFont.truetype(ARIAL, 34), wave(80, 300, 25, 600)),
              ('Another wave with smaller amplitude here', ImageFont.truetype(ARIAL, 30), wave(80, 550, 12, 500))])
page('arc_bottom_ja', [('円弧に沿った日本語のテキストです', ImageFont.truetype(HIRA, 40), arc(600, -150, 600, 0.5, sign=-1))])
# A gray straight overprint across the arc: RemoveOverlappingWithCurvedBoxes deletes the straight line.
page('arc_stamp', [('ROUND STAMP TEXT HERE', ImageFont.truetype(ARIAL, 40), arc(600, 900, 500, -0.5)),
                   ('ROUND STAMP', ImageFont.truetype(ARIAL, 40), lambda s: (420 + s, 425, 0.0), (160, 160, 160)),
                   ('Plain footer line of text', ImageFont.truetype(ARIAL, 30), lambda s: (400 + s, 650, 0.0))])
print(sorted(os.listdir(OUT)))
