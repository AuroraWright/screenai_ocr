"""Layout-path test images (testdata/extra_paths/): inputs that drive analyzer branches the other sets never reach.
Gray straight text printed over a curved (arc) line makes RemoveOverlappingWithCurvedBoxes act."""
import os, math
from PIL import Image, ImageDraw, ImageFont
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'extra_paths')
ARIAL = '/System/Library/Fonts/Supplemental/Arial.ttf'
GRAY = (160, 160, 160)

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

def arc(cx, cy, R, start):
    return lambda s: (cx + R * math.sin(start + s / R), cy - R * math.cos(start + s / R), start + s / R)

def straight(x0, y0):
    return lambda s: (x0 + s, y0, 0.0)

def page(name, items, size=(1200, 800)):
    img = Image.new('RGB', size, 'white')
    for text, font, path, *fill in items: draw_on_path(img, text, font, path, *fill)
    img.save(os.path.join(OUT, name + '.png'))

A40 = ImageFont.truetype(ARIAL, 40)
STAMP = ('ROUND STAMP TEXT HERE', A40, arc(600, 900, 500, -0.5))
# A straight line above the arc's apex: only its middle word overlaps the arc's symbols, so the straight line is
# removed by its marked word alone ('ab OVERLAPPED cd' -> 'ab cd', boxes/confidences recomputed).
txt = 'ab OVERLAPPED cd'
page('curved_partial', [STAMP, (txt, A40, straight(600 - A40.getlength(txt) / 2, 385), GRAY)])
# A second curved line (the detector bends the gray straight overprint along the arc): the lower-scored curved line
# is deleted whole.
page('curved_vs_curved', [STAMP, ('ok ROUND STAMP', A40, straight(330, 425), GRAY)])
print(sorted(os.listdir(OUT)))
