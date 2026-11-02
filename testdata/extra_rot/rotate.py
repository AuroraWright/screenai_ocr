"""Rotates a rendered page by the given angle (degrees, counter-clockwise) on a white background."""
import sys
from PIL import Image

src, dst, angle = sys.argv[1], sys.argv[2], float(sys.argv[3])
img = Image.open(src).convert('RGB')
img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255)).save(dst)
