# SPDX-License-Identifier: MPL-2.0
"""Check detector pyramid hypotheses against a tfl dump: det_pyr_check.py img.png dumpdir"""
import sys, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'trace')
from PIL import Image
from screenai.libyuv_scale import scale_plane
from load_dump import load
rgba = np.array(Image.open(sys.argv[1]).convert('RGBA')).astype(np.int32)
g = ((77 * rgba[..., 0] + 150 * rgba[..., 1] + 29 * rgba[..., 2] + 128) >> 8).astype(np.uint8)
H, W = g.shape
ins = load(sys.argv[2])[0]['in']
f32 = np.float32
def sc(img, s, f):
    h, w = img.shape
    return scale_plane(img, int(np.rint(f32(s) * f32(w))), int(np.rint(f32(s) * f32(h))), f)
s48 = max(min(f32(1280) / f32(W), f32(1280) / f32(H)), f32(1))
s44 = min(min(f32(2048) / f32(H), f32(2048) / f32(W)), s48)
R = sc(g, s44, 3)
print('R', R.shape, 's44', s44)
v104 = min(f32(1280) / f32(R.shape[0]), f32(1280) / f32(R.shape[1]))
lv = [R]
for k in range(1, 5):
    s = v104 if k == 1 else f32(0.5)
    lv.append(lv[-1] if abs(s - 1) < 0.001 else sc(lv[-1], s, 3 if k == 1 else 2))
for t in ins:
    a = t['arr'][0, :, :, 0]
    best = None
    for k, L in enumerate(lv):
        if L.shape[0] <= a.shape[0] and L.shape[1] <= a.shape[1]:
            pad = np.zeros_like(a); pad[:L.shape[0], :L.shape[1]] = L
            d = int((pad != a).sum())
            if best is None or d < best[1]: best = (k, d, L.shape)
    print(t['name'], a.shape, 'best level', best)
