# SPDX-License-Identifier: MPL-2.0
"""Randomized equivalence test: cpp ScalePlane (cpp/build/scale_test) vs the Python libyuv port."""
import numpy as np, subprocess, sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from screenai.libyuv_scale import scale_plane
rng = np.random.default_rng(1)
S = tempfile.mkdtemp(dir=os.environ.get('TMPDIR'))
shapes = [(900, 300, 1280, 427), (1280, 427, 640, 214), (640, 214, 320, 107), (320, 107, 160, 54), (64, 32, 32, 16),
          (100, 40, 75, 30), (96, 48, 36, 18), (400, 100, 100, 25), (50, 20, 100, 40), (51, 21, 101, 41),
          (30, 25, 168, 32), (500, 38, 221, 17), (700, 60, 373, 32), (20, 32, 7, 32), (33, 33, 33, 77),
          (800, 40, 640, 32), (37, 90, 148, 360), (1000, 10, 999, 9), (2, 2, 1, 1), (3, 2, 1, 1)]
for _ in range(80):
    sw, sh = rng.integers(2, 400, 2); dw, dh = rng.integers(1, 500, 2); shapes.append((int(sw), int(sh), int(dw), int(dh)))
for _ in range(40):  # downscales that hit the box paths
    sw, sh = rng.integers(20, 600, 2); f = rng.uniform(0.05, 0.49); shapes.append((int(sw), int(sh), max(1, int(sw * f)), max(1, int(sh * f))))
lines, exp = [], []
for k, (sw, sh, dw, dh) in enumerate(shapes):
    for mode in range(4):
        img = rng.integers(0, 256, (sh, sw), dtype=np.uint8)
        if k % 3 == 0: img = (np.add.outer(np.arange(sh), np.arange(sw)) * 7 % 256).astype(np.uint8)
        p = f'{S}/{k}_{mode}'; img.tofile(p + '.in')
        lines.append(f'{sw} {sh} {dw} {dh} {mode} {p}.in {p}.out'); exp.append((p, scale_plane(img, dw, dh, mode), (sw, sh, dw, dh, mode)))
subprocess.run(['cpp/build/scale_test'], input='\n'.join(lines) + '\n', text=True, check=True)
bad = 0
for p, e, c in exp:
    o = np.fromfile(p + '.out', np.uint8).reshape(e.shape)
    if not np.array_equal(o, e): bad += 1; print('MISMATCH', c, (o != e).sum())
print('cases', len(exp), 'bad', bad)
