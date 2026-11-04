# SPDX-License-Identifier: MPL-2.0
"""Exercises the C API of a libchromescreenai.so directly (timing + edge cases). usage: api_check.py <lib> <model_dir> <images...>
API_CHECK_REPS=<n>: report the median of n timed runs per image (default 1)."""
import ctypes, sys, time, os, hashlib
from PIL import Image
lib, model_dir, images = sys.argv[1], sys.argv[2], sys.argv[3:]
S = ctypes.CDLL(lib)
SIZE = ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_char_p)
CONTENT = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_void_p)
@SIZE
def size(p):
    f = os.path.join(model_dir, p.decode()); return os.path.getsize(f) if os.path.exists(f) else 0
@CONTENT
def content(p, n, buf):
    ctypes.memmove(buf, open(os.path.join(model_dir, p.decode()), 'rb').read(n), n)
class Bitmap(ctypes.Structure):
    _fields_ = [('ref', ctypes.c_void_p), ('pixels', ctypes.c_void_p), ('row_bytes', ctypes.c_size_t), ('cs', ctypes.c_void_p),
                ('color_type', ctypes.c_int32), ('alpha_type', ctypes.c_int32), ('w', ctypes.c_int32), ('h', ctypes.c_int32), ('flags', ctypes.c_uint32)]
S.SetFileContentFunctions.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
S.InitOCRUsingCallback.restype = ctypes.c_bool
S.PerformOCR.argtypes = [ctypes.POINTER(Bitmap), ctypes.POINTER(ctypes.c_uint32)]
S.PerformOCR.restype = ctypes.c_void_p
S.FreeLibraryAllocatedCharArray.argtypes = [ctypes.c_void_p]
S.GetMaxImageDimension.restype = ctypes.c_uint32
S.SetFileContentFunctions(size, content)
t = time.time(); ok = S.InitOCRUsingCallback(); print(f'init {ok} {time.time() - t:.2f}s max_dim {S.GetMaxImageDimension()}')
def run(data, w, h, ct):
    b = Bitmap(pixels=ctypes.cast(ctypes.c_char_p(data), ctypes.c_void_p), row_bytes=w * (1 if ct == 14 else 4), color_type=ct, alpha_type=1, w=w, h=h)
    n = ctypes.c_uint32(0); p = S.PerformOCR(ctypes.byref(b), ctypes.byref(n))
    out = None if not p else ctypes.string_at(p, n.value)
    if p: S.FreeLibraryAllocatedCharArray(p)
    return out
for img in images:
    im = Image.open(img).convert('RGBA'); data = im.tobytes()
    run(data, im.width, im.height, 4)  # warm-up (lazy script recognizers)
    ts = []
    for _ in range(int(os.environ.get('API_CHECK_REPS', '1'))):
        t = time.perf_counter(); out = run(data, im.width, im.height, 4); ts.append(time.perf_counter() - t)
    dt = sorted(ts)[len(ts) // 2]
    print(f'{os.path.basename(img)}: {dt * 1000:.0f} ms  {len(out)} bytes  sha {hashlib.sha1(out).hexdigest()[:10]}')
for name, args in [('0x0', (b'', 0, 0, 4)), ('unsupported color 7', (b'\0' * 64, 4, 4, 7)), ('tiny 1x1', (b'\xff' * 4, 1, 1, 4))]:
    out = run(*args); print(f'{name}: {"NULL" if out is None else f"{len(out)} bytes"}')
g = Image.open(images[0]).convert('L'); out = run(g.tobytes(), g.width, g.height, 14); print(f'gray8 {os.path.basename(images[0])}: {"NULL" if out is None else f"{len(out)} bytes sha {hashlib.sha1(out).hexdigest()[:10]}"}')
