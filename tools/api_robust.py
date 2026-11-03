"""Robustness checks for a libchromescreenai.so through its C API (original or ours, release or sanitizer build).

usage: api_robust.py <lib> <model_dir> <check> [args]
  fuzz <n> [seed] [text]  random bitmaps (sizes, row bytes, color types, contents; `text`: only distorted real pages
                      with tight rows and plausible sizes); prints one line per case:
                      "<case> <result>" with result = sha1 prefix + length, EMPTY, NULL or CRASH(<signal>)
  lifecycle           PerformOCR before init, double init, uninit/reinit, missing callbacks, light mode switching
  faults [files_lib]  model-file faults (missing / truncated / garbage / bit flips) per file the library reads (or
                      the files `files_lib` reads, so two libraries are tested on the same list)
  threads <k> <imgs>  k threads calling PerformOCR concurrently on the images; compares with sequential output
  soak <n> <imgs>     n rounds over the images; prints RSS every round and any output change
Each fuzz/faults case runs in a worker subprocess that is restarted after a crash, so a crash is a result, not an
abort. Run with python -I. For a sanitizer build set API_ROBUST_INSERT to the asan runtime dylib (`clang
-print-file-name=libclang_rt.asan_osx_dynamic.dylib`): it is inserted into every worker process (the runtime strips
DYLD_INSERT_LIBRARIES from the environment, so it cannot simply be inherited). API_ROBUST_LIGHT=1 runs the fuzz
workers in light mode."""
import ctypes, hashlib, json, os, random, resource, subprocess, sys, threading

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


class Bitmap(ctypes.Structure):
    _fields_ = [('ref', ctypes.c_void_p), ('pixels', ctypes.c_void_p), ('row_bytes', ctypes.c_size_t),
                ('cs', ctypes.c_void_p), ('color_type', ctypes.c_int32), ('alpha_type', ctypes.c_int32),
                ('w', ctypes.c_int32), ('h', ctypes.c_int32), ('flags', ctypes.c_uint32)]


SIZE = ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_char_p)
CONTENT = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_void_p)


class Lib:
    """The library with file callbacks over model_dir; `fault` = (relative path, kind) corrupts one file."""

    def __init__(self, lib, model_dir, fault=None):
        self.S = ctypes.CDLL(lib)
        self.model_dir, self.fault, self.reads = model_dir, fault, []
        S = self.S
        S.SetFileContentFunctions.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        S.InitOCRUsingCallback.restype = ctypes.c_bool
        S.PerformOCR.argtypes = [ctypes.POINTER(Bitmap), ctypes.POINTER(ctypes.c_uint32)]
        S.PerformOCR.restype = ctypes.c_void_p
        S.FreeLibraryAllocatedCharArray.argtypes = [ctypes.c_void_p]
        S.GetMaxImageDimension.restype = ctypes.c_uint32
        S.SetOCRLightMode.argtypes = [ctypes.c_bool]
        self._size, self._content = SIZE(self.size_cb), CONTENT(self.content_cb)

    def data(self, rel):
        f = os.path.join(self.model_dir, rel.lstrip('/'))
        d = open(f, 'rb').read() if os.path.isfile(f) else b''
        if self.fault and rel.lstrip('/') == self.fault[0]:
            kind = self.fault[1]
            if kind == 'missing' or kind == 'zero': d = b''
            elif kind == 'truncated': d = d[:len(d) // 2]
            elif kind == 'garbage': d = random.Random(len(d)).randbytes(max(len(d), 64))
            elif kind == 'flipped':  # a few bit flips past the header, same size
                r = random.Random(1); b = bytearray(d)
                for _ in range(8):
                    if len(b) > 64: b[r.randrange(64, len(b))] ^= 1 << r.randrange(8)
                d = bytes(b)
        return d

    def size_cb(self, p):
        rel = p.decode()
        self.reads.append(rel)
        return len(self.data(rel))

    def content_cb(self, p, n, buf):
        d = self.data(p.decode())[:n]
        ctypes.memmove(buf, d, len(d))

    def set_callbacks(self):
        self.S.SetFileContentFunctions(ctypes.cast(self._size, ctypes.c_void_p), ctypes.cast(self._content, ctypes.c_void_p))

    def init(self, light=False):  # as owocr does: set the mode after init
        self.set_callbacks()
        ok = self.S.InitOCRUsingCallback()
        self.S.SetOCRLightMode(light)
        return ok

    def ocr(self, pixels, w, h, ct, row_bytes=None):
        buf = ctypes.create_string_buffer(pixels, max(len(pixels), 1))
        b = Bitmap(pixels=ctypes.cast(buf, ctypes.c_void_p), row_bytes=row_bytes if row_bytes is not None else w * 4,
                   color_type=ct, alpha_type=1, w=w, h=h)
        n = ctypes.c_uint32(0)
        p = self.S.PerformOCR(ctypes.byref(b), ctypes.byref(n))
        if not p: return 'NULL'
        out = ctypes.string_at(p, n.value)
        self.S.FreeLibraryAllocatedCharArray(p)
        return 'EMPTY' if not out else '%s:%d' % (hashlib.sha1(out).hexdigest()[:12], len(out))


def load_rgba(path):
    from PIL import Image
    im = Image.open(path).convert('RGBA')
    return im.tobytes(), im.width, im.height


# ---- fuzz cases -------------------------------------------------------------------------------------------------
SIZES = [1, 2, 3, 7, 8, 15, 16, 31, 32, 33, 63, 64, 100, 127, 128, 129, 255, 256, 640, 1023, 1024, 1279, 1280, 1281,
         2047, 2048, 2049, 4095, 4096, 8191, 8192, 32767, 32768, 32769, 65536]
BAD_DIMS = [0, -1, -2, -32768, -2147483648, 2147483647]
COLOR_TYPES = [4, 6, 14]
OTHER_COLOR_TYPES = [-1, 0, 1, 2, 3, 5, 7, 13, 15, 255]
MAX_AREA = 40_000_000  # keeps one RGBA bitmap below ~160 MB


def make_text_case(r, i, text_images):
    w, h = r.randint(16, 3000), r.randint(16, 3000)
    while w * h > MAX_AREA: w, h = max(1, w // 2), max(1, h // 2)
    return dict(id=i, w=w, h=h, ct=r.choice(COLOR_TYPES), content='text', src=r.choice(text_images),
                mode=r.choice(['resize', 'resize', 'crop', 'tile', 'invert', 'lowcontrast', 'jpeg', 'noise_over']),
                seed=r.randrange(1 << 30), pad=0)


def make_case(r, i, text_images):
    """A (description, kwargs) pair; kwargs feed build_case."""
    kind = r.random()
    if kind < 0.06:  # bad dimensions, with a small buffer: must not be read past
        w = r.choice(BAD_DIMS + SIZES[:6]); h = r.choice(BAD_DIMS + SIZES[:6])
        if w > 0 and h > 0: w = r.choice(BAD_DIMS)
        return dict(id=i, w=w, h=h, ct=r.choice(COLOR_TYPES), content='noise', buf=4096)
    if kind < 0.10:  # unsupported color types
        return dict(id=i, w=r.choice(SIZES[:20]), h=r.choice(SIZES[:20]), ct=r.choice(OTHER_COLOR_TYPES), content='noise')
    ct = r.choice(COLOR_TYPES)
    if kind < 0.45:  # resized/cropped real text pages
        src = r.choice(text_images)
        w = r.choice(SIZES[:-2] + [r.randint(1, 3000) for _ in range(5)])
        h = r.choice(SIZES[:-2] + [r.randint(1, 3000) for _ in range(5)])
        while w * h > MAX_AREA: w, h = max(1, w // 2), max(1, h // 2)
        return dict(id=i, w=w, h=h, ct=ct, content='text', src=src, mode=r.choice(['resize', 'crop', 'tile', 'invert', 'lowcontrast', 'jpeg', 'noise_over']),
                    seed=r.randrange(1 << 30), pad=r.choice([0, 0, 0, 4, 64, 1000]))
    w = r.choice(SIZES); h = r.choice(SIZES)
    while w * h > MAX_AREA: w, h = max(1, w // 2), max(1, h // 2)
    return dict(id=i, w=w, h=h, ct=ct, content=r.choice(['blank', 'black', 'noise', 'stripes', 'checker', 'gradient']),
                seed=r.randrange(1 << 30), pad=r.choice([0, 0, 0, 4, 64]))


def build_case(c):
    """Pixel bytes, row bytes for case c (deterministic)."""
    import numpy as np
    from PIL import Image, ImageOps
    w, h, ct = c['w'], c['h'], c['ct']
    if 'buf' in c or w <= 0 or h <= 0:
        return bytes(c.get('buf', 4096)), 0
    ch = 1 if ct == 14 else 4
    rng = np.random.default_rng(c.get('seed', 0))
    content = c['content']
    if content == 'text':
        im = Image.open(os.path.join(ROOT, c['src'])).convert('RGB')
        mode = c['mode']
        if mode == 'crop':
            x0 = int(rng.integers(0, max(1, im.width - 1))); y0 = int(rng.integers(0, max(1, im.height - 1)))
            im = im.crop((x0, y0, x0 + w, y0 + h))
        elif mode == 'tile':
            t = Image.new('RGB', (w, h), 'white')
            for y in range(0, h, im.height):
                for x in range(0, w, im.width): t.paste(im, (x, y))
            im = t
        else:
            im = im.resize((w, h), Image.Resampling.BILINEAR)
            if mode == 'invert': im = ImageOps.invert(im)
            elif mode == 'lowcontrast': im = Image.eval(im, lambda v: 110 + v // 8)
            elif mode == 'jpeg':
                import io
                b = io.BytesIO(); im.save(b, 'JPEG', quality=int(rng.integers(5, 60))); im = Image.open(b).convert('RGB')
            elif mode == 'noise_over':
                a = np.asarray(im).astype(np.int16) + rng.integers(-60, 60, (h, w, 3))
                im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        im = im.resize((w, h)) if im.size != (w, h) else im
        a = np.asarray(im.convert('L' if ch == 1 else 'RGBA'))
    else:
        shape = (h, w) if ch == 1 else (h, w, 4)
        if content == 'blank': a = np.full(shape, 255, np.uint8)
        elif content == 'black': a = np.zeros(shape, np.uint8)
        elif content == 'noise': a = rng.integers(0, 256, shape, dtype=np.uint8)
        elif content == 'stripes': a = np.broadcast_to(((np.arange(h) // 3) % 2 * 255).astype(np.uint8).reshape((h, 1) + (() if ch == 1 else (1,))), shape).copy()
        elif content == 'checker': a = ((((np.arange(h)[:, None] // 5) + (np.arange(w)[None, :] // 5)) % 2) * 255).astype(np.uint8); a = a if ch == 1 else np.repeat(a[:, :, None], 4, 2)
        else: a = np.broadcast_to((np.arange(w) * 255 // max(1, w - 1)).astype(np.uint8)[None, :] if ch == 1 else (np.arange(w) * 255 // max(1, w - 1)).astype(np.uint8)[None, :, None], shape).copy()
        if ch == 4: a[..., 3] = 255
    a = np.ascontiguousarray(a).reshape(h, w * ch)
    pad = c.get('pad', 0)
    if pad:  # row padding filled with garbage: a library that honours row_bytes must skip it
        a = np.concatenate([a, rng.integers(0, 256, (h, pad), dtype=np.uint8)], 1)
    return a.tobytes(), w * ch + pad


# ---- worker / supervisor ----------------------------------------------------------------------------------------
def worker(lib, model_dir, cases_file):
    L = Lib(lib, model_dir)
    light = os.environ.get('API_ROBUST_LIGHT') == '1'
    L.init(light)
    for line in open(cases_file):
        c = json.loads(line)
        if c.get('skip'): continue
        px, rb = build_case(c)
        print('START', c['id'], flush=True)
        res = L.ocr(px, c['w'], c['h'], c['ct'], rb)
        print('RESULT', c['id'], res, flush=True)
        if res == 'NULL':  # the original stays failed after a failed call until it is re-initialized
            L.S.UninitializeOCR(); L.init(light)


def child_env():
    env = dict(os.environ)
    if env.get('API_ROBUST_INSERT'): env['DYLD_INSERT_LIBRARIES'] = env['API_ROBUST_INSERT']
    return env


def supervise(argv_for_worker, case_ids, env=None):
    """Runs the worker until all cases are done; a case during which the worker died is a CRASH."""
    results, done = {}, set()
    while len(done) < len(case_ids):
        p = subprocess.run(argv_for_worker, capture_output=True, text=True, errors='replace', env=env)
        started = None
        for line in p.stdout.splitlines():
            parts = line.split()
            if parts[0] == 'START': started = int(parts[1])
            elif parts[0] == 'RESULT':
                results[int(parts[1])] = parts[2]; done.add(int(parts[1])); started = None
        if p.returncode != 0:
            if started is None:
                sys.stderr.write(p.stderr[-3000:]); raise SystemExit('worker failed outside a case')
            sig = -p.returncode if p.returncode < 0 else p.returncode
            results[started] = 'CRASH(%d)' % sig; done.add(started)
            tail = [l for l in p.stderr.splitlines() if 'ERROR' in l or 'runtime error' in l or 'SUMMARY' in l][:3]
            results[started] += ' ' + ' | '.join(tail)[:300] if tail else ''
            # skip finished and crashed cases on restart
            lines = [json.loads(l) for l in open(argv_for_worker[-1])]
            for c in lines: c['skip'] = c['id'] in done
            with open(argv_for_worker[-1], 'w') as f:
                for c in lines: f.write(json.dumps(c) + '\n')
        elif len(done) < len(case_ids):
            raise SystemExit('worker exited early')
    return results


def check_fuzz(lib, model_dir, n, seed, text_only=False):
    import glob
    text = sorted(os.path.relpath(p, ROOT) for p in glob.glob(os.path.join(ROOT, 'testdata', '**', '*.png'), recursive=True)
                  if '/fuzz/' not in p and '/extra_fuzz/' not in p)
    r = random.Random(seed)
    cases = [(make_text_case if text_only else make_case)(r, i, text) for i in range(n)]
    cf = os.path.join(os.environ.get('TMPDIR', '/tmp'), 'api_robust_cases_%d.jsonl' % os.getpid())
    with open(cf, 'w') as f:
        for c in cases: f.write(json.dumps(c) + '\n')
    res = supervise([sys.executable, '-I', __file__, lib, model_dir, '_worker', cf], [c['id'] for c in cases], child_env())
    for c in cases:
        desc = '%s %dx%d ct=%d %s' % (c['id'], c['w'], c['h'], c['ct'], c['content'] + ('/' + c['mode'] if 'mode' in c else '') + ('/pad%d' % c['pad'] if c.get('pad') else ''))
        print(desc, res[c['id']])
    os.unlink(cf)


def check_lifecycle(lib, model_dir):
    px, w, h = load_rgba(os.path.join(ROOT, 'testdata', 'en_simple.png'))
    steps = []
    def run_case(name, fn):
        p = subprocess.run([sys.executable, '-I', __file__, lib, model_dir, '_life', name], capture_output=True, text=True, errors='replace',
                           env=child_env())
        out = p.stdout.strip().splitlines()
        print('%-28s %s' % (name, ' '.join(out) if p.returncode == 0 else 'CRASH(%d) %s' % (-p.returncode, ' '.join(out))))
    for name in ['perform_before_init', 'perform_no_callbacks', 'init_no_callbacks', 'double_init', 'uninit_perform',
                 'uninit_reinit', 'uninit_twice', 'maxdim_before_init', 'maxdim_after_uninit', 'light_mode_true', 'light_before_init',
                 'light_toggle', 'light_after_uninit']:
        run_case(name, None)


def life(lib, model_dir, name):
    px, w, h = load_rgba(os.path.join(ROOT, 'testdata', 'en_simple.png'))
    L = Lib(lib, model_dir)
    S = L.S
    if name == 'perform_before_init': L.set_callbacks(); print(L.ocr(px, w, h, 4))
    elif name == 'perform_no_callbacks': print(L.ocr(px, w, h, 4))
    elif name == 'init_no_callbacks': print(S.InitOCRUsingCallback())
    elif name == 'double_init': print(L.init(), L.init(), L.ocr(px, w, h, 4))
    elif name == 'uninit_perform': print(L.init(), S.UninitializeOCR(), L.ocr(px, w, h, 4))
    elif name == 'uninit_reinit': print(L.init(), S.UninitializeOCR(), L.init(), L.ocr(px, w, h, 4))
    elif name == 'uninit_twice': print(L.init(), S.UninitializeOCR(), S.UninitializeOCR(), 'ok')
    elif name == 'maxdim_before_init': print(S.GetMaxImageDimension())
    elif name == 'maxdim_after_uninit': print(L.init(), S.UninitializeOCR(), S.GetMaxImageDimension())
    elif name == 'light_mode_true': L.init(); S.SetOCRLightMode(True); print(L.ocr(px, w, h, 4))
    elif name == 'light_before_init':
        L.set_callbacks(); S.SetOCRLightMode(True); print(S.InitOCRUsingCallback(), L.ocr(px, w, h, 4))
    elif name in ('light_toggle', 'light_after_uninit'):  # a page whose result depends on the mode
        mx, mw, mh = load_rgba(os.path.join(ROOT, 'testdata', 'multiscript.png'))
        if name == 'light_toggle':
            L.init(); a = L.ocr(mx, mw, mh, 4); S.SetOCRLightMode(True); b = L.ocr(mx, mw, mh, 4)
            S.SetOCRLightMode(False); print(a, b, L.ocr(mx, mw, mh, 4))
        else:  # the setting survives UninitializeOCR
            L.init(); S.SetOCRLightMode(True); S.UninitializeOCR(); print(S.InitOCRUsingCallback(), L.ocr(mx, mw, mh, 4))


def check_faults(lib, model_dir, files_lib=None):
    # which files does a full run read? (en_simple + a Cyrillic and an Arabic page to load script recognizers)
    imgs = ['testdata/en_simple.png', 'testdata/extra_lm/cyrl.png', 'testdata/extra_lm/arab.png']
    p = subprocess.run([sys.executable, '-I', __file__, files_lib or lib, model_dir, '_reads'] + imgs, capture_output=True, text=True, errors='replace',
                       env=child_env())
    files = sorted(set(l for l in p.stdout.split('\n') if l.strip()))
    print('files read:', len(files))
    for f in files:
        for kind in ['missing', 'truncated', 'garbage', 'flipped']:
            p = subprocess.run([sys.executable, '-I', __file__, lib, model_dir, '_fault', f, kind] + imgs,
                               capture_output=True, text=True, errors='replace', timeout=600, env=child_env())
            out = ' '.join(p.stdout.split())
            if p.returncode != 0:
                tail = [l for l in p.stderr.splitlines() if 'ERROR' in l or 'runtime error' in l][:2]
                out = 'CRASH(%d) %s %s' % (-p.returncode if p.returncode < 0 else p.returncode, out, ' | '.join(tail)[:200])
            print('%-90s %-9s %s' % (f, kind, out), flush=True)


def reads(lib, model_dir, imgs):
    L = Lib(lib, model_dir)
    L.init()
    for i in imgs: L.ocr(*load_rgba(os.path.join(ROOT, i)), 4)
    print('\n'.join(r.lstrip('/') for r in L.reads))


def fault(lib, model_dir, f, kind, imgs):
    L = Lib(lib, model_dir, (f, kind))
    ok = L.init()
    print('init=%s' % ok, *[L.ocr(*load_rgba(os.path.join(ROOT, i)), 4) for i in imgs] if ok else [])


def rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1 << 20) if sys.platform == 'darwin' else r / 1024


def current_rss_mb():
    out = subprocess.run(['ps', '-o', 'rss=', '-p', str(os.getpid())], capture_output=True, text=True, errors='replace').stdout
    return int(out.strip()) / 1024


def check_threads(lib, model_dir, k, imgs):
    L = Lib(lib, model_dir)
    L.init()
    data = [load_rgba(i) for i in imgs]
    ref = [L.ocr(*d, 4) for d in data]
    errors, lock = [], threading.Lock()
    def run(t):
        for rep in range(3):
            for j in range((t + rep) % len(data), (t + rep) % len(data) + len(data)):
                j %= len(data)
                out = L.ocr(*data[j], 4)
                if out != ref[j]:
                    with lock: errors.append((t, imgs[j], out, ref[j]))
    ths = [threading.Thread(target=run, args=(t,)) for t in range(k)]
    for t in ths: t.start()
    for t in ths: t.join()
    print('threads %d x %d images x 3: %d mismatches' % (k, len(imgs), len(errors)))
    for e in errors[:5]: print('  ', e)


def check_soak(lib, model_dir, n, imgs):
    L = Lib(lib, model_dir)
    L.init()
    data = [load_rgba(i) for i in imgs]
    ref = [L.ocr(*d, 4) for d in data]
    print('after warm-up: rss %.0f MB' % current_rss_mb(), flush=True)
    changed = 0
    for r in range(n):
        for j, d in enumerate(data):
            if L.ocr(*d, 4) != ref[j]: changed += 1
        if r % max(1, n // 10) == 0 or r == n - 1:
            print('round %d: rss %.0f MB, peak %.0f MB, changed outputs %d' % (r, current_rss_mb(), rss_mb(), changed), flush=True)


if __name__ == '__main__':
    lib, model_dir, check = sys.argv[1], sys.argv[2], sys.argv[3]
    a = sys.argv[4:]
    if check == 'fuzz': check_fuzz(lib, model_dir, int(a[0]), int(a[1]) if len(a) > 1 else 1, len(a) > 2 and a[2] == 'text')
    elif check == '_worker': worker(lib, model_dir, a[0])
    elif check == 'lifecycle': check_lifecycle(lib, model_dir)
    elif check == '_life': life(lib, model_dir, a[0])
    elif check == 'faults': check_faults(lib, model_dir, a[0] if a else None)
    elif check == '_reads': reads(lib, model_dir, a)
    elif check == '_fault': fault(lib, model_dir, a[0], a[1], a[2:])
    elif check == 'threads': check_threads(lib, model_dir, int(a[0]), a[1:])
    elif check == 'soak': check_soak(lib, model_dir, int(a[0]), a[1:])
    else: raise SystemExit(__doc__)
