"""lldb script: run the original's own std::sort instantiations (the detector's six sorts) on tie-heavy arrays.

Each site's std::__introsort<...> from libchromescreenai.so is called on generated arrays whose sort keys repeat
a lot; inputs and outputs are written so cpp/tools/detector_sort_check can replay them through detsort::Sort
(cpp/src/detector_sort.h) and require the identical element order.

Usage: DS_OUT=dir lldb -b -o 'command script import trace/det_sort_check.py' -o run -- \
         .venv/bin/python ref/run_ref.py testdata/en_simple.png
Files: <DS_OUT>/<site>.bin: [u32 elem_size][u32 ncases] then per case [u32 n][n*elem input][n*elem output];
every element carries its input index as u32 at offset 0 (or 4 when the key is at 0) and its key elsewhere.
"""
import lldb, os, random, struct

OUT = os.environ.get('DS_OUT', '/tmp/det_sort')
DECODE = 0x1008dcf80
# site: (introsort address, element size, key offset, key format, comparator passed by pointer)
SITES = {
    'line_links': (0x100816f28, 24, 20, 'f', False),      # ConstructLines PairAndCost, cost ascending
    'curve_points': (0x1008e6270, 8, 0, 'f', 'curve'),    # CreateCurveRepresentation Point, projection ascending
    'line_order': (0x100818320, 8, 0, 'f', True),         # pair<float,int>, gtl::OrderBy<First, Less>
    'paragraph_nms': (0x10080c510, 56, 48, 'f', False),   # NonMaximumSuppression ParagraphPrediction, score desc.
    'angle_estimation': (0x100810974, 12, 8, 'f', True),  # RobustAngleEstimation DirectionWithKey, key ascending
    'line_paragraph': (0x100819684, 288, 276, 'i', False),  # GroupRpnDetectorV2 Level1GroupedBox, paragraph asc.
}
done = {'v': False}

def key_seq(n, pattern, rng):
    if pattern.startswith('rand'): k = int(pattern[4:]); return [rng.randrange(k) for _ in range(n)]
    if pattern == 'equal': return [0] * n
    if pattern == 'asc': return [i // 3 for i in range(n)]
    if pattern == 'desc': return [(n - i) // 3 for i in range(n)]
    if pattern == 'organ': return [min(i, n - i) // 2 for i in range(n)]
    return [i % 5 for i in range(n)]

def make_case(n, pattern, rng, size, koff, kfmt):
    out = []
    tag_off = 4 if koff == 0 else 0
    for i, k in enumerate(key_seq(n, pattern, rng)):
        e = bytearray(size)
        struct.pack_into('<I', e, tag_off, i)
        struct.pack_into('<' + kfmt, e, koff, float(k) if kfmt == 'f' else int(k))
        out.append(bytes(e))
    return b''.join(out)

def _eval(frame, expr):
    opts = lldb.SBExpressionOptions(); opts.SetIgnoreBreakpoints(True); opts.SetTryAllThreads(False)
    v = frame.EvaluateExpression(expr, opts)
    if v.GetError().Fail(): raise RuntimeError(v.GetError().GetCString())
    return v.GetValueAsUnsigned()

def run_site(frame, slide, site, rng):
    fn, size, koff, kfmt, comp_kind = SITES[site]
    fn += slide
    proc = frame.GetThread().GetProcess(); err = lldb.SBError()
    sizes = list(range(2, 40)) + [60, 100, 127, 128, 129, 130, 200, 257, 300, 513, 1000]
    cases = [make_case(n, p, rng, size, koff, kfmt) for n in sizes
             for p in ('rand2', 'rand3', 'rand8', 'equal', 'asc', 'desc', 'organ', 'saw')]
    buf = _eval(frame, '(unsigned long)calloc(%d, 1)' % (size * 1100))
    comp = _eval(frame, '(unsigned long)calloc(64, 1)')
    if comp_kind == 'curve':  # {const Vector2<float>* dir, const Point* origin}: dir (1, 0), origin (0, 0)
        proc.WriteMemory(comp + 32, struct.pack('<ffff', 1.0, 0.0, 0.0, 0.0), err)
        proc.WriteMemory(comp, struct.pack('<QQ', comp + 32, comp + 40), err)
    out = [struct.pack('<II', size, len(cases))]
    for d in cases:
        n = len(d) // size
        proc.WriteMemory(buf, d, err)
        depth = 2 * (n.bit_length() - 1)
        if comp_kind:
            expr = '((void(*)(void*,void*,void*,long,bool))0x%x)((void*)0x%x,(void*)0x%x,(void*)0x%x,%d,true), 0' % (
                fn, buf, buf + n * size, comp, depth)
        else:
            expr = '((void(*)(void*,void*,long,bool))0x%x)((void*)0x%x,(void*)0x%x,%d,true), 0' % (
                fn, buf, buf + n * size, depth)
        _eval(frame, expr)
        out += [struct.pack('<I', n), d, proc.ReadMemory(buf, n * size, err)]
    open(os.path.join(OUT, site + '.bin'), 'wb').write(b''.join(out))

def decode_entry(frame, loc, d):
    if done['v']: return False
    done['v'] = True
    slide = frame.GetPC() - DECODE
    for site in SITES: run_site(frame, slide, site, random.Random(site))
    print('det_sort_check: wrote', OUT)
    frame.GetThread().GetProcess().Kill()
    return False

def __lldb_init_module(debugger, d):
    os.makedirs(OUT, exist_ok=True)
    bp = debugger.GetSelectedTarget().BreakpointCreateByName('google_ocr::DecodeRPNModelOutputs', 'libchromescreenai.so')
    bp.SetScriptCallbackFunction('%s.decode_entry' % __name__)
