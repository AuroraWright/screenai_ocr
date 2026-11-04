# SPDX-License-Identifier: MPL-2.0
"""lldb script: run the original's overlap-area routines on generated box pairs and box sets, for
cpp/tools/overlap_check.cc (which replays them through cpp/src/box_overlap.h and requires bit-identical results).

Functions called (libchromescreenai.so, slid):
  Rectangle<float>::CreateFromLTWHA 0x100c4aec4 + Rectangle<float>::OverlapArea 0x100c4b134 (-> IntersectAxisAligned
  NonAxisAligned 0x100c4b288), SafeBoxOverlapArea<Box<float>> 0x10084d5d4 (ClusterSort nesting),
  SafeOverlapBoxes<goodoc::RotatedBoundingBox> 0x100842280, OverlapComputer<goodoc::RotatedBoundingBox> 0x1008689b4
  and OverlapComputer<RotatedBox<float>> 0x10086929c (word pruning / curved overlaps).
Detector (GroupRpnDetectorV2): AxisAlignedBoxOverlapArea<RotatedBox<float>> 0x10080de18, the both-angles-0 branch of the
  inlined overlap in ProcessImagePyramidV2's paragraph NMS and in GetPairwiseConnections (else CreateFromLTWHA x2 +
  OverlapArea), and OverlapComputer<RotatedBox<float> const> 0x100812194 (paragraph assignment: level-1 boxes, then
  paragraph predictions, split = number of level-1 boxes).

Usage: OC_OUT=dir lldb -b -o 'command script import trace/overlap_check.py' -o run -- \
         .venv/bin/python ref/run_ref.py testdata/en_simple.png
Files in OC_OUT (little endian):
  pairs.bin: u32 n, then per pair: u32 category, 10 f32 (a x,y,w,h,angle, b ...), f32 rect_area, f32 safe_area
             (safe_area NaN-boxed 0x7fc0dead when the StatusOr is not OK).
  rbb.bin:   u32 n, then per pair: u32 category, a (i32 l,t,w,h, f32 angle), b (same), u32 overlaps, f32 iou, o1, o2.
  sets.bin:  u32 n, then per set: u32 kind (0 RotatedBoundingBox, 1 RotatedBox<float>), i32 split (-1: none),
             u32 nbox, boxes (kind 0: i32 l,t,w,h + f32 angle; kind 1: 5 f32), u32 nentries, entries (i32 a, i32 b,
             f32 iou, over_first, over_second) in map order.
  det_pairs.bin: u32 n, then per pair: u32 category, 10 f32 (a, b), f32 area(a, b), f32 area(b, a) of the detector's
             overlap (both raw angles == 0 -> AxisAlignedBoxOverlapArea<RotatedBox<float>>, else Rectangle<float>).
  det_sets.bin: sets.bin layout (kind 2: 5 f32 per box) from OverlapComputer<RotatedBox<float> const>.
Categories: see CATEGORIES (det_threshold: I / area(a) around the detector's thresholds DET_THRESHOLDS).
"""
import lldb, math, os, random, struct

OUT = os.environ.get('OC_OUT', '/tmp/overlap_check')
CREATE, OVERLAP, SAFE_BOX = 0x100c4aec4, 0x100c4b134, 0x10084d5d4
SAFE_RBB, OC_RBB, OC_RBF = 0x100842280, 0x1008689b4, 0x10086929c
AA_RBF, OC_RBF_CONST = 0x10080de18, 0x100812194
CATEGORIES = ['random', 'tiny_angle', 'near_0', 'near_90', 'near_180', 'identical', 'nested', 'edge_touch',
              'corner_touch', 'threshold', 'det_threshold']
THRESHOLDS = [0.1, 0.25, 0.3, 0.5, 0.6]
# Detector: level-1 grouping I / area > 0.3 (centers distance < 0.1) or 0.4, paragraph NMS > 0.75, paragraph
# assignment > 0.4.
DET_THRESHOLDS = [0.3, 0.4, 0.75]
done = {'v': False}


def f32(x): return struct.unpack('<f', struct.pack('<f', x))[0]  # round to float32 (exact for + - * / of floats)

def f32_next(x, up):
    i = struct.unpack('<i', struct.pack('<f', x))[0]
    if x == 0: return struct.unpack('<f', struct.pack('<i', 1 if up else -2147483647))[0]
    i += 1 if (up == (x > 0)) else -1
    return struct.unpack('<f', struct.pack('<i', i))[0]


# ---- float32 emulation of Rectangle<float>::OverlapArea, only used to place threshold pairs ----

def _rect(b):
    a = f32(f32(b[4]) * f32(0.017453292))
    return [f32(b[0]), f32(b[1]), f32(b[2]), f32(b[3]), f32(math.cos(a)), f32(math.sin(a))]

def _clip(a, b):
    F = f32
    if b[4] == 1 and b[5] == 0:
        r, bt = F(b[2] + b[0]), F(b[1] + b[3])
        pts = [(b[0], b[1]), (r, b[1]), (r, bt), (b[0], bt)]
    else:
        x1, y1 = F(b[0] + F(b[2] * b[4])), F(b[1] + F(b[2] * b[5]))
        ch, sh = F(b[4] * b[3]), F(b[5] * b[3])
        pts = [(b[0], b[1]), (x1, y1), (F(x1 - sh), F(y1 + ch)), (F(b[0] - sh), F(b[1] + ch))]
    def clip(pts, v, vert, ge):
        out = []
        for i, p in enumerate(pts):
            q = pts[(i + 1) % len(pts)]
            pc, qc = (p[0], q[0]) if vert else (p[1], q[1])
            pin = pc >= v if ge else pc <= v
            cross = (qc < v if ge else qc > v) if pin else (qc > v if ge else qc < v)
            if pin: out.append(p)
            if cross:
                if vert: out.append((v, F(p[1] + F(F(F(v - p[0]) * F(q[1] - p[1])) / F(q[0] - p[0])))))
                else: out.append((F(p[0] + F(F(F(v - p[1]) * F(q[0] - p[0])) / F(q[1] - p[1]))), v))
        return out
    for v, vert, ge in [(a[1], False, True), (F(a[0] + a[2]), True, False), (F(a[1] + a[3]), False, False),
                        (a[0], True, True)]:
        pts = clip(pts, v, vert, ge)
        if len(pts) < 3: return 0.0
    s = 0.0
    for i in range(len(pts) - 1): s = F(s + F(F(pts[i][0] * pts[i + 1][1]) - F(pts[i][1] * pts[i + 1][0])))
    return abs(F(F(s + F(F(pts[-1][0] * pts[0][1]) - F(pts[-1][1] * pts[0][0]))) * 0.5))

def emu_area(A, B):
    F = f32
    a, b = _rect(A), _rect(B)
    if a[4] == 1 and a[5] == 0: return _clip(a, b)
    if b[4] == 1 and b[5] == 0: return _clip(b, a)
    c, s = a[4], a[5]
    ra = [F(F(c * a[0]) + F(s * a[1])), F(F(c * a[1]) - F(s * a[0])), a[2], a[3], 1.0, 0.0]
    rb = [F(F(c * b[0]) + F(s * b[1])), F(F(c * b[1]) - F(s * b[0])), b[2], b[3], F(F(c * b[4]) + F(s * b[5])),
          F(F(c * b[5]) - F(s * b[4]))]
    return _clip(ra, rb)


# ---- generated inputs ----

def tiny(rng): return rng.choice([-1, 1]) * 10 ** rng.uniform(-3, math.log10(0.5))

def rand_box(rng, angle=None):
    return [round(rng.uniform(0, 800), rng.choice([0, 0, 2])), round(rng.uniform(0, 800), rng.choice([0, 0, 2])),
            rng.choice([rng.randint(1, 400), rng.uniform(1, 400)]), rng.choice([rng.randint(1, 120), rng.uniform(1, 120)]),
            rng.uniform(-180, 180) if angle is None else angle]

def near(rng, a, angle):
    return [a[0] + rng.uniform(-0.6, 0.6) * a[2], a[1] + rng.uniform(-0.8, 0.8) * a[3],
            a[2] * rng.uniform(0.3, 1.5), a[3] * rng.uniform(0.5, 1.5), angle]

def along(a, d, e=0.0):  # point a's origin moved by d along its x axis and e along its y axis
    t = math.radians(a[4])
    return a[0] + d * math.cos(t) - e * math.sin(t), a[1] + d * math.sin(t) + e * math.cos(t)

def gen_pairs(rng):
    out = []
    add = lambda cat, a, b: out.append((CATEGORIES.index(cat), [f32(v) for v in a + b]))
    for _ in range(30000):
        a = rand_box(rng); add('random', a, near(rng, a, rng.uniform(-180, 180)))
    for _ in range(30000):
        a = rand_box(rng, rng.choice([0.0, tiny(rng)]))
        add('tiny_angle', a, near(rng, a, rng.choice([0.0, tiny(rng), a[4], a[4] + tiny(rng)])))
    for base, cat in [(0.0, 'near_0'), (90.0, 'near_90'), (-90.0, 'near_90'), (180.0, 'near_180'), (-180.0, 'near_180')]:
        for _ in range(4000):
            eps = rng.choice([0.0, 1e-5, -1e-5, 1e-3, -1e-3, 0.01, -0.01, tiny(rng)])
            a = rand_box(rng, base + eps)
            add(cat, a, near(rng, a, rng.choice([base, base - eps, base + tiny(rng), 0.0])))
    for _ in range(4000):
        a = rand_box(rng, rng.choice([0.0, tiny(rng), rng.uniform(-180, 180)])); add('identical', a, list(a))
    for _ in range(6000):
        a = rand_box(rng, rng.choice([0.0, tiny(rng), rng.uniform(-180, 180)]))
        fx, fy = rng.uniform(0, 0.4), rng.uniform(0, 0.4)
        x, y = along(a, fx * a[2], fy * a[3])
        b = [x, y, a[2] * rng.uniform(0.1, 1 - fx), a[3] * rng.uniform(0.1, 1 - fy), a[4] + rng.choice([0.0, tiny(rng)])]
        add('nested', *(rng.choice([(a, b), (b, a)])))
    for _ in range(6000):
        a = rand_box(rng, rng.choice([0.0, 0.0, tiny(rng), rng.uniform(-180, 180)]))
        side = rng.randrange(4)
        d, e = [(a[2], rng.uniform(-0.5, 0.5) * a[3]), (-a[2], 0.0), (rng.uniform(-0.5, 0.5) * a[2], a[3]),
                (0.0, -a[3])][side]
        x, y = along(a, d, e)
        b = [x, y, a[2], a[3], a[4]]
        if side == 1: b[0], b[1] = along(a, -a[2], 0.0)
        if side == 3: b[0], b[1] = along(a, 0.0, -a[3])
        add('edge_touch', a, b)
    for _ in range(4000):
        a = rand_box(rng, rng.choice([0.0, 0.0, tiny(rng), rng.uniform(-180, 180)]))
        x, y = along(a, a[2], a[3]); add('corner_touch', a, [x, y, rng.uniform(1, 200), rng.uniform(1, 80), a[4]])
    threshold_pairs(rng, add, 'threshold', THRESHOLDS, 1500, True)
    return out

def threshold_pairs(rng, add, cat, thresholds, count, iou):
    # Threshold pairs: b = a shifted along x by d with I / area(a) (or the iou) near t; then +-8 ulps of b.x.
    for k in range(count):
        t = thresholds[k % len(thresholds)]
        use_iou = iou and k % 7 == 0
        a = rand_box(rng, rng.choice([0.0, tiny(rng), tiny(rng), rng.uniform(-3, 3)]))
        b0 = list(a); b0[4] = a[4] + rng.choice([0.0, tiny(rng)]); b0[3] = a[3] * rng.choice([1.0, rng.uniform(0.8, 1.2)])
        area_a, area_b = f32(f32(a[2]) * f32(a[3])), f32(f32(b0[2]) * f32(b0[3]))
        def ratio(d):
            b = list(b0); b[0], b[1] = along(a, d, 0.0)
            i = emu_area(a, b)
            if i == 0: return 0.0, b
            return f32(i / f32(f32(area_a + area_b) - i)) if use_iou else f32(i / area_a), b
        lo, hi = 0.0, float(a[2])
        for _ in range(60):
            mid = (lo + hi) / 2
            if ratio(mid)[0] > t: lo = mid
            else: hi = mid
        b = ratio(lo)[1]
        x = f32(b[0])
        for s in range(-8, 9):
            bx = x
            for _ in range(abs(s)): bx = f32_next(bx, s > 0)
            add(cat, a, [bx] + b[1:])

def gen_det_pairs(rng, pairs):
    """The layout pairs (random, tiny angles, near 0 / +-90 / 180, identical, nested, touching, thresholds) plus
    sweeps around the detector's thresholds, with detector-like boxes (raw boxes 1..80 px, paragraphs larger)."""
    out = list(pairs)
    add = lambda cat, a, b: out.append((CATEGORIES.index(cat), [f32(v) for v in a + b]))
    threshold_pairs(rng, add, 'det_threshold', DET_THRESHOLDS, 3000, False)
    return out

def gen_det_sets(rng):
    """Level-1 boxes (small, near-horizontal or any angle) then paragraph predictions around them (split = number of
    level-1 boxes), as the paragraph assignment builds them; some sets without split; degenerate / touching boxes."""
    sets = []
    for k in range(3000):
        n1, n2 = rng.randint(1, 8), rng.randint(1, 6)
        ang = rng.choice([0.0, 0.0, tiny(rng), rng.uniform(-10, 10), rng.uniform(-180, 180)])
        base = rand_box(rng, ang)
        boxes = []
        for i in range(n1 + n2):
            b = near(rng, base, rng.choice([0.0, ang, ang + tiny(rng), tiny(rng), rng.uniform(-180, 180)]))
            if i >= n1 and rng.random() < 0.5:  # paragraph around several level-1 boxes
                b[0] -= rng.uniform(0, 30); b[1] -= rng.uniform(0, 30)
                b[2] *= rng.uniform(1, 3); b[3] *= rng.uniform(1, 4)
            if rng.random() < 0.05: b[rng.choice([2, 3])] = rng.choice([0.0, -3.0, 1e-3])  # degenerate
            if rng.random() < 0.1:  # touching the previous box
                p = boxes[-1] if boxes else base
                b = [p[0] + p[2], p[1], b[2], b[3], 0.0]
            if rng.random() < 0.05 and boxes: b = list(boxes[-1])  # identical
            boxes.append([f32(x) for x in b])
        split = n1 if k % 5 else -1
        sets.append((2, split, boxes))
    return sets

def gen_rbb_pairs(rng, pairs):
    out = []
    for cat, v in pairs[::3]:
        a = [int(round(x)) for x in v[0:4]] + [v[4]]
        b = [int(round(x)) for x in v[5:9]] + [v[9]]
        out.append((cat, a, b))
    return out

def gen_sets(rng):
    sets = []
    for k in range(3000):
        kind = k % 2
        n = rng.randint(2, 9)
        base = rand_box(rng, rng.choice([0.0, tiny(rng), rng.uniform(-10, 10)]))
        boxes = []
        for _ in range(n):
            b = near(rng, base, rng.choice([0.0, base[4], base[4] + tiny(rng), tiny(rng)]))
            if rng.random() < 0.08: b[rng.choice([2, 3])] = rng.choice([0, -3])  # degenerate
            if rng.random() < 0.15:  # touching the previous box
                p = boxes[-1] if boxes else base
                b = [p[0] + p[2], p[1], b[2], b[3], 0.0]
            boxes.append([int(round(x)) for x in b[:4]] + [f32(b[4])])
        split = rng.randint(1, n - 1) if rng.random() < 0.4 else -1
        sets.append((kind, split, boxes))
    return sets


# ---- calling the original ----

def _eval(frame, expr):
    opts = lldb.SBExpressionOptions(); opts.SetIgnoreBreakpoints(True); opts.SetTryAllThreads(False)
    opts.SetTimeoutInMicroSeconds(600 * 1000000)
    v = frame.EvaluateExpression(expr, opts)
    if v.GetError().Fail() and 'unknown error' not in (v.GetError().GetCString() or ''):
        raise RuntimeError('%s: %s' % (expr[:200], v.GetError().GetCString()))
    return v.GetValueAsUnsigned()

def _alloc(frame, n): return _eval(frame, '(unsigned long)calloc(%d, 1)' % max(n, 1))

def run_pairs(frame, slide, pairs):
    proc = frame.GetThread().GetProcess(); err = lldb.SBError()
    n = len(pairs)
    inp = _alloc(frame, 40 * n); out = _alloc(frame, 8 * n)
    proc.WriteMemory(inp, b''.join(struct.pack('<10f', *v) for _, v in pairs), err)
    expr = '''
struct R6 { float v[6]; };
struct SO { unsigned long s; float v; ~SO() {} };
struct BX { float x, y, w, h, a; int pad; void* cur; void* ver; };
typedef R6 (*CF)(float, float, float, float, float);
typedef float (*OA)(const void*, const void*);
typedef SO (*SB)(const BX&, const BX&);
float* in = (float*)0x%x; float* out = (float*)0x%x;
for (int i = 0; i < %d; i++) {
  float* p = in + 10 * i;
  R6 a = ((CF)0x%x)(p[0], p[1], p[2], p[3], p[4]);
  R6 b = ((CF)0x%x)(p[5], p[6], p[7], p[8], p[9]);
  out[2 * i] = ((OA)0x%x)(&a, &b);
  BX ba = {p[0], p[1], p[2], p[3], p[4], 0, 0, 0}, bb = {p[5], p[6], p[7], p[8], p[9], 0, 0, 0};
  SO r = ((SB)0x%x)(ba, bb);
  if (r.s == 1) out[2 * i + 1] = r.v; else ((unsigned*)out)[2 * i + 1] = 0x7fc0deadu;
}
0''' % (inp, out, n, CREATE + slide, CREATE + slide, OVERLAP + slide, SAFE_BOX + slide)
    _eval(frame, expr)
    res = proc.ReadMemory(out, 8 * n, err)
    with open(os.path.join(OUT, 'pairs.bin'), 'wb') as f:
        f.write(struct.pack('<I', n))
        for k, (cat, v) in enumerate(pairs):
            f.write(struct.pack('<I10f', cat, *v) + res[8 * k:8 * k + 8])

def _rbb(b): return struct.pack('<24x4if4x', b[0], b[1], b[2], b[3], b[4])  # goodoc::RotatedBoundingBox, 48 bytes

def run_rbb(frame, slide, pairs):
    proc = frame.GetThread().GetProcess(); err = lldb.SBError()
    n = len(pairs)
    inp = _alloc(frame, 96 * n); out = _alloc(frame, 16 * n)
    proc.WriteMemory(inp, b''.join(_rbb(a) + _rbb(b) for _, a, b in pairs), err)
    expr = '''
struct SO { unsigned long s; bool v; ~SO() {} };
typedef SO (*SF)(const void*, const void*, float*, float*, float*);
char* in = (char*)0x%x; float* out = (float*)0x%x;
for (int i = 0; i < %d; i++) {
  SO r = ((SF)0x%x)(in + 96 * i, in + 96 * i + 48, out + 4 * i + 1, out + 4 * i + 2, out + 4 * i + 3);
  ((unsigned*)out)[4 * i] = r.s == 1 ? (r.v ? 1u : 0u) : 2u;
}
0''' % (inp, out, n, SAFE_RBB + slide)
    _eval(frame, expr)
    res = proc.ReadMemory(out, 16 * n, err)
    with open(os.path.join(OUT, 'rbb.bin'), 'wb') as f:
        f.write(struct.pack('<I', n))
        for k, (cat, a, b) in enumerate(pairs):
            f.write(struct.pack('<I4if4if', cat, *a, *b) + res[16 * k:16 * k + 16])

def _read_map(proc, obj):
    err = lldb.SBError()
    q = lambda a: struct.unpack('<Q', proc.ReadMemory(a, 8, err))[0]
    out, stack, node = [], [], q(obj + 8)
    while stack or node:
        while node: stack.append(node); node = q(node)
        node = stack.pop()
        out.append(struct.unpack('<iifff', proc.ReadMemory(node + 28, 20, err)))
        node = q(node + 8)
    return out

def run_sets(frame, slide, sets):
    proc = frame.GetThread().GetProcess(); err = lldb.SBError()
    buf = _alloc(frame, 48 * 16); obj = _alloc(frame, 64); vec = _alloc(frame, 32)
    with open(os.path.join(OUT, 'sets.bin'), 'wb') as f:
        f.write(struct.pack('<I', len(sets)))
        for kind, split, boxes in sets:
            if kind == 0: data, size, fn = b''.join(_rbb(b) for b in boxes), 48, OC_RBB
            else: data, size, fn = b''.join(struct.pack('<5f', *b) for b in boxes), 20, OC_RBF
            proc.WriteMemory(buf, data, err)
            proc.WriteMemory(vec, struct.pack('<QQQ', buf, buf + len(data), buf + len(data)), err)
            proc.WriteMemory(obj, b'\0' * 64, err)
            opt = 0 if split < 0 else (1 << 32) | split
            _eval(frame, '((void*(*)(void*, const void*, unsigned long, bool))0x%x)((void*)0x%x, (const void*)0x%x, '
                  '%dul, false), 0' % (fn + slide, obj, vec, opt))
            entries = _read_map(proc, obj)
            f.write(struct.pack('<Iii', kind, split, len(boxes)))
            for b in boxes:
                f.write(struct.pack('<4if', *b) if kind == 0 else struct.pack('<5f', *b))
            f.write(struct.pack('<I', len(entries)) + b''.join(struct.pack('<iifff', *e) for e in entries))

def run_det_pairs(frame, slide, pairs):
    proc = frame.GetThread().GetProcess(); err = lldb.SBError()
    n = len(pairs)
    inp = _alloc(frame, 40 * n); out = _alloc(frame, 8 * n)
    proc.WriteMemory(inp, b''.join(struct.pack('<10f', *v) for _, v in pairs), err)
    expr = '''
struct R6 { float v[6]; };
typedef R6 (*CF)(float, float, float, float, float);
typedef float (*OA)(const void*, const void*);
typedef float (*AA)(const float*, const float*);
float* in = (float*)0x%x; float* out = (float*)0x%x;
for (int i = 0; i < %d; i++) {
  for (int o = 0; o < 2; o++) {
    float* p = in + 10 * i + 5 * o; float* q = in + 10 * i + 5 * (1 - o);
    if (p[4] == 0.0f && q[4] == 0.0f) {
      out[2 * i + o] = ((AA)0x%x)(p, q);
    } else {
      R6 a = ((CF)0x%x)(p[0], p[1], p[2], p[3], p[4]);
      R6 b = ((CF)0x%x)(q[0], q[1], q[2], q[3], q[4]);
      out[2 * i + o] = ((OA)0x%x)(&a, &b);
    }
  }
}
0''' % (inp, out, n, AA_RBF + slide, CREATE + slide, CREATE + slide, OVERLAP + slide)
    _eval(frame, expr)
    res = proc.ReadMemory(out, 8 * n, err)
    with open(os.path.join(OUT, 'det_pairs.bin'), 'wb') as f:
        f.write(struct.pack('<I', n))
        for k, (cat, v) in enumerate(pairs):
            f.write(struct.pack('<I10f', cat, *v) + res[8 * k:8 * k + 8])

def run_det_sets(frame, slide, sets):
    proc = frame.GetThread().GetProcess(); err = lldb.SBError()
    buf = _alloc(frame, 20 * 32); ptrs = _alloc(frame, 8 * 32); obj = _alloc(frame, 64); vec = _alloc(frame, 32)
    with open(os.path.join(OUT, 'det_sets.bin'), 'wb') as f:
        f.write(struct.pack('<I', len(sets)))
        for kind, split, boxes in sets:
            proc.WriteMemory(buf, b''.join(struct.pack('<5f', *b) for b in boxes), err)
            proc.WriteMemory(ptrs, b''.join(struct.pack('<Q', buf + 20 * i) for i in range(len(boxes))), err)
            end = ptrs + 8 * len(boxes)
            proc.WriteMemory(vec, struct.pack('<QQQ', ptrs, end, end), err)
            proc.WriteMemory(obj, b'\0' * 64, err)
            opt = 0 if split < 0 else (1 << 32) | split
            _eval(frame, '((void*(*)(void*, const void*, unsigned long, bool))0x%x)((void*)0x%x, (const void*)0x%x, '
                  '%dul, false), 0' % (OC_RBF_CONST + slide, obj, vec, opt))
            entries = _read_map(proc, obj)
            f.write(struct.pack('<Iii', kind, split, len(boxes)))
            for b in boxes: f.write(struct.pack('<5f', *b))
            f.write(struct.pack('<I', len(entries)) + b''.join(struct.pack('<iifff', *e) for e in entries))

def on_entry(frame, loc, d):
    if done['v']: return False
    done['v'] = True
    t = frame.GetThread().GetProcess().GetTarget()
    mod = t.FindModule(lldb.SBFileSpec('libchromescreenai.so'))
    slide = mod.GetObjectFileHeaderAddress().GetLoadAddress(t) - 0x100000000
    rng = random.Random(20261101)
    pairs = gen_pairs(rng)
    run_pairs(frame, slide, pairs)
    run_rbb(frame, slide, gen_rbb_pairs(rng, pairs))
    run_sets(frame, slide, gen_sets(rng))
    rng2 = random.Random(20261102)  # detector inputs (after the layout ones, which stay as they were)
    det_pairs = gen_det_pairs(rng2, pairs)
    run_det_pairs(frame, slide, det_pairs)
    run_det_sets(frame, slide, gen_det_sets(rng2))
    print('overlap_check: wrote', OUT, len(pairs), 'pairs,', len(det_pairs), 'detector pairs')
    frame.GetThread().GetProcess().Kill()
    return False

def __lldb_init_module(debugger, d):
    os.makedirs(OUT, exist_ok=True)
    bp = debugger.GetSelectedTarget().BreakpointCreateByName('google_ocr::DecodeRPNModelOutputs', 'libchromescreenai.so')
    bp.SetScriptCallbackFunction('%s.on_entry' % __name__)
