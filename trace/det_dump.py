"""lldb script: dump detector intermediates of libchromescreenai.so.

Usage: DD_OUT=dir lldb -b -o 'command script import trace/det_dump.py' -o run -- .venv/bin/python ref/run_ref.py img.png
Writes <DD_OUT>/pairs_<k>.bin: ConstructLines' PairAndCost links [u32 n][n*24 discovery order][n*24 sorted].
Writes <DD_OUT>/rawboxes_<k>.bin: for each DecodeRPNModelOutputs call k, [u32 nlists] then per list [u32 n][n*68 bytes RawBox].
DD_INJECT=<file>: edit the model outputs before DecodeRPNModelOutputs call k (= pass k: 0 horizontal heads, 1 vertical
heads) decodes them; the format is the C++ SCREENAI_DET_INJECT one (see OutputEdits in cpp/src/detector.cc).
Writes <DD_OUT>/inject.txt (per pass: tensor shapes, number of edited floats).
"""
import lldb, os, struct

OUT = os.environ.get('DD_OUT', '/tmp/det_dump')
state = {'decode': 0, 'pending': {}}

def _rd(proc, addr, n):
    if n == 0: return b''
    err = lldb.SBError(); b = proc.ReadMemory(addr, n, err)
    if not err.Success(): raise RuntimeError(err.GetCString())
    return b
def _q(proc, a): return struct.unpack('<Q', _rd(proc, a, 8))[0]

def _vecvec(proc, beg, end, elem):
    out = [struct.pack('<I', (end - beg) // 24)]
    for v in range(beg, end, 24):
        b, e = _q(proc, v), _q(proc, v + 8)
        out.append(struct.pack('<I', (e - b) // elem)); out.append(_rd(proc, b, e - b))
    return b''.join(out)

def _ret_bp(frame, fn):
    t = frame.GetThread().GetProcess().GetTarget()
    lr = frame.FindRegister('lr').GetValueAsUnsigned()
    bp = t.BreakpointCreateByAddress(lr)
    bp.SetOneShot(True)
    bp.SetScriptCallbackFunction('%s.%s' % (__name__, fn))

DECODE = 0x1008dcf80
L1_DONE = 0x100806500  # ProcessImagePyramidV2: trace_end("ComputeLevel1Boxes"); GroupedRawPredictions at sp+0x7d0

def _bp_at(frame, addr, fn):
    if ('bp', addr) in state: return
    t = frame.GetThread().GetProcess().GetTarget()
    bp = t.BreakpointCreateByAddress(addr + state['slide'])
    bp.SetScriptCallbackFunction('%s.%s' % (__name__, fn))
    state[('bp', addr)] = bp

PAIR_SORT_CALL = 0x100807888  # ConstructLines: BL std::__introsort<..PairAndCost*..>(x0=first, x1=last, ...)

def pair_sort(frame, bp_loc, d):
    proc = frame.GetThread().GetProcess()
    b, e = frame.FindRegister('x0').GetValueAsUnsigned(), frame.FindRegister('x1').GetValueAsUnsigned()
    state['pairs_pre'] = struct.pack('<I', (e - b) // 24) + _rd(proc, b, e - b)
    state['pairs_rng'] = (b, e)
    _bp_at(frame, PAIR_SORT_CALL + 4, 'pair_sorted')
    return False

def pair_sorted(frame, bp_loc, d):
    proc = frame.GetThread().GetProcess()
    b, e = state['pairs_rng']
    k = state.setdefault('pairs', 0)
    with open(os.path.join(OUT, 'pairs_%d.bin' % k), 'wb') as f: f.write(state['pairs_pre'] + _rd(proc, b, e - b))
    state['pairs'] += 1
    return False

def l1_done(frame, bp_loc, d):
    proc = frame.GetThread().GetProcess()
    g = frame.FindRegister('sp').GetValueAsUnsigned() + 0x7d0
    rb, re_ = _q(proc, g), _q(proc, g + 8)
    lb, le = _q(proc, g + 24), _q(proc, g + 32)
    out = [struct.pack('<I', (le - lb) // 288)]
    for e in range(lb, le, 288):
        out.append(_rd(proc, e, 288))
        mb, me = _q(proc, e + 208), _q(proc, e + 216)
        ptrs = struct.unpack('<%dQ' % ((me - mb) // 8), _rd(proc, mb, me - mb))
        out.append(struct.pack('<I', len(ptrs)) + b''.join(struct.pack('<i', (p - rb) // 68) for p in ptrs))
    with open(os.path.join(OUT, 'level1_%d.bin' % state.setdefault('l1', 0)), 'wb') as f: f.write(b''.join(out))
    with open(os.path.join(OUT, 'allraw_%d.bin' % state['l1']), 'wb') as f:
        f.write(struct.pack('<II', 1, (re_ - rb) // 68) + _rd(proc, rb, re_ - rb))
    state['l1'] += 1
    return False

def _edits():
    path = os.environ.get('DD_INJECT')
    if not path: return []
    out = []
    for line in open(path):
        f = line.split('#')[0].split()
        if not f: continue
        if f[0] == 'set' and len(f) == 9:
            v = f[8]
            v = float.fromhex(v) if v.lower().lstrip('+-').startswith('0x') else float(v)
            out.append(('set', int(f[1]), -1 if f[2] == '*' else int(f[2]), [int(x) for x in f[3:7]], int(f[7]),
                        struct.unpack('<f', struct.pack('<f', v))[0]))
        elif f[0] == 'copy' and len(f) == 9:
            out.append(('copy', int(f[1]), -1 if f[2] == '*' else int(f[2]), [int(x) for x in f[3:9]], 0, 0.0))
        else:
            raise ValueError('DD_INJECT: bad line: ' + line)
    return out

def _inject(proc, k, arr, n):
    edits = [e for e in _edits() if e[1] == k]
    log = []
    for ti in range(n):
        t = arr + 56 * ti
        data = _q(proc, t)
        sb, se = _q(proc, t + 24), _q(proc, t + 32)
        shape = struct.unpack('<%di' % ((se - sb) // 4), _rd(proc, sb, se - sb))
        _, H, W, C = shape
        mine = [e for e in edits if e[2] in (-1, ti)]
        if not mine:
            log.append('pass %d tensor %d shape %s edited 0' % (k, ti, shape)); continue
        buf = bytearray(_rd(proc, data, H * W * C * 4))
        orig = bytes(buf)
        for op, _, _, a, ch, val in mine:
            if op == 'set':
                if not 0 <= ch < C: continue
                y1 = min(H, a[0] + a[2]) if a[2] else H
                x1 = min(W, a[1] + a[3]) if a[3] else W
                pv = struct.pack('<f', val)
                for y in range(max(0, a[0]), y1):
                    for x in range(max(0, a[1]), x1):
                        o = ((y * W + x) * C + ch) * 4; buf[o:o + 4] = pv
            else:
                bh, bw = a[4] or H, a[5] or W
                snap = {}
                for i in range(bh):
                    for j in range(bw):
                        y, x = a[0] + i, a[1] + j
                        if 0 <= y < H and 0 <= x < W:
                            o = (y * W + x) * C * 4; snap[i, j] = bytes(buf[o:o + C * 4])
                for (i, j), v in snap.items():
                    y, x = a[2] + i, a[3] + j
                    if 0 <= y < H and 0 <= x < W:
                        o = (y * W + x) * C * 4; buf[o:o + C * 4] = v
        changed = sum(1 for o in range(0, len(buf), 4) if buf[o:o + 4] != orig[o:o + 4])
        err = lldb.SBError(); proc.WriteMemory(data, bytes(buf), err)
        if not err.Success(): raise RuntimeError(err.GetCString())
        log.append('pass %d tensor %d shape %s edited %d' % (k, ti, shape, changed))
    with open(os.path.join(OUT, 'inject.txt'), 'a') as f: f.write('\n'.join(log) + '\n')

def decode_entry(frame, bp_loc, d):
    if 'slide' not in state:
        state['slide'] = frame.GetPC() - DECODE
        _bp_at(frame, L1_DONE, 'l1_done')
        _bp_at(frame, PAIR_SORT_CALL, 'pair_sort')
    state['pending'][frame.GetThread().GetThreadID()] = frame.FindRegister('x8').GetValueAsUnsigned()
    if os.environ.get('DD_INJECT'):
        _inject(frame.GetThread().GetProcess(), state['decode'], frame.FindRegister('x2').GetValueAsUnsigned(),
                frame.FindRegister('x3').GetValueAsUnsigned())
    _ret_bp(frame, 'decode_ret')
    return False

def decode_ret(frame, bp_loc, d):
    proc = frame.GetThread().GetProcess()
    res = state['pending'].pop(frame.GetThread().GetThreadID())
    if _q(proc, res) == 1:
        data = _vecvec(proc, _q(proc, res + 8), _q(proc, res + 16), 68)
        with open(os.path.join(OUT, 'rawboxes_%d.bin' % state['decode']), 'wb') as f: f.write(data)
    state['decode'] += 1
    return False

def __lldb_init_module(debugger, d):
    os.makedirs(OUT, exist_ok=True)
    t = debugger.GetSelectedTarget()
    bp = t.BreakpointCreateByName('google_ocr::DecodeRPNModelOutputs', 'libchromescreenai.so')
    bp.SetScriptCallbackFunction('%s.decode_entry' % __name__)
