# SPDX-License-Identifier: MPL-2.0
"""lldb script: dump every TFLite Subgraph::Invoke input/output tensor of libchromescreenai.so as raw .bin + .json metadata
(lldb's embedded Python has no numpy; load with trace/load_dump.py).

Usage: lldb -b -o 'command script import trace/tfl_dump.py' -o run -- .venv/bin/python ref/run_ref.py img.png
Env: TFL_DUMP_DIR (output dir), TFL_DUMP_MAX (max invokes to dump, default 100000).
"""
import lldb, os, struct, json

OUT = os.environ.get('TFL_DUMP_DIR', '/tmp/tfl_dump')
MAXN = int(os.environ.get('TFL_DUMP_MAX', '100000'))
ENTRY, POST = 0x1002cccd4, 0x1002ccce8  # tflite::Subgraph::Invoke entry; instruction after its InvokeImpl call
state = {'n': 0}

def _rd(proc, addr, n):
    err = lldb.SBError(); b = proc.ReadMemory(addr, n, err)
    if not err.Success(): raise RuntimeError(err.GetCString())
    return b
def _q(proc, a): return struct.unpack('<Q', _rd(proc, a, 8))[0]

def _tensors(proc, sg, vec_off):
    beg, end = _q(proc, sg + vec_off), _q(proc, sg + vec_off + 8)
    idx = struct.unpack('<%di' % ((end - beg) // 4), _rd(proc, beg, end - beg)) if end > beg else ()
    tbase = _q(proc, sg + 56)
    res = []
    for i in idx:
        t = tbase + 112 * i
        typ, = struct.unpack('<i', _rd(proc, t, 4))
        data, dims = _q(proc, t + 8), _q(proc, t + 16)
        scale, zp = struct.unpack('<fi', _rd(proc, t + 24, 8))
        nbytes = _q(proc, t + 40)
        name_p = _q(proc, t + 56)
        name = ''
        if name_p:
            raw = _rd(proc, name_p, 128); name = raw.split(b'\0')[0].decode('utf8', 'replace')
        nd, = struct.unpack('<i', _rd(proc, dims, 4))
        shape = struct.unpack('<%di' % nd, _rd(proc, dims + 4, 4 * nd))
        arr = _rd(proc, data, nbytes) if (data and nbytes and typ != 5) else None
        res.append(dict(index=i, name=name, type=typ, shape=list(shape), scale=scale, zp=zp, arr=arr))
    return res

def _save(kind, ts, n=None):
    n = state['n'] if n is None else n
    meta = []
    for k, t in enumerate(ts):
        if t['arr'] is not None:
            with open(os.path.join(OUT, '%05d_%s_%d.bin' % (n, kind, k)), 'wb') as f: f.write(t['arr'])
        meta.append({x: t[x] for x in ('index', 'name', 'type', 'shape', 'scale', 'zp')})
    with open(os.path.join(OUT, '%05d_%s.json' % (n, kind)), 'w') as f: json.dump(meta, f)

def pre(frame, bp_loc, d):
    if 'post' not in state:  # first hit: compute the ASLR slide and plant the post-invoke breakpoint
        t = frame.GetThread().GetProcess().GetTarget()
        slide = frame.GetPC() - ENTRY
        bp = t.BreakpointCreateByAddress(POST + slide)
        bp.SetScriptCallbackFunction('%s.post' % __name__)
        state['post'] = bp
    if state['n'] >= MAXN: return False
    proc = frame.GetThread().GetProcess()
    sg = frame.FindRegister('x0').GetValueAsUnsigned()
    # invokes run concurrently on worker threads: keep inputs per thread until the matching post hit
    state.setdefault('pending', {})[frame.GetThread().GetThreadID()] = _tensors(proc, sg, 37 * 8)
    return False

def post(frame, bp_loc, d):
    if state['n'] >= MAXN: return False
    proc = frame.GetThread().GetProcess()
    sg = frame.FindRegister('x19').GetValueAsUnsigned()
    ins = state.get('pending', {}).pop(frame.GetThread().GetThreadID(), None)
    if ins is not None: _save('in', ins)
    if os.environ.get('TFL_DUMP_SGINFO'):
        import hashlib
        ntens = _q(proc, sg + 40); tbase = _q(proc, sg + 56); h = hashlib.md5(); nro = 0
        for i in range(ntens):
            t = tbase + 112 * i
            alloc, = struct.unpack('<i', _rd(proc, t + 32, 4))
            if alloc == 1 and nro < 40:
                data, nb = _q(proc, t + 8), _q(proc, t + 40)
                if data and nb: h.update(_rd(proc, data, min(nb, 4096))); nro += 1
        with open(os.path.join(OUT, '%05d_sg.json' % state['n']), 'w') as f:
            json.dump(dict(sg=sg, thread=frame.GetThread().GetThreadID(), ntensors=ntens, weights_md5=h.hexdigest()), f)
    if os.environ.get('TFL_DUMP_INPOST'): _save('inpost', _tensors(proc, sg, 37 * 8))
    _save('out', _tensors(proc, sg, 40 * 8))
    state['n'] += 1
    return False

def __lldb_init_module(debugger, d):
    os.makedirs(OUT, exist_ok=True)
    t = debugger.GetSelectedTarget()
    bp = t.BreakpointCreateByName('tflite::Subgraph::Invoke()', 'libchromescreenai.so')
    bp.SetScriptCallbackFunction('%s.pre' % __name__)
