"""lldb script: for each recognizer-sized Subgraph::Invoke (ntensors==HS_NT), save its input and then the input tensor of
every HardSwishEval<optimized> call made on that thread during the invoke.  Files: <inv>_in.bin, <inv>_hs<k>.bin (+ .json)."""
import lldb, struct, os, json
OUT = os.environ.get('HS_OUT', '/tmp/hs'); NT = int(os.environ.get('HS_NT', '398')); MAXI = int(os.environ.get('HS_MAX', '4'))
st = {'n': 0, 'cur': {}}
def _rd(proc, a, n):
    e = lldb.SBError(); return proc.ReadMemory(a, n, e)
def _q(proc, a): return struct.unpack('<Q', _rd(proc, a, 8))[0]
def _tensor(proc, t):
    typ, = struct.unpack('<i', _rd(proc, t, 4)); data, dims = _q(proc, t + 8), _q(proc, t + 16)
    scale, zp = struct.unpack('<fi', _rd(proc, t + 24, 8)); nb = _q(proc, t + 40)
    nd, = struct.unpack('<i', _rd(proc, dims, 4)); shape = list(struct.unpack('<%di' % nd, _rd(proc, dims + 4, 4 * nd)))
    name_p = _q(proc, t + 56); name = proc.ReadCStringFromMemory(name_p, 256, lldb.SBError()) if name_p else ''
    return dict(type=typ, shape=shape, scale=scale, zp=zp, name=name), _rd(proc, data, nb)
def inv(frame, bp_loc, d):
    proc = frame.GetThread().GetProcess(); sg = frame.FindRegister('x0').GetValueAsUnsigned()
    if _q(proc, sg + 40) != NT or st['n'] >= MAXI: return False
    tid = frame.GetThread().GetThreadID(); n = st['n']; st['n'] += 1; st['cur'][tid] = [n, 0]
    ib, ie = _q(proc, sg + 37 * 8), _q(proc, sg + 37 * 8 + 8); idx, = struct.unpack('<i', _rd(proc, ib, 4))
    meta, raw = _tensor(proc, _q(proc, sg + 56) + 112 * idx)
    open(os.path.join(OUT, '%02d_in.bin' % n), 'wb').write(raw); json.dump(meta, open(os.path.join(OUT, '%02d_in.json' % n), 'w'))
    return False
def hs(frame, bp_loc, d):
    tid = frame.GetThread().GetThreadID()
    if tid not in st['cur']: return False
    n, k = st['cur'][tid]; st['cur'][tid][1] += 1
    proc = frame.GetThread().GetProcess(); ctx = frame.FindRegister('x0').GetValueAsUnsigned(); node = frame.FindRegister('x1').GetValueAsUnsigned()
    ins = _q(proc, node); idx, = struct.unpack('<i', _rd(proc, ins + 4, 4))
    meta, raw = _tensor(proc, _q(proc, ctx + 16) + 112 * idx); meta['tensor_index'] = idx
    open(os.path.join(OUT, '%02d_hs%02d.bin' % (n, k)), 'wb').write(raw); json.dump(meta, open(os.path.join(OUT, '%02d_hs%02d.json' % (n, k)), 'w'))
    return False
def __lldb_init_module(debugger, d):
    os.makedirs(OUT, exist_ok=True); t = debugger.GetSelectedTarget()
    b = t.BreakpointCreateByName('tflite::Subgraph::Invoke()', 'libchromescreenai.so'); b.SetScriptCallbackFunction('hs_dump.inv')
    b = t.BreakpointCreateByRegex(r'HardSwishEval<\(tflite::ops::builtin::activations::KernelType\)1>', 'libchromescreenai.so'); b.SetScriptCallbackFunction('hs_dump.hs')
