"""lldb script: at entry of a function in libchromescreenai.so, serialize proto messages passed in registers.

Env: PD_FUNC (demangled name, e.g. 'google_ocr::GroupRpnDetectorV2'), PD_REGS (e.g. 'x1,x2'), PD_OUT (dir), PD_MAX (hits).
Each hit writes <out>/<hit>_<reg>.pb using proto2::MessageLite::AppendToString.
"""
import lldb, os, struct
APPEND = 0x101bbc138  # proto2::MessageLite::AppendToString(std::string*) const
FUNC = os.environ.get('PD_FUNC'); REGS = os.environ.get('PD_REGS', 'x0').split(',')
OUT = os.environ.get('PD_OUT', '/tmp/pd'); MAXN = int(os.environ.get('PD_MAX', '3'))
st = {'n': 0}

def hit(frame, bp_loc, d):
    if st['n'] >= MAXN: return False
    t = frame.GetThread().GetProcess().GetTarget(); proc = t.GetProcess()
    mod = t.FindModule(lldb.SBFileSpec('libchromescreenai.so'))
    slide = mod.GetObjectFileHeaderAddress().GetLoadAddress(t) - 0x100000000
    for r in REGS:
        msg = frame.FindRegister(r).GetValueAsUnsigned()
        opts = lldb.SBExpressionOptions(); opts.SetIgnoreBreakpoints(True); opts.SetTryAllThreads(False); opts.SetStopOthers(True)
        v = frame.EvaluateExpression(
            'void *s = (void*)calloc(1, 24); ((void(*)(void*, void*))0x%x)((void*)0x%x, s); s' % (APPEND + slide, msg), opts)
        s = v.GetValueAsUnsigned(); err = lldb.SBError()
        raw = proc.ReadMemory(s, 24, err)
        # libc++ alternate string layout (as built here): long = {data, size, cap|0x80<<56}, short = {chars[23], size}
        if raw[23] & 0x80:
            data, size = struct.unpack_from('<QQ', raw, 0)
        else:
            size, data = raw[23] & 0x7f, s
        b = proc.ReadMemory(data, size, err) if size else b''
        with open(os.path.join(OUT, '%03d_%s.pb' % (st['n'], r)), 'wb') as f: f.write(b)
    st['n'] += 1
    return False

def __lldb_init_module(debugger, d):
    os.makedirs(OUT, exist_ok=True)
    t = debugger.GetSelectedTarget()
    bp = t.BreakpointCreateByRegex('^%s\\(' % FUNC.replace('::', '::'), 'libchromescreenai.so')
    bp.SetScriptCallbackFunction('%s.hit' % __name__)
