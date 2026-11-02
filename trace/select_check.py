"""lldb script: run the original's LanguageBasedLineRecognizerConfigSelector::Select (0x100148160) on a grid of
synthetic LINE entities (trace/select_grid.py) and record the recognizer key it returns.

At the first Select call of a real run, the live selector (this, holding the shipped multi-pass config) is reused:
every case is parsed into a fresh goodoc::PageLayoutEntity (ParsePartialFromArray) and passed to Select, all in one
expression. Output <SEL_DIR>/orig.txt: one key per case ('!ERR' when Select returns an error status), in grid
order; cpp/tools/select_check replays grid.txt through our SelectRecognizerKey and compares.

Usage: SEL_DIR=dir lldb -b -o 'command script import trace/select_check.py' -o run -- \
         .venv/bin/python ref/run_ref.py testdata/en_simple.png
"""
import lldb, os, struct

SELECT = 0x100148160
CTOR = 0x1014647d0      # goodoc::PageLayoutEntity::PageLayoutEntity(proto2::Arena*)
PARSE = 0x101bbb2a0     # proto2::MessageLite::ParsePartialFromArray(void const*, int)
DIR = os.environ.get('SEL_DIR', '/tmp/sel')
done = {'v': False}

def slide_of(t):
    mod = t.FindModule(lldb.SBFileSpec('libchromescreenai.so'))
    return mod.GetObjectFileHeaderAddress().GetLoadAddress(t) - 0x100000000

def _eval(frame, expr):
    opts = lldb.SBExpressionOptions(); opts.SetIgnoreBreakpoints(True); opts.SetTryAllThreads(False)
    opts.SetTimeoutInMicroSeconds(0)
    v = frame.EvaluateExpression(expr, opts)
    if v.GetError().Fail(): raise RuntimeError(v.GetError().GetCString())
    return v.GetValueAsUnsigned()

def read_string(proc, p):
    err = lldb.SBError(); raw = proc.ReadMemory(p, 24, err)
    if raw[23] & 0x80:  # libc++ long string
        data, size = struct.unpack('<QQ', raw[:16])
        return proc.ReadMemory(data, size, err).decode() if size else ''
    return raw[:raw[23]].decode()

def log(m):
    with open(os.path.join(DIR, 'log.txt'), 'a') as f: f.write(m + '\n')

def on_select(frame, loc, d):
    if done['v'] or frame.GetPC() != SELECT + slide_of(frame.GetThread().GetProcess().GetTarget()):
        return False  # the regex also hits Select's helpers (stable_sort of its $_0)
    done['v'] = True
    loc.GetBreakpoint().SetEnabled(False)  # the grid calls Select itself; trapping each call is slow
    proc = frame.GetThread().GetProcess(); t = proc.GetTarget(); err = lldb.SBError()
    slide = slide_of(t)
    this = frame.FindRegister('x0').GetValueAsUnsigned()
    log('hit Select this=0x%x slide=0x%x' % (this, slide))
    blob = open(os.path.join(DIR, 'grid.bin'), 'rb').read()
    n = struct.unpack_from('<I', blob)[0]
    if os.environ.get('SEL_N'): n = min(n, int(os.environ['SEL_N']))
    offs, pos = [], 4
    for _ in range(struct.unpack_from('<I', blob)[0]):
        ln = struct.unpack_from('<I', blob, pos)[0]; offs.append(pos + 4); pos += 4 + ln
    inp = _eval(frame, '(unsigned long)malloc(%d)' % len(blob))
    proc.WriteMemory(inp, blob, err)
    out = _eval(frame, '(unsigned long)calloc(%d, 24)' % n)
    offp = _eval(frame, '(unsigned long)malloc(%d)' % (4 * n))
    proc.WriteMemory(offp, struct.pack('<%dI' % n, *offs[:n]), err)
    ent = _eval(frame, '(unsigned long)calloc(1, 4096)')
    _eval(frame, '((void(*)(void*,void*))0x%x)((void*)0x%x, (void*)0), 0' % (CTOR + slide, ent))
    for i0 in range(0, n, 2000):  # chunks: a failing case shows up in the log with its range
        i1 = min(n, i0 + 2000)
        expr = ('struct R { unsigned long a, b, c; };'
                'unsigned char* in = (unsigned char*)0x%x; R* out = (R*)0x%x;'
                'for (int i = %d; i < %d; i++) {'
                '  unsigned pos = ((unsigned*)0x%x)[i];'
                '  unsigned len = *(unsigned*)(in + pos - 4);'
                '  ((bool(*)(void*,const void*,int))0x%x)((void*)0x%x, in + pos, (int)len);'
                '  out[i] = ((R(*)(void*,void*))0x%x)((void*)0x%x, (void*)0x%x);'
                '}'
                '0') % (inp, out, i0, i1, offp, PARSE + slide, ent, SELECT + slide, this, ent)
        log('select: cases %d..%d' % (i0, i1))
        _eval(frame, expr)
    log('select: done')
    res = proc.ReadMemory(out, 24 * n, err)
    cache, lines = {}, []
    for i in range(n):
        a, b, c = struct.unpack_from('<QQQ', res, 24 * i)
        if a != 1: lines.append('!ERR'); continue
        if b not in cache: cache[b] = read_string(proc, b)
        lines.append(cache[b])
    open(os.path.join(DIR, 'orig.txt'), 'w').write('\n'.join(lines) + '\n')
    print('select_check: %d cases' % n)
    return False

def __lldb_init_module(debugger, d):
    t = debugger.GetSelectedTarget()
    b = t.BreakpointCreateByRegex(r'LanguageBasedLineRecognizerConfigSelector::Select\(')
    b.SetScriptCallbackFunction('select_check.on_select')
