"""lldb script: serialize goodoc.PageLayoutProto between OCR pipeline stages of libchromescreenai.so.

At entry of each mutator's MutateSub(ctx) the context's PageLayout (ctx+176) holds the previous stage's
output; its PageLayoutProto* lives at PageLayout+16.  The final layout is captured at entry of
ConvertOCRResultsToVisualAnnotationProto(const PageLayoutProto&).
Env: SD_OUT (dir).  Files: <n>_<stage>.pb
"""
import lldb, os, struct
APPEND = 0x101bbc138  # proto2::MessageLite::AppendToString(std::string*) const
OUT = os.environ.get('SD_OUT', '/tmp/sd')
HOOKS = [  # (function regex, stage name, pointer recipe)
    (r'GocrGroupRpnTextDetectionMutator::MutateSub\(', 'det_in', ('x1', 176, 16)),
    (r'MultiPassLineRecognitionMutator::MutateSub\(google_ocr::PageLayoutMutatorContext ?\*\) ?const$', 'rec_in', ('x1', 176, 16)),
    (r'PageLayoutAnalysisMutator::MutateSub\(google_ocr::PageLayoutMutatorContext ?\*\) ?const$', 'lay_in', ('x1', 176, 16)),
    (r'ConvertOCRResultsToVisualAnnotationProto\(', 'final', ('x1',)),  # x0 = returned VisualAnnotation
]
st = {'n': 0}

def _q(proc, a):
    err = lldb.SBError(); return struct.unpack('<Q', proc.ReadMemory(a, 8, err))[0]

def _serialize(frame, msg):
    t = frame.GetThread().GetProcess().GetTarget(); proc = t.GetProcess()
    mod = t.FindModule(lldb.SBFileSpec('libchromescreenai.so'))
    slide = mod.GetObjectFileHeaderAddress().GetLoadAddress(t) - 0x100000000
    opts = lldb.SBExpressionOptions(); opts.SetIgnoreBreakpoints(True); opts.SetTryAllThreads(False); opts.SetStopOthers(True)
    v = frame.EvaluateExpression('void *s = (void*)calloc(1, 24); ((void(*)(void*, void*))0x%x)((void*)0x%x, s); s' % (APPEND + slide, msg), opts)
    s = v.GetValueAsUnsigned(); err = lldb.SBError()
    raw = proc.ReadMemory(s, 24, err)
    if raw is None:
        raise RuntimeError('expression failed: %s' % v.GetError().GetCString())
    if raw[23] & 0x80:
        data, size = struct.unpack_from('<QQ', raw, 0)
    else:
        size, data = raw[23] & 0x7f, s
    return proc.ReadMemory(data, size, err) if size else b''

def make_cb(stage, recipe):
    def cb(frame, bp_loc, d):
        proc = frame.GetThread().GetProcess()
        p = frame.FindRegister(recipe[0]).GetValueAsUnsigned()
        for off in recipe[1:]:
            p = _q(proc, p + off)
        try:
            b = _serialize(frame, p) if p else b''
        except Exception as e:
            print('stage_dump: %s failed: %r (ptr=0x%x)' % (stage, e, p)); b = b''
        with open(os.path.join(OUT, '%03d_%s.pb' % (st['n'], stage)), 'wb') as f: f.write(b)
        st['n'] += 1
        return False
    return cb

def __lldb_init_module(debugger, d):
    os.makedirs(OUT, exist_ok=True)
    t = debugger.GetSelectedTarget()
    g = globals()
    for i, (rx, stage, recipe) in enumerate(HOOKS):
        # recipe ('x1',176,16): ctx+176 is the PageLayout object (not a pointer) -> read *(ctx+176+16)
        if recipe[1:] == (176, 16):
            recipe = (recipe[0], 192)
        g['cb%d' % i] = make_cb(stage, recipe)
        bp = t.BreakpointCreateByRegex(rx, 'libchromescreenai.so')
        bp.SetScriptCallbackFunction('%s.cb%d' % (__name__, i))
