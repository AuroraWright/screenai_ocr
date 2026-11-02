"""lldb script: serialize the goodoc.PageLayoutProto before every aksara layout-analyzer step of libchromescreenai.so.

Hooks layout_analyzer::Step<Spec>::Analyze(goodoc::PageLayout*, ImageCache*) (x1 = goodoc::PageLayout, its
PageLayoutProto* at +16), PageLayoutAnalyzer::Analyze entry and return, PageLayoutAnalysisMutator::MutateSub entry
and return, and ConvertOCRResultsToVisualAnnotationProto (final layout).
Env: LS_OUT (dir). Files: <n>_<step>.pb (+ <n>_<step>.txt with the hook name).
"""
import lldb, os, re, struct

APPEND = 0x101bbc138  # proto2::MessageLite::AppendToString(std::string*) const
OUT = os.environ.get('LS_OUT', '/tmp/ls')
st = {'n': 0, 'ret': {}}


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
    if raw is None: raise RuntimeError('expression failed: %s' % v.GetError().GetCString())
    if raw[23] & 0x80: data, size = struct.unpack_from('<QQ', raw, 0)
    else: size, data = raw[23] & 0x7f, s
    return proc.ReadMemory(data, size, err) if size else b''


def _dump(frame, name, proto):
    try: b = _serialize(frame, proto) if proto else b''
    except Exception as e: print('lay_steps: %s failed: %r' % (name, e)); b = b''
    with open(os.path.join(OUT, '%03d_%s.pb' % (st['n'], name)), 'wb') as f: f.write(b)
    st['n'] += 1


def _short(frame):
    fn = frame.GetFunctionName() or ''
    m = re.search(r'PageLayoutAnalyzerSpec_(\w+)>::Analyze', fn)
    return m.group(1) if m else re.sub(r'\W+', '_', fn.split('(')[0].split('::')[-1])


def _arm_return(frame, name, proto):
    proc = frame.GetThread().GetProcess(); t = proc.GetTarget()
    ret = frame.FindRegister('lr').GetValueAsUnsigned()
    st['ret'][(frame.GetThread().GetThreadID(), ret)] = (name, proto)
    if ret not in st.setdefault('rbps', {}):
        bp = t.BreakpointCreateByAddress(ret)
        bp.SetScriptCallbackFunction('%s.on_return' % __name__)
        st['rbps'][ret] = bp


def on_return(frame, bp_loc, d):
    r = st['ret'].pop((frame.GetThread().GetThreadID(), frame.GetPC()), None)
    if r: _dump(frame, r[0] + '_out', r[1])
    return False


def on_step(frame, bp_loc, d):
    proc = frame.GetThread().GetProcess()
    pl = frame.FindRegister('x1').GetValueAsUnsigned()
    _dump(frame, _short(frame), _q(proc, pl + 16))
    return False


def on_analyzer(frame, bp_loc, d):
    proc = frame.GetThread().GetProcess()
    proto = _q(proc, frame.FindRegister('x1').GetValueAsUnsigned() + 16)
    _dump(frame, 'Analyzer_in', proto)
    _arm_return(frame, 'Analyzer', proto)
    return False


def on_mutator(frame, bp_loc, d):
    proc = frame.GetThread().GetProcess()
    proto = _q(proc, frame.FindRegister('x1').GetValueAsUnsigned() + 192)
    _dump(frame, 'Mutator_in', proto)
    _arm_return(frame, 'Mutator', proto)
    return False


def on_final(frame, bp_loc, d):
    _dump(frame, 'final', frame.FindRegister('x1').GetValueAsUnsigned())
    return False


def __lldb_init_module(debugger, d):
    os.makedirs(OUT, exist_ok=True)
    t = debugger.GetSelectedTarget()
    for rx, cb in [(r'layout_analyzer::Step<.*>::Analyze\(goodoc::PageLayout ?\*, ?google_ocr::ImageCache ?\*\)$', 'on_step'),
                   (r'aksara::api_internal::PageLayoutAnalyzer::Analyze\(goodoc::PageLayout', 'on_analyzer'),
                   (r'PageLayoutAnalysisMutator::MutateSub\(google_ocr::PageLayoutMutatorContext ?\*\) ?const$', 'on_mutator'),
                   (r'ConvertOCRResultsToVisualAnnotationProto\(', 'on_final')]:
        bp = t.BreakpointCreateByRegex(rx, 'libchromescreenai.so')
        bp.SetScriptCallbackFunction('%s.%s' % (__name__, cb))
        print('lay_steps: %s -> %d locations' % (cb, bp.GetNumLocations()))
