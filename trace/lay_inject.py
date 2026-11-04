# SPDX-License-Identifier: MPL-2.0
"""lldb script: run the original layout analysis on injected goodoc.PageLayoutProto files (differential testing).

At entry of PageLayoutAnalysisMutator::MutateSub(ctx) the context's PageLayout (ctx+176, PageLayoutProto* at +16)
is replaced by a layout parsed from a file with the library's own proto2::MessageLite::ParseFromArray, then
goodoc::PageLayout::Reload rebuilds its entity index. Everything else layout analysis reads comes from the context
of the real run: the gray image (ImageCache, word colors) and the page image metadata / size it writes into the
proto, so the image run must be the one the layout belongs to (or one of the same size).
Per-step dumps as trace/lay_steps.py (each step's input, Analyzer/Mutator in and out, final).

Env: LI_CASES = manifest, one "<layout.pb> <out_dir>" line per MutateSub call, in run order (one per image passed to
ref/run_ref.py). A line "- <out_dir>" dumps without injecting.
usage: LI_CASES=m.txt lldb -b -o 'command script import trace/lay_inject.py' -o run -- \
         .venv/bin/python ref/run_ref.py img1.png img2.png ...
"""
import lldb, os, re, struct, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lay_steps import _serialize  # noqa: E402

PARSE = 0x101bbb1fc   # proto2::MessageLite::ParseFromArray(void const*, int)
RELOAD = 0x100c08ddc  # goodoc::PageLayout::Reload()
st = {'case': -1, 'n': 0, 'ret': {}, 'rbps': {}, 'out': None}
CASES = []


def _q(proc, a):
    err = lldb.SBError(); return struct.unpack('<Q', proc.ReadMemory(a, 8, err))[0]


def _expr(frame, code):
    opts = lldb.SBExpressionOptions(); opts.SetIgnoreBreakpoints(True); opts.SetTryAllThreads(False)
    opts.SetStopOthers(True)
    v = frame.EvaluateExpression(code, opts)
    if v.GetError().Fail() and 'unknown error' not in (v.GetError().GetCString() or ''):
        raise RuntimeError('%s: %s' % (code, v.GetError().GetCString()))
    return v


def _slide(frame):
    t = frame.GetThread().GetProcess().GetTarget()
    mod = t.FindModule(lldb.SBFileSpec('libchromescreenai.so'))
    return mod.GetObjectFileHeaderAddress().GetLoadAddress(t) - 0x100000000


def _inject(frame, page_layout, data):
    proc = frame.GetThread().GetProcess(); slide = _slide(frame)
    buf = _expr(frame, '(void*)malloc(%d)' % max(len(data), 1)).GetValueAsUnsigned()
    err = lldb.SBError()
    if data: proc.WriteMemory(buf, data, err)
    if err.Fail(): raise RuntimeError('WriteMemory: %s' % err.GetCString())
    proto = _q(proc, page_layout + 16)
    ok = _expr(frame, '((bool(*)(void*, const void*, int))0x%x)((void*)0x%x, (const void*)0x%x, %d)'
               % (PARSE + slide, proto, buf, len(data))).GetValueAsUnsigned() & 1
    if not ok: raise RuntimeError('ParseFromArray failed')
    _expr(frame, '((void(*)(void*))0x%x)((void*)0x%x); (int)0' % (RELOAD + slide, page_layout))
    _expr(frame, '(void)free((void*)0x%x); (int)0' % buf)


def _dump(frame, name, proto):
    if st['out'] is None: return
    try: b = _serialize(frame, proto) if proto else b''
    except Exception as e: print('lay_inject: %s failed: %r' % (name, e)); b = b''
    with open(os.path.join(st['out'], '%03d_%s.pb' % (st['n'], name)), 'wb') as f: f.write(b)
    st['n'] += 1


def _short(frame):
    fn = frame.GetFunctionName() or ''
    m = re.search(r'PageLayoutAnalyzerSpec_(\w+)>::Analyze', fn)
    return m.group(1) if m else re.sub(r'\W+', '_', fn.split('(')[0].split('::')[-1])


def _arm_return(frame, name, proto):
    t = frame.GetThread().GetProcess().GetTarget()
    ret = frame.FindRegister('lr').GetValueAsUnsigned()
    st['ret'][(frame.GetThread().GetThreadID(), ret)] = (name, proto)
    if ret not in st['rbps']:
        bp = t.BreakpointCreateByAddress(ret)
        bp.SetScriptCallbackFunction('%s.on_return' % __name__)
        st['rbps'][ret] = bp


def on_return(frame, bp_loc, d):
    r = st['ret'].pop((frame.GetThread().GetThreadID(), frame.GetPC()), None)
    if r: _dump(frame, r[0] + '_out', r[1])
    return False


def on_step(frame, bp_loc, d):
    proc = frame.GetThread().GetProcess()
    _dump(frame, _short(frame), _q(proc, frame.FindRegister('x1').GetValueAsUnsigned() + 16))
    return False


def on_analyzer(frame, bp_loc, d):
    proc = frame.GetThread().GetProcess()
    proto = _q(proc, frame.FindRegister('x1').GetValueAsUnsigned() + 16)
    _dump(frame, 'Analyzer_in', proto)
    _arm_return(frame, 'Analyzer', proto)
    return False


def on_mutator(frame, bp_loc, d):
    proc = frame.GetThread().GetProcess()
    st['case'] += 1; st['n'] = 0
    if st['case'] >= len(CASES):
        print('lay_inject: no case for MutateSub call %d' % st['case']); st['out'] = None
        return False
    src, out = CASES[st['case']]
    os.makedirs(out, exist_ok=True); st['out'] = out
    page_layout = frame.FindRegister('x1').GetValueAsUnsigned() + 176
    if src != '-':
        try:
            _inject(frame, page_layout, open(src, 'rb').read())
        except Exception as e:
            print('lay_inject: case %d (%s) injection failed: %r' % (st['case'], src, e))
            open(os.path.join(out, 'INJECT_FAILED'), 'w').write(repr(e)); st['out'] = None
            return False
    proto = _q(proc, page_layout + 16)
    _dump(frame, 'Mutator_in', proto)
    _arm_return(frame, 'Mutator', proto)
    return False


def on_final(frame, bp_loc, d):
    _dump(frame, 'final', frame.FindRegister('x1').GetValueAsUnsigned())
    st['out'] = None
    return False


def __lldb_init_module(debugger, d):
    for line in open(os.environ['LI_CASES']):
        if line.strip(): CASES.append(tuple(line.split()))
    t = debugger.GetSelectedTarget()
    for rx, cb in [(r'layout_analyzer::Step<.*>::Analyze\(goodoc::PageLayout ?\*, ?google_ocr::ImageCache ?\*\)$', 'on_step'),
                   (r'aksara::api_internal::PageLayoutAnalyzer::Analyze\(goodoc::PageLayout', 'on_analyzer'),
                   (r'PageLayoutAnalysisMutator::MutateSub\(google_ocr::PageLayoutMutatorContext ?\*\) ?const$', 'on_mutator'),
                   (r'ConvertOCRResultsToVisualAnnotationProto\(', 'on_final')]:
        bp = t.BreakpointCreateByRegex(rx, 'libchromescreenai.so')
        bp.SetScriptCallbackFunction('%s.%s' % (__name__, cb))
        print('lay_inject: %s -> %d locations, %d cases' % (cb, bp.GetNumLocations(), len(CASES)))
