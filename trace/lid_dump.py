# SPDX-License-Identifier: MPL-2.0
"""lldb script: dump every language-id invocation of libchromescreenai.so (MobileLangID V2 / tflite_langid.tflite).

For each Subgraph::Invoke whose first input is a string tensor (the langid model's `Placeholder`), writes
  NNNNN.json  {"text": [<input string>], "status": <TfLiteStatus of the invoke>, "tensors": [{index,name,type,shape}]}
  NNNNN_<tensor index>.bin  raw bytes of every non-string tensor after the invoke (intermediates may be clobbered
                            by the arena planner; the subgraph outputs 28/29 are always valid)
and for each MobileLangIdV2::IdentifyLanguage call the returned (language, confidence) in lid_results.jsonl.

LID_INJECT=<file> (one hex-encoded UTF-8 text per line) replaces the text argument of successive IdentifyLanguage
calls with these strings, so arbitrary inputs can be run through the original (render an image with enough lines).

Usage: LID_DUMP_DIR=out lldb -b -o 'command script import trace/lid_dump.py' -o run -- .venv/bin/python ref/run_ref.py img.png
"""
import lldb, os, struct, json

OUT = os.environ.get('LID_DUMP_DIR', '/tmp/lid_dump')
ENTRY, POST = 0x1002cccd4, 0x1002ccce8  # tflite::Subgraph::Invoke entry; instruction after its InvokeImpl call
IDENTIFY = 0x1002a8390                  # google_ocr::MobileLangIdV2::IdentifyLanguage(absl::string_view) const
state = {'n': 0, 'pending': {}, 'ident': {}, 'inject': None, 'k': 0}


def _rd(proc, addr, n):
    err = lldb.SBError(); b = proc.ReadMemory(addr, n, err)
    if not err.Success(): raise RuntimeError(err.GetCString())
    return b
def _q(proc, a): return struct.unpack('<Q', _rd(proc, a, 8))[0]
def _i(proc, a): return struct.unpack('<i', _rd(proc, a, 4))[0]


def _tensor(proc, sg, i, with_data):
    t = _q(proc, sg + 56) + 112 * i
    typ = _i(proc, t)
    data, dims, nbytes, name_p = _q(proc, t + 8), _q(proc, t + 16), _q(proc, t + 40), _q(proc, t + 56)
    name = _rd(proc, name_p, 256).split(b'\0')[0].decode('utf8', 'replace') if name_p else ''
    nd = _i(proc, dims) if dims else 0
    shape = list(struct.unpack('<%di' % nd, _rd(proc, dims + 4, 4 * nd))) if nd else []
    arr = _rd(proc, data, nbytes) if (with_data and data and nbytes) else None
    return dict(index=i, name=name, type=typ, shape=shape), arr


def _string_tensor(arr):
    n, = struct.unpack_from('<i', arr, 0)
    offs = struct.unpack_from('<%di' % (n + 1), arr, 4)
    return [arr[offs[k]:offs[k + 1]] for k in range(n)]


def _cstr(proc, a):  # libc++ std::string at a
    b = _rd(proc, a, 24)
    if b[23] & 0x80:
        ptr, size = struct.unpack_from('<QQ', b, 0)
        return _rd(proc, ptr, size) if size else b''
    return b[:b[23]]


def pre(frame, bp_loc, d):
    if 'post' not in state:
        t = frame.GetThread().GetProcess().GetTarget()
        slide = frame.GetPC() - ENTRY
        bp = t.BreakpointCreateByAddress(POST + slide)
        bp.SetScriptCallbackFunction('%s.post' % __name__)
        state['post'] = bp
    proc = frame.GetThread().GetProcess()
    sg = frame.FindRegister('x0').GetValueAsUnsigned()
    beg = _q(proc, sg + 37 * 8)
    first = _i(proc, beg)
    meta, arr = _tensor(proc, sg, first, True)
    if meta['type'] == 5 and arr:
        state['pending'][frame.GetThread().GetThreadID()] = [s.decode('utf8', 'surrogateescape') for s in _string_tensor(arr)]
    return False


def post(frame, bp_loc, d):
    texts = state['pending'].pop(frame.GetThread().GetThreadID(), None)
    if texts is None: return False
    proc = frame.GetThread().GetProcess()
    sg = frame.FindRegister('x19').GetValueAsUnsigned()
    n = state['n']; state['n'] += 1
    metas = []
    for i in range(_q(proc, sg + 40)):
        meta, arr = _tensor(proc, sg, i, True)
        if meta['type'] == 5 or arr is None:
            metas.append(meta); continue
        with open(os.path.join(OUT, '%05d_%d.bin' % (n, i)), 'wb') as f: f.write(arr)
        metas.append(meta)
    with open(os.path.join(OUT, '%05d.json' % n), 'w') as f:
        json.dump(dict(text=texts, thread=frame.GetThread().GetThreadID(), tensors=metas,
                       status=frame.FindRegister('x0').GetValueAsUnsigned() & 0xffffffff), f)  # InvokeImpl result
    return False


def ident_entry(frame, bp_loc, d):
    proc = frame.GetThread().GetProcess()
    if state['inject'] is not None and state['k'] < len(state['inject']):
        data = state['inject'][state['k']]; state['k'] += 1
        err = lldb.SBError()
        buf = proc.AllocateMemory(max(len(data), 1) + 1, lldb.ePermissionsReadable | lldb.ePermissionsWritable, err)
        proc.WriteMemory(buf, data + b'\0', err)
        frame.FindRegister('x1').SetValueFromCString(str(buf))
        frame.FindRegister('x2').SetValueFromCString(str(len(data)))
    ptr, size = frame.FindRegister('x1').GetValueAsUnsigned(), frame.FindRegister('x2').GetValueAsUnsigned()
    text = _rd(proc, ptr, size) if size else b''
    ret = frame.FindRegister('lr').GetValueAsUnsigned()
    out = frame.FindRegister('x8').GetValueAsUnsigned()
    tid = frame.GetThread().GetThreadID()
    state['ident'][tid] = (text, out)
    t = proc.GetTarget()
    bp = state.setdefault('ret_bps', {}).get(ret)
    if bp is None:
        bp = t.BreakpointCreateByAddress(ret)
        bp.SetScriptCallbackFunction('%s.ident_ret' % __name__)
        state['ret_bps'][ret] = bp
    return False


def ident_ret(frame, bp_loc, d):
    tid = frame.GetThread().GetThreadID()
    if tid not in state['ident']: return False
    text, out = state['ident'].pop(tid)
    proc = frame.GetThread().GetProcess()
    status = _q(proc, out)
    rec = dict(text=text.decode('utf8', 'surrogateescape'), status=status)
    if status == 1:
        rec['language'] = _cstr(proc, out + 8).decode('utf8', 'replace')
        rec['confidence'] = struct.unpack('<f', _rd(proc, out + 32, 4))[0]
        rec['confidence_bits'] = _i(proc, out + 32) & 0xffffffff
    with open(os.path.join(OUT, 'lid_results.jsonl'), 'a') as f: f.write(json.dumps(rec) + '\n')
    return False


def __lldb_init_module(debugger, d):
    os.makedirs(OUT, exist_ok=True)
    if os.environ.get('LID_INJECT'):
        state['inject'] = [bytes.fromhex(l.strip()) for l in open(os.environ['LID_INJECT']) if l.strip() or l == '\n']
    t = debugger.GetSelectedTarget()
    bp = t.BreakpointCreateByName('tflite::Subgraph::Invoke()', 'libchromescreenai.so')
    bp.SetScriptCallbackFunction('%s.pre' % __name__)
    bp = t.BreakpointCreateByName('google_ocr::MobileLangIdV2::IdentifyLanguage(absl::string_view) const', 'libchromescreenai.so')
    bp.SetScriptCallbackFunction('%s.ident_entry' % __name__)
