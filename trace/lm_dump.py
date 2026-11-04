# SPDX-License-Identifier: MPL-2.0
"""lldb script: dump google_ocr::CTCDecoder::DecodeNth inputs/outputs (LM beam-search decoder).

Usage: LM_OUT=dir lldb -b -o 'command script import trace/lm_dump.py' -o run -- .venv/bin/python ref/run_ref.py img.png

Per DecodeNth call k writes <out>/call_<k>.bin:
  'LMIN' u64 decoder, i32 n, i32 B, i32 T, i32 K, i32 len, i32 has_idx, f32 logits[T*K] (batch n), f32 idx[T*K]
and at its return <out>/call_<k>_out.bin: i64 status, then raw CTCDecoderOutput::LabelInfo[] (40 bytes each).
Once per decoder <out>/dec_<addr>.bin: name, prior floats, LM scalars and the class -> LM-symbol table.
Env LM_BEAMS=1 additionally dumps every beam (vector<BeamItem>) after each ExpandAndPruneBeam of call 0..LM_BEAMS_MAX.
"""
import lldb, os, struct, threading

OUT = os.environ.get('LM_OUT', '/tmp/lm_dump')
st = {'n': 0, 'decs': set(), 'pending': {}, 'lock': threading.Lock()}


def rd(proc, addr, n):
    err = lldb.SBError()
    b = proc.ReadMemory(addr, n, err) if n else b''
    if n and not err.Success():
        raise RuntimeError('read %x %d: %s' % (addr, n, err))
    return b


def q(proc, addr):
    return struct.unpack('<Q', rd(proc, addr, 8))[0]


def cxxstr(proc, addr):
    raw = rd(proc, addr, 24)
    if raw[23] & 0x80:
        data, size = struct.unpack_from('<QQ', raw, 0)
        return rd(proc, data, size)
    return raw[:raw[23] & 0x7f]


def dump_decoder(proc, dec):
    name = cxxstr(proc, dec + 8)
    out = [b'LMDC', struct.pack('<Q', dec), struct.pack('<I', len(name)), name]
    num_classes = struct.unpack('<i', rd(proc, dec + 160, 4))[0]
    out.append(struct.pack('<i', num_classes))
    lms_b, lms_e = q(proc, dec + 184), q(proc, dec + 192)
    nlm = (lms_e - lms_b) // 96
    out.append(struct.pack('<i', nlm))
    for i in range(nlm):
        e = lms_b + 96 * i
        lname = cxxstr(proc, e)
        lm = q(proc, e + 24)
        lmw = struct.unpack('<f', rd(proc, e + 32, 4))[0]
        out += [struct.pack('<I', len(lname)), lname, struct.pack('<f', lmw)]
        # OcrLanguageModel scalars
        minp, maxc = struct.unpack('<dd', rd(proc, lm + 64, 16))
        ncls, eos = struct.unpack('<ii', rd(proc, lm + 80, 8))
        unk = struct.unpack('<i', rd(proc, lm + 128, 4))[0]
        defarc = rd(proc, lm + 188, 16)
        eoslm = struct.unpack('<i', rd(proc, lm + 204, 4))[0]
        fst = q(proc, lm + 168)
        out += [struct.pack('<ddiii', minp, maxc, ncls, eos, unk), defarc, struct.pack('<iQ', eoslm, fst)]
        tb, te = q(proc, lm + 88), q(proc, lm + 96)
        n = (te - tb) // 24
        raw = rd(proc, tb, 24 * n)
        out.append(struct.pack('<i', n))
        for c in range(n):
            b, e2 = struct.unpack_from('<QQ', raw, 24 * c)
            ids = rd(proc, b, e2 - b) if e2 > b else b''
            out += [struct.pack('<i', len(ids) // 4), ids]
    with open(os.path.join(OUT, 'dec_%x.bin' % dec), 'wb') as f:
        f.write(b''.join(out))


def entry(frame, bp_loc, d):
    proc = frame.GetThread().GetProcess()
    dec = frame.FindRegister('x0').GetValueAsUnsigned()
    inp = frame.FindRegister('x1').GetValueAsUnsigned()
    n = frame.FindRegister('x3').GetValueAsUnsigned() & 0xffffffff
    x8 = frame.FindRegister('x8').GetValueAsUnsigned()
    lr = frame.FindRegister('lr').GetValueAsUnsigned()
    with st['lock']:
        k = st['n']; st['n'] += 1
        newdec = dec not in st['decs']; st['decs'].add(dec)
    if newdec:
        dump_decoder(proc, dec)
    w = struct.unpack('<13Q', rd(proc, inp, 104))
    has_idx = rd(proc, inp + 104, 1)[0] & 1
    logits, B, T, K, lens = w[0], w[1], w[2], w[3], w[4]
    length = struct.unpack('<i', rd(proc, lens + 4 * n, 4))[0]
    lg = rd(proc, logits + 4 * n * T * K, 4 * T * K)
    ix = rd(proc, w[9] + 4 * w[11] * w[12] * n, 4 * T * K) if has_idx else b''
    prior_p, prior_n, prior_flag = w[6], w[7], w[8]
    with open(os.path.join(OUT, 'call_%04d.bin' % k), 'wb') as f:
        f.write(b'LMIN' + struct.pack('<Qiiiiii', dec, n, B, T, K, length, has_idx) + lg + ix)
    if newdec and (prior_flag & 1):
        with open(os.path.join(OUT, 'dec_%x_prior.bin' % dec), 'wb') as f:
            f.write(rd(proc, prior_p, 4 * prior_n))
    # breakpoint at the return address (one per address); pending calls keyed by (thread, sp)
    t = proc.GetTarget()
    sp = frame.FindRegister('sp').GetValueAsUnsigned()
    tid = frame.GetThread().GetThreadID()
    with st['lock']:
        st['pending'][(tid, sp)] = (k, x8)
        if lr not in st.setdefault('retbps', set()):
            st['retbps'].add(lr)
            bp = t.BreakpointCreateByAddress(lr)
            bp.SetScriptCallbackFunction('%s.ret' % __name__)
    return False


def ret(frame, bp_loc, d):
    proc = frame.GetThread().GetProcess()
    key = (frame.GetThread().GetThreadID(), frame.FindRegister('sp').GetValueAsUnsigned())
    with st['lock']:
        if key not in st['pending']:
            return False
        k, x8 = st['pending'].pop(key)
    status = q(proc, x8)
    body = b''
    if status == 1:
        b, e = q(proc, x8 + 8), q(proc, x8 + 16)
        body = rd(proc, b, e - b)
    with open(os.path.join(OUT, 'call_%04d_out.bin' % k), 'wb') as f:
        f.write(struct.pack('<q', status) + body)
    return False


def __lldb_init_module(debugger, d):
    os.makedirs(OUT, exist_ok=True)
    t = debugger.GetSelectedTarget()
    bp = t.BreakpointCreateByRegex('^google_ocr::CTCDecoder::DecodeNth\\(', 'libchromescreenai.so')
    bp.SetScriptCallbackFunction('%s.entry' % __name__)
