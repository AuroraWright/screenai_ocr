"""lldb script: overwrite the language of LINE entities at LanguageBasedLineRecognizerConfigSelector::Select entry
(0x100148160), i.e. after the first pass and just before the second-pass recognizer is chosen, then continue.
Combine with trace/stage_dump.py to dump the stages; ours does the same with SCREENAI_DEBUG_SELECT_LANG.

INJ_LANG=<tag>: every Select call; INJ_LANG=<n>:<tag>,...: only the n-th call. INJ_LOG=file logs each call.
Usage: INJ_LANG=zh-Hant SD_OUT=dir lldb -b -o 'command script import trace/stage_dump.py' \
         -o 'command script import trace/select_inject.py' -o run -- .venv/bin/python ref/run_ref.py img.png
"""
import lldb, os, struct

st = {'calls': 0}

def rd(proc, a, n):
    err = lldb.SBError(); return proc.ReadMemory(a, n, err)
def q(proc, a): return struct.unpack('<Q', rd(proc, a, 8))[0]

def read_string(proc, p):
    raw = rd(proc, p, 24)
    if raw[23] & 0x80:
        data, size = struct.unpack('<QQ', raw[:16])
        return rd(proc, data, size).decode() if size else ''
    return raw[:raw[23]].decode()

def write_string(proc, p, s):
    err = lldb.SBError(); b = s.encode(); raw = rd(proc, p, 24)
    if raw[23] & 0x80:
        data, size, cap = struct.unpack('<QQQ', raw)
        if (cap & ~(1 << 63)) <= len(b): return False
        proc.WriteMemory(data, b + b'\0', err); proc.WriteMemory(p + 8, struct.pack('<Q', len(b)), err)
    else:
        if len(b) > 22: return False
        proc.WriteMemory(p, b + b'\0' * (23 - len(b)) + bytes([len(b)]), err)
    return True

SELECT = 0x100148160

def slide_of(t):
    mod = t.FindModule(lldb.SBFileSpec('libchromescreenai.so'))
    return mod.GetObjectFileHeaderAddress().GetLoadAddress(t) - 0x100000000

def on_select(frame, loc, d):
    proc = frame.GetThread().GetProcess()
    if frame.GetPC() != SELECT + slide_of(proc.GetTarget()): return False  # the regex also hits Select's helpers
    n = st['calls']; st['calls'] += 1
    spec = os.environ.get('INJ_LANG', '')
    tag = None
    if ':' not in spec: tag = spec or None
    else:
        for item in spec.split(','):
            k, _, v = item.partition(':')
            if int(k) == n: tag = v
    ent = frame.FindRegister('x1').GetValueAsUnsigned()
    mix = q(proc, ent + 128)
    old = None
    if mix:
        rep = q(proc, mix + 24); cnt = struct.unpack('<i', rd(proc, mix + 32, 4))[0]
        if cnt > 0:
            lang = q(proc, rep - 1 + 8) if rep & 1 else rep
            sp = q(proc, lang + 24) & ~3
            old = read_string(proc, sp)
            if tag is not None and old != '' and not write_string(proc, sp, tag): old += ' (not written)'
    if os.environ.get('INJ_LOG'):
        with open(os.environ['INJ_LOG'], 'a') as f: f.write('%d %r -> %r\n' % (n, old, tag))
    return False

def __lldb_init_module(debugger, d):
    t = debugger.GetSelectedTarget()
    b = t.BreakpointCreateByRegex(r'LanguageBasedLineRecognizerConfigSelector::Select\(')
    b.SetScriptCallbackFunction('select_inject.on_select')
