# SPDX-License-Identifier: MPL-2.0
"""lldb script: print registers at entry of functions matched by regex in libchromescreenai.so.

Env: RA_RX=';'-separated regexes, RA_REGS=comma-separated registers (default x0..x7), RA_MAX=max hits per bp.
Usage: lldb -b -o 'command script import trace/rec_args.py' -o run -- .venv/bin/python ref/run_ref.py img.png
"""
import lldb, os
RX = [r for r in os.environ.get('RA_RX', '').split(';') if r]
REGS = os.environ.get('RA_REGS', 'x0,x1,x2,x3,x4,x5,x6,x7').split(',')
MAX = int(os.environ.get('RA_MAX', '20'))
hits = [0] * len(RX)

def make_cb(i):
    def cb(frame, bp_loc, d):
        hits[i] += 1
        if hits[i] <= MAX:
            vals = ' '.join('%s=%s' % (r, frame.FindRegister(r).GetValue()) for r in REGS)
            print('ARGS %d %d %s %s' % (i, hits[i], frame.GetFunctionName()[:80], vals))
        return False
    return cb

def __lldb_init_module(debugger, d):
    t = debugger.GetSelectedTarget()
    g = globals()
    for i, rx in enumerate(RX):
        g['cb%d' % i] = make_cb(i)
        bp = t.BreakpointCreateByRegex(rx, 'libchromescreenai.so')
        bp.SetScriptCallbackFunction('%s.cb%d' % (__name__, i))
