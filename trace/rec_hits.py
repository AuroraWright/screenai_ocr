# SPDX-License-Identifier: MPL-2.0
"""lldb script: count hits of functions (by regex on symbol names) in libchromescreenai.so.

Env: RH_RX="ReduceActivationsGatesAndSoftMaxAarch64;ReduceActivationsGatesAndSoftMax\(" (';'-separated regexes).
Counts are printed by every hit as 'HIT <i> <count>' (last line per i is the total).
Usage: lldb -b -o 'command script import trace/rec_hits.py' -o run -- .venv/bin/python ref/run_ref.py img.png
"""
import lldb, os
RX = [r for r in os.environ.get('RH_RX', '').split(';') if r]
hits = [0] * len(RX)

def make_cb(i):
    def cb(frame, bp_loc, d):
        hits[i] += 1
        print('HIT %d %d %s' % (i, hits[i], frame.GetFunctionName()))
        return False
    return cb

def __lldb_init_module(debugger, d):
    t = debugger.GetSelectedTarget()
    g = globals()
    for i, rx in enumerate(RX):
        g['cb%d' % i] = make_cb(i)
        bp = t.BreakpointCreateByRegex(rx, 'libchromescreenai.so')
        bp.SetScriptCallbackFunction('%s.cb%d' % (__name__, i))
