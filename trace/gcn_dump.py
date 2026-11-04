# SPDX-License-Identifier: MPL-2.0
"""lldb script: like tfl_dump.py, but only dumps TFLite invokes of the layout cluster-sort GCN
(gocr/layout/cluster_sort/model_v2.tflite: inputs serving_default_inputs_1/_2/inputs). Same file format.

Usage: TFL_DUMP_DIR=<dir> lldb -b -o 'command script import trace/gcn_dump.py' -o run -- .venv/bin/python ref/run_ref.py img.png
"""
import os, struct, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tfl_dump as T


def _is_gcn(proc, sg):
    beg, end = T._q(proc, sg + 37 * 8), T._q(proc, sg + 37 * 8 + 8)
    if end - beg != 12: return False
    first, = struct.unpack('<i', T._rd(proc, beg, 4))
    name_p = T._q(proc, T._q(proc, sg + 56) + 112 * first + 56)
    return bool(name_p) and T._rd(proc, name_p, 32).split(b'\0')[0] == b'serving_default_inputs_1:0'


def pre(frame, bp_loc, d):
    proc = frame.GetThread().GetProcess()
    if _is_gcn(proc, frame.FindRegister('x0').GetValueAsUnsigned()):
        return T.pre(frame, bp_loc, d)
    if 'post' not in T.state: T.pre(frame, bp_loc, d); T.state['pending'].clear()
    return False


def post(frame, bp_loc, d):
    if frame.GetThread().GetThreadID() in T.state.get('pending', {}): return T.post(frame, bp_loc, d)
    return False


def __lldb_init_module(debugger, d):
    T.OUT = os.environ.get('TFL_DUMP_DIR', '/tmp/gcn_dump')
    os.makedirs(T.OUT, exist_ok=True)
    T.__name__ = __name__  # route the post-invoke breakpoint planted by tfl_dump.pre to our post()
    t = debugger.GetSelectedTarget()
    bp = t.BreakpointCreateByName('tflite::Subgraph::Invoke()', 'libchromescreenai.so')
    bp.SetScriptCallbackFunction('%s.pre' % __name__)
