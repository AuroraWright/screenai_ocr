"""lldb script: at the first EP_MAX Subgraph::Invoke entries, print the execution plan (builtin code[:custom name] per node).
Subgraph layout (from IDA): execution_plan_ vector<int> at +392, nodes_and_registration_ (168-byte elements: TfLiteNode
then TfLiteRegistration at +80; builtin_code at reg+40, custom_name at reg+48) at +264."""
import lldb, struct, os
st = {'n': 0}
MAXN = int(os.environ.get('EP_MAX', '3'))
def _rd(proc, a, n):
    e = lldb.SBError(); return proc.ReadMemory(a, n, e)
def _q(proc, a): return struct.unpack('<Q', _rd(proc, a, 8))[0]
def cb(frame, bp_loc, d):
    if st['n'] >= MAXN: return False
    st['n'] += 1
    proc = frame.GetThread().GetProcess(); e = lldb.SBError()
    sg = frame.FindRegister('x0').GetValueAsUnsigned()
    pb, pe = _q(proc, sg + 392), _q(proc, sg + 400)
    ids = struct.unpack('<%di' % ((pe - pb) // 4), _rd(proc, pb, pe - pb))
    nodes = _q(proc, sg + 264)
    out = []
    for nid in ids:
        reg = nodes + 168 * nid + 80
        code = struct.unpack('<i', _rd(proc, reg + 40, 4))[0]
        cn = _q(proc, reg + 48)
        name = proc.ReadCStringFromMemory(cn, 64, e) if cn else ''
        out.append('%d%s' % (code, ':' + name if name else ''))
    print('PLAN sg=0x%x ntensors=%d nodes=%d: %s' % (sg, _q(proc, sg + 40), len(ids), ' '.join(out)))
    return False
def __lldb_init_module(debugger, d):
    t = debugger.GetSelectedTarget()
    bp = t.BreakpointCreateByName('tflite::Subgraph::Invoke()', 'libchromescreenai.so')
    bp.SetScriptCallbackFunction('exec_plan.cb')
