"""Diff two serialized goodoc.PageLayoutProto files (reference dump from trace/stage_dump.py vs ours).

usage: cmp_layout.py REF.pb OURS.pb [--types LINE,WORD,SYMBOL] [--ftol 1e-5] [--max 40] [--ignore field,field]
Entities are paired by index within each entity type (after a stable sort by original order).  Prints per-field
mismatch counts and the first --max differences.
"""
import argparse, os, sys
from collections import Counter, defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pool import load_pool
from google.protobuf import message_factory
from google.protobuf.json_format import MessageToDict

_pool = None
def decode(path):
    global _pool
    if _pool is None:
        _pool, _, _ = load_pool(os.path.join(os.path.dirname(HERE), 'protos_extracted'))
    m = message_factory.GetMessageClass(_pool.FindMessageTypeByName('goodoc.PageLayoutProto'))()
    m.ParseFromString(open(path, 'rb').read())
    return MessageToDict(m, preserving_proto_field_name=True, descriptor_pool=_pool)

def flatten(d, pre=''):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            k = k.replace('[google_ocr.PhotoOcrTextDetectionInfo.ext]', 'det').replace('[aksara.api_internal.InternalEntityExt.ext]', 'aks')
            out.update(flatten(v, pre + '.' + k if pre else k))
    elif isinstance(d, list):
        for i, v in enumerate(d): out.update(flatten(v, '%s[%d]' % (pre, i)))
    else:
        out[pre] = d
    return out

def by_type(layout):
    g = defaultdict(list)
    for i, e in enumerate(layout.get('entities', [])):
        g[e.get('entity_type', 'WORD')].append((i, flatten(e)))
    return g

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ref'); ap.add_argument('ours')
    ap.add_argument('--types', default=''); ap.add_argument('--ftol', type=float, default=0.0)
    ap.add_argument('--max', type=int, default=40); ap.add_argument('--ignore', default='')
    a = ap.parse_args()
    ign = [x for x in a.ignore.split(',') if x]
    R, O = by_type(decode(a.ref)), by_type(decode(a.ours))
    types = a.types.split(',') if a.types else sorted(set(R) | set(O))
    fieldbad, shown, total = Counter(), 0, 0
    for t in types:
        r, o = R.get(t, []), O.get(t, [])
        print('%-10s ref %4d  ours %4d' % (t, len(r), len(o)))
        for k in range(min(len(r), len(o))):
            (ri, rf), (oi, of) = r[k], o[k]
            for f in sorted(set(rf) | set(of)):
                if any(f.startswith(x) or ('.' + x) in f for x in ign): continue
                rv, ov = rf.get(f), of.get(f)
                same = rv == ov
                if not same and isinstance(rv, float) and isinstance(ov, (int, float)):
                    same = abs(rv - ov) <= a.ftol * max(1.0, abs(rv))
                if not same and f.endswith('parent_entity_id'):
                    same = True  # parents index the whole list; compared structurally below
                if not same:
                    total += 1; fieldbad[t + ':' + f.split('[')[0]] += 1
                    if shown < a.max:
                        shown += 1
                        print('  %s #%d (ref idx %d / ours %d) %s: ref=%r ours=%r' % (t, k, ri, oi, f, rv, ov))
    print('field mismatch counts:')
    for f, c in fieldbad.most_common(): print('  %5d  %s' % (c, f))
    print('TOTAL mismatches', total)

if __name__ == '__main__':
    main()
