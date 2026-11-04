# SPDX-License-Identifier: MPL-2.0
"""Print message schemas from the extracted descriptor pool: schema.py Msg [Msg...] (recurses into goodoc/aksara types)."""
import sys; sys.path.insert(0, 'tools')
from pool import load_pool
pool, _, _ = load_pool('protos_extracted')
TN = ['', 'double', 'float', 'int64', 'uint64', 'int32', 'fixed64', 'fixed32', 'bool', 'string', 'group', 'msg', 'bytes',
      'uint32', 'enum', 'sfixed32', 'sfixed64', 'sint32', 'sint64']
seen = set()
def show(n, depth, maxd, pref):
    if n in seen or depth > maxd: return
    seen.add(n)
    d = pool.FindMessageTypeByName(n)
    print('message', n)
    for f in d.fields:
        t = f.message_type.full_name if f.message_type else (f.enum_type.full_name if f.enum_type else TN[f.type])
        rep = 'repeated' if getattr(f, 'is_repeated', False) or getattr(f, 'label', 1) == 3 else ''
        print('   %3d %-9s %-36s %s' % (f.number, rep, f.name, t))
    for f in d.fields:
        if f.message_type and f.message_type.full_name.startswith(pref): show(f.message_type.full_name, depth + 1, maxd, pref)
    for f in d.fields:
        if f.enum_type and f.enum_type.full_name not in seen and f.enum_type.full_name.startswith(pref):
            seen.add(f.enum_type.full_name)
            print('enum', f.enum_type.full_name, ' '.join('%s=%d' % (v.name, v.number) for v in f.enum_type.values))
maxd = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 3
for n in sys.argv[1].split(','): show(n, 0, maxd, n.split('.')[0])
