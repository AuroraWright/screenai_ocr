# SPDX-License-Identifier: MPL-2.0
"""Field-level diff of two goodoc.PageLayoutProto dumps, pairing entities by index (no type grouping).

usage: lay_diff.py A.pb B.pb [--max N]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cmp_layout import decode, flatten

def main():
    a, b = decode(sys.argv[1]), decode(sys.argv[2])
    mx = int(sys.argv[sys.argv.index('--max') + 1]) if '--max' in sys.argv else 200
    top_a = {k: v for k, v in a.items() if k != 'entities'}; top_b = {k: v for k, v in b.items() if k != 'entities'}
    for k in sorted(set(flatten(top_a)) | set(flatten(top_b))):
        fa, fb = flatten(top_a).get(k), flatten(top_b).get(k)
        if fa != fb: print('page.%s: %r -> %r' % (k, fa, fb))
    ea, eb = a.get('entities', []), b.get('entities', [])
    print('entities %d -> %d' % (len(ea), len(eb)))
    n = 0
    for i in range(max(len(ea), len(eb))):
        fa = flatten(ea[i]) if i < len(ea) else {}
        fb = flatten(eb[i]) if i < len(eb) else {}
        for k in sorted(set(fa) | set(fb)):
            if fa.get(k) != fb.get(k):
                if n < mx: print('[%d %s] %s: %r -> %r' % (i, fb.get('entity_type', fa.get('entity_type', 'WORD')), k, fa.get(k), fb.get(k)))
                n += 1
    print('%d differences' % n)

if __name__ == '__main__':
    main()
