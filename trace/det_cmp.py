"""Compare detector intermediate dumps: det_cmp.py REFDIR OURDIR [names...] (bit-exact, prints first diffs)."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from det_load import vecvec
ref, ours = sys.argv[1], sys.argv[2]
names = sys.argv[3:] or sorted(f for f in os.listdir(ours) if f.startswith('rawboxes'))
for n in names:
    a, b = vecvec(os.path.join(ref, n)), vecvec(os.path.join(ours, n))
    print(n, 'lists', [len(x) for x in a], [len(x) for x in b])
    for i, (x, y) in enumerate(zip(a, b)):
        if len(x) != len(y): print('  list', i, 'len differs'); continue
        if len(x) == 0: continue
        xb, yb = x.view(np.uint8).reshape(len(x), -1), y.view(np.uint8).reshape(len(y), -1)
        bad = np.where((xb != yb).any(1))[0]
        if len(bad):
            print('  list', i, 'rows differ:', len(bad), 'of', len(x))
            for r in bad[:3]: print('   ref', x[r], '\n   our', y[r])
