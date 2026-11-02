"""Load tensors dumped by tfl_dump.py."""
import json, os, glob
import numpy as np
DT = {1: np.float32, 2: np.int32, 3: np.uint8, 4: np.int64, 6: np.bool_, 7: np.int16, 9: np.int8, 10: np.float16}
def load(d):
    calls = []
    for mf in sorted(glob.glob(os.path.join(d, '*_in.json'))):
        n = os.path.basename(mf).split('_')[0]; call = {}
        for kind in ('in', 'out'):
            p = os.path.join(d, '%s_%s.json' % (n, kind))
            if not os.path.exists(p): continue
            ts = json.load(open(p))
            for k, t in enumerate(ts):
                b = os.path.join(d, '%s_%s_%d.bin' % (n, kind, k))
                t['arr'] = np.fromfile(b, DT[t['type']]).reshape(t['shape']) if os.path.exists(b) and t['type'] in DT else None
            call[kind] = ts
        calls.append(call)
    return calls
