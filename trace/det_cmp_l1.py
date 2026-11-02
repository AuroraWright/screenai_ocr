"""Compare Level1 groups: det_cmp_l1.py REF/level1_0.bin OUR/level1_0.bin"""
import sys, os, struct, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from det_load import level1
def ours(path):
    b = open(path, 'rb').read(); n, = struct.unpack_from('<I', b, 0); o = 4; out = []
    for _ in range(n):
        f = np.frombuffer(b, '<f4', 7, o); o += 28
        m, = struct.unpack_from('<I', b, o); o += 4
        out.append((f, list(struct.unpack_from('<%di' % m, b, o)))); o += 4 * m
    return out
R, O = level1(sys.argv[1]), ours(sys.argv[2])
print('groups', len(R), len(O))
bad = 0
for k, ((rf, ri, ridx), (of, oidx)) in enumerate(zip(R, O)):
    ref = np.array(list(rf[:5]) + [rf[63], rf[67]], np.float32)
    if ridx != oidx or (ref.view(np.int32) != of.view(np.int32)).any():
        bad += 1
        if bad <= 5: print(k, 'ref', ref, ridx[:10], '\n  our', of, oidx[:10])
print('mismatching groups', bad)
