"""Load det_dump.py outputs."""
import struct, numpy as np
RAW = np.dtype([('x', '<f4'), ('y', '<f4'), ('w', '<f4'), ('h', '<f4'), ('angle', '<f4'), ('pts', '<f4', (8,)),
                ('cx', '<f4'), ('cy', '<f4'), ('score', '<f4'), ('batch', '<i4')])
def vecvec(path, dt=RAW):
    b = open(path, 'rb').read(); o = 4; out = []
    for _ in range(struct.unpack_from('<I', b, 0)[0]):
        n, = struct.unpack_from('<I', b, o); o += 4
        out.append(np.frombuffer(b, dt, n, o)); o += n * dt.itemsize
    return out

def level1(path):
    """Level1GroupedBox dumps: list of (raw 288 bytes as float32/int32 views, member indices)."""
    b = open(path, 'rb').read(); n, = struct.unpack_from('<I', b, 0); o = 4; out = []
    for _ in range(n):
        e = b[o:o + 288]; o += 288
        m, = struct.unpack_from('<I', b, o); o += 4
        idx = list(struct.unpack_from('<%di' % m, b, o)); o += 4 * m
        out.append((np.frombuffer(e, '<f4'), np.frombuffer(e, '<i4'), idx))
    return out
