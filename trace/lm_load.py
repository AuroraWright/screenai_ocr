# SPDX-License-Identifier: MPL-2.0
"""Load dumps written by trace/lm_dump.py (python -I trace/lm_load.py DIR prints a summary)."""
import os, struct, sys, glob


def load_call(path):
    b = open(path, 'rb').read()
    assert b[:4] == b'LMIN'
    dec, n, B, T, K, length, has_idx = struct.unpack_from('<Qiiiiii', b, 4)
    o = 4 + 32
    lg = struct.unpack_from('<%df' % (T * K), b, o); o += 4 * T * K
    ix = struct.unpack_from('<%df' % (T * K), b, o) if has_idx else None
    out = None
    op = path[:-4] + '_out.bin'
    if os.path.exists(op):
        ob = open(op, 'rb').read()
        status, = struct.unpack_from('<q', ob, 0)
        labels = [struct.unpack_from('<iidddd', ob, 8 + 40 * i) for i in range((len(ob) - 8) // 40)]
        out = (status, labels)
    return dict(dec=dec, n=n, B=B, T=T, K=K, length=length, logits=lg, idx=ix, out=out)


def load_decoder(path):
    b = open(path, 'rb').read()
    assert b[:4] == b'LMDC'
    o = 4
    dec, = struct.unpack_from('<Q', b, o); o += 8
    nl, = struct.unpack_from('<I', b, o); o += 4
    name = b[o:o + nl].decode(); o += nl
    num_classes, nlm = struct.unpack_from('<ii', b, o); o += 8
    lms = []
    for _ in range(nlm):
        nl, = struct.unpack_from('<I', b, o); o += 4
        lname = b[o:o + nl].decode(); o += nl
        lmw, = struct.unpack_from('<f', b, o); o += 4
        minp, maxc, ncls, eos, unk = struct.unpack_from('<ddiii', b, o); o += 28
        defarc = struct.unpack_from('<iifi', b, o); o += 16
        eoslm, fst = struct.unpack_from('<iQ', b, o); o += 12
        n, = struct.unpack_from('<i', b, o); o += 4
        table = []
        for _ in range(n):
            m, = struct.unpack_from('<i', b, o); o += 4
            table.append(list(struct.unpack_from('<%di' % m, b, o))); o += 4 * m
        lms.append(dict(name=lname, lm_weight=lmw, min_prob=minp, max_cost=maxc, num_classes=ncls, eos=eos,
                        unk=unk, default_arc=defarc, eos_lm=eoslm, table=table))
    prior = None
    pp = path[:-4] + '_prior.bin'
    if os.path.exists(pp):
        pb = open(pp, 'rb').read()
        prior = struct.unpack('<%df' % (len(pb) // 4), pb)
    return dict(dec=dec, name=name, num_classes=num_classes, lms=lms, prior=prior)


if __name__ == '__main__':
    d = sys.argv[1]
    for p in sorted(glob.glob(os.path.join(d, 'dec_*.bin'))):
        if p.endswith('_prior.bin'):
            continue
        x = load_decoder(p)
        print('decoder %x %r num_classes=%d prior=%s' % (x['dec'], x['name'], x['num_classes'], x['prior'] and len(x['prior'])))
        for lm in x['lms']:
            t = lm['table']
            print('  lm %r w=%g maxc=%r ncls=%d eos=%d unk=%d eos_lm=%d defarc=%s table=%d empty=%d multi=%d' % (
                lm['name'], lm['lm_weight'], lm['max_cost'], lm['num_classes'], lm['eos'], lm['unk'], lm['eos_lm'],
                lm['default_arc'], len(t), sum(1 for e in t if not e), sum(1 for e in t if len(e) > 1)))
    for p in sorted(glob.glob(os.path.join(d, 'call_*.bin'))):
        if p.endswith('_out.bin'):
            continue
        c = load_call(p)
        print('%s dec=%x n=%d B=%d T=%d K=%d len=%d idx=%s out=%s' % (
            os.path.basename(p), c['dec'], c['n'], c['B'], c['T'], c['K'], c['length'], c['idx'] is not None,
            c['out'] and (c['out'][0], [l[0] for l in c['out'][1]])))
