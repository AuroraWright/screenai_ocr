# SPDX-License-Identifier: MPL-2.0
"""Compare cpp run_langid against language-id outputs of the original library.

Usage: .venv/bin/python tests/test_langid.py <run_langid binary> <model_dir> <dir>...
A dir is either a trace/lid_dump.py output (NNNNN.json + NNNNN_28/29.bin + lid_results.jsonl) or a packed
reference set with ref.jsonl (testdata/extra_lid). Checks per model invoke that softmax and logits are bit-identical
(or that the invoke fails in both), and per MobileLangIdV2::IdentifyLanguage call the (language, confidence) result.
"""
import json, os, struct, subprocess, sys


def load(d):
    name = os.path.basename(d.rstrip('/'))
    p = os.path.join(d, 'ref.jsonl')
    if os.path.exists(p):
        for k, line in enumerate(open(p)):
            r = json.loads(line)
            yield bytes.fromhex(r['text_hex']), r, '%s/%d' % (name, k)
        return
    for f in sorted(os.listdir(d)):
        if f.endswith('.json') and f[:5].isdigit():
            meta = json.load(open(os.path.join(d, f)))
            r = dict(kind='invoke', status=meta.get('status', 0))
            if r['status'] == 0:
                rd = lambda i: open(os.path.join(d, '%s_%d.bin' % (f[:5], i)), 'rb').read()
                r.update(softmax=list(struct.unpack('<111I', rd(29))), logits=list(struct.unpack('<111I', rd(28))))
            yield meta['text'][0].encode('utf8', 'surrogateescape'), r, '%s/%s' % (name, f[:5])
    p = os.path.join(d, 'lid_results.jsonl')
    if os.path.exists(p):
        for k, line in enumerate(open(p)):
            r = json.loads(line)
            if r['status'] == 1:
                r['kind'] = 'result'
                yield r['text'].encode('utf8', 'surrogateescape'), r, '%s/result%d' % (name, k)


def main():
    binary, model_dir, dirs = sys.argv[1], sys.argv[2], sys.argv[3:]
    cases = [c for d in dirs for c in load(d)]
    inp = ''.join(t.hex() + '\n' for t, _, _ in cases)
    out = subprocess.run([binary, model_dir, '--hex'], input=inp, capture_output=True, text=True, check=True).stdout
    lines = out.splitlines()
    assert len(lines) == len(cases), (len(lines), len(cases))
    bad = 0
    for (text, ref, origin), line in zip(cases, lines):
        lang, _, cbits, smo, lgo = line.split(' ')
        errs = []
        if ref['kind'] == 'invoke':
            if ref['status'] != 0 or smo == '-':
                if (ref['status'] != 0) != (smo == '-'): errs.append('invoke status differs (original %d)' % ref['status'])
            else:
                smo = [int(x, 16) for x in smo.split(',')]; lgo = [int(x, 16) for x in lgo.split(',')]
                if ref['softmax'] != smo: errs.append('softmax %d/111 differ' % sum(a != b for a, b in zip(ref['softmax'], smo)))
                if ref['logits'] != lgo: errs.append('logits %d/111 differ' % sum(a != b for a, b in zip(ref['logits'], lgo)))
        elif (lang, int(cbits, 16)) != (ref['language'], ref['confidence_bits']):
            errs.append('result %s/%s vs original %s/%08x' % (lang, cbits, ref['language'], ref['confidence_bits']))
        if errs:
            bad += 1
            print('MISMATCH %s %r: %s' % (origin, text, '; '.join(errs)))
    print('%d cases (%d invokes, %d results), %d mismatches' % (
        len(cases), sum(r['kind'] == 'invoke' for _, r, _ in cases), sum(r['kind'] == 'result' for _, r, _ in cases), bad))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
