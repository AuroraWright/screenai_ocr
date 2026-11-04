# SPDX-License-Identifier: MPL-2.0
"""Detector tie tests with injected model outputs (testdata/det_inject/<case>.txt, from testdata/make_det_inject.py).
usage: det_inject_check.py OUTDIR [--ref] [case ...]
  --ref: run the original under lldb with DD_INJECT (trace/det_dump.py) and store testdata/det_inject/<case>/
         (001_rec_in.pb, inject.txt = the tensors edited);
  else:  run cpp/build_det/run_detector with SCREENAI_DET_INJECT + SCREENAI_DET_DUMP, compare the LINE/PARAGRAPH
         entities with the reference exactly and print the ties.txt counters of the five sort sites."""
import os, shutil, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
src = open(os.path.join(ROOT, 'tools/cmp_layout.py')).read().split('if __name__')[0]
g = {'__file__': os.path.join(ROOT, 'tools/cmp_layout.py')}; exec(src, g)
DIR = os.path.join(ROOT, 'testdata/det_inject')
SITES = ('line_links', 'curve_points', 'line_order', 'paragraph_nms', 'angle_estimation')
RES = os.path.expanduser('~/.config/screen_ai/resources')

args = sys.argv[1:]
out = args.pop(0); os.makedirs(out, exist_ok=True)
ref = '--ref' in args; args = [a for a in args if a != '--ref']
cases = args or sorted(f[:-4] for f in os.listdir(DIR) if f.endswith('.txt'))
bad = []
for c in cases:
    edits = os.path.join(DIR, c + '.txt')
    img = os.path.join(ROOT, open(edits).readline().split('image:')[1].strip())
    if ref:
        d = os.path.join(out, c); shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
        env = dict(os.environ, DD_INJECT=edits, SD_OUT=d, DD_OUT=d)
        subprocess.run(['lldb', '-b', '-o', 'command script import trace/stage_dump.py', '-o',
                        'command script import trace/det_dump.py', '-o', 'run', '--',
                        os.path.join(ROOT, '.venv/bin/python'), 'ref/run_ref.py', img],
                       cwd=ROOT, env=env, capture_output=True)
        dst = os.path.join(DIR, c); os.makedirs(dst, exist_ok=True)
        for f in ('001_rec_in.pb', 'inject.txt'): shutil.copy(os.path.join(d, f), dst)
        print(c, 'reference stored')
        continue
    d = os.path.join(out, c); os.makedirs(d, exist_ok=True)
    o = os.path.join(d, 'out.pb')
    if os.path.exists(o): os.remove(o)
    env = dict(os.environ, SCREENAI_DET_INJECT=edits, SCREENAI_DET_DUMP=d)
    subprocess.run([os.path.join(ROOT, 'cpp/build_det/run_detector'), RES, img, o], env=env, capture_output=True)
    r = g['decode'](os.path.join(DIR, c, '001_rec_in.pb'))
    ok = os.path.exists(o) and r.get('entities') == g['decode'](o).get('entities')
    ties = {}
    for line in open(os.path.join(d, 'ties.txt')):
        f = line.split()
        if len(f) == 5: ties[f[0]] = (int(f[2]), int(f[4]))
    events = [l.split()[0][6:] + '=' + l.split()[1] for l in open(os.path.join(d, 'ties.txt')) if l.startswith('event:')]
    print('%-26s %-6s %s  %s' % (c, 'exact' if ok else 'DIFFER',
                                 ' '.join('%s=%d/%d' % (s, *ties.get(s, (0, 0))) for s in SITES), ' '.join(events)))
    if not ok: bad.append(c)
if not ref: print('cases', len(cases), 'exact', len(cases) - len(bad), 'differ', bad)
