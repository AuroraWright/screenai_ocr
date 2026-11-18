# SPDX-License-Identifier: MPL-2.0
"""Run cpp/build/run_detector on every image that has an original stage dump and compare the LINE/PARAGRAPH
entities with <set>/<img>/001_rec_in.pb exactly.  usage: det_check_all.py OUTDIR [set ...]"""
import glob, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'tools'))
src = open(os.path.join(ROOT, 'tools/cmp_layout.py')).read().split('if __name__')[0]
g = {'__file__': os.path.join(ROOT, 'tools/cmp_layout.py')}; exec(src, g)
IMGDIR = {'stages': 'testdata', 'stages_e2e': 'testdata/e2e', 'stages_holdout2': 'testdata/holdout2',
          'stages_extra_lm': 'testdata/extra_lm', 'stages_extra_indic': 'testdata/extra_indic',
          'stages_extra_rot': 'testdata/extra_rot', 'stages_extra_curve': 'testdata/extra_curve',
          'stages_extra_ties': 'testdata/extra_ties', 'stages_extra_det': 'testdata/extra_det'}
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
sets = sys.argv[2:] or [s for s in IMGDIR if os.path.isdir(os.path.join(ROOT, 'testdata', s))]
tot, bad = 0, []
for st in sets:
    for img in sorted(os.listdir(os.path.join(ROOT, 'testdata', st))):
        if not os.path.isdir(os.path.join(ROOT, 'testdata', st, img)): continue
        png = os.path.join(ROOT, IMGDIR[st], img + '.png')
        o = os.path.join(out, '%s_%s.pb' % (st, img))
        subprocess.run([os.path.join(ROOT, 'cpp/build/run_detector'), os.path.expanduser('~/.config/screen_ai/resources'), png, o], capture_output=True)
        r = g['decode'](os.path.join(ROOT, 'testdata', st, img, '001_rec_in.pb'))
        ok = os.path.exists(o) and r.get('entities') == g['decode'](o).get('entities')
        tot += 1
        if not ok: bad.append('%s/%s' % (st, img))
print('images', tot, 'exact', tot - len(bad), 'differ', bad)
