"""Compare cpp run_layout (layout analysis) against the original library's layout stage.

Usage: .venv/bin/python tests/test_layout.py <run_layout binary> <model_dir> [<stage dir>...]
A stage dir testdata/stages[_<set>]/<name> holds 002_lay_in.pb and 003_final.pb (trace/stage_dump.py); its image is
testdata/<name>.png (testdata/stages) or testdata/<set>/<name>.png (testdata/stages_<set>, e.g. e2e, extra_lm,
extra_indic). Default: every testdata/stages* set except holdouts. Runs run_layout on 002_lay_in.pb and
requires 0 tools/cmp_layout.py mismatches against 003_final.pb plus equal page language_mix and creator_id.
"""
import glob, os, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import cmp_layout  # noqa: E402


SETS = ['stages', 'stages_e2e', 'stages_extra_lm', 'stages_extra_indic', 'stages_extra_rot', 'stages_extra_curve',
        'stages_extra_paths', 'stages_fuzz']


def image_for(stage_dir):
    d = os.path.abspath(stage_dir.rstrip('/'))
    name, stage_set = os.path.basename(d), os.path.basename(os.path.dirname(d))
    sub = stage_set[len('stages_'):] if stage_set.startswith('stages_') else ''
    p = os.path.join(ROOT, 'testdata', sub, name + '.png')
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return p


def main():
    binary, model_dir = sys.argv[1], sys.argv[2]
    dirs = sys.argv[3:] or [d for s in SETS for d in sorted(glob.glob(os.path.join(ROOT, 'testdata', s, '*')))]
    failed = 0
    with tempfile.TemporaryDirectory() as tmp:
        for d in dirs:
            name = os.path.basename(d.rstrip('/'))
            out = os.path.join(tmp, name + '.pb')
            subprocess.run([binary, model_dir, image_for(d), os.path.join(d, '002_lay_in.pb'), out],
                           capture_output=True, check=True)
            ref = os.path.join(d, '003_final.pb')
            r = subprocess.run([sys.executable, os.path.join(ROOT, 'tools', 'cmp_layout.py'), ref, out],
                               capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
            a, b = cmp_layout.decode(ref), cmp_layout.decode(out)
            page_ok = all(a.get(k) == b.get(k) for k in ('language_mix', 'creator_id'))
            ok = r.endswith(' 0') and page_ok
            failed += not ok
            print('%-16s %s%s %s' % (name, r, '' if page_ok else ' (page fields differ)', 'OK' if ok else 'FAIL'))
    print('%d/%d cases match' % (len(dirs) - failed, len(dirs)))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
