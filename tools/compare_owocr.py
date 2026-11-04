# SPDX-License-Identifier: MPL-2.0
"""Runs owocr's ChromeScreenAI wrapper against two libchromescreenai builds and compares the parsed results.

usage: compare_owocr.py <original model_dir> <our libchromescreenai.so> <images...>
A scratch model dir is assembled from symlinks (the user's model dir is never modified)."""
import os, sys, tempfile, json, subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent

def make_dir(model_dir, lib):
    d = Path(tempfile.mkdtemp(prefix='screenai_models_'))
    for p in Path(model_dir).iterdir():
        if p.name != 'libchromescreenai.so':
            os.symlink(p, d / p.name)
    os.symlink(Path(lib).resolve(), d / 'libchromescreenai.so')
    return d

def main():
    model_dir, lib, images = sys.argv[1], sys.argv[2], sys.argv[3:]
    ours = make_dir(model_dir, lib)
    results = {}
    for name, d in (('orig', model_dir), ('ours', str(ours))):
        out = subprocess.run([sys.executable, '-I', str(HERE / 'owocr_dump.py'), d] + images, capture_output=True, text=True)
        if out.returncode:
            sys.exit(f'{name} failed:\n{out.stderr[-3000:]}')
        results[name] = json.loads(out.stdout)
    bad = 0
    for img in images:
        ra, rb = results['orig'][img], results['ours'][img]
        same_generic = ra['generic'] == rb['generic']
        same = same_generic and ra['raw'] == rb['raw']
        bad += not same
        status = 'identical' if same else ('owocr result identical, raw annotation differs' if same_generic else 'DIFFERENT')
        print(f'{os.path.basename(img)}: {status}')
        if not same:
            a, b = (ra['raw'] or {}).get('lines', []), (rb['raw'] or {}).get('lines', [])
            for i, (la, lb) in enumerate(zip(a, b)):
                if la != lb:
                    print('  line', i, '\n    orig', json.dumps(la, ensure_ascii=False)[:400], '\n    ours', json.dumps(lb, ensure_ascii=False)[:400])
                    break
            if len(a) != len(b):
                print(f'  line count {len(a)} vs {len(b)}')
    print(f'{len(images) - bad}/{len(images)} identical')

if __name__ == '__main__':
    main()
