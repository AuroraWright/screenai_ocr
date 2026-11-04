# SPDX-License-Identifier: MPL-2.0
"""Runs owocr's own ChromeScreenAI engine on images with the given model dir; prints JSON
{image: {"raw": MessageToDict(VisualAnnotation), "generic": owocr OcrResult as dict}}."""
import os, sys, json, types, logging, tempfile, dataclasses
from pathlib import Path

model_dir = Path(sys.argv[1]).resolve()
home = Path(tempfile.mkdtemp(prefix='owocr_home_'))
(home / '.config' / 'screen_ai').mkdir(parents=True)
os.symlink(model_dir, home / '.config' / 'screen_ai' / 'resources')
os.environ['HOME'] = str(home)

# owocr imports these at module level; the Screen AI engine does not use them.
sys.modules['loguru'] = types.SimpleNamespace(logger=logging.getLogger('owocr'))
sys.modules['curl_cffi'] = types.ModuleType('curl_cffi')
sys.path.insert(0, os.environ.get('OWOCR_ROOT', '/Users/aurora/Developer/owocr'))
import owocr.ocr as ocr

# Keep the raw VisualAnnotation dict too: owocr's generic result drops fields such as language.
last_response = {}
to_generic = ocr.ChromeScreenAI._to_generic_result
def capture(self, response, *args):
    last_response['r'] = response
    return to_generic(self, response, *args)
ocr.ChromeScreenAI._to_generic_result = capture

engine = ocr.ChromeScreenAI()
assert engine.available, 'engine not available'
out = {}
for img in sys.argv[2:]:
    last_response.clear()
    ok, res = engine(Path(img))
    out[img] = {'ok': ok, 'raw': last_response.get('r'), 'generic': dataclasses.asdict(res) if ok else str(res)}
print(json.dumps(out, ensure_ascii=False))
