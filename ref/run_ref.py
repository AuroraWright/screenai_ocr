# SPDX-License-Identifier: MPL-2.0
"""Run the original libchromescreenai.so on images and dump VisualAnnotation as JSON (reference outputs)."""
import ctypes, os, sys, json
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'third_party'))
from chrome_screen_ai_pb2 import VisualAnnotation
from google.protobuf.json_format import MessageToDict

MODEL_DIR = Path(os.environ.get('SCREEN_AI_DIR', Path.home() / '.config/screen_ai/resources'))

@ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_char_p)
def _size(p):
    path = MODEL_DIR / p.decode()
    return os.path.getsize(path) if path.exists() else 0

@ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_void_p)
def _content(p, s, ptr):
    path = MODEL_DIR / p.decode()
    if path.exists():
        ctypes.memmove(ptr, path.read_bytes()[:s], s)

class SkColorInfo(ctypes.Structure):
    _fields_ = [('fColorSpace', ctypes.c_void_p), ('fColorType', ctypes.c_int32), ('fAlphaType', ctypes.c_int32)]
class SkISize(ctypes.Structure):
    _fields_ = [('fWidth', ctypes.c_int32), ('fHeight', ctypes.c_int32)]
class SkImageInfo(ctypes.Structure):
    _fields_ = [('fColorInfo', SkColorInfo), ('fDimensions', SkISize)]
class SkPixmap(ctypes.Structure):
    _fields_ = [('fPixels', ctypes.c_void_p), ('fRowBytes', ctypes.c_size_t), ('fInfo', SkImageInfo)]
class SkBitmap(ctypes.Structure):
    _fields_ = [('fPixelRef', ctypes.c_void_p), ('fPixmap', SkPixmap), ('fFlags', ctypes.c_uint32)]

_lib = None
def lib():
    global _lib
    if _lib is None:
        l = ctypes.CDLL(str(MODEL_DIR / 'libchromescreenai.so'), mode=os.RTLD_LAZY)
        l.SetFileContentFunctions.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        l.InitOCRUsingCallback.restype = ctypes.c_bool
        l.SetOCRLightMode.argtypes = [ctypes.c_bool]
        l.PerformOCR.argtypes = [ctypes.POINTER(SkBitmap), ctypes.POINTER(ctypes.c_uint32)]
        l.PerformOCR.restype = ctypes.c_void_p
        l.FreeLibraryAllocatedCharArray.argtypes = [ctypes.c_void_p]
        l.GetMaxImageDimension.restype = ctypes.c_uint32
        l.SetFileContentFunctions(_size, _content)
        assert l.InitOCRUsingCallback()
        l.SetOCRLightMode(False)
        _lib = l
    return _lib

def ocr_raw(img: Image.Image) -> bytes:
    l = lib()
    data = img.convert('RGBA').tobytes()
    bm = SkBitmap()
    bm.fPixmap.fPixels = ctypes.cast(ctypes.c_char_p(data), ctypes.c_void_p)
    bm.fPixmap.fRowBytes = img.width * 4
    bm.fPixmap.fInfo.fColorInfo.fColorType = 4
    bm.fPixmap.fInfo.fColorInfo.fAlphaType = 1
    bm.fPixmap.fInfo.fDimensions.fWidth = img.width
    bm.fPixmap.fInfo.fDimensions.fHeight = img.height
    n = ctypes.c_uint32(0)
    ptr = l.PerformOCR(ctypes.byref(bm), ctypes.byref(n))
    assert ptr
    out = ctypes.string_at(ptr, n.value)
    l.FreeLibraryAllocatedCharArray(ptr)
    return out

def ocr(img):
    va = VisualAnnotation(); va.ParseFromString(ocr_raw(img))
    return MessageToDict(va, preserving_proto_field_name=True)

if __name__ == '__main__':
    for p in sys.argv[1:]:
        r = ocr(Image.open(p))
        out = Path(p).with_suffix('.ref.json')
        out.write_text(json.dumps(r, ensure_ascii=False, indent=1))
        print(p, '->', out, len(r.get('lines', [])), 'lines')
        for l in r.get('lines', []): print('  ', l.get('block_id'), l.get('direction'), repr(l.get('utf8_string')))
