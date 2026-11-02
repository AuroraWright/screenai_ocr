"""Runs the original library's PerformOCR on one image N times (for timing under lldb)."""
import sys; sys.path.insert(0, sys.path[0] + '/../ref')
from run_ref import ocr_raw
from PIL import Image
img = Image.open(sys.argv[1])
for _ in range(int(sys.argv[2])): ocr_raw(img)
