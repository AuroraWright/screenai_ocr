# SPDX-License-Identifier: MPL-2.0
"""lldb script: wall-clock timestamps at the original's stage boundaries (mutator entries + VisualAnnotation conversion).
Breakpoints auto-continue; prints deltas per PerformOCR call. Overhead is a few ms per stop."""
import lldb, time
HOOKS = [(r'GocrGroupRpnTextDetectionMutator::MutateSub\(', 'detect_start'),
         (r'MultiPassLineRecognitionMutator::MutateSub\(google_ocr::PageLayoutMutatorContext ?\*\) ?const$', 'recognize_start'),
         (r'PageLayoutAnalysisMutator::MutateSub\(google_ocr::PageLayoutMutatorContext ?\*\) ?const$', 'layout_start'),
         (r'ConvertOCRResultsToVisualAnnotationProto\(', 'convert_start')]
marks = []
def hit(frame, bp_loc, extra, internal_dict):
    name = HOOKS[bp_loc.GetBreakpoint().GetID() - 1][1] if bp_loc.GetBreakpoint().GetID() <= len(HOOKS) else '?'
    marks.append((name, time.perf_counter()))
    if name == 'convert_start':
        t = dict(marks)
        if 'detect_start' in t:
            print('STAGES detect %.1f ms  recognize %.1f ms  layout %.1f ms' % (
                (t['recognize_start'] - t['detect_start']) * 1e3, (t['layout_start'] - t['recognize_start']) * 1e3,
                (t['convert_start'] - t['layout_start']) * 1e3), flush=True)
        marks.clear()
    return False
def __lldb_init_module(debugger, d):
    t = debugger.GetSelectedTarget()
    for rx, name in HOOKS:
        bp = t.BreakpointCreateByRegex(rx, 'libchromescreenai.so')
        bp.SetScriptCallbackFunction('stage_time.hit')
        bp.SetAutoContinue(True)
