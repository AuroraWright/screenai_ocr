# SPDX-License-Identifier: MPL-2.0
"""lldb script: print every PageLayoutMutator::MutateSub / Calculator::Process invoked (name + thread)."""
import lldb
def cb(frame, bp_loc, d):
    print('HIT', frame.GetThread().GetIndexID(), frame.GetFunctionName())
    return False
def __lldb_init_module(debugger, d):
    t = debugger.GetSelectedTarget()
    for rx in [r'::MutateSub\(google_ocr::PageLayoutMutatorContext', r'Calculator::Process\(mediapipe::CalculatorContext']:
        bp = t.BreakpointCreateByRegex(rx, 'libchromescreenai.so')
        bp.SetScriptCallbackFunction('list_mutators.cb')
