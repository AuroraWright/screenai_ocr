// SPDX-License-Identifier: MPL-2.0
// The original's x86-64 build (the Windows DLL) takes x86-specific integer paths in a few image kernels where its
// arm64 build matches the portable reference: libyuv's ScaleFilterCols_SSSE3 (libyuv_scale.cc) and the Q15
// interior of the line warp (recognizer.cc). SCREENAI_X86_COMPAT selects those paths: on for x86-64 targets, off
// elsewhere; -DSCREENAI_X86_COMPAT=0/1 overrides it (e.g. to test the x86 paths on arm).
#pragma once

#ifndef SCREENAI_X86_COMPAT
#if defined(__x86_64__) || (defined(_M_X64) && !defined(_M_ARM64EC))
#define SCREENAI_X86_COMPAT 1
#else
#define SCREENAI_X86_COMPAT 0
#endif
#endif
