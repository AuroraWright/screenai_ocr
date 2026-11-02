// Bit-exact reimplementation of libyuv::ScalePlane (8-bit, single channel) as used by the OCR pipeline.
// Only the code paths reachable for positive widths/heights are implemented; arithmetic follows libyuv's C
// reference row kernels (which its NEON kernels match bit-for-bit).
#pragma once

#include <cstdint>

namespace screenai {

enum FilterMode { kFilterNone = 0, kFilterLinear = 1, kFilterBilinear = 2, kFilterBox = 3 };

// Scales a src_width x src_height plane into dst (dst_stride bytes per row).
void ScalePlane(const uint8_t* src, int src_stride, int src_width, int src_height, uint8_t* dst, int dst_stride,
                int dst_width, int dst_height, FilterMode filtering);

}  // namespace screenai
