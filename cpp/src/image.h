// SPDX-License-Identifier: MPL-2.0
// Image containers and the pixel conversions used by the original pipeline.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace screenai {

struct GrayImage {
  int width = 0, height = 0;
  std::vector<uint8_t> data;  // row-major, stride == width
  uint8_t* row(int y) { return data.data() + static_cast<size_t>(y) * width; }
  const uint8_t* row(int y) const { return data.data() + static_cast<size_t>(y) * width; }
};

struct RgbaImage {
  int width = 0, height = 0;
  std::vector<uint8_t> data;  // R,G,B,A bytes, stride == 4 * width
};

// visionkit ImageConversion: gray = (77 R + 150 G + 29 B + 128) >> 8 (alpha ignored).
GrayImage RgbaToGray(const RgbaImage& rgba);

// Loads PNG/JPEG/etc. (tools only; the library itself receives raw pixels).
bool LoadImageFile(const std::string& path, RgbaImage* out);

}  // namespace screenai
