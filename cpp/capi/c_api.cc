// Drop-in replacement for the OCR entry points of Chrome's libchromescreenai (chrome_screen_ai/ocr.cc and its C
// exports), so existing callers (e.g. owocr) work unchanged. Main-content extraction is not implemented.
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

#include "ocr_engine.h"

#define SCREENAI_EXPORT extern "C" __attribute__((visibility("default")))

namespace {

// Skia's SkBitmap layout as the library reads it (SkPixmap fields; row bytes are ignored, rows must be tight).
struct SkBitmap {
  void* pixel_ref;
  const uint8_t* pixels;
  size_t row_bytes;
  void* color_space;
  int32_t color_type;
  int32_t alpha_type;
  int32_t width, height;
};
constexpr int kRGBA_8888 = 4, kBGRA_8888 = 6, kGray_8 = 14;

screenai::Resources::SizeFn g_size_fn = nullptr;
screenai::Resources::ContentFn g_content_fn = nullptr;
std::unique_ptr<screenai::OcrEngine> g_engine;
int g_max_image_dimension = 0;

char* CopyOut(const std::string& s, uint32_t* length) {
  *length = static_cast<uint32_t>(s.size());
  char* out = new char[s.size()];
  std::memcpy(out, s.data(), s.size());
  return out;
}

}  // namespace

SCREENAI_EXPORT void SetFileContentFunctions(screenai::Resources::SizeFn size_fn,
                                             screenai::Resources::ContentFn content_fn) {
  g_size_fn = size_fn;
  g_content_fn = content_fn;
}

SCREENAI_EXPORT bool InitOCRUsingCallback() {
  if (g_engine) return true;
  std::shared_ptr<screenai::Resources> resources;
  if (g_size_fn && g_content_fn) {
    resources = screenai::Resources::FromCallbacks(g_size_fn, g_content_fn);
  } else if (const char* dir = std::getenv("SCREEN_AI_MODEL_DIR")) {  // convenience for callers without callbacks
    resources = screenai::Resources::FromDirectory(dir);
  } else {
    return false;
  }
  g_engine = screenai::OcrEngine::Create(std::move(resources));
  if (!g_engine) return false;
  g_max_image_dimension = 2048;
  return true;
}

// The light pipeline swaps in a different line recognizer; only the default pipeline is reimplemented.
SCREENAI_EXPORT void SetOCRLightMode(bool) {}

SCREENAI_EXPORT uint32_t GetMaxImageDimension() { return g_max_image_dimension; }

// The original keeps reporting the max image dimension after uninitialization.
SCREENAI_EXPORT void UninitializeOCR() { g_engine.reset(); }

SCREENAI_EXPORT char* PerformOCR(const SkBitmap* bitmap, uint32_t* length) {
  *length = 0;
  if (!g_engine && !InitOCRUsingCallback()) return nullptr;
  const int w = bitmap->width, h = bitmap->height;
  // "Unsupported image size." (the original aborts on negative sizes; here they are rejected the same way)
  if (w <= 0 || h <= 0 || w > 32768 || h > 32768) return new char[0];
  screenai::RgbaImage rgba;
  rgba.width = w;
  rgba.height = h;
  const size_t n = static_cast<size_t>(w) * h;
  if (bitmap->color_type == kRGBA_8888 || bitmap->color_type == kBGRA_8888) {
    // The original hands both orders to the pipeline as the same 4-channel format.
    rgba.data.assign(bitmap->pixels, bitmap->pixels + 4 * n);
  } else if (bitmap->color_type == kGray_8) {
    rgba.data.resize(4 * n);
    for (size_t i = 0; i < n; i++) std::memset(&rgba.data[4 * i], bitmap->pixels[i], 3), rgba.data[4 * i + 3] = 255;
  } else {
    return new char[0];  // "Unsupported color type"
  }
  std::string out;
  if (!g_engine->Run(rgba, &out)) return nullptr;
  return CopyOut(out, length);
}

SCREENAI_EXPORT void FreeLibraryAllocatedCharArray(char* p) { delete[] p; }
SCREENAI_EXPORT void FreeLibraryAllocatedInt32Array(int32_t* p) { delete[] p; }

SCREENAI_EXPORT void GetLibraryVersion(uint32_t* major, uint32_t* minor) {
  *major = 153;  // the version string of the reimplemented library: "153.05"
  *minor = 5;
}

SCREENAI_EXPORT void EnableDebugMode() {}
