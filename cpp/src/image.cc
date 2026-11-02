#include "image.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_ONLY_BMP
#include "stb_image.h"

namespace screenai {

GrayImage RgbaToGray(const RgbaImage& rgba) {
  GrayImage g;
  g.width = rgba.width;
  g.height = rgba.height;
  g.data.resize(static_cast<size_t>(g.width) * g.height);
  const uint8_t* p = rgba.data.data();
  for (size_t i = 0; i < g.data.size(); i++, p += 4) g.data[i] = static_cast<uint8_t>((77 * p[0] + 150 * p[1] + 29 * p[2] + 128) >> 8);
  return g;
}

bool LoadImageFile(const std::string& path, RgbaImage* out) {
  int w, h, c;
  unsigned char* px = stbi_load(path.c_str(), &w, &h, &c, 4);
  if (!px) return false;
  out->width = w;
  out->height = h;
  out->data.assign(px, px + static_cast<size_t>(w) * h * 4);
  stbi_image_free(px);
  return true;
}

}  // namespace screenai
