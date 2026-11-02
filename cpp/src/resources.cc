#include "resources.h"

#include <fstream>
#include <iterator>

namespace screenai {

std::shared_ptr<Resources> Resources::FromDirectory(const std::string& root) {
  auto r = std::make_shared<Resources>();
  r->read_ = [root](const std::string& rel, std::string* out) {
    std::ifstream f(root + "/" + rel, std::ios::binary);
    if (!f) return false;
    out->assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    return !out->empty();
  };
  return r;
}

std::shared_ptr<Resources> Resources::FromCallbacks(SizeFn size_fn, ContentFn content_fn) {
  auto r = std::make_shared<Resources>();
  r->read_ = [size_fn, content_fn](const std::string& rel, std::string* out) {
    const uint32_t n = size_fn(rel.c_str());
    if (n == 0) return false;
    out->assign(n, '\0');
    content_fn(rel.c_str(), n, out->data());
    return true;
  };
  return r;
}

bool Resources::Read(const std::string& relative_path, std::string* out) const { return read_(relative_path, out); }

}  // namespace screenai
