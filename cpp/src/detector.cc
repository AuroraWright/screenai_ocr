// SPDX-License-Identifier: MPL-2.0
#include "detector.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <string_view>

#include "box_overlap.h"
#include "detector_sort.h"
#include "libyuv_scale.h"
#include "proto_wire.h"
#include "tflite_model.h"

namespace screenai {

namespace {

constexpr char kModelPath[] = "gocr/gocr_models/detection/gocr_group_rpn_text_detection_model_2024_q4.tflite";
constexpr char kConfigPath[] =
    "gocr/gocr_models/detection/gocr_group_rpn_text_detection_config_2024_q4_chrome.binarypb";

}  // namespace

// google_ocr.GroupRpnTextDetectionMutatorConfig (only the fields the Chrome config sets / the code reads).
struct TextDetector::Config {
  // model_runner_config.tflite_model_pooled_cached_runner_config
  int num_threads = 4;
  std::vector<std::string> input_names, output_names;
  // model_settings (GocrDetectorModelConfig)
  float confidence_threshold = 0.5f;
  std::vector<float> heatmap_width_scaling, heatmap_height_scaling, max_width, max_height;
  std::vector<int> subsampling;  // multi_head_heatmap_subsampling_factor
  bool heatmap_log_scaling = false, apply_sigmoid = false, project_cos_sin = false;
  bool combine_vertical_and_horizontal = false;
  float anchor_x_center = 0, anchor_y_center = 0, allowable_anchor_shift = 0;
  // options (GroupRpnTextDetectionMutatorRuntimeOptions)
  struct Level1 {  // GocrDetectorLevel1GroupingConfig
    float max_height_ratio = 0, search_radius_factor = 0, maximum_angle_difference = 0,
          max_centers_relative_distance = 0, optimal_centers_relative_distance = 0, min_overlap = 0,
          min_overlap_with_optimal_centers_aligment = 0;
    int minimum_cluster_size_to_curve = 0;
    float maximum_height_ratio_to_curve = 0, minimum_angle_variance_to_curve = 0;
  } level1;
  struct Merge {  // GocrDetectorMergeLevel1Config
    float v[19] = {};  // fields 1..18 (index = field - 1)
    bool use_experimental_curved_line_construction = false, enable_reclustering = false;
  } merge;
  std::vector<std::pair<int, int>> predefined_shapes;  // input_image_config.predefined_shapes (width, height)
  float box_width_padding = 0, box_top_height_padding = 0, box_bottom_height_padding = 0,
        box_width_padding_left_portion = 0;  // post_process_config
  float canonical_text_line_height = 0;
  std::vector<int> packing_sequence, center_box_packing_sequence;
  bool skip_level1_nms = false, build_nuggets = false;
  int version = 0;
};

namespace {

using Config = TextDetector::Config;

void ReadPackedFloats(const pb::Reader& r, std::vector<float>* out) {
  if (r.wire_type() != 2) {
    out->push_back(r.float32());
    return;
  }
  std::string_view b = r.bytes();
  for (size_t i = 0; i + 4 <= b.size(); i += 4) {
    float f;
    std::memcpy(&f, b.data() + i, 4);
    out->push_back(f);
  }
}

void ReadPackedInts(const pb::Reader& r, std::vector<int>* out) {
  if (r.wire_type() != 2) {
    out->push_back(static_cast<int>(r.int64()));
    return;
  }
  std::string_view b = r.bytes();
  uint64_t v = 0;
  int shift = 0;
  for (char c : b) {
    v |= static_cast<uint64_t>(c & 0x7F) << shift;
    shift += 7;
    if (!(c & 0x80)) {
      out->push_back(static_cast<int>(v));
      v = 0, shift = 0;
    }
  }
}

void ParseModelSettings(std::string_view b, Config* c) {
  pb::Reader r(b);
  while (r.Next()) {
    switch (r.field()) {
      case 1: c->confidence_threshold = r.float32(); break;
      case 4: ReadPackedFloats(r, &c->heatmap_width_scaling); break;
      case 5: ReadPackedFloats(r, &c->heatmap_height_scaling); break;
      case 6: c->heatmap_log_scaling = r.varint() != 0; break;
      case 7: c->apply_sigmoid = r.varint() != 0; break;
      case 8: c->anchor_x_center = r.float32(); break;
      case 9: c->anchor_y_center = r.float32(); break;
      case 10: c->allowable_anchor_shift = r.float32(); break;
      case 11: c->project_cos_sin = r.varint() != 0; break;
      case 12: ReadPackedInts(r, &c->subsampling); break;
      case 18: c->combine_vertical_and_horizontal = r.varint() != 0; break;
      case 19: ReadPackedFloats(r, &c->max_width); break;
      case 20: ReadPackedFloats(r, &c->max_height); break;
    }
  }
}

void ParseOptions(std::string_view b, Config* c) {
  pb::Reader r(b);
  while (r.Next()) {
    switch (r.field()) {
      case 2: {
        pb::Reader l(r.bytes());
        Config::Level1& o = c->level1;
        while (l.Next()) {
          switch (l.field()) {
            case 1: o.max_height_ratio = l.float32(); break;
            case 2: o.search_radius_factor = l.float32(); break;
            case 3: o.maximum_angle_difference = l.float32(); break;
            case 4: o.max_centers_relative_distance = l.float32(); break;
            case 5: o.optimal_centers_relative_distance = l.float32(); break;
            case 6: o.min_overlap = l.float32(); break;
            case 7: o.min_overlap_with_optimal_centers_aligment = l.float32(); break;
            case 8: o.minimum_cluster_size_to_curve = static_cast<int>(l.int64()); break;
            case 9: o.maximum_height_ratio_to_curve = l.float32(); break;
            case 10: o.minimum_angle_variance_to_curve = l.float32(); break;
          }
        }
        break;
      }
      case 4: {
        pb::Reader m(r.bytes());
        while (m.Next()) {
          if (m.field() >= 1 && m.field() <= 18) c->merge.v[m.field() - 1] = m.float32();
          if (m.field() == 19) c->merge.use_experimental_curved_line_construction = m.varint() != 0;
          if (m.field() == 20) c->merge.enable_reclustering = m.varint() != 0;
        }
        break;
      }
      case 6: {
        pb::Reader ic(r.bytes());
        while (ic.Next()) {
          if (ic.field() != 6) continue;
          std::pair<int, int> s{0, 0};
          pb::Reader sr(ic.bytes());
          while (sr.Next()) (sr.field() == 1 ? s.first : s.second) = static_cast<int>(sr.int64());
          c->predefined_shapes.push_back(s);
        }
        break;
      }
      case 7: {
        pb::Reader p(r.bytes());
        while (p.Next()) {
          if (p.field() == 2) c->box_width_padding = p.float32();
          if (p.field() == 3) c->box_top_height_padding = p.float32();
          if (p.field() == 4) c->box_bottom_height_padding = p.float32();
          if (p.field() == 5) c->box_width_padding_left_portion = p.float32();
        }
        break;
      }
      case 9: c->canonical_text_line_height = r.float32(); break;
      case 11: c->build_nuggets = r.varint() != 0; break;
      case 13: ReadPackedInts(r, &c->packing_sequence); break;
      case 14: c->skip_level1_nms = r.varint() != 0; break;
      case 16: ReadPackedInts(r, &c->center_box_packing_sequence); break;
      case 17: c->version = static_cast<int>(r.int64()); break;
    }
  }
}

bool ParseConfig(std::string_view any_bytes, Config* c) {
  // The file is a google.protobuf.Any wrapping GroupRpnTextDetectionMutatorConfig.
  std::string_view cfg;
  pb::Reader a(any_bytes);
  while (a.Next())
    if (a.field() == 2) cfg = a.bytes();
  pb::Reader r(cfg);
  while (r.Next()) {
    switch (r.field()) {
      case 1: {
        pb::Reader mr(r.bytes());
        while (mr.Next()) {
          if (mr.field() != 7) continue;  // tflite_model_pooled_cached_runner_config
          pb::Reader t(mr.bytes());
          while (t.Next()) {
            if (t.field() == 3) c->num_threads = static_cast<int>(t.int64());
            if (t.field() == 8) c->output_names.emplace_back(t.bytes());
            if (t.field() == 12) c->input_names.emplace_back(t.bytes());
          }
        }
        break;
      }
      case 2: ParseModelSettings(r.bytes(), c); break;
      case 3: ParseOptions(r.bytes(), c); break;
    }
  }
  return r.ok() && !c->output_names.empty() && !c->packing_sequence.empty() && !c->predefined_shapes.empty();
}

// ---------------------------------------------------------------------------------------------------------------
// Debug dumps (SCREENAI_DET_DUMP=<dir>), same binary layout as trace/det_dump.py.

const char* DumpDir() {
  static const char* d = std::getenv("SCREENAI_DET_DUMP");
  return d;
}

void DumpFile(const std::string& name, const std::string& data) {
  if (!DumpDir()) return;
  std::string path = std::string(DumpDir()) + "/" + name;
  if (FILE* f = std::fopen(path.c_str(), "wb")) {
    std::fwrite(data.data(), 1, data.size(), f);
    std::fclose(f);
  }
}

template <typename T>
void AppendPod(std::string* s, const T& v) {
  s->append(reinterpret_cast<const char*>(&v), sizeof(T));
}

// Model output edits (SCREENAI_DET_INJECT=<file>, same format as trace/det_dump.py's DD_INJECT), applied to the
// output tensors before they are decoded, to give the post-processing exactly equal inputs. One edit per line:
//   set  <pass> <tensor> <y0> <x0> <h> <w> <ch> <value>   channel ch of the h x w cells at (y0, x0) = value
//   copy <pass> <tensor> <sy> <sx> <dy> <dx> <h> <w>       the h x w cells at (sy, sx) (all channels) to (dy, dx)
// pass 0 = horizontal heads, 1 = vertical heads; tensor = index within the pass or *; h / w 0 = to the edge;
// value = strtod (decimal or hex float), rounded to float. Blocks are clipped to the tensor; # starts a comment.
struct OutputEdit {
  bool copy = false;
  int pass = 0, tensor = -1, a[6] = {}, ch = 0;
  float value = 0.0f;
};

const std::vector<OutputEdit>& OutputEdits() {
  static const std::vector<OutputEdit> edits = [] {
    std::vector<OutputEdit> v;
    const char* path = std::getenv("SCREENAI_DET_INJECT");
    if (!path) return v;
    FILE* f = std::fopen(path, "r");
    if (!f) return v;
    char line[512];
    while (std::fgets(line, sizeof line, f)) {
      if (char* h = std::strchr(line, '#')) *h = 0;
      char op[16], tensor[16];
      OutputEdit e;
      char val[64];
      int n = std::sscanf(line, "%15s %d %15s %d %d %d %d", op, &e.pass, tensor, &e.a[0], &e.a[1], &e.a[2], &e.a[3]);
      if (n <= 0) continue;
      e.tensor = tensor[0] == '*' ? -1 : std::atoi(tensor);
      if (std::strcmp(op, "set") == 0 &&
          std::sscanf(line, "%*s %*d %*s %d %d %d %d %d %63s", &e.a[0], &e.a[1], &e.a[2], &e.a[3], &e.ch, val) == 6) {
        e.value = static_cast<float>(std::strtod(val, nullptr));
      } else if (std::strcmp(op, "copy") == 0 &&
                 std::sscanf(line, "%*s %*d %*s %d %d %d %d %d %d", &e.a[0], &e.a[1], &e.a[2], &e.a[3], &e.a[4],
                             &e.a[5]) == 6) {
        e.copy = true;
      } else {
        std::fprintf(stderr, "SCREENAI_DET_INJECT: bad line: %s", line);
        continue;
      }
      v.push_back(e);
    }
    std::fclose(f);
    return v;
  }();
  return edits;
}

void ApplyOutputEdits(int pass, int tensor, float* t, int H, int W, int C) {
  for (const OutputEdit& e : OutputEdits()) {
    if (e.pass != pass || (e.tensor != -1 && e.tensor != tensor)) continue;
    const int h = e.copy ? e.a[4] : e.a[2], w = e.copy ? e.a[5] : e.a[3];
    auto cell = [&](int y, int x) { return t + (static_cast<size_t>(y) * W + x) * C; };
    if (!e.copy) {
      if (e.ch < 0 || e.ch >= C) continue;
      for (int y = std::max(0, e.a[0]); y < (h ? std::min(H, e.a[0] + h) : H); y++)
        for (int x = std::max(0, e.a[1]); x < (w ? std::min(W, e.a[1] + w) : W); x++) cell(y, x)[e.ch] = e.value;
      continue;
    }
    const int bh = h ? h : H, bw = w ? w : W;
    std::vector<float> snap(static_cast<size_t>(bh) * bw * C);
    std::vector<bool> valid(static_cast<size_t>(bh) * bw);
    for (int i = 0; i < bh; i++)
      for (int j = 0; j < bw; j++) {
        const int y = e.a[0] + i, x = e.a[1] + j;
        if (y < 0 || y >= H || x < 0 || x >= W) continue;
        valid[static_cast<size_t>(i) * bw + j] = true;
        std::memcpy(&snap[(static_cast<size_t>(i) * bw + j) * C], cell(y, x), C * sizeof(float));
      }
    for (int i = 0; i < bh; i++)
      for (int j = 0; j < bw; j++) {
        const int y = e.a[2] + i, x = e.a[3] + j;
        if (!valid[static_cast<size_t>(i) * bw + j] || y < 0 || y >= H || x < 0 || x >= W) continue;
        std::memcpy(cell(y, x), &snap[(static_cast<size_t>(i) * bw + j) * C], C * sizeof(float));
      }
  }
}

// Per sort site: number of sorts, and of adjacent element pairs left tied (neither compares less) after sorting.
std::map<std::string, std::pair<int, int>>& TieCounts() {
  static std::map<std::string, std::pair<int, int>> m;
  return m;
}

// Branch counters for the SCREENAI_DET_DUMP ties.txt (which ConstructLines paths an input reaches).
void CountEvent(const char* name) {
  if (DumpDir()) TieCounts()[std::string("event:") + name].first++;
}

// detsort::Sort, recording ties for the SCREENAI_DET_DUMP ties.txt.
template <class It, class C>
void SortCounted(const char* site, It first, It last, C comp) {
  detsort::Sort(first, last, comp);
  if (!DumpDir()) return;
  auto& t = TieCounts()[site];
  t.first++;
  for (It i = first; i != last && i + 1 != last; ++i)
    if (!comp(*i, *(i + 1)) && !comp(*(i + 1), *i)) t.second++;
}

// ---------------------------------------------------------------------------------------------------------------
// Image pyramid (GroupRpnDetectorV2 + group_rpn_detector_tensor_utils).

struct Plane {
  int width = 0, height = 0;
  std::vector<uint8_t> data;
};

// google_ocr::ScaleImage: dims rintf(scale * dim) but at least 1, libyuv ScalePlane with the given filter.
Plane ScaleImage(const Plane& src, float scale, FilterMode filter) {
  Plane dst;
  dst.width = std::max(1, static_cast<int>(rintf(scale * static_cast<float>(src.width))));
  dst.height = std::max(1, static_cast<int>(rintf(scale * static_cast<float>(src.height))));
  dst.data.resize(static_cast<size_t>(dst.width) * dst.height);
  ScalePlane(src.data.data(), src.width, src.width, src.height, dst.data.data(), dst.width, dst.width, dst.height,
             filter);
  return dst;
}

// ---------------------------------------------------------------------------------------------------------------
// Trigonometry.

constexpr float kDegToRad = 0.017453292f;  // 0x3C8EFA35
constexpr float kRadToDeg = 57.2957763671875f;  // 0x42652EE0

// The original calls Apple libm's __sincosf_stret / __sincos_stret, whose last bits differ from sinf/cosf.
struct SinCos {
  float s, c;
};
inline SinCos SinCosF(float x) {
#ifdef __APPLE__
  const __float2 r = __sincosf_stret(x);
  return {r.__sinval, r.__cosval};
#else
  return {sinf(x), cosf(x)};
#endif
}
inline void SinCosD(double x, double* s, double* c) {
#ifdef __APPLE__
  const __double2 r = __sincos_stret(x);
  *s = r.__sinval, *c = r.__cosval;
#else
  *s = sin(x), *c = cos(x);
#endif
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------
// Raw boxes (ocr::photo::HeatmapToBoundingBoxes + google_ocr::TensorsToRawBoxes).

namespace {

struct RawBox {  // google_ocr::RawBox (68 bytes in the original)
  float x, y, w, h, angle;  // top-left corner (before rotation about it), size, degrees
  float pts[8];             // corners: (x,y), (x+w*cos, y+w*sin), +(-sin*h, cos*h), (x-sin*h, y+cos*h)
  float cx, cy;
  float score;
  int batch;
};
static_assert(sizeof(RawBox) == 68, "RawBox layout");

// One model output head [1, H, W, 7]: channels = logit, dx, dy, log w, log h, cos, sin.
void HeatmapToRawBoxes(const float* t, int H, int W, int head, const Config& c, std::vector<RawBox>* out) {
  const float min_logit = -logf(1.0f / c.confidence_threshold + -1.0f);
  const float sub = static_cast<float>(c.subsampling[head]);
  const float wscale = c.heatmap_width_scaling[head], hscale = c.heatmap_height_scaling[head];
  const float max_w = c.max_width[head], max_h = c.max_height[head];
  for (int y = 0; y < H; y++) {
    for (int x = 0; x < W; x++) {
      const float* p = t + (static_cast<size_t>(y) * W + x) * 7;
      float score = p[0];
      if (c.apply_sigmoid) {
        if (!(score > min_logit)) continue;
        score = 1.0f / (expf(-score) + 1.0f);
      } else if (!(score > c.confidence_threshold)) {
        continue;
      }
      const float dx = p[1], dy = p[2], lw = p[3], lh = p[4], cs = p[5], sn = p[6];
      float angle;
      if (c.project_cos_sin) {
        const float n = sqrtf(cs * cs + sn * sn) + 0.00001f;
        angle = static_cast<float>(static_cast<double>(atan2f(sn / n, cs / n) * 180.0f) / 3.141592653589793);
      } else {
        float v = sn <= 1.0f ? sn : 1.0f;
        if (v < -1.0f) v = -1.0f;
        double a = static_cast<double>(asinf(v) * 180.0f) / 3.141592653589793;
        if (cs <= 0.0f) a = a + 180.0;
        angle = static_cast<float>(a);
      }
      const double rad = static_cast<double>(angle) * 3.141592653589793 / 180.0;
      double sinv, cosv;
      SinCosD(rad, &sinv, &cosv);
      float w, h;
      if (c.heatmap_log_scaling) {
        w = expf(lw) * wscale;
        h = expf(lh) * hscale;
      } else {
        w = lw * wscale;
        h = lh * hscale;
      }
      if (!(w >= 1.0f) || !(h >= 1.0f)) continue;
      const float bcx = (c.anchor_x_center + (dx + static_cast<float>(x))) * sub;
      const float bcy = (c.anchor_y_center + (dy + static_cast<float>(y))) * sub;
      const double c64 = static_cast<float>(cosv), s64 = static_cast<float>(sinv);
      const double hw = static_cast<double>(w) * 0.5, hh = static_cast<double>(h) * 0.5;
      RawBox b;
      b.x = static_cast<float>((static_cast<double>(bcx) - c64 * hw) + s64 * hh);
      b.y = static_cast<float>((static_cast<double>(bcy) - s64 * hw) - c64 * hh);
      b.w = w;
      b.h = h;
      b.angle = angle;
      b.score = score;
      b.batch = 0;
      // TensorsToRawBoxes
      if ((b.w > max_w && max_w > 0.0f) || (b.h > max_h && max_h > 0.0f)) continue;
      float p1x, p1y, p2x, p2y, p3x, p3y;
      if (b.angle == 0.0f) {
        p1x = b.x + b.w, p1y = b.y;
        p2x = b.x + b.w, p2y = b.y + b.h;
        p3x = b.x, p3y = b.y + b.h;
      } else {
        const SinCos sc = SinCosF(b.angle * 0.017453292f);
        const float sa = sc.s, ca = sc.c;
        p1x = b.x + b.w * ca;
        p1y = b.y + b.w * sa;
        const float t2 = -sa * b.h;
        p2x = t2 + p1x;
        p2y = ca * b.h + p1y;
        p3x = b.x + t2;
        p3y = b.y + ca * b.h;
      }
      b.pts[0] = b.x, b.pts[1] = b.y, b.pts[2] = p1x, b.pts[3] = p1y;
      b.pts[4] = p2x, b.pts[5] = p2y, b.pts[6] = p3x, b.pts[7] = p3y;
      b.cx = (((b.x + 0.0f) + p1x) + p2x + p3x) * 0.25f;
      b.cy = (((b.y + 0.0f) + p1y) + p2y + p3y) * 0.25f;
      out->push_back(b);
    }
  }
}

std::string SerializeRawLists(const std::vector<std::vector<RawBox>>& lists) {
  std::string s;
  AppendPod(&s, static_cast<uint32_t>(lists.size()));
  for (const auto& l : lists) {
    AppendPod(&s, static_cast<uint32_t>(l.size()));
    s.append(reinterpret_cast<const char*>(l.data()), l.size() * sizeof(RawBox));
  }
  return s;
}

// ---------------------------------------------------------------------------------------------------------------
// Geometry helpers (google_ocr::box_util, google_ocr::Rectangle<float>, rotation_util).


using RBox = overlap::Box;  // google_ocr::box_util::RotatedBox<float>

float NormalizeAngle(float a) {  // (-180, 180]
  while (a <= -180.0f) a = a + 360.0f;
  while (a > 180.0f) a = a + -360.0f;
  return a;
}

// Overlap area as the detector's call sites compute it (paragraph NMS in ProcessImagePyramidV2, level-1 grouping in
// GetPairwiseConnections): both raw angles 0 -> AxisAlignedBoxOverlapArea<RotatedBox<float>> (0x10080de18), else
// Rectangle<float>::CreateFromLTWHA x2 + OverlapArea, i.e. overlap::BoxOverlapArea (replayed against the library by
// trace/overlap_check.py det_pairs.bin). Both call sites pass (box i, box j) as the original does.
using overlap::BoxOverlapArea;

// rotation_util::RobustAngleEstimation<float>: median of the directions relative to the dominant quadrant.
float RobustAngleEstimation(const std::vector<float>& angles) {
  if (angles.empty()) return 0.0f;
  static const float kDirs[4][2] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
  int hist[4] = {0, 0, 0, 0};
  std::vector<std::pair<float, float>> cs;
  cs.reserve(angles.size());
  for (float a : angles) {
    const SinCos sc = SinCosF(a * kDegToRad);
    const float s = sc.s, c = sc.c;
    int best = 0;
    for (int k = 1; k < 4; k++) {
      const float db = (c * kDirs[best][0] + 0.0f) + s * kDirs[best][1];
      const float dk = (c * kDirs[k][0] + 0.0f) + s * kDirs[k][1];
      if (db < dk) best = k;
    }
    hist[best]++;
    cs.push_back({c, s});
  }
  int dom = 0;
  for (int k = 1; k < 4; k++)
    if (hist[dom] < hist[k]) dom = k;
  const float dx = kDirs[dom][0], dy = kDirs[dom][1];
  struct DirectionWithKey {
    float c, s, key;
  };
  std::vector<DirectionWithKey> v;
  v.reserve(cs.size());
  for (const auto& [c, s] : cs) v.push_back({c, s, atan2f(dx * s - dy * c, (dx * c + 0.0f) + dy * s)});
  SortCounted("angle_estimation", v.begin(), v.end(),
              [](const DirectionWithKey& a, const DirectionWithKey& b) { return a.key < b.key; });
  const size_t n = v.size();
  float mc, ms;
  if (n & 1) {
    mc = v[n / 2].c, ms = v[n / 2].s;
  } else {
    mc = (v[n / 2].c + v[n / 2 - 1].c) * 0.5f;
    ms = (v[n / 2].s + v[n / 2 - 1].s) * 0.5f;
  }
  const float a = atan2f(ms, mc) * kRadToDeg;
  return (ms == 0.0f && mc == 0.0f) ? 0.0f : a;
}

// Mean direction of two angles (degrees): atan2 of the summed unit vectors; for opposite directions the one closer
// to 0 (inlined rotation_util helper, shared by the level-1 and line grouping).
float MeanAngle(float a0, float a1) {
  const SinCos s0 = SinCosF(a0 * kDegToRad), s1 = SinCosF(a1 * kDegToRad);
  const float c = (s0.c + 0.0f) + s1.c;
  const float s = (s0.s + 0.0f) + s1.s;
  if (fabsf(c) > 0.00001f || fabsf(s) > 0.00001f)
    return (fabsf(s) > 0.00001f || c > -1.99998f) ? atan2f(s, c) * kRadToDeg : 180.0f;  // -1.99998f: 0xBFFFFF58
  float a = NormalizeAngle(a0);
  float aa = fabsf(a);
  if (aa >= 180.0f) a = 180.0f, aa = 180.0f;
  const float b = NormalizeAngle(a1);
  if (b == -a) a = fabsf(a);
  if (fabsf(b) < aa) a = b;
  return a;
}

// box_util::ExpandBoxToContainBox(src, dst): grow dst (keeping its angle) to contain src.
void ExpandBoxToContainBox(const RBox& src, RBox* dst) {
  if (dst->w <= 0.0f || dst->h <= 0.0f) {
    *dst = src;
    dst->angle = NormalizeAngle(src.angle);
    return;
  }
  const float sa = NormalizeAngle(src.angle), da = NormalizeAngle(dst->angle);
  if (sa == 0.0f && da == 0.0f) {
    float x = src.x, w = dst->w, h = dst->h;
    const float dxl = dst->x - src.x;
    if (dxl <= 0.0f) {
      x = dst->x;
    } else {
      dst->x = x;
      w = w + dxl;
      dst->w = w;
    }
    float y = src.y;
    const float dyt = dst->y - src.y;
    if (dyt <= 0.0f) {
      y = dst->y;
    } else {
      dst->y = y;
      h = h + dyt;
      dst->h = h;
    }
    const float r = src.x + src.w;
    if (r > x + w) dst->w = r - x;
    const float b = src.y + src.h;
    if (b > y + h) dst->h = b - y;
    return;
  }
  const float ox = dst->x, oy = dst->y, ow = dst->w, oh = dst->h;
  const SinCos sc1 = SinCosF(da * -0.017453292f);
  const float s1 = sc1.s, c1 = sc1.c;
  const float ddy = src.y - oy;
  const float k = NormalizeAngle(src.angle - da);
  const float p0x = (ox + c1 * (src.x - ox)) - s1 * ddy;
  const float p0y = (oy + s1 * (src.x - ox)) + c1 * ddy;
  float px[4], py[4];
  px[0] = p0x, py[0] = p0y;
  if (k == 0.0f) {
    px[1] = src.w + p0x, py[1] = p0y;
    px[2] = src.w + p0x, py[2] = src.h + p0y;
    px[3] = p0x, py[3] = src.h + p0y;
  } else {
    const SinCos sc2 = SinCosF(k * kDegToRad);
    const float s2 = sc2.s, c2 = sc2.c;
    px[1] = p0x + src.w * c2, py[1] = p0y + src.w * s2;
    px[2] = px[1] - s2 * src.h, py[2] = src.h * c2 + py[1];
    px[3] = p0x - s2 * src.h, py[3] = p0y + src.h * c2;
  }
  float mnx = ox, mny = oy, mxx = ox + ow, mxy = oy + oh;
  for (int i = 0; i < 4; i++) {
    if (mnx > px[i]) mnx = px[i];
    if (mny > py[i]) mny = py[i];
    if (px[i] > mxx) mxx = px[i];
    if (py[i] > mxy) mxy = py[i];
  }
  const SinCos sc3 = SinCosF(da * kDegToRad);
  const float s3 = sc3.s, c3 = sc3.c;
  const float ex = mnx - ox, ey = mny - oy;
  dst->x = (ox + c3 * ex) - s3 * ey;
  dst->y = (oy + s3 * ex) + c3 * ey;
  dst->w = mxx - mnx;
  dst->h = mxy - mny;
}

// box_util::FindContainingBoxWithRobustAngle over raw boxes.
RBox FindContainingBoxWithRobustAngle(const std::vector<const RawBox*>& boxes) {
  std::vector<float> angles;
  angles.reserve(boxes.size());
  for (const RawBox* b : boxes) angles.push_back(b->angle);
  const float angle = RobustAngleEstimation(angles);
  RBox r;
  for (const RawBox* b : boxes) {
    if (r.w <= 0.0f || r.h <= 0.0f) r = {b->x, b->y, 1.0f, 1.0f, angle};
    ExpandBoxToContainBox({b->x, b->y, b->w, b->h, b->angle}, &r);
  }
  return r;
}

// ---------------------------------------------------------------------------------------------------------------
// cityscan::KdTree<Vector2<float>, 2> (nanoflann KDTreeSingleIndexAdaptor, leaf_max_size 10, L2_Simple), only the
// unsorted radius search the detector uses. The neighbor order matters (it decides union-find roots).

class KdTree2 {
 public:
  explicit KdTree2(std::vector<std::pair<float, float>> pts) : pts_(std::move(pts)) {
    const size_t n = pts_.size();
    vind_.resize(n);
    for (size_t i = 0; i < n; i++) vind_[i] = i;
    if (n == 0) return;
    float bb[4] = {pts_[0].first, pts_[0].first, pts_[0].second, pts_[0].second};
    for (size_t i = 1; i < n; i++) {
      const float x = pts_[i].first, y = pts_[i].second;
      if (x < bb[0]) bb[0] = x;
      if (x > bb[1]) bb[1] = x;
      if (y < bb[2]) bb[2] = y;
      if (y > bb[3]) bb[3] = y;
    }
    for (int i = 0; i < 4; i++) root_bbox_[i] = bb[i];
    nodes_.reserve(2 * n);
    DivideTree(0, n, bb);
  }

  // Indices with squared distance < r2, in traversal order.
  void RadiusSearch(float qx, float qy, float r2, std::vector<int>* out) const {
    out->clear();
    if (pts_.empty()) return;
    float dists[2] = {0, 0};
    float distsq = 0.0f;
    if (qx < root_bbox_[0]) {
      dists[0] = (qx - root_bbox_[0]) * (qx - root_bbox_[0]);
      distsq = dists[0];
    } else if (qx > root_bbox_[1]) {
      dists[0] = (qx - root_bbox_[1]) * (qx - root_bbox_[1]);
      distsq = dists[0];
    }
    if (qy < root_bbox_[2]) {
      dists[1] = (qy - root_bbox_[2]) * (qy - root_bbox_[2]);
      distsq = distsq + dists[1];
    } else if (qy > root_bbox_[3]) {
      dists[1] = (qy - root_bbox_[3]) * (qy - root_bbox_[3]);
      distsq = distsq + dists[1];
    }
    const float q[2] = {qx, qy};
    SearchLevel(q, r2, 0, distsq, dists, out);
  }

 private:
  struct Node {
    int divfeat = 0;
    float divlow = 0, divhigh = 0;
    size_t left = 0, right = 0;  // leaf range in vind_
    size_t child2 = 0;           // 0 for leaves; child1 is the next node
  };

  float Coord(size_t vi, int d) const { return d == 0 ? pts_[vind_[vi]].first : pts_[vind_[vi]].second; }

  // Returns the node index; bbox (low0, high0, low1, high1) is updated to the subtree's point bounds.
  size_t DivideTree(size_t left, size_t right, float* bbox) {
    const size_t id = nodes_.size();
    nodes_.emplace_back();
    const size_t count = right - left;
    if (count <= 10) {
      nodes_[id].left = left, nodes_[id].right = right;
      float lo0 = Coord(left, 0), hi0 = lo0, lo1 = Coord(left, 1), hi1 = lo1;
      for (size_t k = left + 1; k < right; k++) {
        const float x = Coord(k, 0), y = Coord(k, 1);
        if (lo0 > x) lo0 = x;
        if (hi0 < x) hi0 = x;
        if (lo1 > y) lo1 = y;
        if (hi1 < y) hi1 = y;
      }
      bbox[0] = lo0, bbox[1] = hi0, bbox[2] = lo1, bbox[3] = hi1;
      return id;
    }
    // middleSplit_
    float max_span = bbox[1] - bbox[0];
    if (bbox[3] - bbox[2] > max_span) max_span = bbox[3] - bbox[2];
    const float lim = max_span * 0.99999f;
    auto min_max = [&](int d, float* mn, float* mx) {
      *mn = *mx = Coord(left, d);
      for (size_t k = left + 1; k < right; k++) {
        const float v = Coord(k, d);
        if (v < *mn) *mn = v;
        if (*mx < v) *mx = v;
      }
    };
    int cutfeat = 0;
    float mn0 = 0, mx0 = 0, spread0 = 0, mn1 = 0, mx1 = 0;
    const bool use0 = bbox[1] - bbox[0] >= lim;
    if (use0) min_max(0, &mn0, &mx0), spread0 = mx0 - mn0;
    float mn = mn0, mx = mx0;
    if (bbox[3] - bbox[2] >= lim) {
      min_max(1, &mn1, &mx1);
      if (!(use0 && mx1 - mn1 <= spread0)) cutfeat = 1, mn = mn1, mx = mx1;
    }
    const float lo = bbox[2 * cutfeat], span = bbox[2 * cutfeat + 1] - lo;
    float cutval = lo + span * 0.5f;
    if (cutval < mn) cutval = mn;
    if (cutval > mx) cutval = mx;
    // Three-way partition around cutval.
    size_t lt = 0, i = 0, gt = count;
    size_t* ind = vind_.data() + left;
    while (i < gt) {
      const float v = cutfeat == 0 ? pts_[ind[i]].first : pts_[ind[i]].second;
      if (v < cutval) {
        std::swap(ind[lt], ind[i]);
        lt++, i++;
      } else if (v > cutval) {
        gt--;
        std::swap(ind[i], ind[gt]);
      } else {
        i++;
      }
    }
    size_t idx;
    if (lt > count / 2)
      idx = lt;
    else
      idx = i < count / 2 ? i : count / 2;
    float lb[4] = {bbox[0], bbox[1], bbox[2], bbox[3]};
    lb[2 * cutfeat + 1] = cutval;
    DivideTree(left, left + idx, lb);
    float rb[4] = {bbox[0], bbox[1], bbox[2], bbox[3]};
    rb[2 * cutfeat] = cutval;
    const size_t c2 = DivideTree(left + idx, right, rb);
    nodes_[id].divfeat = cutfeat;
    nodes_[id].divlow = lb[2 * cutfeat + 1];
    nodes_[id].divhigh = rb[2 * cutfeat];
    nodes_[id].child2 = c2;
    for (int d = 0; d < 2; d++) {
      bbox[2 * d] = lb[2 * d] > rb[2 * d] ? rb[2 * d] : lb[2 * d];
      bbox[2 * d + 1] = lb[2 * d + 1] > rb[2 * d + 1] ? lb[2 * d + 1] : rb[2 * d + 1];
    }
    return id;
  }

  void SearchLevel(const float* q, float r2, size_t node, float mindistsq, float* dists, std::vector<int>* out) const {
    const Node& nd = nodes_[node];
    if (nd.child2 == 0) {
      for (size_t i = nd.left; i < nd.right; i++) {
        const float dx = q[0] - pts_[vind_[i]].first, dy = q[1] - pts_[vind_[i]].second;
        if (dx * dx + dy * dy < r2) out->push_back(static_cast<int>(vind_[i]));
      }
      return;
    }
    const int f = nd.divfeat;
    const float val = q[f];
    const float diff1 = val - nd.divlow, diff2 = val - nd.divhigh;
    size_t best, other;
    float cut_dist;
    if (diff1 + diff2 < 0.0f) {
      best = node + 1, other = nd.child2, cut_dist = diff2 * diff2;
    } else {
      best = nd.child2, other = node + 1, cut_dist = diff1 * diff1;
    }
    SearchLevel(q, r2, best, mindistsq, dists, out);
    const float dst = dists[f];
    mindistsq = (mindistsq + cut_dist) - dst;
    dists[f] = cut_dist;
    if (!(1.0f * mindistsq > r2)) SearchLevel(q, r2, other, mindistsq, dists, out);
    dists[f] = dst;
  }

  std::vector<std::pair<float, float>> pts_;
  std::vector<size_t> vind_;
  std::vector<Node> nodes_;
  float root_bbox_[4] = {0, 0, 0, 0};
};

// ---------------------------------------------------------------------------------------------------------------
// Level-1 grouping (GetPairwiseConnections + DisjointSet + CreateClusterList + "ComputeLevel1Boxes").

struct Level1Box {  // google_ocr::Level1GroupedBox (the fields used downstream)
  RBox box;
  std::vector<int> members;  // indices into the concatenated raw boxes
  float avg_height = 0, avg_score = 0;
  int paragraph = -1, paragraph2 = -1;  // best / runner-up overlapping paragraph prediction (+276 / +280)
  // Set by ConstructLines: corners (+172), rounded axis-aligned bounds (+20), corner centroid (+232).
  float pts[8] = {};
  int ax = 0, ay = 0, aw = 0, ah = 0;
  float cx = 0, cy = 0;
};

// adjacency[i]: boxes j (in search order) box i connects to.
std::vector<std::vector<int>> GetPairwiseConnections(const Config::Level1& cfg, const std::vector<RawBox>& raw) {
  const int n = static_cast<int>(raw.size());
  std::vector<std::pair<float, float>> centers(n);
  for (int i = 0; i < n; i++) centers[i] = {raw[i].cx, raw[i].cy};
  KdTree2 tree(centers);
  std::vector<std::vector<int>> adj(n);
  std::set<std::pair<int, int>> seen;
  std::vector<int> nb;
  for (int i = 0; i < n; i++) {
    const RawBox& bi = raw[i];
    int r = static_cast<int>(bi.h);
    if (r <= static_cast<int>(bi.w)) r = static_cast<int>(bi.w);
    const float rr = cfg.search_radius_factor * static_cast<float>(r);
    tree.RadiusSearch(bi.cx, bi.cy, rr * rr, &nb);
    for (int j : nb) {
      if (!seen.insert({std::min(i, j), std::max(i, j)}).second) continue;
      const RawBox& bj = raw[j];
      const float hmin_f = bj.h >= bi.h ? bi.h : bj.h;
      const int hmin = static_cast<int>(hmin_f);
      if (cfg.max_height_ratio > 0.0f) {
        const float hmax = bi.h < bj.h ? bj.h : bi.h;
        if (static_cast<float>(static_cast<int>(hmax)) / static_cast<float>(hmin) > cfg.max_height_ratio) continue;
      }
      float d = fmodf(fabsf(bi.angle - bj.angle), 360.0f);
      if (d > 180.0f) d = 360.0f - d;
      if (!(d <= cfg.maximum_angle_difference)) continue;
      float centers_dist = 1.0f;
      if (cfg.max_centers_relative_distance > 0.0f) {
        const float mean = MeanAngle(bi.angle, bj.angle);
        const SinCos scm = SinCosF(mean * kDegToRad);
        const float sm = scm.s, cm = scm.c;
        const float px = sm * (bi.cx - bj.cx), py = cm * (bi.cy - bj.cy);
        centers_dist = fabsf(py - px) / static_cast<float>(hmin);
        if (centers_dist > cfg.max_centers_relative_distance) continue;
      }
      const float ov = BoxOverlapArea({bi.x, bi.y, bi.w, bi.h, bi.angle}, {bj.x, bj.y, bj.w, bj.h, bj.angle});
      if (ov == 0.0f) continue;
      const float oi = ov / (bi.w * bi.h), oj = ov / (bj.w * bj.h);
      const float thr = centers_dist < cfg.optimal_centers_relative_distance
                            ? cfg.min_overlap_with_optimal_centers_aligment
                            : cfg.min_overlap;
      if (oi > thr || oj > thr) adj[i].push_back(j);
    }
  }
  return adj;
}

class DisjointSet {
 public:
  explicit DisjointSet(int n) : parent_(n), rank_(n, 0) {
    for (int i = 0; i < n; i++) parent_[i] = i;
  }
  int GetRoot(int i) {
    if (parent_[i] != i) parent_[i] = GetRoot(parent_[i]);
    return parent_[i];
  }
  void Union(int a, int b) {
    const int ra = GetRoot(a), rb = GetRoot(b);
    if (ra == rb) return;
    if (rank_[ra] < rank_[rb]) {
      parent_[ra] = rb;
    } else {
      parent_[rb] = ra;
      if (rank_[ra] <= rank_[rb]) rank_[ra]++;
    }
  }
  // Clusters ordered by root index; members ascending.
  std::vector<std::vector<int>> Clusters() {
    const int n = static_cast<int>(parent_.size());
    std::map<int, int> id;
    for (int i = 0; i < n; i++)
      if (GetRoot(i) == i) id.emplace(i, static_cast<int>(id.size()));
    std::vector<std::vector<int>> out(id.size());
    for (int i = 0; i < n; i++) out[id[GetRoot(i)]].push_back(i);
    return out;
  }

 private:
  std::vector<int> parent_, rank_;
};

std::vector<Level1Box> ComputeLevel1Boxes(const Config::Level1& cfg, const std::vector<RawBox>& raw) {
  const auto adj = GetPairwiseConnections(cfg, raw);
  DisjointSet ds(static_cast<int>(raw.size()));
  for (int i = 0; i < static_cast<int>(adj.size()); i++)
    for (int j : adj[i]) ds.Union(i, j);
  std::vector<Level1Box> out;
  for (const auto& cl : ds.Clusters()) {
    std::vector<const RawBox*> boxes;
    for (int i : cl) boxes.push_back(&raw[i]);
    Level1Box l;
    l.box = FindContainingBoxWithRobustAngle(boxes);
    l.members = cl;
    float sh = 0.0f, ss = 0.0f;
    for (int i : cl) ss = ss + raw[i].score, sh = sh + raw[i].h;
    l.avg_height = sh / static_cast<float>(static_cast<int>(cl.size()));
    l.avg_score = ss / static_cast<float>(static_cast<int>(cl.size()));
    out.push_back(std::move(l));
  }
  return out;
}

std::string SerializeLevel1(const std::vector<Level1Box>& l1) {
  // Compact form: per box [5 floats box][avg_height][avg_score][u32 n][n ints].
  std::string s;
  AppendPod(&s, static_cast<uint32_t>(l1.size()));
  for (const auto& l : l1) {
    AppendPod(&s, l.box);
    AppendPod(&s, l.avg_height);
    AppendPod(&s, l.avg_score);
    AppendPod(&s, static_cast<uint32_t>(l.members.size()));
    for (int m : l.members) AppendPod(&s, m);
  }
  return s;
}

// ---------------------------------------------------------------------------------------------------------------
// Paragraph predictions (vertical heads): NonMaximumSuppression, then each level-1 box is tagged with the
// predictions covering most of it (box_util::OverlapComputer between level-1 boxes and paragraphs).

struct ParagraphPrediction {
  RBox box;
  float score;
};

std::vector<ParagraphPrediction> ParagraphNms(const std::vector<std::vector<RawBox>>& lists) {
  std::vector<ParagraphPrediction> p;
  for (const auto& l : lists)
    for (const RawBox& b : l) p.push_back({{b.x, b.y, b.w, b.h, b.angle}, b.score});
  SortCounted("paragraph_nms", p.begin(), p.end(),
              [](const ParagraphPrediction& a, const ParagraphPrediction& b) { return a.score > b.score; });
  std::vector<bool> suppressed(p.size(), false);
  std::vector<ParagraphPrediction> kept;
  for (size_t i = 0; i < p.size(); i++) {
    if (suppressed[i]) continue;
    kept.push_back(p[i]);
    for (size_t j = 0; j < p.size(); j++) {
      const float ov = BoxOverlapArea(p[i].box, p[j].box);
      if (ov == 0.0f) continue;
      if (ov / (p[i].box.w * p[i].box.h) > 0.75f || ov / (p[j].box.w * p[j].box.h) > 0.75f) suppressed[j] = true;
    }
  }
  return kept;
}

void AssignParagraphs(const std::vector<ParagraphPrediction>& paras, std::vector<Level1Box>* l1) {
  // OverlapComputer<RotatedBox<float> const> (0x100812194) over [level-1 boxes, paragraphs] split after the level-1
  // boxes (replayed by trace/overlap_check.py det_sets.bin): entries (i, n1 + j) in map order, over_first =
  // I / area(level-1 box).
  const int n1 = static_cast<int>(l1->size());
  std::vector<RBox> boxes;
  for (const Level1Box& l : *l1) boxes.push_back(l.box);
  for (const auto& p : paras) boxes.push_back(p.box);
  const overlap::OverlapMap ov = overlap::ComputeOverlaps(boxes, n1);
  for (int i = 0; i < n1; i++) {
    Level1Box& l = (*l1)[i];
    int best = -1, second = -1;
    float vbest = 0.4f, vsecond = 0.4f;
    for (auto it = ov.lower_bound({i, n1}); it != ov.end() && it->first.first == i; ++it) {
      const float r = it->second.over_first;
      const int j = it->first.second - n1;
      if (r > vbest) {
        if (best > -1) vsecond = vbest, second = best;
        vbest = r, best = j;
      } else if (r > vsecond) {
        vsecond = r, second = j;
      }
    }
    if (best < 0) continue;
    l.paragraph = best;
    if (second < 0) continue;
    if (vbest - vsecond < 0.25f && paras[best].score <= paras[second].score) std::swap(best, second);
    l.paragraph = best;
    l.paragraph2 = second;
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Lines (ConstructLines): level-1 boxes are chained left-to-right by the cheapest admissible neighbour pairs,
// each chain becomes a MergedBoxesStruct.

struct MergedLine {  // google_ocr::MergedBoxesStruct (the fields used downstream)
  RBox box;
  std::vector<int> l1;  // level-1 boxes (indices), cluster order
  int count = 0;        // raw boxes (+176)
  float avg_height = 0, avg_score = 0, min_cos = 1.0f;  // +180, +208, returned alongside
  float left_padding = 0, right_padding = 0;            // +188, +192
  int paragraph = -1;                                   // +216
  // Curved lines (CreateCurveRepresentation): +200 flag, thickness (+144), top / bottom polylines (+24 / +64).
  bool curved = false;
  float thickness = 0;
  std::vector<std::pair<float, float>> top, bottom;
};

MergedLine ConstructMergedBoxesStruct(const std::vector<Level1Box>& l1, const std::vector<RawBox>& raw,
                                      const std::vector<int>& members) {
  MergedLine m;
  m.l1 = members;
  std::vector<const RawBox*> boxes;
  float sh = 0.0f, ss = 0.0f;
  for (int i : members) {
    for (int r : l1[i].members) boxes.push_back(&raw[r]);
    const int n = static_cast<int>(l1[i].members.size());
    ss = ss + l1[i].avg_score * static_cast<float>(n);
    m.count += n;
    sh = sh + l1[i].avg_height * static_cast<float>(n);
  }
  m.avg_score = ss / static_cast<float>(m.count);
  m.avg_height = sh / static_cast<float>(m.count);
  std::vector<float> angles;
  for (const RawBox* b : boxes) angles.push_back(b->angle);
  const float angle = RobustAngleEstimation(angles);
  const SinCos sr = SinCosF(angle * kDegToRad);
  for (float a : angles) {
    const SinCos sa = SinCosF(a * kDegToRad);
    const float d = sr.s * sa.s + (sr.c * sa.c + 0.0f);
    if (d < m.min_cos) m.min_cos = d;
  }
  m.box = {0, 0, 0, 0, angle};
  for (const RawBox* b : boxes) {
    if (m.box.w <= 0.0f || m.box.h <= 0.0f) m.box = {b->x, b->y, 1.0f, 1.0f, angle};
    ExpandBoxToContainBox({b->x, b->y, b->w, b->h, b->angle}, &m.box);
  }
  return m;
}

int RoundToInt(float v) {  // PointsToAxisAlignedBox<..., AlignedBox<int>> with rounding
  const float r = roundf(v);
  if (r >= 2147500000.0f) return 0x7FFFFFFF;
  if (truncf(r) < -2147500000.0f) return static_cast<int>(0x80000000u);
  return static_cast<int>(r);
}

float ComputeNormalizedCentersDistance(float ax, float ay, float bx, float by, float angle, float norm) {
  const SinCos sc = SinCosF(static_cast<float>(static_cast<double>(angle) * 3.141592653589793 / 180.0));
  const float mx = (ax + bx) * -0.5f, my = (ay + by) * -0.5f;
  return (fabsf(sc.c * (ay + my) - sc.s * (ax + mx)) + fabsf(sc.c * (by + my) - sc.s * (bx + mx))) / norm;
}

// CreateCurveRepresentation: the line's raw boxes give their left/right edge midpoints; sorted along the line
// direction and smoothed with a 6-point moving average (keeping a point only once it moved more than half the
// thickness), they form the mid line, offset by half the thickness on both sides into top / bottom polylines.
void CreateCurveRepresentation(const std::vector<const RawBox*>& raws, float dir_x, float dir_y, float ox, float oy,
                               MergedLine* m) {
  constexpr int kWindow = 6;
  m->curved = true;
  std::vector<std::pair<float, float>> pts;
  float t = 0.0f;
  for (const RawBox* r : raws) {
    pts.push_back({(r->pts[0] + r->pts[6]) * 0.5f, (r->pts[1] + r->pts[7]) * 0.5f});
    pts.push_back({(r->pts[2] + r->pts[4]) * 0.5f, (r->pts[3] + r->pts[5]) * 0.5f});
    if (t < r->h) t = r->h;  // max height (use_max_height)
  }
  m->thickness = t;
  auto key = [&](const std::pair<float, float>& p) { return (dir_x * (p.first - ox) + 0.0f) + dir_y * (p.second - oy); };
  SortCounted("curve_points", pts.begin(), pts.end(),
              [&](const std::pair<float, float>& a, const std::pair<float, float>& b) { return key(a) < key(b); });
  std::vector<std::pair<float, float>> mid;
  const size_t n = pts.size();
  if (n == 0) return;
  if (n > 1) {
    float buf[kWindow][2] = {};
    int next = 0, count = 0;
    float sx = 0.0f, sy = 0.0f;
    auto add = [&](const std::pair<float, float>& p) {
      sx = (sx - buf[next][0]) + p.first;
      sy = (sy - buf[next][1]) + p.second;
      buf[next][0] = p.first, buf[next][1] = p.second;
      count = count + 1 <= kWindow ? count + 1 : kWindow;
      next = (next + 1) % kWindow;
      const float c = static_cast<float>(count);
      return std::pair<float, float>{sx / c, sy / c};
    };
    mid.push_back(add(pts[0]));
    for (size_t i = 1; i + 1 < n; i++) {
      const auto a = add(pts[i]);
      const float dx = a.first - mid.back().first, dy = a.second - mid.back().second;
      if (dx * dx + dy * dy > t * (t * 0.25f)) mid.push_back(a);
    }
  }
  mid.push_back(pts[n - 1]);
  const float half = t * 0.5f;
  for (size_t i = 1; i < mid.size(); i++) {
    const auto& p = mid[i - 1];
    const auto& q = mid[i];
    const SinCos sc = SinCosF(atan2f(q.first - p.first, p.second - q.second));
    const float ox2 = sc.c * half, oy2 = sc.s * half;
    m->top.push_back({p.first - ox2, p.second - oy2});
    m->bottom.push_back({ox2 + p.first, oy2 + p.second});
    if (i + 1 == mid.size()) {
      m->top.push_back({q.first - ox2, q.second - oy2});
      m->bottom.push_back({ox2 + q.first, oy2 + q.second});
    }
  }
}

struct PairAndCost {
  int left, right;
  float gap, centers, height_ratio, cost;
};

// geometry2d::RTreeBase<R2BoundingBox, MergedBox const*> (Guttman R-tree, at most 10 entries per node, linear
// split), ported for its query result order: ConstructLines evaluates neighbour pairs in Find() order.
class RTree {
 public:
  struct Box {
    double x0, y0, x1, y1;
  };

  void Insert(const Box& b, int item) {
    if (!root_) {
      root_ = NewNode(0);
      root_->entries.push_back({b, nullptr, item});
      return;
    }
    Node* node = root_;
    while (node->height != 0) {
      Node* best = nullptr;
      double best_growth = 0.0, best_area = 0.0;
      for (const Entry& e : node->entries) {
        const double w = e.box.x1 - e.box.x0, h = e.box.y1 - e.box.y0;
        Box u = e.box;
        if (w == 0.0 && h == 0.0) {
          u = b;
        } else if (b.x1 - b.x0 != 0.0 || b.y1 - b.y0 != 0.0) {
          if (b.x0 < u.x0) u.x0 = b.x0;
          if (b.y0 < u.y0) u.y0 = b.y0;
          if (u.x1 < b.x1) u.x1 = b.x1;
          if (u.y1 < b.y1) u.y1 = b.y1;
        }
        const double area = (u.y1 - u.y0) * (u.x1 - u.x0);
        const double growth = area - w * h;
        if (!best || growth < best_growth || (growth == best_growth && area < best_area))
          best = e.child, best_growth = growth, best_area = area;
      }
      node = best;
    }
    node->entries.push_back({b, nullptr, item});
    AdjustTree(node, node->entries.size() >= kSplitSize ? Split(node) : nullptr);
  }

  // Items whose boxes intersect q (closed), depth-first in entry order.
  void Find(const Box& q, std::vector<int>* out) const {
    if (root_) Find(root_, q, out);
  }

 private:
  static constexpr size_t kSplitSize = 11;
  struct Node;
  struct Entry {
    Box box;
    Node* child;
    int item;
  };
  struct Node {
    int height;  // 0 = leaf, -1 = new root not yet filled
    std::vector<Entry> entries;
    Node* parent = nullptr;
  };

  Node* NewNode(int height) {
    nodes_.push_back(std::make_unique<Node>());
    nodes_.back()->height = height;
    return nodes_.back().get();
  }

  static bool Empty(const Box& b) { return b.x1 - b.x0 == 0.0 && b.y1 - b.y0 == 0.0; }

  static Box Bounds(const Node* n) {
    Box u{0, 0, 0, 0};
    for (const Entry& e : n->entries) {
      if (Empty(u)) {
        u = e.box;
      } else if (!Empty(e.box)) {
        if (u.x0 > e.box.x0) u.x0 = e.box.x0;
        if (u.y0 > e.box.y0) u.y0 = e.box.y0;
        if (u.x1 < e.box.x1) u.x1 = e.box.x1;
        if (!(u.y1 >= e.box.y1)) u.y1 = e.box.y1;
      }
    }
    return u;
  }

  static Box Union(const Box& a, const Box& e) {  // as in Node::Split
    if (Empty(a)) return e;
    if (Empty(e)) return a;
    return {e.x0 >= a.x0 ? a.x0 : e.x0, e.y0 >= a.y0 ? a.y0 : e.y0, a.x1 >= e.x1 ? a.x1 : e.x1,
            a.y1 >= e.y1 ? a.y1 : e.y1};
  }

  static void PickSplitSeeds(const Node* n, int* s1, int* s2) {
    const auto& v = n->entries;
    double max_x0 = v[0].box.x0, max_y0 = v[0].box.y0, min_x1 = v[0].box.x1, min_y1 = v[0].box.y1;
    int i_max_x0 = 0, i_max_y0 = 0, i_min_x1 = 0, i_min_y1 = 0;
    for (int k = 1; k < static_cast<int>(v.size()); k++) {
      const Box& b = v[k].box;
      if (b.x0 <= max_x0) {
        if (b.x1 < min_x1) min_x1 = b.x1, i_min_x1 = k;
      } else {
        max_x0 = b.x0, i_max_x0 = k;
      }
      if (b.y0 > max_y0) {
        max_y0 = b.y0, i_max_y0 = k;
      } else if (b.y1 < min_y1) {
        min_y1 = b.y1, i_min_y1 = k;
      }
    }
    const Box all = Bounds(n);
    double sx = fabs(max_x0 - min_x1) / (all.x1 - all.x0);
    const double sy = fabs(max_y0 - min_y1) / (all.y1 - all.y0);
    if (i_max_x0 == i_min_x1) sx = 0.0;
    const double ty = i_max_y0 != i_min_y1 ? sy : 0.0;
    if (sx <= ty)
      *s1 = i_max_y0, *s2 = i_min_y1;
    else
      *s1 = i_max_x0, *s2 = i_min_x1;
    if (*s1 == *s2) *s2 = static_cast<int>(v.size()) - 1;
  }

  // Moves part of `n`'s entries into a new sibling, which is returned.
  Node* Split(Node* n) {
    int s1, s2;
    PickSplitSeeds(n, &s1, &s2);
    std::vector<Entry> a{n->entries[s1]}, b{n->entries[s2]};
    Box ba = a[0].box, bb = b[0].box;
    for (int k = 0; k < static_cast<int>(n->entries.size()); k++) {
      if (k == s1 || k == s2) continue;
      const Entry& e = n->entries[k];
      if (a.size() == 8) {
        b.push_back(e);
        continue;
      }
      if (b.size() == 8) {
        a.push_back(e);
        continue;
      }
      const Box ua = Union(ba, e.box), ub = Union(bb, e.box);
      const double area_a = (ua.y1 - ua.y0) * (ua.x1 - ua.x0);
      const double grow_a = area_a - (ba.y1 - ba.y0) * (ba.x1 - ba.x0);
      const double area_b = (ub.y1 - ub.y0) * (ub.x1 - ub.x0);
      const double grow_b = area_b - (bb.y1 - bb.y0) * (bb.x1 - bb.x0);
      if (grow_a > grow_b || (grow_a == grow_b && area_a >= area_b)) {
        b.push_back(e), bb = ub;
      } else {
        a.push_back(e), ba = ua;
      }
    }
    Node* sib = NewNode(n->height);
    n->entries = std::move(a);
    sib->entries = std::move(b);
    if (sib->height != 0)
      for (Entry& e : sib->entries) e.child->parent = sib;
    return sib;
  }

  static void AddNode(Node* n, Node* child) {
    if (n->entries.empty() && n->height == -1) n->height = child->height + 1;
    n->entries.push_back({Bounds(child), child, -1});
    child->parent = n;
  }

  void AdjustTree(Node* n, Node* split) {
    while (n != root_) {
      Node* parent = n->parent;
      for (Entry& e : parent->entries)
        if (e.child == n) {
          e.box = Bounds(n);
          break;
        }
      if (split) {
        AddNode(parent, split);
        split = parent->entries.size() >= kSplitSize ? Split(parent) : nullptr;
      }
      n = parent;
    }
    if (split) {
      Node* r = NewNode(-1);
      AddNode(r, root_);
      AddNode(r, split);
      root_ = r;
    }
  }

  static void Find(const Node* n, const Box& q, std::vector<int>* out) {
    for (const Entry& e : n->entries) {
      if (e.box.x0 <= q.x1 && !(e.box.x1 < q.x0 || e.box.y0 > q.y1) && e.box.y1 >= q.y0) {
        if (n->height != 0)
          Find(e.child, q, out);
        else
          out->push_back(e.item);
      }
    }
  }

  Node* root_ = nullptr;
  std::vector<std::unique_ptr<Node>> nodes_;
};

std::vector<MergedLine> ConstructLines(const Config::Merge& cfg, const std::vector<RawBox>& raw,
                                       std::vector<Level1Box>* l1p) {
  std::vector<Level1Box>& l1 = *l1p;
  const int n = static_cast<int>(l1.size());
  for (Level1Box& l : l1) {
    const RBox& b = l.box;
    float* p = l.pts;
    p[0] = b.x, p[1] = b.y;
    if (b.angle == 0.0f) {
      p[2] = b.x + b.w, p[3] = b.y;
      p[4] = p[2], p[5] = b.y + b.h;
      p[6] = b.x, p[7] = p[5];
    } else {
      const SinCos sc = SinCosF(b.angle * kDegToRad);
      p[2] = b.x + sc.c * b.w, p[3] = b.y + sc.s * b.w;
      p[4] = p[2] - sc.s * b.h, p[5] = p[3] + sc.c * b.h;
      p[6] = b.x - sc.s * b.h, p[7] = b.y + sc.c * b.h;
    }
    float mnx = p[0], mny = p[1], mxx = p[0], mxy = p[1];
    for (int k = 1; k < 4; k++) {
      if (mnx > p[2 * k]) mnx = p[2 * k];
      if (mny > p[2 * k + 1]) mny = p[2 * k + 1];
      if (p[2 * k] > mxx) mxx = p[2 * k];
      if (p[2 * k + 1] > mxy) mxy = p[2 * k + 1];
    }
    l.ax = RoundToInt(mnx), l.ay = RoundToInt(mny);
    l.aw = RoundToInt(mnx + (mxx - mnx)) - l.ax, l.ah = RoundToInt(mny + (mxy - mny)) - l.ay;
    l.cx = ((((0.0f + p[0]) + p[2]) + p[4]) + p[6]) * 0.25f;
    l.cy = ((((0.0f + p[1]) + p[3]) + p[5]) + p[7]) * 0.25f;
  }

  // Candidate neighbours: R-tree query with the box widened along its reading direction.
  struct Q {
    double x0, y0, x1, y1;
  };
  std::vector<Q> query(n);
  for (int i = 0; i < n; i++) {
    const Level1Box& l = l1[i];
    float a = l.box.angle;
    if (a <= -180.0f)
      do a = a + 360.0f;
      while (a <= -180.0f);
    while (a > 180.0f) a = a + -360.0f;
    if (a < 0.0f) a = a + 360.0f;
    const int64_t quadrant = static_cast<int64_t>(rintf(a / 90.0f));
    Q q{static_cast<double>(l.ax), static_cast<double>(l.ay), static_cast<double>(l.aw + l.ax),
        static_cast<double>(l.ah + l.ay)};
    const float e = l.box.h * cfg.v[0];  // neighbor_search_expansion
    if (quadrant & 1)
      q.y0 = q.y0 - e, q.y1 = e + q.y1;
    else
      q.x0 = q.x0 - e, q.x1 = e + q.x1;
    query[i] = q;
  }
  RTree tree;
  for (int i = 0; i < n; i++) {
    const Level1Box& l = l1[i];
    tree.Insert({static_cast<double>(l.ax), static_cast<double>(l.ay), static_cast<double>(l.aw + l.ax),
                 static_cast<double>(l.ah + l.ay)},
                i);
  }

  // Pairs in the original's discovery order (it decides the order of equal-cost links in the sort below).
  std::vector<PairAndCost> pairs;
  std::set<std::pair<int, int>> seen;
  std::vector<int> found;
  for (int i = 0; i < n; i++) {
    found.clear();
    tree.Find({query[i].x0, query[i].y0, query[i].x1, query[i].y1}, &found);
    if (DumpDir() && !std::is_sorted(found.begin(), found.end())) CountEvent("rtree_query_not_index_order");
    for (int j : found) {
      if (j == i) continue;
      const int lo = std::min(i, j), hi = std::max(i, j);
      if (!seen.insert({lo, hi}).second) continue;
      const Level1Box& a = l1[lo];
      const Level1Box& b = l1[hi];
      const float hmax = a.avg_height >= b.avg_height ? a.avg_height : b.avg_height;
      const float hmin = b.avg_height < a.avg_height ? b.avg_height : a.avg_height;
      float d = fmodf(fabsf(a.box.angle - b.box.angle), 360.0f);
      if (d > 180.0f) d = 360.0f - d;
      if (!(d <= 45.0f)) continue;  // (the original then rejects it in every branch)
      const float mean = MeanAngle(a.box.angle, b.box.angle);
      const float centers = ComputeNormalizedCentersDistance(a.cx, a.cy, b.cx, b.cy, mean, hmin);
      const SinCos sc = SinCosF(static_cast<float>(static_cast<double>(mean) * 3.141592653589793 / -180.0));
      // Corners in the frame rotated by -mean: (y', x') per corner; min/max over each box.
      auto extent = [&](const Level1Box& l, float mn[2], float mx[2]) {
        mn[0] = mn[1] = 3.4028235e38f, mx[0] = mx[1] = -3.4028235e38f;
        for (int k = 0; k < 4; k++) {
          const float px = l.pts[2 * k], py = l.pts[2 * k + 1];
          const float r[2] = {sc.s * px + sc.c * py, sc.c * px - sc.s * py};
          for (int c = 0; c < 2; c++) {
            if (mn[c] > r[c]) mn[c] = r[c];
            if (r[c] > mx[c]) mx[c] = r[c];
          }
        }
      };
      float amn[2], amx[2], bmn[2], bmx[2];
      extent(b, bmn, bmx);
      extent(a, amn, amx);
      const bool b_right = bmn[1] > amn[1];
      float lo2[2], hi2[2];
      for (int c = 0; c < 2; c++) {
        lo2[c] = bmn[c] > amn[c] ? bmn[c] : amn[c];
        hi2[c] = amx[c] > bmx[c] ? bmx[c] : amx[c];
      }
      const float xov = lo2[1] < hi2[1] ? hi2[1] - lo2[1] : 0.0f;
      const float yov_abs = hi2[0] > lo2[0] ? hi2[0] - lo2[0] : 0.0f;
      const float gap_abs = hi2[1] > lo2[1] ? 0.0f : lo2[1] - hi2[1];
      const float hmin_box = b.box.h < a.box.h ? b.box.h : a.box.h;
      const float yov = yov_abs / hmin_box;
      const float gap = gap_abs / hmin;
      const float wmin = b.box.w < a.box.w ? b.box.w : a.box.w;
      const float xovr = xov / wmin;
      bool reject = yov == 0.0f;
      const int pa = a.paragraph, pb = b.paragraph, pa2 = a.paragraph2, pb2 = b.paragraph2;
      if (pa == -1) {
        if (gap > 0.5f && pb != -1) reject = true;
      } else if (pb == -1) {
        if (gap > 0.5f) reject = true;
      } else {
        const bool share = pa == pb || (pb2 != -1 && pa == pb2) || (pa2 != -1 && (pa2 == pb || pa2 == pb2));
        if (!share) reject = true;
      }
      if (reject) continue;
      if (xovr > cfg.v[6] && yov < cfg.v[7]) continue;  // overlapping_not_aligned_pair_*
      if (xovr > cfg.v[4] && yov > cfg.v[5]) continue;  // duplicate_pair_*
      if (yov < cfg.v[2] && centers > cfg.v[3]) continue;  // not_aligned_pair_*
      const float ratio = hmin / hmax;
      const float cost = static_cast<float>(static_cast<double>(1.0f - ratio) * 0.5 +
                                            static_cast<double>(gap + centers));
      pairs.push_back({b_right ? lo : hi, b_right ? hi : lo, gap, centers, ratio, cost});
    }
  }
  std::string pairs_dump;
  if (DumpDir()) {
    AppendPod(&pairs_dump, static_cast<uint32_t>(pairs.size()));
    pairs_dump.append(reinterpret_cast<const char*>(pairs.data()), pairs.size() * sizeof(PairAndCost));
  }
  SortCounted("line_links", pairs.begin(), pairs.end(),
              [](const PairAndCost& a, const PairAndCost& b) { return a.cost < b.cost; });
  if (DumpDir()) {
    pairs_dump.append(reinterpret_cast<const char*>(pairs.data()), pairs.size() * sizeof(PairAndCost));
    DumpFile("pairs_0.bin", pairs_dump);
  }
  DisjointSet ds(n);
  std::set<int> has_right, has_left;
  for (const PairAndCost& p : pairs) {
    if (has_right.count(p.left) || has_left.count(p.right)) continue;
    has_right.insert(p.left);
    has_left.insert(p.right);
    if (p.cost < -cfg.v[1]) ds.Union(p.left, p.right);  // minimum_box_dissimilarity
  }

  std::vector<MergedLine> lines;
  for (const auto& cl : ds.Clusters()) {
    MergedLine m = ConstructMergedBoxesStruct(l1, raw, cl);
    if (m.min_cos >= 0.99f && m.box.h <= m.avg_height * cfg.v[9]) {  // split_merged_box_height_ratio
      CountEvent("line_low");
      lines.push_back(std::move(m));
      continue;
    }
    // Order the level-1 boxes along the line by the projection of their left-edge midpoints.
    const SinCos sc = SinCosF(m.box.angle * kDegToRad);
    std::vector<std::pair<float, int>> order;
    for (int i : cl) {
      const Level1Box& l = l1[i];
      const float px = ((l.pts[0] + l.pts[6]) * 0.5f - m.box.x) * sc.c;
      const float py = ((l.pts[1] + l.pts[7]) * 0.5f - m.box.y) * sc.s;
      order.push_back({(px + 0.0f) + py, i});
    }
    SortCounted("line_order", order.begin(), order.end(),
                [](const std::pair<float, int>& a, const std::pair<float, int>& b) { return a.first < b.first; });
    if (m.min_cos < 0.99f) {
      std::vector<const RawBox*> raws;
      for (const auto& o : order)
        for (int r : l1[o.second].members) raws.push_back(&raw[r]);
      CreateCurveRepresentation(raws, sc.c, sc.s, m.box.x, m.box.y, &m);
      CountEvent("line_curved");
      lines.push_back(std::move(m));
      continue;
    }
    const int first = order[0].second;
    if (l1[first].avg_height <= m.avg_height + m.avg_height) {
      CountEvent("line_straight");
      lines.push_back(std::move(m));
      continue;
    }
    // A leading box much taller than the line becomes its own line.
    CountEvent("line_leading_split");
    lines.push_back(ConstructMergedBoxesStruct(l1, raw, {first}));
    std::vector<int> rest;
    for (size_t k = 1; k < order.size(); k++) rest.push_back(order[k].second);
    lines.push_back(ConstructMergedBoxesStruct(l1, raw, rest));
  }
  return lines;
}

// GroupRpnDetectorV2 CropDetections, which crops every line image for the recognizer and in doing so changes the
// line's box. The line is cropped from the most downscaled pyramid image that keeps it at least kCropLineHeight px
// tall, or from the page itself when that image's scale is >= 0.999 (the scale is reported as
// detection_image_scale, per mille). For a downscaled image the box is scaled into it with integer truncation
// (ocr::photo::box_util::ScaleBoundingBox) and back afterwards, which can move its edges by a pixel; an unrotated,
// uncurved box is also clipped to the image it is cropped from (ImageCropRotateAndScaleGray).
// (The original first inserts half-scale images between pyramid levels more than 2x apart, starting from the
// reference level; with the predefined shapes the reference and first levels are at most 2048 / 1280 apart, so that
// never happens.)
constexpr int kCropLineHeight = 64;  // GroupRpnTextDetectionMutatorRuntimeOptions float at +112, as int

struct PyramidImages {
  std::vector<float> scales;
  std::vector<bool> present;            // the level's image exists (a reused level leaves an empty slot)
  std::vector<std::pair<int, int>> dims;  // width, height
};

void ScaleDetectionBox(float s, RotatedBox* b, std::optional<Curve>* curve) {
  const int right = static_cast<int>(s * static_cast<float>(b->left + b->width));
  const int bottom = static_cast<int>(s * static_cast<float>(b->top + b->height));
  b->left = static_cast<int>(s * static_cast<float>(b->left));
  b->top = static_cast<int>(s * static_cast<float>(b->top));
  b->width = right - b->left;
  b->height = bottom - b->top;
  if (!*curve) return;
  const double sd = static_cast<double>(s);
  (*curve)->thickness = (*curve)->thickness * sd;
  for (auto& p : (*curve)->points) p.first = p.first * sd, p.second = p.second * sd;
}

// Leptonica boxCreate + boxClipToRectangle as the crop uses them; false when nothing of the box is inside.
bool ClipToImage(int w, int h, RotatedBox* b) {
  int x = b->left, y = b->top, bw = b->width, bh = b->height;
  if (bw < 0 || bh < 0) return false;
  if (x < 0) {
    bw += x, x = 0;
    if (bw < 1) return false;
  }
  if (y < 0) {
    bh += y, y = 0;
    if (bh < 1) return false;
  }
  if (x >= w || y >= h || bw + x < 1 || bh + y < 1) return false;
  if (bw + x > w) bw = w - x;
  if (bh + y > h) bh = h - y;
  if (bw != b->width || bh != b->height) b->left = x, b->top = y, b->width = bw, b->height = bh;
  return true;
}

// Returns detection_image_scale; adjusts the box (and curve) as the crop does.
int CropDetection(const PyramidImages& p, int page_w, int page_h, RotatedBox* box, std::optional<Curve>* curve) {
  const double height = *curve ? (*curve)->thickness : static_cast<double>(box->height);
  const float min_scale =
      static_cast<float>(kCropLineHeight) / static_cast<float>(static_cast<int>(height)) + -0.001f;
  int chosen = -1;
  for (size_t k = 0; k < p.scales.size(); k++)
    if (p.present[k] && p.scales[k] >= min_scale) chosen = static_cast<int>(k);
  const bool scaled = chosen >= 0 && p.scales[chosen] < 0.999f;
  const float s = scaled ? p.scales[chosen] : 1.0f;
  if (scaled) ScaleDetectionBox(s, box, curve);
  // A failed crop (box entirely outside the image) leaves detection_image_scale unset in the original; unreachable
  // for boxes the detector found inside the page.
  if (!*curve && box->angle == 0.0f)
    ClipToImage(scaled ? p.dims[chosen].first : page_w, scaled ? p.dims[chosen].second : page_h, box);
  if (scaled) ScaleDetectionBox(1.0f / s, box, curve);
  return static_cast<int>(s * 1000.0f);
}

// Post-processing (PostProcessConfig): pad the line box and round it to whole pixels.
void PadLine(const Config& c, MergedLine* m) {
  RBox& b = m->box;
  const float a = NormalizeAngle(b.angle);
  const SinCos sn = SinCosF(a * kDegToRad);
  const float h = b.h, w = b.w;
  const float cx = (b.x - (h * 0.5f) * sn.s) + sn.c * (w * 0.5f);
  const float cy = (b.y + (h * 0.5f) * sn.c) + sn.s * (w * 0.5f);
  float pad = c.box_width_padding * h;
  if (pad > 16.0f) pad = 16.0f;
  if (pad < 4.0f) pad = 4.0f;
  const SinCos so = SinCosF(b.angle * kDegToRad);
  float top = c.box_top_height_padding * h;
  if (top > 8.0f) top = 8.0f;
  const float ht = h * 0.5f + top;
  const float left = c.box_width_padding_left_portion * pad;
  const float wl = w * 0.5f + left;
  const float x = (cx + so.s * ht) - so.c * wl;
  const float y = (cy - so.c * ht) - so.s * wl;
  float bottom = c.box_bottom_height_padding * h;
  if (bottom > 8.0f) bottom = 8.0f;
  if (bottom < 1.0f) bottom = 1.0f;
  const float hh = static_cast<float>((static_cast<double>(h) * 0.5 + static_cast<double>(ht)) +
                                      static_cast<double>(bottom));
  b.w = static_cast<float>(static_cast<int64_t>(rintf(w + pad)));
  b.x = static_cast<float>(static_cast<int64_t>(rintf(x)));
  b.y = static_cast<float>(static_cast<int64_t>(rintf(y)));
  b.h = static_cast<float>(static_cast<int64_t>(rintf(hh)));
  m->left_padding = left;
  m->right_padding = pad - left;
  if (!m->curved) return;
  // Curved: thicken by the vertical paddings and extend both polyline ends along their end segments.
  const float t = m->thickness;
  float ct = c.box_top_height_padding * t;
  if (ct > 8.0f) ct = 8.0f;
  float cb = c.box_bottom_height_padding * t;
  if (cb > 8.0f) cb = 8.0f;
  if (cb < 1.0f) cb = 1.0f;
  m->thickness = t + (ct + cb);
  const size_t n = m->top.size();
  if (n < 2) return;
  float cw = c.box_width_padding * t;
  if (cw > 16.0f) cw = 16.0f;
  if (cw < 4.0f) cw = 4.0f;
  struct Dir {
    float x, y, inv;
  };
  auto dir = [](const std::pair<float, float>& a, const std::pair<float, float>& b) {
    Dir d{a.first - b.first, a.second - b.second, 0.0f};
    const float l = d.x * d.x + d.y * d.y;
    const float r = sqrtf(l);
    d.inv = l == 0.0f ? r : 1.0f / r;
    return d;
  };
  auto& tp = m->top;
  auto& bt = m->bottom;
  const Dir t0 = dir(tp[0], tp[1]), tn = dir(tp[n - 1], tp[n - 2]);
  const Dir b0 = dir(bt[0], bt[1]), bn = dir(bt[n - 1], bt[n - 2]);
  const float cl = c.box_width_padding_left_portion * cw, cr = cw - cl;
  auto move = [](std::pair<float, float>* p, const Dir& d, float len) {
    p->first = (d.x * d.inv) * len + p->first;
    p->second = (d.y * d.inv) * len + p->second;
  };
  move(&tp[0], t0, cl);
  move(&tp[n - 1], tn, cr);
  move(&bt[0], b0, cl);
  move(&bt[n - 1], bn, cr);
}

// GroupRpnDetectorV2: the paragraph of a line is the one whose level-1 boxes cover the largest area.
int LineParagraph(const std::vector<Level1Box>& l1, const MergedLine& m) {
  std::vector<const Level1Box*> v;
  for (int i : m.l1) v.push_back(&l1[i]);
  SortCounted("line_paragraph", v.begin(), v.end(),
              [](const Level1Box* a, const Level1Box* b) { return a->paragraph < b->paragraph; });
  int best = -1;
  float vbest = -1.0f, run = 0.0f;
  for (size_t k = 0; k < v.size(); k++) {
    const float acc = run + v[k]->box.w * v[k]->box.h;
    if (k + 1 < v.size() && v[k]->paragraph == v[k + 1]->paragraph) {
      run = acc;
    } else {
      run = 0.0f;
      if (acc > vbest) best = v[k]->paragraph, vbest = acc;
    }
  }
  return best;
}

// The image pyramid of GroupRpnDetectorV2 (also rebuilt by the line recognizer, which crops lines of large text
// from it: DetectorPyramid).
void BuildPyramid(const TextDetector::Config& c, const Plane& src, std::vector<std::unique_ptr<Plane>>* levels_out,
                  std::vector<float>* scales_out) {
  const int W = src.width, H = src.height;
  // Reference image: box-filter scale so the image fits the largest predefined shape, upscaling (if at all) only
  // up to the first shape (GroupRpnDetectorV2, fixed_shape_inference = false).
  const auto& first = c.predefined_shapes.front();
  const auto& last = c.predefined_shapes.back();
  float up = static_cast<float>(first.first) / static_cast<float>(W);
  if (static_cast<float>(first.second) / static_cast<float>(H) < up)
    up = static_cast<float>(first.second) / static_cast<float>(H);
  if (up <= 1.0f) up = 1.0f;
  float s0 = static_cast<float>(last.second) / static_cast<float>(H);
  if (s0 >= static_cast<float>(last.first) / static_cast<float>(W))
    s0 = static_cast<float>(last.first) / static_cast<float>(W);
  if (s0 >= up) s0 = up;

  // Pyramid levels: [reference, fit-to-first-shape, then halvings]; one level per packing_sequence entry.
  const int num_levels = static_cast<int>(c.packing_sequence.size());
  std::vector<std::unique_ptr<Plane>>& levels = *levels_out;
  std::vector<float>& scales = *scales_out;
  levels.clear();
  levels.resize(num_levels);
  scales.assign(num_levels, 0.0f);
  levels[0] = std::make_unique<Plane>(ScaleImage(src, s0, kFilterBox));
  scales[0] = s0;
  for (int k = 1; k < num_levels; k++) {
    const Plane& prev = *levels[k - 1];
    float s;
    if (k == 1) {
      s = static_cast<float>(first.second) / static_cast<float>(prev.height);
      if (s >= static_cast<float>(first.first) / static_cast<float>(prev.width))
        s = static_cast<float>(first.first) / static_cast<float>(prev.width);
    } else {
      s = 0.5f;
    }
    if (fabsf(s + -1.0f) < 0.001f)
      levels[k] = std::move(levels[k - 1]);  // reused, the previous level slot becomes empty
    else
      levels[k] = std::make_unique<Plane>(ScaleImage(prev, s, k == 1 ? kFilterBox : kFilterBilinear));
    scales[k] = s * scales[k - 1];
  }
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------

TextDetector::~TextDetector() = default;

std::vector<DetectorPyramidLevel> DetectorPyramid(const Resources& res, const GrayImage& gray) {
  std::string cfg_bytes;
  TextDetector::Config c;
  if (gray.width < 1 || gray.height < 1 || gray.width >= 32768 || gray.height >= 32768) return {};
  if (!res.Read(kConfigPath, &cfg_bytes) || !ParseConfig(cfg_bytes, &c)) return {};
  Plane src;
  src.width = gray.width, src.height = gray.height;
  src.data = gray.data;
  std::vector<std::unique_ptr<Plane>> levels;
  std::vector<float> scales;
  BuildPyramid(c, src, &levels, &scales);
  levels[0].reset();  // never fed to the model, and gone when the lines are cropped (see Detect)
  std::vector<DetectorPyramidLevel> out(levels.size());
  for (size_t k = 0; k < levels.size(); k++) {
    out[k].scale = scales[k];
    if (!levels[k]) continue;
    out[k].image.emplace();
    out[k].image->width = levels[k]->width, out[k].image->height = levels[k]->height;
    out[k].image->data = std::move(levels[k]->data);
  }
  return out;
}

std::unique_ptr<TextDetector> TextDetector::Create(const Resources& res) {
  std::string cfg_bytes, model_bytes;
  if (!res.Read(kConfigPath, &cfg_bytes) || !res.Read(kModelPath, &model_bytes)) return nullptr;
  std::unique_ptr<TextDetector> d(new TextDetector());
  d->config_ = std::make_unique<Config>();
  if (!ParseConfig(cfg_bytes, d->config_.get())) return nullptr;
  d->model_ = TfliteModel::Create(std::move(model_bytes), d->config_->num_threads);
  if (!d->model_) return nullptr;
  return d;
}

bool TextDetector::Detect(const GrayImage& gray, PageLayout* out) {
  const Config& c = *config_;
  const int W = gray.width, H = gray.height;
  if (W < 1 || H < 1) return false;
  // google_ocr::ScaleImage refuses grayscale inputs with a side >= 32768 ("Large image failure"), which fails the
  // whole OCR call; the API itself only rejects sides > 32768.
  if (W >= 32768 || H >= 32768) return false;

  Plane src;
  src.width = W, src.height = H;
  src.data = gray.data;
  const int num_levels = static_cast<int>(c.packing_sequence.size());
  std::vector<std::unique_ptr<Plane>> levels;
  std::vector<float> scales;
  BuildPyramid(c, src, &levels, &scales);
  if (DumpDir()) {
    std::string d;
    for (const auto& [w, h] : c.predefined_shapes) d += "shape " + std::to_string(w) + "x" + std::to_string(h) + "\n";
    for (int k = 0; k < num_levels; k++)
      d += "level " + std::to_string(k) + " " + (levels[k] ? std::to_string(levels[k]->width) + "x" +
                                                                 std::to_string(levels[k]->height)
                                                           : std::string("empty")) +
           " scale " + std::to_string(scales[k]) + " heads " + std::to_string(c.packing_sequence[k]) + "/" +
           std::to_string(c.center_box_packing_sequence[k]) + "\n";
    char cfg[512];
    snprintf(cfg, sizeof cfg,
             "threshold %g sigmoid %d log_scaling %d cos_sin %d anchor %g %g level1 radius %g max_angle %g "
             "max_h_ratio %g\n",
             c.confidence_threshold, c.apply_sigmoid, c.heatmap_log_scaling, c.project_cos_sin, c.anchor_x_center,
             c.anchor_y_center, c.level1.search_radius_factor, c.level1.maximum_angle_difference,
             c.level1.max_height_ratio);
    d += cfg;
    d += "merge";
    for (float v : c.merge.v) d += " " + std::to_string(v);
    snprintf(cfg, sizeof cfg,
             " curved_exp %d recluster %d\nlevel1 centers %g/%g min_overlap %g/%g curve size %d hratio %g angvar %g\n",
             c.merge.use_experimental_curved_line_construction, c.merge.enable_reclustering,
             c.level1.max_centers_relative_distance, c.level1.optimal_centers_relative_distance, c.level1.min_overlap,
             c.level1.min_overlap_with_optimal_centers_aligment, c.level1.minimum_cluster_size_to_curve,
             c.level1.maximum_height_ratio_to_curve, c.level1.minimum_angle_variance_to_curve);
    d += cfg;
    for (size_t i = 0; i < c.subsampling.size(); i++)
      d += "head " + std::to_string(i) + " sub " + std::to_string(c.subsampling[i]) + " wscale " +
           std::to_string(c.heatmap_width_scaling[i]) + " hscale " + std::to_string(c.heatmap_height_scaling[i]) +
           " max " + std::to_string(c.max_width[i]) + "x" + std::to_string(c.max_height[i]) + "\n";
    DumpFile("levels.txt", d);
  }
  // The reference level is never fed to the model (input_features stays a 32x32 zero tensor); the boxes its heads
  // produce are dropped like those of an empty level.
  levels[0].reset();
  // CropDetections sees the same pyramid: the reference level is gone there too (testdata/extra_det/big_text_huge:
  // a line the 0.32 reference level would keep 64 px tall is cropped from the page instead).
  PyramidImages pyramid{scales, std::vector<bool>(num_levels), std::vector<std::pair<int, int>>(num_levels)};
  for (int k = 0; k < num_levels; k++) {
    pyramid.present[k] = levels[k] != nullptr;
    if (levels[k]) pyramid.dims[k] = {levels[k]->width, levels[k]->height};
  }

  // Model inputs: one per level with heads (packing_sequence or center_box_packing_sequence > 0), padded with 0
  // to a multiple of 32; empty levels get a zero 32x32 tensor.
  TfliteModel& m = *model_;
  std::vector<int> input_level;
  for (int k = 0; k < num_levels; k++)
    if (c.packing_sequence[k] != 0 || c.center_box_packing_sequence[k] != 0) input_level.push_back(k);
  if (input_level.size() != c.input_names.size()) return false;
  for (size_t i = 0; i < input_level.size(); i++) {
    const Plane* p = levels[input_level[i]].get();
    const int iw = p ? (p->width + 31) / 32 * 32 : 32, ih = p ? (p->height + 31) / 32 * 32 : 32;
    const int idx = m.InputIndex(c.input_names[i]);
    if (idx < 0 || !m.Resize(idx, {1, ih, iw, 1})) return false;
  }
  if (!m.AllocateTensors()) return false;
  for (size_t i = 0; i < input_level.size(); i++) {
    const Plane* p = levels[input_level[i]].get();
    TfLiteTensor* t = m.input(m.InputIndex(c.input_names[i]));
    uint8_t* d = t->data.uint8;
    std::memset(d, 0, t->bytes);
    if (p) {
      const int iw = t->dims->data[2];
      for (int y = 0; y < p->height; y++) std::memcpy(d + static_cast<size_t>(y) * iw, p->data.data() + y * p->width, p->width);
    }
  }
  if (!m.Invoke()) return false;

  // Decode: horizontal heads (packing_sequence) then vertical heads (center_box_packing_sequence), in output order.
  std::vector<std::vector<RawBox>> hraw, vraw;
  std::vector<float> hinv, vinv;
  std::vector<bool> hnull, vnull;
  int head = 0;
  std::string cells_dump;
  for (int pass = 0; pass < 2; pass++) {
    const std::vector<int>& seq = pass == 0 ? c.packing_sequence : c.center_box_packing_sequence;
    int local = 0;
    for (int k = 0; k < num_levels; k++) {
      for (int j = 0; j < seq[k]; j++, head++, local++) {
        const TfLiteTensor* t = m.output(c.output_names[head]);
        if (!t || t->dims->size != 4) return false;
        std::vector<RawBox> boxes;
        if (!OutputEdits().empty())
          ApplyOutputEdits(pass, local, t->data.f, t->dims->data[1], t->dims->data[2], t->dims->data[3]);
        HeatmapToRawBoxes(t->data.f, t->dims->data[1], t->dims->data[2], local, c, &boxes);
        if (DumpDir()) {  // cells.txt: per output tensor, its shape and the cells above the threshold
          const int H = t->dims->data[1], W = t->dims->data[2], C = t->dims->data[3];
          char buf[160];
          snprintf(buf, sizeof buf, "tensor %d %d %dx%dx%d level %d\n", pass, local, H, W, C, k);
          cells_dump += buf;
          const float min_logit = -logf(1.0f / c.confidence_threshold + -1.0f);
          for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
              const float* q = t->data.f + (static_cast<size_t>(y) * W + x) * C;
              if (!(q[0] > min_logit)) continue;
              snprintf(buf, sizeof buf, "%d %d %a %a %a %a %a %a %a\n", y, x, q[0], q[1], q[2], q[3], q[4], q[5],
                       q[6]);
              cells_dump += buf;
            }
        }
        (pass == 0 ? hraw : vraw).push_back(std::move(boxes));
        (pass == 0 ? hinv : vinv).push_back(1.0f / scales[k]);
        (pass == 0 ? hnull : vnull).push_back(levels[k] == nullptr);
      }
    }
  }
  DumpFile("cells.txt", cells_dump);
  DumpFile("rawboxes_0.bin", SerializeRawLists(hraw));
  DumpFile("rawboxes_1.bin", SerializeRawLists(vraw));

  // Back to page coordinates; heads of an empty pyramid level (never fed to the model) are dropped.
  auto rescale = [](std::vector<RawBox>* l, float inv) {
    for (RawBox& b : *l) {
      b.x *= inv, b.y *= inv, b.w *= inv, b.h *= inv;
      for (float& p : b.pts) p *= inv;
      b.cx *= inv, b.cy *= inv;
    }
  };
  std::vector<RawBox> all;
  for (size_t i = 0; i < hraw.size(); i++) {
    if (hnull[i]) continue;
    rescale(&hraw[i], hinv[i]);
    all.insert(all.end(), hraw[i].begin(), hraw[i].end());
  }
  std::vector<std::vector<RawBox>> vlists;
  for (size_t i = 0; i < vraw.size(); i++) {
    if (vnull[i]) continue;
    rescale(&vraw[i], vinv[i]);
    vlists.push_back(std::move(vraw[i]));
  }
  DumpFile("allraw_0.bin", SerializeRawLists({all}));

  const std::vector<ParagraphPrediction> paragraphs = ParagraphNms(vlists);
  std::vector<Level1Box> l1 = ComputeLevel1Boxes(c.level1, all);
  DumpFile("level1_0.bin", SerializeLevel1(l1));
  AssignParagraphs(paragraphs, &l1);
  std::vector<MergedLine> lines = ConstructLines(c.merge, all, &l1);

  // Entities: LINE, then its PARAGRAPH (shared by lines whose dominant paragraph prediction is the same).
  std::map<int, int> paragraph_entity;
  std::string dump;
  for (MergedLine& m : lines) {
    const RBox unpadded = m.box;
    PadLine(c, &m);
    m.paragraph = LineParagraph(l1, m);
    const int line_index = static_cast<int>(out->entities.size());
    out->entities.emplace_back();
    int parent;
    auto it = m.paragraph == -1 ? paragraph_entity.end() : paragraph_entity.find(m.paragraph);
    if (it != paragraph_entity.end()) {
      parent = it->second;
    } else {
      parent = static_cast<int>(out->entities.size());
      Entity p;
      p.type = EntityType::kParagraph;
      out->entities.push_back(std::move(p));
      if (m.paragraph != -1) paragraph_entity[m.paragraph] = parent;
    }
    Entity& e = out->entities[line_index];
    e.type = EntityType::kLine;
    e.parent = parent;
    RotatedBox box;
    box.left = static_cast<int>(m.box.x);
    box.top = static_cast<int>(m.box.y);
    box.width = static_cast<int>(static_cast<int64_t>(rintf(m.box.w)));
    box.height = static_cast<int>(static_cast<int64_t>(rintf(m.box.h)));
    box.angle = m.box.angle;
    std::optional<Curve> curve;
    if (m.curved) {  // TransformBoxesToPhotoOCR: mid line = deduplicated midpoints of the top / bottom polylines
      curve.emplace();
      curve->thickness = static_cast<double>(m.thickness);
      for (size_t k = 0; k < m.top.size() && k < m.bottom.size(); k++) {
        const float x = (m.top[k].first + m.bottom[k].first) * 0.5f;
        const float y = (m.top[k].second + m.bottom[k].second) * 0.5f;
        if (k > 0 && static_cast<float>(curve->points.back().first) == x &&
            static_cast<float>(curve->points.back().second) == y)
          continue;
        curve->points.push_back({static_cast<double>(x), static_cast<double>(y)});
      }
    }
    const int image_scale = CropDetection(pyramid, W, H, &box, &curve);
    e.box = box;
    if (curve && curve->points.size() >= 2) {  // BoundingBoxToBoundingPolygon: curved_box needs >= 2 points
      e.box->curve = curve;
      e.box->curve->top_to_bottom = false;
    }
    e.detection_confidence = m.avg_score;
    DetectionInfo d;
    d.detection_box = box;
    d.detection_box.curve = curve;
    d.average_raw_detections_height = m.avg_height;
    d.detection_nms_score = m.avg_score;
    d.num_raw_boxes = m.count;
    d.sum_raw_boxes_scores = m.avg_score * static_cast<float>(m.count);
    d.left_padding = m.left_padding;
    d.right_padding = m.right_padding;
    d.detection_image_scale = image_scale;
    e.detection = d;
    if (DumpDir()) {
      char buf[256];
      snprintf(buf, sizeof buf, "line %d par %d n %d box %.9g %.9g %.9g %.9g %.9g unpadded %.9g %.9g %.9g %.9g l1",
               line_index, m.paragraph, m.count, m.box.x, m.box.y, m.box.w, m.box.h, m.box.angle, unpadded.x,
               unpadded.y, unpadded.w, unpadded.h);
      dump += buf;
      for (int i : m.l1) dump += " " + std::to_string(i) + "/" + std::to_string(l1[i].paragraph);
      dump += "\n";
    }
  }
  DumpFile("lines.txt", dump);
  if (DumpDir()) {
    std::string ties;
    for (const auto& [site, t] : TieCounts())
      ties += site.rfind("event:", 0) == 0
                  ? site + " " + std::to_string(t.first) + "\n"
                  : site + " sorts " + std::to_string(t.first) + " tied_adjacent " + std::to_string(t.second) + "\n";
    DumpFile("ties.txt", ties);
  }
  return true;
}

}  // namespace screenai
