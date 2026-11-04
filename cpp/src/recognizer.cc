// SPDX-License-Identifier: MPL-2.0
#include "recognizer.h"

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <map>
#include <thread>

#include "ctc_decoder.h"
#include "detector.h"
#include "language_id.h"
#include "libyuv_scale.h"
#include "proto_wire.h"
#include "recognizer_bidi.h"
#include "recognizer_curve.h"
#include "recognizer_indic.h"
#include "recognizer_langcode.h"
#include "recognizer_words.h"
#include "tflite_model.h"
#include "x86_compat.h"

namespace screenai {
namespace rec {
namespace {

constexpr const char* kModelRoot = "gocr/gocr_models/";
constexpr const char* kModelDir = "gocr/gocr_models/line_recognition_mobile_convnext320_omni/";
constexpr const char* kMultiPassConfig =
    "gocr/gocr_models/gocr_line_recognition_omni_mobile_chrome_multiscript_2024_q4.binarypb";

// Fixed line geometry of every shipped GocrLineRecognizerConfig (scaled_image_height, chunk_width,
// chunk_left/right_padding, frame_width, num_chunk_per_batch).
constexpr int kLineHeight = 32, kChunkWidth = 136, kPad = 16, kFrameWidth = 4, kBatch = 8;
constexpr int kChunkInput = kPad + kChunkWidth + kPad;  // 168

// google_ocr.GocrLineRecognizerConfig (the fields that differ between the shipped recognizers).
struct Config {
  std::string recognizer_name, model_path, label_map_path, script_id;
  std::vector<std::string> output_names;
  // TfliteModelPooledRunnerConfig: a pool of num_inference_threads interpreters (each with interpreter_num_threads
  // threads) that run the batches concurrently.
  int pool_size = 1, interpreter_threads = 1;
  bool base_dir_rtl = false;
  bool has_ctc_decoder = false;  // ctc_decoder_config with an LM: CtcBeamDecoder instead of greedy decoding
  bool has_confidence_scorer = false;
  std::vector<std::string> script_names, supported_scripts, content_type_names;
  // output indices of the optional heads (0 = not used: the line gets default symbol boxes, no script/content
  // type/direction prediction)
  int symbol_box_index = 0, script_index = 0, content_type_index = 0, direction_index = 0;
};

bool ParseConfig(std::string_view line_recognizer_config, Config* c) {
  // LineRecognizerConfig{name=1, sub_config=2 (Any{type_url=1, value=2})}
  std::string_view sub;
  pb::Reader r(line_recognizer_config);
  while (r.Next())
    if (r.field() == 2) {
      pb::Reader any(r.bytes());
      while (any.Next())
        if (any.field() == 2) sub = any.bytes();
    }
  if (sub.empty()) return false;
  pb::Reader g(sub);
  while (g.Next()) {
    switch (g.field()) {
      case 1: c->recognizer_name = g.bytes(); break;
      case 12: c->base_dir_rtl = g.varint() != 0; break;
      case 14: c->label_map_path = g.bytes(); break;
      case 17: c->script_id = g.bytes(); break;
      case 13: {  // TensorFlowModelRunnerConfig.tflite_model_pooled_runner_config (3)
        pb::Reader t(g.bytes());
        while (t.Next()) {
          if (t.field() != 3) continue;
          pb::Reader p(t.bytes());
          while (p.Next()) {
            if (p.field() == 1) c->model_path = p.bytes();
            if (p.field() == 2) c->pool_size = std::max(1, static_cast<int>(p.int64()));
            if (p.field() == 3) c->interpreter_threads = std::max(1, static_cast<int>(p.int64()));
            if (p.field() == 8) c->output_names.emplace_back(p.bytes());
          }
        }
        break;
      }
      case 102: {  // GocrCTCDecoderRecognizerConfig
        pb::Reader d(g.bytes());
        while (d.Next()) {
          switch (d.field()) {
            case 1: {
              pb::Reader cd(d.bytes());
              while (cd.Next())
                if (cd.field() == 1) c->has_ctc_decoder = true;  // lm_spec
              break;
            }
            case 7: c->has_confidence_scorer = true; break;
            case 10: c->script_names.emplace_back(d.bytes()); break;
            case 11: c->supported_scripts.emplace_back(d.bytes()); break;
            case 12: c->content_type_names.emplace_back(d.bytes()); break;
            case 19: c->symbol_box_index = static_cast<int>(d.varint()); break;
            case 20: c->script_index = static_cast<int>(d.varint()); break;
            case 21: c->content_type_index = static_cast<int>(d.varint()); break;
            case 22: c->direction_index = static_cast<int>(d.varint()); break;
          }
        }
        break;
      }
    }
  }
  return g.ok() && !c->model_path.empty();
}

// ---------------------------------------------------------------------------------------------------------------
// Line image (GocrLineRecognizer::CreateLineImage): deskewed crop of the line box at page resolution.

int RoundHalfAway(float v) { return static_cast<int>(roundf(v)); }

void AlignedBox(const RotatedBox& b, int* L, int* T, int* W, int* H) {
  const float a = b.angle;
  if (a == 0.0f) {
    *L = b.left, *T = b.top, *W = b.width, *H = b.height;
    return;
  }
  const float r = a * 0.0174532924f;
  const float s = static_cast<float>(sin(static_cast<double>(r))), c = static_cast<float>(cos(static_cast<double>(r)));
  const float fw = static_cast<float>(b.width - 1), fh = static_cast<float>(b.height - 1);
  const float fl = static_cast<float>(b.left), ft = static_cast<float>(b.top);
  const float x1 = c * fw + fl, y1 = s * fw + ft;
  const float x2 = x1 + -(s * fh), y2 = y1 + c * fh;
  const float x3 = -(s * fh) + fl, y3 = c * fh + ft;
  const float mnx = std::min({fl, x1, x2, x3}), mxx = std::max({fl, x1, x2, x3});
  const float mny = std::min({ft, y1, y2, y3}), mxy = std::max({ft, y1, y2, y3});
  *L = RoundHalfAway(mnx), *T = RoundHalfAway(mny);
  const int R = RoundHalfAway(mnx + (mxx - mnx)), B = RoundHalfAway(mny + (mxy - mny));
  *W = R - *L + 1, *H = B - *T + 1;
}

void Clip(int* l, int* t, int* w, int* h, int W, int H) {
  if (*l < 0) *w = *l + *w, *l = 0;
  if (*t < 0) *h = *t + *h, *t = 0;
  if (*l + *w > W) *w = W - *l;
  if (*t + *h > H) *h = H - *t;
}

// Inverse-affine bilinear warp with 10-bit fixed-point coordinates and 5-bit interpolation weights. Interior pixels
// use float weights in the arm64 original and Q15 integer weights (as at the border) in the x86-64 one.
GrayImage Warp(const GrayImage& page, int cl, int ct, int sw, int sh, const float M[6], int dw, int dh) {
  GrayImage out;
  out.width = dw, out.height = dh;
  out.data.assign(static_cast<size_t>(dw) * dh, 0);
  int fp[6];
  for (int i = 0; i < 6; i++) fp[i] = static_cast<int>(lrintf(M[i] * 1024.0f));
  auto src = [&](int x, int y) { return static_cast<int>(page.data[static_cast<size_t>(ct + y) * page.width + cl + x]); };
  for (int v = 0; v < dh; v++) {
    for (int u = 0; u < dw; u++) {
      const int32_t xf = fp[4] + 16 + u * fp[0] + v * fp[1];
      const int32_t yf = fp[5] + 16 + u * fp[2] + v * fp[3];
      const int x = static_cast<int16_t>(xf >> 10), y = static_cast<int16_t>(yf >> 10);
      const int idx = (yf & 0x3E0) | ((xf >> 5) & 0x1F);
      const double fx = (idx & 31) / 32.0, fy = (idx >> 5) / 32.0;
      const double w[4] = {(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy};
      int value;
      if (!SCREENAI_X86_COMPAT && (x & 0xFFFF) < sw - 1 && (y & 0xFFFF) < sh - 1) {
        const float val = ((static_cast<float>(w[0]) * static_cast<float>(src(x, y)) +
                            static_cast<float>(w[1]) * static_cast<float>(src(x + 1, y))) +
                           static_cast<float>(w[2]) * static_cast<float>(src(x, y + 1))) +
                          static_cast<float>(w[3]) * static_cast<float>(src(x + 1, y + 1));
        value = std::min(255, std::max(0, static_cast<int>(nearbyintf(val))));
      } else {
        const int x0 = x < 0 ? 0 : std::min(x, sw - 1), x1 = x < -1 ? 0 : std::min(x + 1, sw - 1);
        const int y0 = y < 0 ? 0 : std::min(y, sh - 1), y1 = y < -1 ? 0 : std::min(y + 1, sh - 1);
        int q[4];
        for (int i = 0; i < 4; i++) q[i] = static_cast<int>(nearbyint(w[i] * 32768));
        value = (q[0] * src(x0, y0) + q[1] * src(x1, y0) + q[2] * src(x0, y1) + q[3] * src(x1, y1) + 0x4000) >> 15;
      }
      out.data[static_cast<size_t>(v) * dw + u] = static_cast<uint8_t>(value);
    }
  }
  return out;
}

GrayImage LineImage(const GrayImage& page, const RotatedBox& b) {
  if (b.curve) return CurvedLineImage(page, *b.curve);  // ImageCrop of a curved_box polygon
  GrayImage out;
  if (b.angle == 0.0f) {
    int l = b.left, t = b.top, w = b.width, h = b.height;
    Clip(&l, &t, &w, &h, page.width, page.height);
    if (w <= 0 || h <= 0) return out;
    out.width = w, out.height = h;
    out.data.resize(static_cast<size_t>(w) * h);
    for (int y = 0; y < h; y++) memcpy(out.row(y), page.row(t + y) + l, w);
    return out;
  }
  int l, t, w, h;
  AlignedBox(b, &l, &t, &w, &h);
  Clip(&l, &t, &w, &h, page.width, page.height);
  if (w <= 0 || h <= 0 || b.width <= 0 || b.height <= 0) return out;
  const float r = static_cast<float>(static_cast<double>(b.angle) * 3.141592653589793 / 180.0);
  const float s = static_cast<float>(sin(static_cast<double>(r))), c = static_cast<float>(cos(static_cast<double>(r)));
  const float M[6] = {c, -s, s, c, static_cast<float>(b.left - l), static_cast<float>(b.top - t)};
  return Warp(page, l, t, w, h, M, b.width, b.height);
}

GrayImage ScaleToHeight(const GrayImage& img) {
  if (img.height == kLineHeight) return img;
  const float sc = static_cast<float>(kLineHeight) / static_cast<float>(img.height);
  GrayImage out;
  out.width = std::max(1, static_cast<int>(nearbyintf(sc * static_cast<float>(img.width))));
  out.height = std::max(1, static_cast<int>(nearbyintf(sc * static_cast<float>(img.height))));
  out.data.resize(static_cast<size_t>(out.width) * out.height);
  ScalePlane(img.data.data(), img.width, img.width, img.height, out.data.data(), out.width, out.width, out.height,
             kFilterBox);
  return out;
}

// Chunks of 136 columns with 16 columns of context on each side (zero outside the line).
std::vector<std::vector<uint8_t>> Chunks(const GrayImage& img) {
  std::vector<std::vector<uint8_t>> out;
  const int W = img.width, H = img.height;
  for (int s = 0;;) {
    const int left = std::min(s, kPad);
    int n = W - s;
    if (s + kChunkWidth + kPad <= W) n = kChunkWidth + kPad;
    std::vector<uint8_t> c(static_cast<size_t>(H) * kChunkInput, 0);
    for (int y = 0; y < H; y++)
      memcpy(&c[static_cast<size_t>(y) * kChunkInput + kPad - left], img.row(y) + s - left, left + n);
    out.push_back(std::move(c));
    s += kChunkWidth;
    if (s >= W) break;
  }
  return out;
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------
// Per-line model outputs after GocrLineRecognizer::DecodeLines (chunks merged, gated heads reduced).
struct LineTensors {
  int frames = 0;
  std::vector<float> logits;  // [frames][k][2] {class, value}
  std::vector<float> left, right, top, height;
  std::vector<float> script, content, direction;  // softmax over the gated activations (empty if absent)
};

class ScriptRecognizer {
 public:
  static std::unique_ptr<ScriptRecognizer> Create(const Resources& res, const std::string& config_file,
                                                  std::string* error);
  const Config& config() const { return cfg_; }
  const CtcBeamDecoder* decoder() const { return decoder_.get(); }

  // Runs the model over all lines (batches of 8 chunks) and returns the merged per-line tensors.
  std::vector<LineTensors> Run(const std::vector<GrayImage>& scaled_lines);
  const std::string* Label(int id) const {
    auto it = labels_.find(id);
    return it == labels_.end() ? nullptr : &it->second;
  }
  int num_labels() const { return static_cast<int>(labels_.size()); }

 private:
  Config cfg_;
  std::vector<std::unique_ptr<TfliteModel>> models_;  // interpreter pool
  std::map<int, std::string> labels_;
  std::unique_ptr<CtcBeamDecoder> decoder_;
  std::map<std::string, int> out_tensor_;  // signature output name -> tensor index
};

std::unique_ptr<ScriptRecognizer> ScriptRecognizer::Create(const Resources& res, const std::string& config_file,
                                                           std::string* error) {
  std::unique_ptr<ScriptRecognizer> r(new ScriptRecognizer());
  std::string bytes;
  if (!res.Read(std::string(kModelRoot) + config_file, &bytes) || !ParseConfig(bytes, &r->cfg_)) {
    *error = "cannot read recognizer config " + config_file;
    return nullptr;
  }
  std::string model;
  if (!res.Read(kModelDir + r->cfg_.model_path, &model)) {
    *error = "cannot read " + r->cfg_.model_path;
    return nullptr;
  }
  for (int i = 0; i < r->cfg_.pool_size; i++) {
    r->models_.push_back(TfliteModel::Create(model, r->cfg_.interpreter_threads));
    if (!r->models_.back()) {
      *error = "cannot load " + r->cfg_.model_path;
      return nullptr;
    }
  }
  for (const auto& [name, idx] : r->models_[0]->interpreter()->signature_outputs("serving_default"))
    r->out_tensor_[name] = static_cast<int>(idx);
  std::string lm;
  if (!res.Read(kModelDir + r->cfg_.label_map_path, &lm)) {
    *error = "cannot read " + r->cfg_.label_map_path;
    return nullptr;
  }
  pb::Reader lr(lm);
  while (lr.Next()) {
    if (lr.field() != 1) continue;
    std::string name;
    int id = 0;
    pb::Reader e(lr.bytes());
    while (e.Next()) {
      if (e.field() == 1) name = e.bytes();
      if (e.field() == 2) id = static_cast<int>(e.int64());
    }
    r->labels_.emplace(id, std::move(name));
  }
  // Every shipped model has labels 0..C-2 plus the CTC blank (C = decode_logits classes). A label map that does
  // not (truncated / corrupt file) is rejected, which fails the OCR call like the original does, instead of
  // decoding to labels that do not exist.
  int classes = -1;
  if (auto it = r->out_tensor_.find("decode_logits"); it != r->out_tensor_.end()) {
    const TfLiteTensor* t = r->models_[0]->interpreter()->tensor(it->second);
    classes = t->dims->data[t->dims->size - 1];
  }
  if (!lr.ok() || classes < 2 || static_cast<int>(r->labels_.size()) != classes - 1 ||
      r->labels_.begin()->first != 0 || r->labels_.rbegin()->first != classes - 2) {
    *error = "label map " + r->cfg_.label_map_path + " does not match the model";
    return nullptr;
  }
  if (r->cfg_.has_ctc_decoder) {
    const std::string file = config_file.substr(config_file.find('/') + 1);
    r->decoder_ = CtcBeamDecoder::Create(res, "gocr/gocr_models/line_recognition_mobile_convnext320_omni", file, error);
    if (!r->decoder_) return nullptr;
  }
  return r;
}

namespace {

// TfliteModelPooledRunner dense dequantization (vectorized blocks of 8 use FMLA with a double-rounded bias).
void DequantizeDense(const uint8_t* q, size_t n, float scale, int zp, float* out) {
  const float bias = static_cast<float>(static_cast<double>(scale) * static_cast<double>(-zp));
  size_t i = 0;
  for (; i + 8 <= n; i += 8)
    for (size_t j = i; j < i + 8; j++) out[j] = std::fma(static_cast<float>(q[j]), scale, bias);
  for (; i < n; i++) out[i] = static_cast<float>(static_cast<double>(scale) * static_cast<double>(q[i] - zp));
}

// tensor_utils::ReduceActivationsGatesAndSoftMax for one line.
std::vector<float> ReduceGatedSoftmax(const std::vector<float>& act, const std::vector<float>& gate, int frames,
                                      int classes) {
  std::vector<float> acc(classes, 0.0f);
  float gsum = 0.0f;
  for (int t = 0; t < frames; t++) {
    for (int c = 0; c < classes; c++) acc[c] += gate[t] * act[static_cast<size_t>(t) * classes + c];
    gsum += gate[t];
  }
  float mx = -FLT_MAX;
  for (int c = 0; c < classes; c++) {
    acc[c] = acc[c] / (gsum + 1.0f);
    if (acc[c] > mx) mx = acc[c];
  }
  float sum = 0.0f;
  for (int c = 0; c < classes; c++) acc[c] = expf(acc[c] - mx), sum += acc[c];
  for (int c = 0; c < classes; c++) acc[c] = acc[c] / sum;
  return acc;
}

}  // namespace

std::vector<LineTensors> ScriptRecognizer::Run(const std::vector<GrayImage>& lines) {
  struct ChunkRef {
    int line, offset, width;
  };
  std::vector<std::vector<uint8_t>> chunks;
  std::vector<ChunkRef> refs;
  for (size_t li = 0; li < lines.size(); li++) {
    if (lines[li].width <= 0) continue;
    auto cs = Chunks(lines[li]);
    for (size_t c = 0; c < cs.size(); c++) {
      const int off = static_cast<int>(c) * kChunkWidth;
      refs.push_back({static_cast<int>(li), off, std::min(kChunkWidth, lines[li].width - off)});
      chunks.push_back(std::move(cs[c]));
    }
  }
  const int frames_per_chunk = kChunkInput / kFrameWidth;  // 42
  const int first_frame = kPad / kFrameWidth;              // 4
  std::vector<LineTensors> out(lines.size());
  // Batch inputs, then the invokes spread over the interpreter pool, then the outputs in batch order.
  const size_t batch_bytes = static_cast<size_t>(kBatch) * kLineHeight * kChunkInput;
  const size_t num_batches = (chunks.size() + kBatch - 1) / kBatch;
  std::vector<std::vector<uint8_t>> inputs(num_batches, std::vector<uint8_t>(batch_bytes, 0));
  for (size_t b = 0; b < num_batches; b++) {
    for (size_t j = 0; j < static_cast<size_t>(kBatch) && b * kBatch + j < chunks.size(); j++)
      memcpy(inputs[b].data() + j * kLineHeight * kChunkInput, chunks[b * kBatch + j].data(),
             chunks[b * kBatch + j].size());
    if (const char* dir = std::getenv("SCREENAI_REC_DUMP")) {  // debug: model input of every invoke
      static int seq = 0;
      char name[64];
      std::snprintf(name, sizeof(name), "/%05d_in.bin", seq++);
      if (FILE* f = std::fopen((std::string(dir) + name).c_str(), "wb")) {
        std::fwrite(inputs[b].data(), 1, batch_bytes, f);
        std::fclose(f);
      }
    }
  }
  struct Output {
    std::vector<uint8_t> data;
    float scale = 0;
    int zero_point = 0, last_dim = 0;
  };
  std::vector<std::map<std::string, Output>> outputs(num_batches);
  std::atomic<size_t> next_batch{0};
  std::atomic<bool> ok{true};
  auto worker = [&](TfliteModel* model) {
    if (!model->Resize(0, {kBatch, kLineHeight, kChunkInput, 1}) || !model->AllocateTensors()) return void(ok = false);
    for (size_t b; (b = next_batch.fetch_add(1)) < num_batches && ok;) {
      memcpy(model->input(0)->data.uint8, inputs[b].data(), batch_bytes);
      if (!model->Invoke()) return void(ok = false);
      for (const auto& [name, idx] : out_tensor_) {
        const TfLiteTensor* t = model->interpreter()->tensor(idx);
        Output& o = outputs[b][name];
        o.data.assign(t->data.uint8, t->data.uint8 + t->bytes);
        o.scale = t->params.scale;
        o.zero_point = t->params.zero_point;
        o.last_dim = t->dims->data[t->dims->size - 1];
      }
    }
  };
  const size_t pool = std::min(models_.size(), num_batches);
  if (pool <= 1) {
    worker(models_[0].get());
  } else {
    std::vector<std::thread> threads;
    for (size_t i = 0; i < pool; i++) threads.emplace_back(worker, models_[i].get());
    for (std::thread& t : threads) t.join();
  }
  if (!ok) return {};
  for (size_t b0 = 0; b0 < chunks.size(); b0 += kBatch) {
    const std::map<std::string, Output>& batch_out = outputs[b0 / kBatch];
    auto tensor = [&](const char* name) -> const Output* {
      auto it = batch_out.find(name);
      return it == batch_out.end() ? nullptr : &it->second;
    };
    struct Dense {
      std::vector<float> v;
      int inner = 0;
    };
    auto dense = [&](const char* name) {
      Dense d;
      const Output* t = tensor(name);
      if (!t) return d;
      const size_t n = t->data.size();
      d.v.resize(n);
      DequantizeDense(t->data.data(), n, t->scale, t->zero_point, d.v.data());
      d.inner = static_cast<int>(n / (kBatch * frames_per_chunk));
      return d;
    };
    const char* names[] = {"symbol_left_offset", "symbol_right_offset", "symbol_top",
                           "symbol_height",      "script_activations",  "script_gates",
                           "content_type_activations", "content_type_gates", "direction_activations",
                           "direction_gates"};
    Dense d[10];
    for (int i = 0; i < 10; i++) d[i] = dense(names[i]);
    const Output* lt = tensor("decode_logits");
    const int C = lt->last_dim;
    for (int j = 0; j < kBatch && b0 + j < chunks.size(); j++) {
      const ChunkRef& r = refs[b0 + j];
      LineTensors& L = out[r.line];
      const int nf = (r.width + kFrameWidth - 1) / kFrameWidth;
      for (int f = 0; f < nf; f++) {
        const int src = j * frames_per_chunk + first_frame + f;
        float sp[5][2];
        SparseDequantizeFrame(lt->data.data() + static_cast<size_t>(src) * C, C, lt->zero_point, lt->scale, 5,
                              &sp[0][0]);
        L.logits.insert(L.logits.end(), &sp[0][0], &sp[0][0] + 10);
        std::vector<float>* dst[] = {&L.left, &L.right, &L.top, &L.height};
        for (int i = 0; i < 4; i++)
          if (!d[i].v.empty()) dst[i]->push_back(d[i].v[src]);
        L.frames++;
      }
    }
    // gated heads: keep the raw per-frame activations until all chunks of the line are merged
    for (int j = 0; j < kBatch && b0 + j < chunks.size(); j++) {
      const ChunkRef& r = refs[b0 + j];
      LineTensors& L = out[r.line];
      const int nf = (r.width + kFrameWidth - 1) / kFrameWidth;
      for (int h = 0; h < 3; h++) {
        const Dense& act = d[4 + 2 * h];
        const Dense& gate = d[5 + 2 * h];
        if (act.v.empty()) continue;
        std::vector<float>& dst = h == 0 ? L.script : h == 1 ? L.content : L.direction;
        for (int f = 0; f < nf; f++) {
          const int src = j * frames_per_chunk + first_frame + f;
          dst.insert(dst.end(), act.v.begin() + static_cast<size_t>(src) * act.inner,
                     act.v.begin() + static_cast<size_t>(src + 1) * act.inner);
          dst.push_back(gate.v[src]);
        }
      }
    }
  }
  // reduce the gated heads: stored per frame as [act x C, gate]
  for (LineTensors& L : out) {
    if (cfg_.symbol_box_index <= 0) L.left.clear(), L.right.clear(), L.top.clear(), L.height.clear();
    if (cfg_.script_index <= 0) L.script.clear();
    if (cfg_.content_type_index <= 0) L.content.clear();
    if (cfg_.direction_index <= 0) L.direction.clear();
    for (std::vector<float>* v : {&L.script, &L.content, &L.direction}) {
      if (v->empty() || L.frames == 0) continue;
      const int stride = static_cast<int>(v->size()) / L.frames;
      const int C = stride - 1;
      std::vector<float> act(static_cast<size_t>(L.frames) * C), gate(L.frames);
      for (int t = 0; t < L.frames; t++) {
        std::copy(v->begin() + static_cast<size_t>(t) * stride, v->begin() + static_cast<size_t>(t) * stride + C,
                  act.begin() + static_cast<size_t>(t) * C);
        gate[t] = (*v)[static_cast<size_t>(t) * stride + C];
      }
      *v = ReduceGatedSoftmax(act, gate, L.frames, C);
    }
  }
  return out;
}

namespace {

struct Label {
  int id, frame;
  float prob;
};

// recognition::GreedyCTCLineDecoding over the sparse top-k logits.
std::vector<Label> GreedyDecode(const LineTensors& L, const ScriptRecognizer& r) {
  const int blank = r.num_labels();
  std::vector<Label> out;
  int prev = blank;
  for (int t = 0; t < L.frames; t++) {
    const float* e = &L.logits[static_cast<size_t>(t) * 10];
    int best = 0;
    for (int i = 1; i < 5; i++)
      if (e[2 * i + 1] > e[2 * best + 1]) best = i;
    const float mx = e[2 * best + 1];
    float sum = 0;
    for (int i = 0; i < 5; i++) sum += expf(e[2 * i + 1] - mx);
    const float prob = 1.0f / sum;
    int id = static_cast<int>(rintf(e[2 * best]));
    if (!r.Label(id)) id = blank;
    if (id != blank) {
      if (id != prev) {
        out.push_back({id, t, prob});
      } else {
        out.back().prob = prob;
      }
    }
    prev = id;
  }
  return out;
}

float BitsToFloat(uint32_t u) {
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// Eigen NEON pexp<Packet4f> lane (FMLA = fused multiply-add), as compiled into the original.
float EigenPexp(float x0) {
  const float x = std::min(std::max(x0, BitsToFloat(0xC2D00000)), BitsToFloat(0x42B1722D));
  const float t = x * BitsToFloat(0x3FB8AA3B);
  const float m = __builtin_roundevenf(t);
  float r = std::fma(BitsToFloat(0xBF318000), m, x);
  r = std::fma(BitsToFloat(0x395E8083), m, r);
  const float r2 = r * r;
  const float a = std::fma(BitsToFloat(0x3AB42872), r2, BitsToFloat(0x3D2AADCC));
  const float b = std::fma(BitsToFloat(0x3C091DE6), r2, BitsToFloat(0x3E2AAA47));
  float c = std::fma(a, r2, BitsToFloat(0x3EFFFFFC));
  const float low = r + 1.0f;
  c = std::fma(b, r, c);
  const float y = std::fma(c, r2, low);
  const uint32_t e = static_cast<uint32_t>(static_cast<int32_t>(m) + 254);
  const uint32_t h = e >> 1;
  const float out = (y * BitsToFloat(h << 23)) * BitsToFloat((e - h) << 23);
  return std::max(out, x0);
}

// Eigen inner-most SumReducer over exp(x[0..n)): packets of 4 (pexp, faddp reduction) + scalar expf tail.
float EigenSumExp(const float* x, int n) {
  const int vec = n / 4 * 4;
  float p[4] = {0, 0, 0, 0};
  for (int i = 0; i < vec; i += 4)
    for (int k = 0; k < 4; k++) p[k] = p[k] + EigenPexp(x[i + k]);
  float s = 0;
  for (int i = vec; i < n; i++) s = s + expf(x[i]);
  return s + ((p[0] + p[1]) + (p[2] + p[3]));
}

// CTCDecoder + CTCDecoderConfidenceScorer (CTCDecoderConfidenceFeatureExtractor logits_top_k with all costs
// disabled, AvgLogitsConfidenceScorer): label confidence = 1 / sum_j exp(topk_j - logit(label)).
std::vector<Label> BeamDecode(const LineTensors& L, const CtcBeamDecoder& dec) {
  constexpr int kK = 5, kTopK = 5;
  std::vector<float> values(static_cast<size_t>(L.frames) * kK), ids(values.size());
  for (size_t i = 0; i < values.size(); i++) ids[i] = L.logits[2 * i], values[i] = L.logits[2 * i + 1];
  CtcDecoderInput in;
  in.num_frames = L.frames, in.k = kK, in.values = values.data(), in.class_ids = ids.data();
  CtcDecoderOutput out;
  std::vector<Label> labels;
  if (!dec.Decode(in, &out)) return labels;
  for (const CtcDecoderOutput::Label& l : out.labels) {
    const float* row = &values[static_cast<size_t>(l.frame) * kK];
    const float* rid = &ids[static_cast<size_t>(l.frame) * kK];
    const int top = std::min(kK - 1, kTopK);
    std::vector<float> v(row, row + kK);
    std::nth_element(v.begin(), v.begin() + top, v.end(), std::greater<float>());
    v.resize(top + 1);
    std::sort(v.begin(), v.end(), std::greater<float>());
    int j = 0;
    while (j < kK && static_cast<int>(rintf(rid[j])) != l.label) j++;
    if (j == kK) j = 0;  // original: error status
    const float logit = row[j];
    float f[kTopK];
    for (int i = 0; i < kTopK; i++) f[i] = v[i] - logit;
    const float conf = 1.0f / EigenSumExp(f, kTopK);
    labels.push_back({l.label, l.frame, static_cast<float>(static_cast<double>(conf))});
  }
  return labels;
}

int Argmax(const std::vector<float>& v, int n) {
  int best = 0;
  for (int i = 1; i < n; i++)
    if (v[i] > v[best]) best = i;
  return best;
}

// RotateOrientationClockwise<Box<float>> + BoundingBoxToBoundingPolygon (rounded). A curved box keeps its curve,
// with points and thickness rounded to float by the Box<float> round trip: each quadrant clears top_to_bottom if
// set, else reverses the mid-line points and sets it.
RotatedBox RotateClockwise(const RotatedBox& b, int quadrants) {
  const int q = quadrants & 3;
  if (q == 0) return b;
  const float x = static_cast<float>(b.left), y = static_cast<float>(b.top);
  const float w = static_cast<float>(b.width), h = static_cast<float>(b.height);
  float s, c;
  SinCos(b.angle * 0.0174532924f, &s, &c);
  float nx, ny;
  if (q == 1) {
    nx = x + w * c, ny = y + w * s;
  } else if (q == 2) {
    nx = (x + w * c) - h * s, ny = h * c + (y + w * s);
  } else {
    nx = x - h * s, ny = y + h * c;
  }
  float a = b.angle + static_cast<float>(90 * q);
  while (a <= -180.0f) a = a + 360.0f;
  while (a > 180.0f) a = a + -360.0f;
  RotatedBox out;
  out.left = static_cast<int>(roundf(nx)), out.top = static_cast<int>(roundf(ny));
  out.width = (q & 1) ? b.height : b.width, out.height = (q & 1) ? b.width : b.height;
  out.angle = a;
  if (b.curve) {
    out.curve = b.curve;
    for (auto& p : out.curve->points)
      p = {static_cast<double>(static_cast<float>(p.first)), static_cast<double>(static_cast<float>(p.second))};
    out.curve->thickness = static_cast<double>(static_cast<float>(out.curve->thickness));
    if (out.curve->points.size() >= 2) {
      bool ttb = out.curve->top_to_bottom.value_or(false);
      for (int i = 0; i < q; i++) {
        if (!ttb) std::reverse(out.curve->points.begin(), out.curve->points.end());
        ttb = !ttb;
      }
      out.curve->top_to_bottom = ttb;
    }
  }
  return out;
}

int OrientationFromAngle(float a) {
  if (a >= -45.0f && a < 45.0f) return 0;
  if (a >= 45.0f && a < 135.0f) return 1;
  if (a < 135.0f && a >= -135.0f) return 3;
  return 2;
}

// GocrCTCDecoderRecognizer::BuildDefaultBoundingBox (clamp_symbol_box).
struct BoxPredictions {
  const std::vector<float>* left = nullptr;
  const std::vector<float>* right = nullptr;
  const std::vector<float>* top = nullptr;
  const std::vector<float>* height = nullptr;
};

}  // namespace

struct LineDecode {
  std::vector<RecSymbol> symbols;  // line-relative boxes
  std::string text;
  float confidence = 0;
};

namespace {

LineDecode DecodeLine(const LineTensors& L, const ScriptRecognizer& r, const GrayImage& line, bool reversed) {
  LineDecode d;
  LineTensors rev;
  if (reversed) {  // logits of the 180-degree rotated line
    rev.frames = L.frames;
    rev.logits.resize(L.logits.size());
    for (int t = 0; t < L.frames; t++)
      std::copy_n(&L.logits[static_cast<size_t>(L.frames - 1 - t) * 10], 10, &rev.logits[static_cast<size_t>(t) * 10]);
  }
  const LineTensors& LL = reversed ? rev : L;
  const std::vector<Label> labels = r.decoder() ? BeamDecode(LL, *r.decoder()) : GreedyDecode(LL, r);
  const float W = static_cast<float>(line.width), H = static_cast<float>(line.height);
  const double scale = static_cast<double>(static_cast<float>(W / static_cast<float>(L.frames)));
  std::vector<float> left = L.left, right = L.right, top = L.top, height = L.height;
  if (reversed && !left.empty()) {
    // GetSymbolBoxPredictions(reverse): predictions of the 180-degree rotated line
    const int n = L.frames;
    std::vector<float> l2(n), r2(n), t2(n), h2(n);
    for (int t = 0; t < n; t++) {
      const int s = n - 1 - t;
      l2[t] = right[s], r2[t] = left[s], h2[t] = height[s];
      t2[t] = (1.0f - top[s]) - height[s];
    }
    left = l2, right = r2, top = t2, height = h2;
  }
  const bool have = !left.empty() && !right.empty() && !top.empty() && !height.empty();
  // first non-space label's vertical extent (for space symbols)
  int ns_top = 0, ns_h = static_cast<int>(H);
  for (const Label& l : labels) {
    const std::string* s = r.Label(l.id);
    if (!s || *s == " ") continue;
    if (have) {
      ns_top = std::max(0, std::min(static_cast<int>(llroundf(top[l.frame] * H)), static_cast<int>(H - 1.0f)));
      ns_h = std::max(1, static_cast<int>(llroundf(H * height[l.frame])));
      ns_h = std::min(ns_h, static_cast<int>(H - static_cast<float>(ns_top)));
    }
    break;
  }
  double sum = 0;
  for (const Label& l : labels) {
    RecSymbol s;
    s.text = *r.Label(l.id);
    s.confidence = l.prob;
    const int f = l.frame;
    RotatedBox& b = s.box;
    if (!have || s.text == " ") {
      b.left = std::min(static_cast<int>(llround(scale * static_cast<double>(f))), static_cast<int>(W - 1.0f));
      b.width = std::max(1, static_cast<int>(llround(scale)));
      if (!have) {
        b.top = 0, b.height = static_cast<int>(H);
      } else {
        b.top = ns_top, b.height = ns_h;
      }
    } else {
      // The center goes to float before the subtraction (fcvt s, d; fsub s): 279.49999 - 6e-6 rounds to 280
      // (testdata/extra_ties/sym_arc).
      const float center = static_cast<float>((static_cast<double>(f) + 0.5) * scale);
      const float lo = left[f] * H;
      b.left = std::max(0, std::min(static_cast<int>(llroundf(center - lo)),
                                    static_cast<int>(W - 1.0f)));
      b.width = std::max(1, static_cast<int>(llroundf(lo + H * right[f])));
      b.width = std::min(b.width, static_cast<int>(W - static_cast<float>(b.left)));
      b.top = std::max(0, std::min(static_cast<int>(llroundf(H * top[f])), static_cast<int>(H - 1.0f)));
      b.height = std::max(1, static_cast<int>(llroundf(H * height[f])));
      b.height = std::min(b.height, static_cast<int>(H - static_cast<float>(b.top)));
    }
    d.text += s.text;
    sum += s.confidence;
    d.symbols.push_back(std::move(s));
  }
  d.confidence = d.symbols.empty() ? 0.0f : static_cast<float>(sum / static_cast<double>(d.symbols.size()));
  return d;
}

}  // namespace
}  // namespace rec

// ---------------------------------------------------------------------------------------------------------------

LineRecognizer::~LineRecognizer() = default;

std::unique_ptr<LineRecognizer> LineRecognizer::Create(const Resources& res) {
  std::unique_ptr<LineRecognizer> r(new LineRecognizer());
  r->res_ = &res;
  std::string bytes;
  if (!res.Read(rec::kMultiPassConfig, &bytes)) return nullptr;
  // google.protobuf.Any{value=2: MultiPassLineRecognitionMutatorConfig}
  std::string_view cfg;
  pb::Reader any(bytes);
  while (any.Next())
    if (any.field() == 2) cfg = any.bytes();
  pb::Reader m(cfg);
  while (m.Next()) {
    if (m.field() == 1) {  // map<string, LineRecognizerConfig{config_file=2}>
      std::string key, file;
      pb::Reader e(m.bytes());
      while (e.Next()) {
        if (e.field() == 1) key = e.bytes();
        if (e.field() == 2) {
          pb::Reader v(e.bytes());
          while (v.Next())
            if (v.field() == 2) file = v.bytes();
        }
      }
      r->config_files_[key] = file;
    } else if (m.field() == 2) {  // LineRecognizerConfigSelectorConfig{default_key=3}
      pb::Reader s(m.bytes());
      while (s.Next())
        if (s.field() == 3) r->default_key_ = s.bytes();
    } else if (m.field() == 5) {
      r->maximum_iterations_ = static_cast<int>(m.int64());
    }
  }
  if (!r->config_files_.count(r->default_key_)) return nullptr;
  // Language ID is optional, as in the original: without a loadable model, lines get no language from it.
  r->langid_ = LanguageIdentifier::Create(res);
  if (!r->Get(r->default_key_)) return nullptr;
  return r;
}

rec::ScriptRecognizer* LineRecognizer::Get(const std::string& key) {
  auto it = recognizers_.find(key);
  if (it != recognizers_.end()) return it->second.get();
  auto f = config_files_.find(key);
  if (f == config_files_.end()) return nullptr;
  std::string err;
  auto r = rec::ScriptRecognizer::Create(*res_, f->second, &err);
  if (!r) fprintf(stderr, "LineRecognizer: %s\n", err.c_str());
  return (recognizers_[key] = std::move(r)).get();
}

namespace {

struct LineState {
  rec::LineDecode decode;
  std::vector<rec::RecWord> words;
  GrayImage image;  // unscaled line image of the last pass
  std::string key;  // recognizer used
  GrayImage base_image;  // line image of the first pass (every pass starts from it)
  float box_scale = 1.0f;  // line-image -> page pixels (1000 / detection_image_scale)
  bool flipped = false;    // box turned by 180 degrees (direction head) since the first pass
};

// Lines of large text (detection_image_scale < 1000) come with the image GroupRpnDetectorV2's CropDetections cut
// from the detector pyramid level of that scale (ImageCache key = the line polygon). Their page box went through
// box_util::ScaleBoundingBox by s and back by 1/s (integer truncation, see detector.cc CropDetection); the crop box
// is recovered by inverting the way back, which is injective since 1/s > 1.
int UnscaleTruncated(int v, float s) {
  const float r = 1.0f / s;
  const int x = static_cast<int>(floorf(static_cast<float>(v) * s));
  for (int d : {0, 1, -1, 2, -2, 3, -3})
    if (static_cast<int>(r * static_cast<float>(x + d)) == v) return x + d;
  return x;
}

double UnscaleDouble(double v, float s) {
  const double r = static_cast<double>(1.0f / s);
  double c = v / r;
  for (int i = 0; i < 4 && c * r != v; i++) c = std::nextafter(c, c * r < v ? HUGE_VAL : -HUGE_VAL);
  return c;
}

RotatedBox PyramidCropBox(const RotatedBox& page, float s) {
  RotatedBox b = page;
  b.left = UnscaleTruncated(page.left, s);
  b.top = UnscaleTruncated(page.top, s);
  b.width = UnscaleTruncated(page.left + page.width, s) - b.left;
  b.height = UnscaleTruncated(page.top + page.height, s) - b.top;
  if (b.curve) {
    b.curve->thickness = UnscaleDouble(b.curve->thickness, s);
    for (auto& p : b.curve->points) p = {UnscaleDouble(p.first, s), UnscaleDouble(p.second, s)};
  }
  return b;
}

// GenerateLineWordsMaybeSetColors: symbol and word-break boxes (floor/ceil of the edges) and color sampling points
// (truncated) of a pyramid line image are scaled to page pixels before they are mapped onto the page.
void ScaleLineBoxes(float f, std::vector<rec::RecSymbol>* symbols, std::vector<rec::RecWord>* words) {
  auto box = [f](RotatedBox* b) {
    const int l = static_cast<int>(floorf(static_cast<float>(b->left) * f));
    const int t = static_cast<int>(floorf(static_cast<float>(b->top) * f));
    b->width = static_cast<int>(ceilf(static_cast<float>(b->width + b->left) * f)) - l;
    b->height = static_cast<int>(ceilf(static_cast<float>(b->height + b->top) * f)) - t;
    b->left = l, b->top = t;
  };
  auto pt = [f](std::pair<int, int>* p) {
    p->first = static_cast<int>(static_cast<float>(p->first) * f);
    p->second = static_cast<int>(static_cast<float>(p->second) * f);
  };
  for (rec::RecSymbol& sym : *symbols) box(&sym.box);
  for (rec::RecWord& w : *words) {
    if (w.brk && w.brk->box) box(&*w.brk->box);
    if (w.colors) {
      for (auto& p : w.colors->foreground_samples) pt(&p);
      for (auto& p : w.colors->background_samples) pt(&p);
    }
  }
}

// LanguageBasedLineRecognizerConfigSelector::Select: the recognizer key for a line after the first pass.
// recognizer_script_id is taken as is for CUSTOM (101) content. Otherwise, for each language (one per line here),
// with key suffixes "-t-i0-handwrit:vert", ":vert", "-t-i0-handwrit" (HANDWRITTEN_TEXT / top-to-bottom lines
// only) and "": the tag itself, its language subtag (not und/mul), its most likely script, and Hani for
// Hans/Hant. The shipped keys are bare script codes and "und", so in practice only the "" suffix and the tag or
// its script can match. Unused selector options: the low-confidence key (fields not in the shipped config).
// nullopt: the original's error status ("-t-i0-handwrit needs to come at the very end"), which fails the OCR call.
std::optional<std::string> SelectRecognizerKey(const std::map<std::string, std::string>& keys,
                                               const std::string& default_key, const Entity& e) {
  if (e.recognizer_script_id && !e.recognizer_script_id->empty() && e.content_type == 101 &&
      keys.count(*e.recognizer_script_id))
    return *e.recognizer_script_id;
  static const std::string kHandwrit = "-t-i0-handwrit";
  for (std::string lang : e.languages) {
    bool handwritten = e.content_type == 1;
    const size_t h = lang.rfind(kHandwrit);
    if (h != std::string::npos) {
      if (h + kHandwrit.size() != lang.size()) return std::nullopt;
      lang.resize(h);
      handwritten = true;
    }
    std::vector<std::string> suffixes = {""};
    if (handwritten) suffixes.push_back(kHandwrit);
    if (e.writing_direction == 2) {
      suffixes.push_back(":vert");
      if (handwritten) suffixes.push_back(kHandwrit + ":vert");
    }
    std::reverse(suffixes.begin(), suffixes.end());
    std::string language = lang.substr(0, lang.find_first_of("-_"));
    for (char& c : language) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    const std::string script = rec::MostLikelyScript(lang);
    for (const std::string& s : suffixes) {
      if (keys.count(lang + s)) return lang + s;
      if (language != "und" && language != "mul" && keys.count(language + s)) return language + s;
      if (keys.count(script + s)) return script + s;
      if ((script == "Hans" || script == "Hant") && keys.count("Hani" + s)) return "Hani" + s;
    }
  }
  return default_key;
}

GrayImage Rotate180(const GrayImage& img) {
  GrayImage out = img;
  std::reverse(out.data.begin(), out.data.end());
  return out;
}

}  // namespace

std::optional<std::string> LineRecognizer::SelectKey(const Entity& line) const {
  return SelectRecognizerKey(config_files_, default_key_, line);
}

bool LineRecognizer::Recognize(const RgbaImage& rgba, const GrayImage& gray, PageLayout* layout) {
  (void)rgba;
  std::vector<int> line_idx;
  for (size_t i = 0; i < layout->entities.size(); i++)
    if (layout->entities[i].type == EntityType::kLine) line_idx.push_back(static_cast<int>(i));
  std::vector<LineState> state(line_idx.size());

  std::optional<std::vector<DetectorPyramidLevel>> pyramid;  // built for the first line of large text

  // Recognizes `which` lines with recognizer `key`; first pass also handles the direction head.
  auto run_pass = [&](const std::string& key, const std::vector<int>& which) {
    rec::ScriptRecognizer* r = Get(key);
    if (!r) return false;
    const rec::Config& cfg = r->config();
    std::vector<GrayImage> images, scaled;
    for (int w : which) {
      // Later passes cut the line from its pass-1 box and turn the image when the direction head flipped the
      // box (not the same pixels as cutting the flipped box: that is off by one pixel in x and y).
      const Entity& e = layout->entities[line_idx[w]];
      LineState& st = state[w];
      if (st.key.empty() && e.box) {
        const int dis = e.detection ? e.detection->detection_image_scale : 1000;
        const DetectorPyramidLevel* level = nullptr;
        if (dis > 0 && dis < 1000) {
          if (!pyramid) pyramid = DetectorPyramid(*res_, gray);
          for (const DetectorPyramidLevel& l : *pyramid)
            if (l.image && static_cast<int>(l.scale * 1000.0f) == dis) level = &l;
        }
        if (level) {
          // The detection box keeps the curve in double precision (the polygon's points are floats).
          RotatedBox box = *e.box;
          if (box.curve && e.detection->detection_box.curve) {
            box.curve->points = e.detection->detection_box.curve->points;
            box.curve->thickness = e.detection->detection_box.curve->thickness;
          }
          const RotatedBox crop = PyramidCropBox(box, level->scale);
          st.base_image = crop.curve ? rec::CurvedLineImage(*level->image, *crop.curve, false)
                                     : rec::LineImage(*level->image, crop);
          st.box_scale = 1000.0f / static_cast<float>(dis);
        } else {
          st.base_image = rec::LineImage(gray, *e.box);
        }
      }
      images.push_back(st.flipped ? Rotate180(st.base_image) : st.base_image);
      scaled.push_back(images.back().width > 0 ? rec::ScaleToHeight(images.back()) : GrayImage());
    }
    const std::vector<rec::LineTensors> tensors = r->Run(scaled);
    if (tensors.size() != which.size()) return false;
    for (size_t k = 0; k < which.size(); k++) {
      Entity& e = layout->entities[line_idx[which[k]]];
      LineState& st = state[which[k]];
      const rec::LineTensors& L = tensors[k];
      st.key = key;
      st.image = images[k];
      if (L.frames == 0) continue;
      bool reversed = false;
      if (!L.direction.empty()) {
        const int d = rec::Argmax(L.direction, 4);
        const RotatedBox rotated = rec::RotateClockwise(*e.box, d);
        if (d == 1 || d == 2) {
          e.box = rec::RotateClockwise(*e.box, 2);
          st.flipped = !st.flipped;
          reversed = true;
        }
        e.orientation = rec::OrientationFromAngle(rotated.angle);
        e.writing_direction = (d & ~2) == 1 ? 2 : 0;
        e.textline_order = (d & ~2) == 1 ? 1 : 2;
        e.predicted_direction = d;
        e.direction_confidence = L.direction[d];
        if (e.detection) e.detection->garbage_prob = L.direction[4];
      }
      st.decode = rec::DecodeLine(L, *r, images[k], reversed);
      {  // StandardTextReorderer::JoinAtomsToSymbols (Indic symbol manipulators)
        const size_t before = st.decode.symbols.size();
        rec::JoinSymbolsLogical(&st.decode.symbols);
        if (st.decode.symbols.size() != before) {
          double sum = 0;
          for (const rec::RecSymbol& s : st.decode.symbols) sum += s.confidence;
          st.decode.confidence =
              st.decode.symbols.empty() ? 0.0f : static_cast<float>(sum / static_cast<double>(st.decode.symbols.size()));
        }
      }
      // StandardTextReorderer::SymbolsToResult: recognition (visual) order -> logical order.
      {
        std::vector<std::string> texts;
        for (const rec::RecSymbol& s : st.decode.symbols) texts.push_back(s.text);
        std::vector<int> offsets;
        rec::SymbolsVisualToLogical(texts, cfg.base_dir_rtl, &st.decode.text, &offsets);
        if (std::find(offsets.begin(), offsets.end(), -1) == offsets.end()) {
          std::vector<int> perm(offsets.size());
          for (size_t i = 0; i < perm.size(); i++) perm[i] = static_cast<int>(i);
          std::stable_sort(perm.begin(), perm.end(), [&](int a, int b) { return offsets[a] < offsets[b]; });
          std::vector<rec::RecSymbol> sorted;
          for (int i : perm) sorted.push_back(std::move(st.decode.symbols[i]));
          st.decode.symbols = std::move(sorted);
        }
      }
      const bool rtl = cfg.base_dir_rtl && rec::HasAnyRtl(st.decode.text);
      if (rtl) e.writing_direction = 1;
      if (!L.content.empty() || !e.content_type) e.content_type = 0;  // script passes keep the und prediction
      if (!L.content.empty()) {
        const int c = rec::Argmax(L.content, static_cast<int>(L.content.size()));
        e.content_type = c < static_cast<int>(cfg.content_type_names.size()) &&
                                 cfg.content_type_names[c] == "HANDWRITTEN_TEXT"
                             ? 1
                             : 0;
        e.type_confidence = L.content[c];
      }
      std::string language;
      std::optional<std::string> predicted_script;
      if (!L.script.empty()) {
        const int s = rec::Argmax(L.script, static_cast<int>(L.script.size()));
        const std::string name = s < static_cast<int>(cfg.script_names.size()) ? cfg.script_names[s] : "";
        predicted_script = name;
        if (std::find(cfg.supported_scripts.begin(), cfg.supported_scripts.end(), name) !=
            cfg.supported_scripts.end())
          language = langid_ ? langid_->Identify(st.decode.text) : "";
        else
          language = name;
      } else {
        language = langid_ ? langid_->Identify(st.decode.text) : "";
      }
      e.confidence = st.decode.confidence;
      e.languages = {language};
      e.recognizer_script_id = cfg.script_id.empty() ? key : cfg.script_id;
      AksaraResult a;
      if (!st.decode.symbols.empty()) {  // a line decoded to nothing has neither (arc_gentle_180)
        a.text = st.decode.text;
        a.confidence = st.decode.confidence;
      }
      a.language = language;
      a.predicted_script = predicted_script;
      if (cfg.base_dir_rtl) a.right_to_left = rtl;  // set by RTL recognizers: paragraph direction of the text
      e.aksara = a;
      st.words = rec::BuildWords(st.decode.symbols, st.decode.text);
      if (!rtl) {
        rec::RecomputeSpaceWordBreakBoxes(st.decode.symbols, e.writing_direction == 2, &st.words);
        rec::EstimateWordColors(images[k], st.decode.symbols, &st.words);
      }
      if (st.box_scale != 1.0f) ScaleLineBoxes(st.box_scale, &st.decode.symbols, &st.words);
    }
    return true;
  };

  // Lines in hierarchy order (grouped under their parents, parents in entity order), as goodoc::PageLayout
  // traverses them.
  std::vector<int> order(line_idx.size());
  for (size_t i = 0; i < order.size(); i++) order[i] = static_cast<int>(i);
  auto parent_rank = [&](int k) {
    const int p = layout->entities[line_idx[k]].parent;
    return p >= 0 && p < static_cast<int>(layout->entities.size()) ? p : line_idx[k];
  };
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return parent_rank(a) < parent_rank(b); });
  if (!run_pass(default_key_, order)) return false;
  if ((light_ ? 1 : maximum_iterations_) > 1) {
    // Second pass: lines whose language selects another recognizer are re-recognized by it; the multi-pass
    // mutator moves them behind the others, grouped by recognizer key.
    std::map<std::string, std::vector<int>> second;
    std::vector<int> kept;
    // Debug (trace/select_inject.py does the same to the original at Select): SCREENAI_DEBUG_SELECT_LANG=<tag>
    // replaces the language of every line before its recognizer is selected; "<n>:<tag>,..." only that of the
    // n-th line in selection order.
    const char* inject = std::getenv("SCREENAI_DEBUG_SELECT_LANG");
    int select_call = 0;
    for (int i : order) {
      Entity& e = layout->entities[line_idx[i]];
      if (inject) {
        const std::string spec = inject;
        std::optional<std::string> tag;
        if (spec.find(':') == std::string::npos) {
          tag = spec;
        } else {
          size_t p = 0;
          while (p <= spec.size()) {
            size_t q = spec.find(',', p);
            if (q == std::string::npos) q = spec.size();
            const std::string item = spec.substr(p, q - p);
            const size_t c = item.find(':');
            if (c != std::string::npos && std::atoi(item.substr(0, c).c_str()) == select_call) tag = item.substr(c + 1);
            p = q + 1;
          }
        }
        if (tag && !e.languages.empty()) e.languages[0] = *tag;
      }
      select_call++;
      const std::optional<std::string> key = SelectKey(e);
      if (!key) return false;
      if (*key != state[i].key) {
        second[*key].push_back(i);
      } else {
        kept.push_back(i);
      }
    }
    order = kept;
    for (const auto& [key, which] : second) {
      if (!run_pass(key, which)) return false;
      order.insert(order.end(), which.begin(), which.end());
    }
  }

  // Assemble: non-line entities first, then each line with its words and symbols.
  PageLayout out;
  out.width = layout->width, out.height = layout->height;
  out.language_mix = layout->language_mix;
  out.creator_id = layout->creator_id;
  std::vector<int> remap(layout->entities.size(), -1);
  for (size_t i = 0; i < layout->entities.size(); i++)
    if (layout->entities[i].type != EntityType::kLine) {
      remap[i] = static_cast<int>(out.entities.size());
      out.entities.push_back(layout->entities[i]);
    }
  int next = static_cast<int>(out.entities.size());
  for (int k : order) {
    remap[line_idx[k]] = next++;
    for (const auto& w : state[k].words) next += 1 + static_cast<int>(w.symbols.size());
  }
  for (Entity& e : out.entities)
    if (e.parent >= 0) e.parent = remap[e.parent];
  for (int k : order) {
    Entity line = layout->entities[line_idx[k]];
    if (line.parent >= 0) line.parent = remap[line.parent];
    const int line_id = static_cast<int>(out.entities.size());
    // Vertical (top-to-bottom) lines: the recognizer worked on the line rotated to horizontal; the page boxes of
    // the line and its descendants are turned upright again (RotateOrientationClockwise by 3 quadrants).
    const bool vertical = line.writing_direction == 2;
    auto upright = [&](const RotatedBox& b) { return vertical ? rec::RotateClockwise(b, 3) : b; };
    // Vertical lines re-recognized in a later pass map their boxes from the upright line box of the first pass
    // turned back by one quadrant: the angle round trip moves the page angle of every symbol by an ulp, which shows
    // in the averaged word angles (lldb on FindContainingBoxWithAverageAngle, ja_vertical turned by 180 degrees).
    // Lines kept from the first pass map from their box as recognized. For curved lines the round trip rounds the
    // curve to float (testdata/extra_fuzz/curve_2pass).
    RotatedBox lb = line.box ? *line.box : RotatedBox();
    if (vertical && state[k].key != default_key_)
      lb = rec::RotateClockwise(rec::RotateClockwise(lb, 3), 1);
    // Curved lines: line-image boxes follow the mid-line curve back onto the page.
    std::optional<rec::CurveMapper> mapper;
    if (lb.curve) mapper.emplace(*lb.curve);
    auto to_page = [&](const RotatedBox& rel) {
      return mapper ? rec::CurveToPage(*mapper, rel) : rec::ToPage(lb, rel);
    };
    if (line.box) line.box = upright(*line.box);
    out.entities.push_back(line);
    const LineState& st = state[k];
    for (const rec::RecWord& w : st.words) {
      Entity we;
      we.type = EntityType::kWord;
      we.parent = line_id;
      we.content_type = line.content_type;
      we.orientation = line.orientation;
      we.writing_direction = w.writing_direction && !vertical ? w.writing_direction : line.writing_direction;
      we.textline_order = line.textline_order;
      we.confidence = w.confidence;
      we.languages = line.languages;
      we.text = w.text;
      std::vector<RotatedBox> child_boxes;
      for (int si : w.symbols) child_boxes.push_back(to_page(st.decode.symbols[si].box));
      we.box = upright(rec::ContainingBoxWithAverageAngle(child_boxes));
      if (w.brk) {
        WordBreak b;
        b.type = w.brk->type;
        if (w.brk->box) b.box = upright(to_page(*w.brk->box));
        we.word_break = b;
      }
      if (w.colors) {
        Colors c = *w.colors;
        auto pt = [&](std::pair<int, int> p) {
          RotatedBox pb;
          pb.left = p.first, pb.top = p.second, pb.width = 1, pb.height = 1;
          const RotatedBox q = to_page(pb);
          return std::make_pair(q.left, q.top);
        };
        for (auto& p : c.foreground_samples) p = pt(p);
        for (auto& p : c.background_samples) p = pt(p);
        we.colors = c;
      }
      const int word_id = static_cast<int>(out.entities.size());
      out.entities.push_back(std::move(we));
      for (size_t j = 0; j < w.symbols.size(); j++) {
        const rec::RecSymbol& s = st.decode.symbols[w.symbols[j]];
        Entity se;
        se.type = EntityType::kSymbol;
        se.parent = word_id;
        se.content_type = line.content_type;
        se.confidence = s.confidence;
        se.text = s.text;
        if (s.mirrored) se.writing_direction = 1;
        se.box = upright(child_boxes[j]);
        out.entities.push_back(std::move(se));
      }
    }
  }
  *layout = std::move(out);
  return true;
}

}  // namespace screenai
