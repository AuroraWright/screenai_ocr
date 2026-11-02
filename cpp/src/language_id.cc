#include "language_id.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

#include "flatbuffers/flexbuffers.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/mutable_op_resolver.h"
#include "tensorflow/lite/string_util.h"
#include "tflite_model.h"

namespace screenai {
namespace {

#include "rune_tables.inc"

// ---- Plan 9 libutf (charntorune / runetochar / isalpharune / tolowerrune) as linked into the original ----

constexpr int kRuneError = 0xFFFD;
constexpr int kRuneDelta = 1048576;  // table deltas are stored biased by 2^20

// Decodes one rune from s[0..n). Returns the byte length (0 for a truncated sequence) and sets *r
// (kRuneError on any malformed input).
int CharNToRune(int* r, const unsigned char* s, int n) {
  if (n < 1) return *r = kRuneError, 0;
  const unsigned c = s[0];
  if (c < 0x80) return *r = c, 1;
  if (n == 1) return *r = kRuneError, 0;
  const unsigned c1 = s[1] ^ 0x80;
  if (c1 > 0x3F) return *r = kRuneError, 1;
  if (c <= 0xDF) {
    if (c >= 0xC0) {
      const unsigned v = ((c & 0x1F) << 6) | c1;
      if (v >= 0x80) return *r = static_cast<int>(v), 2;
    }
    return *r = kRuneError, 1;
  }
  if (n < 3) return *r = kRuneError, 0;
  const unsigned c2 = s[2] ^ 0x80;
  if (c2 > 0x3F) return *r = kRuneError, 1;
  if (c <= 0xEF) {
    const unsigned v = ((c & 0xF) << 12) | (c1 << 6);
    if (v >= 0x800) return *r = static_cast<int>(v | c2), 3;
    return *r = kRuneError, 1;
  }
  if (n == 3) return *r = kRuneError, 0;
  const unsigned c3 = s[3] ^ 0x80;
  if (c3 > 0x3F || c > 0xF7) return *r = kRuneError, 1;
  const unsigned v = (((c & 7) << 12) | (c1 << 6) | c2) << 6;
  if (v - 0x110000u < 0xFFF00000u) return *r = kRuneError, 1;
  return *r = static_cast<int>(v | c3), 4;
}

int RuneToChar(char* out, unsigned c) {
  if (c <= 0x7F) return out[0] = static_cast<char>(c), 1;
  if (c <= 0x7FF) {
    out[0] = static_cast<char>(0xC0 | (c >> 6));
    out[1] = static_cast<char>(0x80 | (c & 0x3F));
    return 2;
  }
  if (c > 0x10FFFF) c = kRuneError;
  if (c <= 0xFFFF) {
    out[0] = static_cast<char>(0xE0 | (c >> 12));
    out[1] = static_cast<char>(0x80 | ((c >> 6) & 0x3F));
    out[2] = static_cast<char>(0x80 | (c & 0x3F));
    return 3;
  }
  out[0] = static_cast<char>(0xF0 | (c >> 18));
  out[1] = static_cast<char>(0x80 | ((c >> 12) & 0x3F));
  out[2] = static_cast<char>(0x80 | ((c >> 6) & 0x3F));
  out[3] = static_cast<char>(0x80 | (c & 0x3F));
  return 4;
}

// Plan 9 rbsearch: last entry (of n entries, ne ints each) whose first value is <= c, or null.
const int* RBSearch(int c, const int* t, int n, int ne) {
  while (n > 1) {
    const int m = n >> 1;
    const int* p = t + m * ne;
    if (c >= p[0]) {
      t = p;
      n -= m;
    } else {
      n = m;
    }
  }
  return n && c >= t[0] ? t : nullptr;
}

template <size_t N>
constexpr int Count(const int (&)[N], int ne) {
  return static_cast<int>(N) / ne;
}

bool IsAlphaRune(int c) {
  const int* p = RBSearch(c, kIsalphar, Count(kIsalphar, 2), 2);
  if (p && c >= p[0] && c <= p[1]) return true;
  p = RBSearch(c, kIsalphas, Count(kIsalphas, 1), 1);
  return p && c == p[0];
}

int ToLowerRune(int c) {
  const int* p = RBSearch(c, kTolowerr, Count(kTolowerr, 3), 3);
  if (p && c >= p[0] && c <= p[1]) return c + p[2] - kRuneDelta;
  p = RBSearch(c, kTolowerp, Count(kTolowerp, 3), 3);
  if (p && c >= p[0] && c <= p[1] && !((c - p[0]) & 1)) return c + p[2] - kRuneDelta;
  p = RBSearch(c, kTolowers, Count(kTolowers, 2), 2);
  if (p && c == p[0]) return c + p[1] - kRuneDelta;
  return c;
}

// expander::predictondevice::LowercaseUnicodeStr. (The original loops forever on a truncated trailing sequence;
// here such bytes are copied as U+FFFD and decoding stops.)
std::string LowercaseUnicodeStr(std::string_view s) {
  std::string out;
  const auto* p = reinterpret_cast<const unsigned char*>(s.data());
  const int n = static_cast<int>(s.size());
  for (int i = 0; i < n;) {
    int r;
    int len = CharNToRune(&r, p + i, n - i);
    if (IsAlphaRune(r)) r = ToLowerRune(r);
    char buf[4];
    out.append(buf, RuneToChar(buf, static_cast<unsigned>(r)));
    if (len == 0) break;
    i += len;
  }
  return out;
}

// util_hash::MurmurHash64WithSeed (MurmurHash64A).
uint64_t MurmurHash64WithSeed(const char* data, size_t len, uint64_t seed) {
  constexpr uint64_t m = 0xC6A4A7935BD1E995ULL;
  uint64_t h = seed ^ (len * m);
  const size_t n8 = len & ~size_t{7};
  for (size_t i = 0; i < n8; i += 8) {
    uint64_t k;
    std::memcpy(&k, data + i, 8);
    k *= m;
    k ^= k >> 47;
    k *= m;
    h ^= k;
    h *= m;
  }
  if (len & 7) {
    uint64_t t = 0;
    for (size_t i = len & 7; i-- > 0;) t = (t << 8) | static_cast<unsigned char>(data[n8 + i]);
    h ^= t;
    h *= m;
  }
  h ^= h >> 47;
  h *= m;
  h ^= h >> 47;
  return h;
}

// UTF8StrLen(absl::string_view): bytes that are not continuation bytes.
int Utf8StrLen(std::string_view s) {
  int n = 0;
  for (char c : s) n += static_cast<signed char>(c) >= -64;
  return n;
}

// aksara::util::CountObsoleteRussianChars: matches of [IiѢѣѲѳѴѵ] plus matches of \pL[ъЪ]\PL (non-overlapping).
int CountObsoleteRussianChars(std::string_view s) {
  std::vector<int> runes;
  const auto* p = reinterpret_cast<const unsigned char*>(s.data());
  for (int i = 0, n = static_cast<int>(s.size()); i < n;) {
    int r;
    const int len = CharNToRune(&r, p + i, n - i);
    runes.push_back(r);
    i += len ? len : n;
  }
  int count = 0;
  for (int r : runes)
    count += r == 'I' || r == 'i' || r == 0x462 || r == 0x463 || r == 0x472 || r == 0x473 || r == 0x474 || r == 0x475;
  for (size_t i = 0; i + 2 < runes.size();) {
    if (IsAlphaRune(runes[i]) && (runes[i + 1] == 0x44A || runes[i + 1] == 0x42A) && !IsAlphaRune(runes[i + 2])) {
      count++;
      i += 3;
    } else {
      i++;
    }
  }
  return count;
}

// google_ocr::LangId::SanitizeSaftLanguage.
std::string SanitizeSaftLanguage(std::string_view text, std::string_view lang, float confidence) {
  if (confidence < 0.2 || Utf8StrLen(text) < 5) return "und";
  if (lang.size() >= 5) {
    if (lang.substr(lang.size() - 5) == "-Latn") return "und";
  } else if (lang == "ru") {
    if (static_cast<double>(Utf8StrLen(text)) * 0.005 < static_cast<double>(CountObsoleteRussianChars(text)))
      return "ru-PETR1708";
  }
  return std::string(lang);
}

// ---- NGramHash custom op (learning/expander/pod/deep_pod/tflite_handlers/ngram_hash.cc) ----

struct NGramHashParams {
  uint64_t seed = 0;
  std::vector<int> ngram_lengths, vocab_sizes;
  int max_splits = 128;
  bool lowercase = true;
};

void* NGramInit(TfLiteContext*, const char* buffer, size_t length) {
  auto* p = new NGramHashParams;
  const auto m = flexbuffers::GetRoot(reinterpret_cast<const uint8_t*>(buffer), length).AsMap();
  p->seed = static_cast<uint64_t>(m["seed"].AsInt64());
  for (const char* key : {"ngram_lengths", "vocab_sizes"}) {
    const auto v = m[key].AsTypedVector();
    auto& dst = key[0] == 'n' ? p->ngram_lengths : p->vocab_sizes;
    for (size_t i = 0; i < v.size(); i++) dst.push_back(static_cast<int>(v[i].AsInt64()));
  }
  if (!m["max_splits"].IsNull()) p->max_splits = static_cast<int>(m["max_splits"].AsInt64());
  if (!m["lowercase_input"].IsNull()) p->lowercase = m["lowercase_input"].AsUInt64() != 0;
  return p;
}

void NGramFree(TfLiteContext*, void* buffer) { delete static_cast<NGramHashParams*>(buffer); }

TfLiteStatus NGramPrepare(TfLiteContext* context, TfLiteNode* node) {
  TfLiteTensor* output = tflite::GetOutput(context, node, 0);
  TF_LITE_ENSURE(context, output != nullptr);
  tflite::SetTensorToDynamic(output);
  return kTfLiteOk;
}

TfLiteStatus NGramEval(TfLiteContext* context, TfLiteNode* node) {
  const auto* p = static_cast<const NGramHashParams*>(node->user_data);
  const TfLiteTensor* input = tflite::GetInput(context, node, 0);
  TF_LITE_ENSURE(context, input != nullptr && input->bytes != 0);
  TF_LITE_ENSURE(context, !p->ngram_lengths.empty() && p->ngram_lengths.size() == p->vocab_sizes.size());
  TF_LITE_ENSURE(context, p->max_splits > 0);
  const tflite::StringRef s = tflite::GetString(input, 0);
  const LanguageIdentifier::Tokens tok =
      LanguageIdentifier::Tokenize(std::string_view(s.str, s.len), p->max_splits, p->lowercase);

  TfLiteTensor* output = tflite::GetOutput(context, node, 0);
  TF_LITE_ENSURE(context, output != nullptr && output->allocation_type == kTfLiteDynamic);
  const int num_ngrams = static_cast<int>(p->ngram_lengths.size());
  const int num_tokens = static_cast<int>(tok.spans.size());
  TfLiteIntArray* dims = TfLiteIntArrayCreate(3);
  dims->data[0] = 1;
  dims->data[1] = num_ngrams;
  dims->data[2] = num_tokens;
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, output, dims));
  TF_LITE_ENSURE(context, output->type == kTfLiteInt32);
  int32_t* out = output->data.i32;
  for (int k = 0; k < num_ngrams; k++) {
    const int n = p->ngram_lengths[k];
    const auto vocab = static_cast<uint64_t>(static_cast<int64_t>(p->vocab_sizes[k]));
    for (int i = 0; i < num_tokens; i++) {
      size_t len = 0;
      for (int j = i; j < std::min(num_tokens, i + n); j++) len += tok.spans[j].second;
      const size_t off = tok.spans[i].first;
      len = std::min(len, tok.text.size() - off);
      out[k * num_tokens + i] = static_cast<int32_t>(MurmurHash64WithSeed(tok.text.data() + off, len, p->seed) % vocab + 1);
    }
  }
  return kTfLiteOk;
}

// ---- KmeansEmbeddingLookup custom op ----
// inputs: ids int32 [1, T], codes uint8 [V, M], codebook float [256, B]; output float [1, M * B]:
// mean over the ids (up to the first id 0) of concat_m codebook[codes[id, m]].

TfLiteStatus KmeansPrepare(TfLiteContext* context, TfLiteNode* node) {
  TfLiteTensor* output = tflite::GetOutput(context, node, 0);
  const TfLiteTensor* ids = tflite::GetInput(context, node, 0);
  const TfLiteTensor* codes = tflite::GetInput(context, node, 1);
  const TfLiteTensor* codebook = tflite::GetInput(context, node, 2);
  TF_LITE_ENSURE(context, output && ids && codes && codebook);
  TF_LITE_ENSURE_TYPES_EQ(context, ids->type, kTfLiteInt32);
  TF_LITE_ENSURE_TYPES_EQ(context, codes->type, kTfLiteUInt8);
  TF_LITE_ENSURE_TYPES_EQ(context, codebook->type, kTfLiteFloat32);
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteFloat32);
  TfLiteIntArray* dims = TfLiteIntArrayCreate(2);
  dims->data[0] = 1;
  dims->data[1] = codebook->dims->data[1] * codes->dims->data[1];
  return context->ResizeTensor(context, output, dims);
}

TfLiteStatus KmeansEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* ids = tflite::GetInput(context, node, 0);
  const TfLiteTensor* codes = tflite::GetInput(context, node, 1);
  const TfLiteTensor* codebook = tflite::GetInput(context, node, 2);
  TfLiteTensor* output = tflite::GetOutput(context, node, 0);
  TF_LITE_ENSURE(context, output && ids && codes && codebook);
  TF_LITE_ENSURE_MSG(context, ids->dims->data[0] == 1, "`batch_size` must be == 1.");
  const int num_ids = ids->dims->data[1];
  const int m = codes->dims->data[1];
  const int b = codebook->dims->data[1];
  float* out = output->data.f;
  std::fill(out, out + m * b, 0.0f);
  int count = 0;
  for (; count < num_ids; count++) {
    const int id = ids->data.i32[count];
    if (id == 0) break;
    const uint8_t* code = codes->data.uint8 + static_cast<int64_t>(id) * m;
    for (int j = 0; j < m; j++) {
      const float* row = codebook->data.f + code[j] * b;
      for (int e = 0; e < b; e++) out[j * b + e] += row[e];
    }
  }
  if (count >= 2) {
    const auto denom = static_cast<float>(count);
    for (int j = 0; j < m * b; j++) out[j] /= denom;
  }
  return kTfLiteOk;
}

// ---- SOFTMAX (float) as compiled into the original: tflite::optimized_ops::SoftmaxImpl over its Eigen version ----
// Our TFLite build links a newer Eigen whose pexp differs in the last bit, so the langid graph gets this kernel.

float BitsToFloat(uint32_t u) {
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// Eigen NEON pexp<Packet4f> lane (FMLA = fused multiply-add).
float EigenPexp(float x0) {
  const float x = std::min(std::max(x0, BitsToFloat(0xC2D00000)), BitsToFloat(0x42B1722D));  // [-104, 88.72283]
  const float t = x * BitsToFloat(0x3FB8AA3B);                                               // log2(e)
  const float m = __builtin_roundevenf(t);
  float r = std::fma(BitsToFloat(0xBF318000), m, x);  // -0.693359375
  r = std::fma(BitsToFloat(0x395E8083), m, r);        // 2.12194440e-4
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
  return std::max(out, x0);  // FMAX(out, x)
}

// Eigen redux<scalar_sum_op> over a contiguous float column (NEON, 4 packet accumulators).
float EigenSum(const float* v, int64_t n) {
  const auto addr = reinterpret_cast<uintptr_t>(v);
  int64_t start = static_cast<int64_t>(-(addr >> 2) & 3);
  if (start >= n || (addr & 3)) start = n;
  const int64_t avail = n - start;
  const int64_t aligned = avail & ~int64_t{3};
  if (aligned == 0) {
    float res = v[0];
    for (int64_t i = 1; i < n; i++) res += v[i];
    return res;
  }
  using P = std::array<float, 4>;
  auto load = [&](int64_t i) { return P{v[i], v[i + 1], v[i + 2], v[i + 3]}; };
  auto add = [](const P& a, const P& b) { return P{a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]}; };
  P r0 = load(start);
  const int64_t blocks = avail & ~int64_t{15};
  if (blocks) {
    P r1 = load(start + 4), r2 = load(start + 8), r3 = load(start + 12);
    for (int64_t i = start + 16; i < start + blocks; i += 16) {
      r0 = add(r0, load(i));
      r1 = add(r1, load(i + 4));
      r2 = add(r2, load(i + 8));
      r3 = add(r3, load(i + 12));
    }
    const int64_t rest = avail & 0xC;
    if (rest) {
      const int64_t i = start + blocks;
      r0 = add(r0, load(i));
      if (rest >= 8) r1 = add(r1, load(i + 4));
      if (rest == 12) r2 = add(r2, load(i + 8));
    }
    r0 = add(add(r3, r2), add(r1, r0));
  } else {
    if (aligned >= 5) r0 = add(r0, load(start + 4));
    if (aligned >= 9) r0 = add(r0, load(start + 8));
  }
  float res = (r0[0] + r0[1]) + (r0[2] + r0[3]);
  for (int64_t i = 0; i < start; i++) res += v[i];
  for (int64_t i = start + aligned; i < n; i++) res += v[i];
  return res;
}

TfLiteStatus SoftmaxPrepare(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = tflite::GetInput(context, node, 0);
  TfLiteTensor* output = tflite::GetOutput(context, node, 0);
  TF_LITE_ENSURE(context, input && output);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteFloat32);
  TF_LITE_ENSURE(context, input->dims->size >= 1);
  return context->ResizeTensor(context, output, TfLiteIntArrayCopy(input->dims));
}

TfLiteStatus SoftmaxEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = tflite::GetInput(context, node, 0);
  TfLiteTensor* output = tflite::GetOutput(context, node, 0);
  const float beta = static_cast<const TfLiteSoftmaxParams*>(node->builtin_data)->beta;
  const int depth = input->dims->data[input->dims->size - 1];
  const int64_t total = tflite::NumElements(input);
  const int64_t batches = depth ? total / depth : 0;
  const float* in = input->data.f;
  float* out = output->data.f;
  for (int64_t b = 0; b < batches; b++) {
    const float* row = in + b * depth;
    float mx = row[0];
    for (int i = 1; i < depth; i++) mx = std::max(mx, row[i]);
    for (int i = 0; i < depth; i++) out[b * depth + i] = (row[i] - mx) * beta;
  }
  // exp over the whole block: scalar libm expf until 16-byte aligned and for the last (n - head) % 4, packets between.
  const auto addr = reinterpret_cast<uintptr_t>(out);
  int64_t head = static_cast<int64_t>(-(addr >> 2) & 3);
  if (head >= total || (addr & 3)) head = total;
  for (int64_t i = 0; i < head; i++) out[i] = expf(out[i]);
  int64_t i = head;
  for (; i + 4 <= total; i += 4)
    for (int k = 0; k < 4; k++) out[i + k] = EigenPexp(out[i + k]);
  for (; i < total; i++) out[i] = expf(out[i]);
  for (int64_t b = 0; b < batches; b++) {
    const float scale = 1.0f / EigenSum(out + b * depth, depth);
    for (int k = 0; k < depth; k++) out[b * depth + k] *= scale;
  }
  return kTfLiteOk;
}

}  // namespace

LanguageIdentifier::Tokens LanguageIdentifier::Tokenize(std::string_view utf8_text, int max_splits, bool lowercase) {
  // expander::predictondevice::Tokenize(text, max_splits, /*split_on_non_alpha=*/true)
  std::string lowered;
  if (lowercase) {
    lowered = LowercaseUnicodeStr(utf8_text);
    utf8_text = lowered;
  }
  Tokens t;
  t.text = "^";
  t.spans.emplace_back(0, 1);
  const auto* p = reinterpret_cast<const unsigned char*>(utf8_text.data());
  const int n = static_cast<int>(utf8_text.size());
  for (int i = 0; i < n && static_cast<int>(t.spans.size()) + 1 < max_splits;) {
    int r;
    const int len = CharNToRune(&r, p + i, n - i);
    if (r == kRuneError) return Tokens();
    if (len == 0) break;
    t.spans.emplace_back(t.text.size(), IsAlphaRune(r) ? len : 1);
    if (IsAlphaRune(r))
      t.text.append(utf8_text.data() + i, len);
    else
      t.text += ' ';
    i += len;
  }
  t.spans.emplace_back(t.text.size(), 1);
  t.text += '$';
  return t;
}

void LanguageIdentifier::RegisterCustomOps(tflite::MutableOpResolver* resolver) {
  static TfLiteRegistration ngram = [] {
    TfLiteRegistration r{};
    r.init = NGramInit;
    r.free = NGramFree;
    r.prepare = NGramPrepare;
    r.invoke = NGramEval;
    return r;
  }();
  static TfLiteRegistration kmeans = [] {
    TfLiteRegistration r{};
    r.prepare = KmeansPrepare;
    r.invoke = KmeansEval;
    return r;
  }();
  static TfLiteRegistration softmax = [] {
    TfLiteRegistration r{};
    r.prepare = SoftmaxPrepare;
    r.invoke = SoftmaxEval;
    r.builtin_code = tflite::BuiltinOperator_SOFTMAX;
    return r;
  }();
  resolver->AddCustom("NGramHash", &ngram);
  resolver->AddCustom("KmeansEmbeddingLookup", &kmeans);
  resolver->AddBuiltin(tflite::BuiltinOperator_SOFTMAX, &softmax);
}

std::unique_ptr<LanguageIdentifier> LanguageIdentifier::Create(const Resources& res) {
  std::string bytes;
  if (!res.Read(kModelPath, &bytes)) return nullptr;
  return CreateFromModel(std::move(bytes));
}

std::unique_ptr<LanguageIdentifier> LanguageIdentifier::CreateFromModel(std::string model_bytes) {
  std::unique_ptr<LanguageIdentifier> lid(new LanguageIdentifier());
  // Builtin kernels only: the original runs the hybrid FULLY_CONNECTED layers on the reference/optimized
  // builtin kernels (its graph shows their quantization temporaries), not on XNNPACK.
  lid->model_ = TfliteModel::Create(std::move(model_bytes), 1, &RegisterCustomOps, /*use_xnnpack=*/false);
  if (!lid->model_ || !lid->model_->AllocateTensors()) return nullptr;
  const TfLiteTensor* labels = lid->model_->output("pod/IndexToClassID");
  if (!labels || labels->type != kTfLiteString) return nullptr;
  for (int i = 0, n = tflite::GetStringCount(labels); i < n; i++) {
    const tflite::StringRef s = tflite::GetString(labels, i);
    lid->labels_.emplace_back(s.str, s.len);
  }
  return lid;
}

LanguageIdentifier::~LanguageIdentifier() = default;

bool LanguageIdentifier::Scores(std::string_view utf8_text, std::vector<float>* softmax,
                                std::vector<float>* logits) const {
  std::lock_guard<std::mutex> lock(mu_);
  tflite::DynamicBuffer buf;
  buf.AddString(utf8_text.data(), utf8_text.size());
  buf.WriteToTensorAsVector(model_->input(0));
  if (!model_->Invoke()) return false;
  const TfLiteTensor* sm = model_->output("pod/Softmax");
  const TfLiteTensor* lg = model_->output("pod/batch_norm_2/batch_norm_core/add_1");
  if (!sm || !lg) return false;
  const int n = static_cast<int>(labels_.size());
  if (softmax) softmax->assign(sm->data.f, sm->data.f + n);
  if (logits) logits->assign(lg->data.f, lg->data.f + n);
  return true;
}

std::vector<LanguageIdentifier::Prediction> LanguageIdentifier::TopK(std::string_view utf8_text, int k) const {
  std::vector<float> scores;
  if (!Scores(utf8_text, &scores)) return {};
  std::vector<int> order(scores.size());
  for (size_t i = 0; i < order.size(); i++) order[i] = static_cast<int>(i);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return scores[a] > scores[b]; });
  std::vector<Prediction> out;
  for (int i = 0; i < std::min<int>(k, static_cast<int>(order.size())); i++)
    out.push_back({labels_[order[i]], scores[order[i]]});
  return out;
}

LanguageIdentifier::Result LanguageIdentifier::IdentifyLanguage(std::string_view utf8_text) const {
  std::vector<float> scores;
  if (!Scores(utf8_text, &scores) || scores.empty()) return {"und", 1.0f};
  // MobileLangIdV2::IdentifyLanguage: first strictly-greatest score starting from 0.
  std::string best;
  float conf = 0.0f;
  for (size_t i = 0; i < scores.size(); i++) {
    if (scores[i] > conf) {
      best = labels_[i];
      conf = scores[i];
    }
  }
  if (best == "unknown") best = "und";
  return {SanitizeSaftLanguage(utf8_text, best, conf), conf};
}

}  // namespace screenai
