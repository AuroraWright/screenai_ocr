#include "lm_compact_fst.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>

namespace screenai::lm {
namespace {

constexpr int32_t kFstMagic = 0x7eb2fdd6;
constexpr uint8_t kInfWeight = 254;

size_t Words(size_t bits) { return (bits + 63) >> 6; }

// Position of the r-th (0-based) set bit of w; r < popcount(w).
int SelectInWord(uint64_t w, int r) {
  for (; r > 0; r--) w &= w - 1;
  return std::countr_zero(w);
}

class ByteReader {
 public:
  explicit ByteReader(std::string_view s) : s_(s) {}
  template <typename T>
  bool Get(T* v) {
    if (s_.size() - pos_ < sizeof(T)) return false;
    memcpy(v, s_.data() + pos_, sizeof(T));
    pos_ += sizeof(T);
    return true;
  }
  bool GetString(std::string* v) {
    int32_t n;
    if (!Get(&n) || n < 0 || s_.size() - pos_ < static_cast<size_t>(n)) return false;
    v->assign(s_.data() + pos_, n);
    pos_ += n;
    return true;
  }
  size_t pos() const { return pos_; }

 private:
  std::string_view s_;
  size_t pos_ = 0;
};

}  // namespace

void BitmapIndex::Build(const uint64_t* bits, size_t num_bits) {
  bits_ = bits;
  num_bits_ = num_bits;
  const size_t n = Words(num_bits);
  rank_.assign(n + 1, 0);
  for (size_t i = 0; i < n; i++) {
    uint64_t w = bits[i];
    if (i + 1 == n && (num_bits & 63)) w &= (uint64_t{1} << (num_bits & 63)) - 1;
    rank_[i + 1] = rank_[i] + std::popcount(w);
  }
}

size_t BitmapIndex::Rank1(size_t end) const {
  const size_t w = end >> 6, b = end & 63;
  size_t r = rank_[w];
  if (b) r += std::popcount(bits_[w] & ((uint64_t{1} << b) - 1));
  return r;
}

size_t BitmapIndex::Select1(size_t r) const {
  if (r >= rank_.back()) return num_bits_;
  // last word index w with rank_[w] <= r
  const size_t w = std::upper_bound(rank_.begin(), rank_.end(), r) - rank_.begin() - 1;
  return (w << 6) + SelectInWord(bits_[w], static_cast<int>(r - rank_[w]));
}

size_t BitmapIndex::Select0(size_t r) const {
  if (r >= num_bits_ - rank_.back()) return num_bits_;
  // zeros before word w: 64 w - rank_[w] (non-decreasing in w)
  size_t lo = 0, hi = rank_.size() - 1;  // find last w with zeros(w) <= r
  while (hi - lo > 1) {
    const size_t mid = (lo + hi) / 2;
    if ((mid << 6) - rank_[mid] <= r) lo = mid; else hi = mid;
  }
  const size_t w = lo;
  return (w << 6) + SelectInWord(~bits_[w], static_cast<int>(r - ((w << 6) - rank_[w])));
}

std::unique_ptr<CompactLmFst> CompactLmFst::Read(std::string_view file, std::string* error) {
  auto fail = [error](const char* msg) {
    if (error) *error = msg;
    return nullptr;
  };
  ByteReader r(file);
  int32_t magic, version, flags;
  std::string fst_type, arc_type;
  uint64_t properties;
  int64_t start, num_states, num_arcs;
  if (!r.Get(&magic) || magic != kFstMagic) return fail("not an OpenFst file");
  if (!r.GetString(&fst_type) || !r.GetString(&arc_type)) return fail("truncated header");
  if (fst_type != "compact_lm") return fail("fst type is not compact_lm");
  if (!r.Get(&version) || !r.Get(&flags) || !r.Get(&properties) || !r.Get(&start) || !r.Get(&num_states) ||
      !r.Get(&num_arcs))
    return fail("truncated header");
  // Embedded symbol tables (flags 1, 2) are not used by the shipped models; 4 = data aligned to 16 bytes.
  if (flags & ~4) return fail("unsupported OpenFst header flags");
  auto fst = std::unique_ptr<CompactLmFst>(new CompactLmFst());
  int32_t num_labels, num_unigrams;
  uint8_t unused;
  float scale;
  uint64_t ngram_size;
  if (!r.Get(&num_labels) || !r.Get(&num_unigrams) || !r.Get(&unused) || !r.Get(&scale) || !r.Get(&ngram_size))
    return fail("truncated compact_lm header");
  size_t pos = r.pos();
  if (flags & 4) pos = (pos + 15) & ~size_t{15};
  const size_t label_words = Words(num_labels);
  const size_t payload = ((ngram_size + 7) & ~uint64_t{7}) + 8 * label_words + num_unigrams;
  if (pos > file.size() || file.size() - pos < payload) return fail("truncated compact_lm data");
  fst->data_.assign(payload + 8, '\0');
  // std::string storage of this size is heap allocated with >= 16-byte alignment
  char* base = fst->data_.data();
  memcpy(base, file.data() + pos, payload);
  fst->start_ = static_cast<int>(start);
  fst->num_labels_ = num_labels;
  fst->num_unigrams_ = num_unigrams;
  fst->scale_ = scale;

  const uint64_t* d = reinterpret_cast<const uint64_t*>(base);
  const uint64_t ns = d[0], nf = d[1], nfinal = d[2];
  fst->num_states_ = ns, fst->num_futures_ = nf, fst->num_final_ = nfinal;
  size_t off = 24;
  const uint64_t* context = reinterpret_cast<const uint64_t*>(base + off);
  off += 8 * Words(2 * ns + 1);
  const uint64_t* future = reinterpret_cast<const uint64_t*>(base + off);
  off += 8 * Words(ns + nf + 1);
  const uint64_t* final_bits = reinterpret_cast<const uint64_t*>(base + off);
  off += 8 * Words(ns);
  fst->context_words_ = reinterpret_cast<const uint16_t*>(base + off);
  off += 2 * (ns + 1);
  fst->future_words_ = reinterpret_cast<const uint16_t*>(base + off);
  off += 2 * nf;
  fst->backoff_ = reinterpret_cast<const uint8_t*>(base + off);
  off += ns + 1;
  off += nfinal;  // final probabilities (unused: end of text is scored as a label)
  fst->future_probs_ = reinterpret_cast<const uint8_t*>(base + off);
  off += nf + 1;
  if (off > ngram_size) return fail("inconsistent n-gram data size");
  fst->context_index_.Build(context, 2 * ns + 1);
  fst->future_index_.Build(future, ns + nf + 1);
  fst->final_index_.Build(final_bits, ns);
  fst->select_root_ = fst->context_index_.Select0s(0);
  if (fst->context_index_.Rank1(0) != 0 || fst->select_root_.first != 1 || !fst->context_index_.Get(2))
    return fail("malformed n-gram data");
  fst->root_children_ = fst->context_words_ + fst->context_index_.Rank1(2);

  const size_t lw = (ngram_size + 7) & ~uint64_t{7};
  fst->label_index_.Build(reinterpret_cast<const uint64_t*>(base + lw), num_labels);
  fst->unigram_weights_ = reinterpret_cast<const uint8_t*>(base + lw + 8 * label_words);
  fst->SetInstFuture(0, &fst->root_offset_, &fst->root_num_futures_);
  return fst;
}

float CompactLmFst::Weight(uint8_t q) const {
  return q == kInfWeight ? std::numeric_limits<float>::infinity() : scale_ * static_cast<float>(q);
}

void CompactLmFst::SetInstFuture(int state, size_t* offset, size_t* num_futures) const {
  const auto zeros = future_index_.Select0s(state);
  *num_futures = zeros.second - zeros.first - 1;
  *offset = future_index_.Rank1(zeros.first + 1);
}

void CompactLmFst::Context(size_t node, std::vector<uint16_t>* context) const {
  context->clear();
  while (node != 0) {
    context->push_back(context_words_[context_index_.Rank1(node)]);
    node = context_index_.Select1(context_index_.Rank0(node) - 1);
  }
}

int CompactLmFst::Transition(const std::vector<uint16_t>& context, uint16_t future) const {
  const uint16_t* children = root_children_;
  size_t num_children = select_root_.second - 2;
  const uint16_t* loc = std::lower_bound(children, children + num_children, future);
  if (loc == children + num_children || *loc != future) return static_cast<int>(context_index_.Rank1(0));
  size_t node = 2 + (loc - children);
  size_t node_rank = context_index_.Rank1(node);
  std::pair<size_t, size_t> zeros = node_rank == 0 ? select_root_ : context_index_.Select0s(node_rank);
  size_t first_child = zeros.first + 1;
  if (!context_index_.Get(first_child)) return static_cast<int>(context_index_.Rank1(node));
  size_t last_child = zeros.second - 1;
  for (int word = static_cast<int>(context.size()) - 1; word >= 0; --word) {
    children = context_words_ + context_index_.Rank1(first_child);
    const size_t n = last_child - first_child + 1;
    loc = std::lower_bound(children, children + n, context[word]);
    if (loc == children + n || *loc != context[word]) break;
    node = first_child + (loc - children);
    node_rank = context_index_.Rank1(node);
    zeros = node_rank == 0 ? select_root_ : context_index_.Select0s(node_rank);
    first_child = zeros.first + 1;
    if (!context_index_.Get(first_child)) break;
    last_child = zeros.second - 1;
  }
  return static_cast<int>(context_index_.Rank1(node));
}

bool CompactLmFst::FindLabel(int state, int label, Arc* arc) const {
  if (label <= 0 || label >= num_labels_) return false;
  arc->ilabel = label;
  if (state == 0) {
    // Unigram state: dense arcs, position label - 1 (ArcIterator<CompactLmFst> for state 0).
    if (!label_index_.Get(label)) {
      arc->weight = Weight(unigram_weights_[label - label_index_.Rank1(label)]);
      arc->nextstate = 0;
    } else {
      const size_t pos = static_cast<uint16_t>(label_index_.Rank1(label)) - size_t{1};
      const uint16_t short_label = future_words_[root_offset_ + pos];
      arc->weight = Weight(future_probs_[root_offset_ + pos]);
      arc->nextstate = Transition({}, short_label);
      arc->ilabel = static_cast<int>(label_index_.Select1(short_label));
    }
    return arc->weight != std::numeric_limits<float>::infinity();
  }
  if (!label_index_.Get(label)) return false;
  const uint16_t short_label = static_cast<uint16_t>(label_index_.Rank1(label));
  size_t offset, num_futures;
  SetInstFuture(state, &offset, &num_futures);
  const uint16_t* begin = future_words_ + offset;
  const uint16_t* it = std::lower_bound(begin, begin + num_futures, short_label);
  if (it == begin + num_futures || *it != short_label) return false;
  arc->weight = Weight(future_probs_[offset + (it - begin)]);
  std::vector<uint16_t> context;
  Context(Node(state), &context);
  arc->nextstate = Transition(context, short_label);
  return true;
}

bool CompactLmFst::FindBackoff(int state, Arc* arc) const {
  if (state == 0) return false;
  const size_t node = Node(state);
  arc->ilabel = 0;
  arc->nextstate = static_cast<int>(context_index_.Rank1(context_index_.Select1(context_index_.Rank0(node) - 1)));
  arc->weight = Weight(backoff_[state]);
  return true;
}

}  // namespace screenai::lm
