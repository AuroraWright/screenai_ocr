// SPDX-License-Identifier: MPL-2.0
// Reader for the recognizer's character n-gram language models (<script>_lm.fst): OpenFst files of type
// "compact_lm" (speech::CompactLmFst), an fst::NGramFst (LOUDS-encoded backoff n-gram model) over 16-bit "short"
// labels with 8-bit quantized weights, plus a label bitmap and a quantized unigram table.
//
// File layout (little endian):
//   OpenFst header: i32 magic 0x7eb2fdd6, str fst_type "compact_lm", str arc_type "standard", i32 version,
//                   i32 flags, u64 properties, i64 start, i64 num_states, i64 num_arcs   (str = i32 len + bytes)
//   i32 num_labels      size of the full (symbol table) label space
//   i32 num_unigrams    labels that are not in the n-gram vocabulary (weight table below)
//   u8  (unused here)
//   f32 weight_scale    weight = weight_scale * q for a quantized weight q; q == 254 means +infinity
//   u64 ngram_size
//   ngram_size bytes    fst::NGramFst<ShortArc> data (num_states, num_futures, num_final, context/future/final
//                       bitmaps, context/future words as u16, backoff/final/future weights as u8)
//   pad to 8 bytes
//   u64[ceil(num_labels/64)]  bitmap: bit L set <=> full label L is in the n-gram vocabulary; its short label is
//                             rank1(L) (number of set bits below L)
//   u8[num_unigrams]          weight of label L not in the vocabulary at index L - rank1(L); its arc from the
//                             unigram state loops back to state 0
//
// State 0 is the unigram (root) state; it has no backoff arc. Lookups follow speech::CompactLmFstMatcher.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace screenai::lm {

// fst::BitmapIndex equivalent: rank/select over a borrowed bit array.
class BitmapIndex {
 public:
  void Build(const uint64_t* bits, size_t num_bits);
  size_t num_bits() const { return num_bits_; }
  bool Get(size_t i) const { return (bits_[i >> 6] >> (i & 63)) & 1; }
  size_t Rank1(size_t end) const;  // number of ones in [0, end)
  size_t Rank0(size_t end) const { return end - Rank1(end); }
  size_t Select1(size_t r) const;  // position of the r-th one (0-based), num_bits() if none
  size_t Select0(size_t r) const;  // position of the r-th zero (0-based), num_bits() if none
  std::pair<size_t, size_t> Select0s(size_t r) const { return {Select0(r), Select0(r + 1)}; }

 private:
  const uint64_t* bits_ = nullptr;
  size_t num_bits_ = 0;
  std::vector<uint64_t> rank_;  // ones before each 64-bit word, plus total
};

class CompactLmFst {
 public:
  struct Arc {
    int ilabel = -1;
    float weight = 0;
    int nextstate = -1;
  };

  static std::unique_ptr<CompactLmFst> Read(std::string_view file, std::string* error);

  int Start() const { return start_; }
  int num_labels() const { return num_labels_; }

  // Arc with full label `label` leaving `state` (CompactLmFstMatcher::Find(label) followed by Value()).
  // At the unigram state an arc of weight +inf counts as missing. Returns false if there is none.
  bool FindLabel(int state, int label, Arc* arc) const;
  // Backoff (epsilon) arc of `state`; false for the unigram state.
  bool FindBackoff(int state, Arc* arc) const;

 private:
  float Weight(uint8_t q) const;
  // NGramFstImpl helpers (OpenFst ngram-fst.h)
  void SetInstFuture(int state, size_t* offset, size_t* num_futures) const;
  size_t Node(int state) const { return context_index_.Select1(state); }
  void Context(size_t node, std::vector<uint16_t>* context) const;
  int Transition(const std::vector<uint16_t>& context, uint16_t future) const;

  std::string data_;  // 8-byte aligned copy of the payload after the header
  int start_ = 0;
  int num_labels_ = 0;
  int num_unigrams_ = 0;
  float scale_ = 0;
  uint64_t num_states_ = 0, num_futures_ = 0, num_final_ = 0;
  const uint16_t* context_words_ = nullptr;
  const uint16_t* future_words_ = nullptr;
  const uint8_t* backoff_ = nullptr;
  const uint8_t* future_probs_ = nullptr;
  const uint8_t* unigram_weights_ = nullptr;
  BitmapIndex context_index_, future_index_, final_index_, label_index_;
  std::pair<size_t, size_t> select_root_;
  const uint16_t* root_children_ = nullptr;
  size_t root_offset_ = 0, root_num_futures_ = 0;
};

}  // namespace screenai::lm
