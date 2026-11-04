// SPDX-License-Identifier: MPL-2.0
// CTC beam-search decoder with character n-gram LM and class prior: google_ocr::CTCDecoder (+ CTCBeamOps) as
// configured by GocrCTCDecoderRecognizer for the scripts that ship <script>_lm.fst/.syms, <script>_fst*_config.pb
// and a prior (arab, cyrl, deva, hani, jpan, knda, kore, mlym, taml, telu, thai). Scripts without an LM use greedy
// CTC decoding instead (not here).
//
// Decoding (per line, frames t = 0..num_frames-1, costs are negative log scores):
//   label selection: candidates (class, -logit) of frame t; keep those within label_selection_width of the best
//     and at most label_selection_size (libc++ nth_element + stable_partition), ordered by (cost, class).
//   expansion of every beam item by every candidate (float accumulators per item):
//     optical += -logit
//     blank:                 misc  = float(blank_cost + misc)
//     same class as before:  misc  = float(self_loop_cost + misc)          (CTC repeat, no LM step)
//     new class:             lm    = float(lm_cost * float(lm_weight) + lm)  lm_cost = LM -log p (double)
//                            misc  = float(insertion_penalty + misc)        (not for the eos class)
//                            prior = float(prior_weight * prior[class] + prior)
//   beam pruning on total = ((optical + lm) + misc) + prior: keep within beam_width of the best, at most
//     beam_size, ordered by (total, candidate index), then drop adjacent duplicates (same class and LM state).
//   After the last frame one more expansion with the end-of-sentence class (== blank for the shipped label maps,
//   so it only adds blank_cost). The best final item is traced back; collapsing repeats and blanks gives one
//   Label per emitted class: frame = frame of its smallest per-frame optical cost increment, costs = increments
//   (double) of the accumulators at that frame (first label: the accumulators themselves).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "lm_char_model.h"
#include "resources.h"

namespace screenai {

// google_ocr.CTCDecoderConfig beam/cost options.
struct CtcDecoderOptions {
  int beam_size = 0;
  double beam_width = 0;
  int label_selection_size = 0;
  double label_selection_width = 0;
  double lm_weight = 0, insertion_penalty = 0, prior_weight = 0, blank_cost = 0, self_loop_cost = 0;
};

// One line in the layout of google_ocr::CTCDecoderInput as built by GocrCTCDecoderRecognizer::UpdateLineResult:
// the stitched sparse (top-k) recognizer logits after google_ocr::GetLogits splits the {class, value} pairs.
struct CtcDecoderInput {
  int num_frames = 0;                // logits_lengths[n]: number of valid frames to decode
  int k = 0;                         // entries per frame (dequantize_topk = 5)
  const float* values = nullptr;     // [num_frames][k] dequantized logits
  const float* class_ids = nullptr;  // [num_frames][k] class index stored as float (rintf'ed); nullptr = dense
                                     // logits with k == num_classes() + 1 (column = class, last = blank)
};

// google_ocr::CTCDecoderOutput (label_info only; the lattice is disabled by the shipped configs).
// This is what recognition::CreateTextLineResultFromCTCDecoderOutput consumes after the confidence scorer
// (CTCDecoderConfidenceScorer, AvgLogits) has filled in `confidence` (the decoder itself leaves 1.0).
struct CtcDecoderOutput {
  struct Label {
    int label = 0;              // class id (index into the label map)
    int frame = 0;              // frame of the label (0-based, in decoder frames)
    double optical_cost = 0;    // NegativeLogitsScore increment at that frame
    double lm_cost = 0;         // weighted LM cost increment
    double prior_cost = 0;      // weighted log prior increment
    double confidence = 1.0;
  };
  std::vector<Label> labels;
};

class CtcBeamDecoder {
 public:
  // model_dir: e.g. "gocr/gocr_models/line_recognition_mobile_convnext320_omni"; script: "jpan", "thai", ... (reads
  // <script>_fst_config.pb or <script>_fst_2024_q3_config.pb) or the config file name itself. Loads the label map,
  // prior (prior_path / encoded_prior) and LM named by the config. Returns nullptr on error (message in *error).
  static std::unique_ptr<CtcBeamDecoder> Create(const Resources& res, const std::string& model_dir,
                                                const std::string& script, std::string* error = nullptr);

  // CTCDecoder::DecodeNth for one line.
  bool Decode(const CtcDecoderInput& in, CtcDecoderOutput* out) const;

  int num_classes() const { return blank_; }  // label map size; the CTC blank class id equals it
  int blank_id() const { return blank_; }
  const std::string& label(int class_id) const;  // label map name ("" if unknown)
  const std::string& recognizer_script_id() const { return script_id_; }  // e.g. "Jpan"
  const CtcDecoderOptions& options() const { return opt_; }
  const lm::FstCharLanguageModel& language_model() const { return *lm_; }
  const std::vector<float>& prior() const { return prior_; }

 private:
  struct Item;  // gocr_internal::BeamItemInfo
  struct BeamItem;
  void PruneIdCostPair(std::vector<Item>* items, int n, double width) const;
  void ExpandAndPruneBeam(const std::vector<BeamItem>& beam, const std::vector<Item>& labels,
                          std::vector<BeamItem>* next) const;

  CtcDecoderOptions opt_;
  float lm_weight_ = 0;
  int blank_ = 0;
  std::vector<std::string> labels_;
  std::vector<float> prior_;
  std::unique_ptr<lm::FstCharLanguageModel> lm_;
  std::string script_id_;
};

// ocr::photo::SparseDequantizeBatch for one frame of uint8 logits (TfliteModelPooledRunner with dequantize_topk):
// entries with q >= clamp(int(0.01f / scale + zero_point), 0, 255) are packed as (q << 16 | class), the k largest
// (libc++ nth_element, std::greater) are written as {class, (q - zero_point) * scale}; missing entries stay {0, 0}.
void SparseDequantizeFrame(const uint8_t* logits, int num_classes, int zero_point, float scale, int k,
                           float* out /* [k][2] */);

}  // namespace screenai
