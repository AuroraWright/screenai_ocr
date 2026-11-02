#include "ctc_decoder.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <map>

#include "proto_wire.h"

namespace screenai {
namespace {

// libc++ std::__nth_element (non-branchless __sort3 / __selection_sort), reproduced so that the order of
// elements with equal keys matches the original on any standard library.
template <class T, class Less>
bool Sort3(T* x, T* y, T* z, Less c) {
  if (!c(*y, *x)) {
    if (!c(*z, *y)) return false;
    std::swap(*y, *z);
    if (c(*y, *x)) std::swap(*x, *y);
    return true;
  }
  if (c(*z, *y)) {
    std::swap(*x, *z);
    return true;
  }
  std::swap(*x, *y);
  if (c(*z, *y)) std::swap(*y, *z);
  return true;
}

template <class T, class Less>
void SelectionSort(T* first, T* last, Less c) {
  for (T* lm1 = last - 1; first != lm1; ++first) {
    T* m = first;
    for (T* i = first + 1; i != last; ++i)
      if (c(*i, *m)) m = i;
    if (m != first) std::swap(*first, *m);
  }
}

template <class T, class Less>
void NthElement(T* first, T* nth, T* last, Less comp) {
  constexpr ptrdiff_t kLimit = 7;
  while (true) {
    if (nth == last) return;
    const ptrdiff_t len = last - first;
    switch (len) {
      case 0:
      case 1:
        return;
      case 2:
        if (comp(*--last, *first)) std::swap(*first, *last);
        return;
      case 3: {
        T* m = first;
        Sort3(first, ++m, --last, comp);
        return;
      }
    }
    if (len <= kLimit) {
      SelectionSort(first, last, comp);
      return;
    }
    T* m = first + len / 2;
    T* lm1 = last;
    unsigned n_swaps = Sort3(first, m, --lm1, comp);
    T* i = first;
    T* j = lm1;
    if (!comp(*i, *m)) {
      bool found_guard = false;
      while (true) {
        if (i == --j) break;
        if (comp(*j, *m)) {
          found_guard = true;
          break;
        }
      }
      if (found_guard) {
        std::swap(*i, *j);
        ++n_swaps;
      } else {
        ++i;
        j = last;
        if (!comp(*first, *--j)) {
          while (true) {
            if (i == j) return;
            if (comp(*first, *i)) {
              std::swap(*i, *j);
              ++n_swaps;
              ++i;
              break;
            }
            ++i;
          }
        }
        if (i == j) return;
        while (true) {
          while (!comp(*first, *i)) ++i;
          do --j;
          while (comp(*first, *j));
          if (i >= j) break;
          std::swap(*i, *j);
          ++n_swaps;
          ++i;
        }
        if (nth < i) return;
        first = i;
        continue;
      }
    }
    ++i;
    if (i < j) {
      while (true) {
        while (comp(*i, *m)) ++i;
        do --j;
        while (!comp(*j, *m));
        if (i >= j) break;
        std::swap(*i, *j);
        ++n_swaps;
        if (m == i) m = j;
        ++i;
      }
    }
    if (i != m && comp(*m, *i)) {
      std::swap(*i, *m);
      ++n_swaps;
    }
    if (nth == i) return;
    if (n_swaps == 0) {
      if (nth < i) {
        j = m = first;
        bool sorted = true;
        while (true) {
          if (++j == i) break;
          if (comp(*j, *m)) {
            sorted = false;
            break;
          }
          m = j;
        }
        if (sorted) return;
      } else {
        j = m = i;
        bool sorted = true;
        while (true) {
          if (++j == last) break;
          if (comp(*j, *m)) {
            sorted = false;
            break;
          }
          m = j;
        }
        if (sorted) return;
      }
    }
    if (nth < i) {
      last = i;
    } else {
      first = ++i;
    }
  }
}

bool ReadFile(const Resources& res, const std::string& path, std::string* out, std::string* error) {
  if (res.Read(path, out)) return true;
  if (error) *error = "cannot read " + path;
  return false;
}

// tensorflow.TensorProto (DT_FLOAT) -> values
bool ParseFloatTensor(std::string_view b, std::vector<float>* out) {
  out->clear();
  pb::Reader r(b);
  int dtype = 1;
  while (r.Next()) {
    if (r.field() == 1) dtype = static_cast<int>(r.int64());
    if (r.field() == 4) {
      const std::string_view c = r.bytes();
      out->resize(c.size() / 4);
      memcpy(out->data(), c.data(), out->size() * 4);
    }
    if (r.field() == 5) {
      if (r.wire_type() == pb::kFixed32) {
        out->push_back(r.float32());
      } else if (r.wire_type() == pb::kLen) {
        const std::string_view c = r.bytes();
        for (size_t i = 0; i + 4 <= c.size(); i += 4) {
          float f;
          memcpy(&f, c.data() + i, 4);
          out->push_back(f);
        }
      }
    }
  }
  return r.ok() && dtype == 1;
}

struct DecoderConfig {
  std::string label_map_path, script_id, lm_filename, lm_symbols, prior_path;
  std::string encoded_prior;
  bool has_lm = false;
  CtcDecoderOptions opt;
};

bool ParseConfig(std::string_view line_recognizer_config, DecoderConfig* c) {
  std::string_view gocr;  // google_ocr.GocrLineRecognizerConfig inside LineRecognizerConfig.sub_config (Any)
  pb::Reader r(line_recognizer_config);
  while (r.Next())
    if (r.field() == 2) {
      pb::Reader any(r.bytes());
      while (any.Next())
        if (any.field() == 2) gocr = any.bytes();
    }
  if (gocr.empty()) return false;
  std::string_view rec;
  pb::Reader g(gocr);
  while (g.Next()) {
    if (g.field() == 14) c->label_map_path = g.bytes();
    if (g.field() == 17) c->script_id = g.bytes();
    if (g.field() == 102) rec = g.bytes();
  }
  pb::Reader rr(rec);
  while (rr.Next()) {
    if (rr.field() == 9) c->prior_path = rr.bytes();
    if (rr.field() == 8) c->encoded_prior = rr.bytes();
    if (rr.field() != 1) continue;
    pb::Reader dc(rr.bytes());  // CTCDecoderConfig
    while (dc.Next()) {
      pb::Reader m(dc.bytes());
      if (dc.field() == 1) {  // OcrLanguageModelSpec
        while (m.Next())
          if (m.field() == 7) {  // FstCharLanguageModelSpec
            c->has_lm = true;
            pb::Reader f(m.bytes());
            while (f.Next()) {
              if (f.field() == 1) c->lm_filename = f.bytes();
              if (f.field() == 2) c->lm_symbols = f.bytes();
            }
          }
      } else if (dc.field() == 2) {  // BeamOptions
        while (m.Next()) {
          if (m.field() == 1) c->opt.beam_size = static_cast<int>(m.int64());
          if (m.field() == 2) c->opt.beam_width = m.float64();
          if (m.field() == 3) c->opt.label_selection_size = static_cast<int>(m.int64());
          if (m.field() == 4) c->opt.label_selection_width = m.float64();
        }
      } else if (dc.field() == 3) {  // CostOptions
        while (m.Next()) {
          if (m.field() == 1) c->opt.lm_weight = m.float64();
          if (m.field() == 2) c->opt.insertion_penalty = m.float64();
          if (m.field() == 3) c->opt.prior_weight = m.float64();
          if (m.field() == 4) c->opt.blank_cost = m.float64();
          if (m.field() == 5) c->opt.self_loop_cost = m.float64();
        }
      }
    }
  }
  return true;
}

}  // namespace

// gocr_internal::BeamItemInfo: (id, cost, cost) — id is a class (label selection) or a candidate index (pruning).
struct CtcBeamDecoder::Item {
  int id;
  float cost;
};

// gocr_internal::BeamItem
struct CtcBeamDecoder::BeamItem {
  int label;
  int state;  // OcrFstLanguageModelHistory
  float optical, lm, misc, prior;
  int parent;
};

std::unique_ptr<CtcBeamDecoder> CtcBeamDecoder::Create(const Resources& res, const std::string& model_dir,
                                                       const std::string& script, std::string* error) {
  const std::string dir = model_dir.empty() || model_dir.back() == '/' ? model_dir : model_dir + "/";
  std::string config_bytes;
  bool found = false;
  for (const std::string& name : {script, script + "_fst_config.pb", script + "_fst_2024_q3_config.pb"}) {
    if (name.size() > 3 && name.compare(name.size() - 3, 3, ".pb") == 0 && res.Read(dir + name, &config_bytes)) {
      found = true;
      break;
    }
  }
  if (!found) {
    if (error) *error = "no LM decoder config for " + script;
    return nullptr;
  }
  DecoderConfig cfg;
  if (!ParseConfig(config_bytes, &cfg) || !cfg.has_lm) {
    if (error) *error = "config has no fst_char_lm_spec: " + script;
    return nullptr;
  }
  auto d = std::unique_ptr<CtcBeamDecoder>(new CtcBeamDecoder());
  d->opt_ = cfg.opt;
  d->lm_weight_ = static_cast<float>(cfg.opt.lm_weight);
  d->script_id_ = cfg.script_id;

  // Label map (google_ocr.LabelClassIDMap) -> flat_hash_map<int, string>; blank = its size.
  std::string bytes;
  if (!ReadFile(res, dir + cfg.label_map_path, &bytes, error)) return nullptr;
  std::map<int, std::string> label_map;
  pb::Reader lr(bytes);
  while (lr.Next()) {
    if (lr.field() != 1) continue;
    int id = 0;
    std::string name;
    pb::Reader l(lr.bytes());
    while (l.Next()) {
      if (l.field() == 1) name = l.bytes();
      if (l.field() == 2) id = static_cast<int>(l.int64());
    }
    label_map.emplace(id, std::move(name));
  }
  d->blank_ = static_cast<int>(label_map.size());
  std::vector<std::pair<int, std::string>> labels(label_map.begin(), label_map.end());
  for (const auto& [id, name] : labels) {
    if (id >= static_cast<int>(d->labels_.size())) d->labels_.resize(id + 1);
    d->labels_[id] = name;
  }

  // Prior: encoded_prior or prior_path (tensorflow.TensorProto of log priors, num_classes + 1 values).
  if (!cfg.encoded_prior.empty()) {
    bytes = cfg.encoded_prior;
  } else if (!ReadFile(res, dir + cfg.prior_path, &bytes, error)) {
    return nullptr;
  }
  if (!ParseFloatTensor(bytes, &d->prior_) || static_cast<int>(d->prior_.size()) <= d->blank_) {
    if (error) *error = "bad prior tensor";
    return nullptr;
  }

  std::string fst, syms;
  if (!ReadFile(res, dir + cfg.lm_filename, &fst, error) || !ReadFile(res, dir + cfg.lm_symbols, &syms, error))
    return nullptr;
  d->lm_ = lm::FstCharLanguageModel::Create(fst, syms, labels, error);
  if (!d->lm_) return nullptr;
  return d;
}

const std::string& CtcBeamDecoder::label(int class_id) const {
  static const std::string kEmpty;
  return class_id >= 0 && class_id < static_cast<int>(labels_.size()) ? labels_[class_id] : kEmpty;
}

// CTCBeamOps::PruneIdCostPair / PartitionItems
void CtcBeamDecoder::PruneIdCostPair(std::vector<Item>* items, int n, double width) const {
  if (items->empty()) return;
  Item* begin = items->data();
  Item* end = begin + items->size();
  const Item* best = begin;
  for (const Item* it = begin + 1; it != end; ++it)
    if (it->cost < best->cost) best = it;
  float threshold = static_cast<float>(width + static_cast<double>(best->cost));
  const int k = std::min(n - 1, static_cast<int>(items->size()) - 1);
  NthElement(begin, begin + k, end, [](const Item& a, const Item& b) { return a.cost < b.cost; });
  if (begin[k].cost <= threshold) threshold = begin[k].cost;
  Item* mid = std::stable_partition(begin, end, [threshold](const Item& a) { return a.cost <= threshold; });
  if (mid > begin + n) mid = begin + n;
  items->resize(mid - begin);
  std::stable_sort(items->begin(), items->end(), [](const Item& a, const Item& b) {
    return a.cost != b.cost ? a.cost < b.cost : a.id < b.id;
  });
}

// CTCBeamOps::ExpandAndPruneBeam + PruneBeam
void CtcBeamDecoder::ExpandAndPruneBeam(const std::vector<BeamItem>& beam, const std::vector<Item>& labels,
                                        std::vector<BeamItem>* next) const {
  std::vector<std::vector<std::pair<double, int>>> lm_costs(beam.size());
  std::vector<int> classes;
  for (size_t b = 0; b < beam.size(); b++) {
    classes.clear();
    for (const Item& l : labels)
      if (l.id != blank_ && l.id != beam[b].label) classes.push_back(l.id);
    lm_->GetNLogProbabilities(classes, beam[b].state, &lm_costs[b]);
  }
  std::vector<BeamItem> cand;
  cand.reserve(beam.size() * labels.size());
  for (size_t b = 0; b < beam.size(); b++) {
    size_t li = 0;
    for (const Item& l : labels) {
      BeamItem e = beam[b];
      e.optical = l.cost + beam[b].optical;
      e.parent = static_cast<int>(b);
      if (l.id == blank_) {
        e.misc = static_cast<float>(opt_.blank_cost + static_cast<double>(e.misc));
      } else if (l.id == beam[b].label) {
        e.misc = static_cast<float>(opt_.self_loop_cost + static_cast<double>(e.misc));
      } else {
        const auto& [cost, state] = lm_costs[b][li++];
        e.lm = static_cast<float>(cost * static_cast<double>(lm_weight_) + static_cast<double>(e.lm));
        e.state = state;
        if (l.id != lm_->eos_class()) {
          e.misc = static_cast<float>(opt_.insertion_penalty + static_cast<double>(e.misc));
          e.prior = static_cast<float>(opt_.prior_weight * static_cast<double>(prior_[l.id]) +
                                       static_cast<double>(e.prior));
        }
      }
      e.label = l.id;
      cand.push_back(e);
    }
  }
  std::vector<Item> infos(cand.size());
  for (size_t i = 0; i < cand.size(); i++)
    infos[i] = {static_cast<int>(i), ((cand[i].optical + cand[i].lm) + cand[i].misc) + cand[i].prior};
  PruneIdCostPair(&infos, opt_.beam_size, opt_.beam_width);
  next->clear();
  for (const Item& it : infos) {
    const BeamItem& e = cand[it.id];
    // std::unique on (label, LM history)
    if (!next->empty() && next->back().label == e.label && next->back().state == e.state) continue;
    next->push_back(e);
  }
}

bool CtcBeamDecoder::Decode(const CtcDecoderInput& in, CtcDecoderOutput* out) const {
  out->labels.clear();
  if (!in.class_ids && in.k != blank_ + 1) return false;  // "num_classes must be blank_id + 1"
  const int frames = in.num_frames;
  std::vector<std::vector<BeamItem>> beams(std::max(frames, -2) + 2);
  beams[0].push_back({blank_, lm_->BosState(), 0, 0, 0, 0, -1});
  std::vector<Item> items;
  for (int t = 0; t < frames; t++) {
    items.clear();
    for (int j = 0; j < in.k; j++) {
      const int id = in.class_ids ? static_cast<int>(static_cast<int64_t>(std::rintf(in.class_ids[t * in.k + j]))) : j;
      items.push_back({id, -in.values[t * in.k + j]});
    }
    PruneIdCostPair(&items, opt_.label_selection_size, opt_.label_selection_width);
    ExpandAndPruneBeam(beams[t], items, &beams[t + 1]);
  }
  if (frames < 0) return true;
  items.assign(1, {lm_->eos_class(), 0.0f});
  ExpandAndPruneBeam(beams[frames], items, &beams[frames + 1]);

  // GenerateOutput
  int last = frames;
  const int limit = std::min(frames + 3, static_cast<int>(beams.size()));
  for (int i = 0; i < limit; i++)
    if (beams[i].empty()) {
      last = i - 2;
      break;
    }
  if (last <= 0) return true;
  std::vector<const BeamItem*> path;
  const BeamItem* cur = &beams[last + 1][0];
  for (int v = last; v >= 1; v--) {
    cur = &beams[v][cur->parent];
    path.push_back(cur);
  }
  std::reverse(path.begin(), path.end());
  float prev = 0.0f;
  const BeamItem& first = *path[0];
  if (first.label != blank_ && first.label != -1) {
    out->labels.push_back({first.label, 0, static_cast<double>(first.optical), static_cast<double>(first.lm),
                           static_cast<double>(first.prior), 1.0});
    prev = first.optical;
  }
  for (size_t k = 1; k < path.size(); k++) {
    const BeamItem& p = *path[k - 1];
    const BeamItem& c = *path[k];
    if (c.label == blank_) continue;
    const double d = static_cast<double>(c.optical) - static_cast<double>(p.optical);
    if (p.label == c.label) {
      if (d >= static_cast<double>(prev) || out->labels.empty()) continue;
      out->labels.back().frame = static_cast<int>(k);
      out->labels.back().optical_cost = d;
    } else {
      out->labels.push_back({c.label, static_cast<int>(k), d,
                             static_cast<double>(c.lm) - static_cast<double>(p.lm),
                             static_cast<double>(c.prior) - static_cast<double>(p.prior), 1.0});
    }
    prev = static_cast<float>(d);
  }
  return true;
}

void SparseDequantizeFrame(const uint8_t* logits, int num_classes, int zero_point, float scale, int k, float* out) {
  int threshold = static_cast<int>(0.01f / scale + static_cast<float>(zero_point));
  threshold = std::min(std::max(threshold, 0), 255);
  std::memset(out, 0, sizeof(float) * 2 * k);
  std::vector<int> packed;
  for (int c = 0; c < num_classes; c++)
    if (logits[c] >= threshold) packed.push_back(c + (static_cast<int>(logits[c]) << 16));
  if (k < static_cast<int>(packed.size()))
    NthElement(packed.data(), packed.data() + k - 1, packed.data() + packed.size(), std::greater<int>());
  const int n = std::min(k, static_cast<int>(packed.size()));
  for (int i = 0; i < n; i++) {
    out[2 * i] = static_cast<float>(packed[i] & 0xFFFF);
    out[2 * i + 1] = static_cast<float>(((packed[i] >> 16) & 0xFF) - zero_point) * scale;
  }
}

}  // namespace screenai
