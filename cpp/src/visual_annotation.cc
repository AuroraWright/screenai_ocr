// SPDX-License-Identifier: MPL-2.0
#include "visual_annotation.h"

#include <bit>
#include <cstdint>
#include <map>
#include <optional>
#include <vector>

#include "proto_wire.h"

namespace screenai {
namespace {

struct Rect {
  int x = 0, y = 0, width = 0, height = 0;
  float angle = 0;
};

struct SymbolBox {
  Rect box;
  std::string text;
  float confidence = 0;
};

struct WordBox {
  std::vector<SymbolBox> symbols;
  Rect box, whitespace_box;
  std::string text, language;
  bool has_space_after = false, estimate_color_success = false, estimate_gray_success = false;
  int foreground_gray = 0, background_gray = 0, foreground_rgb = 0, background_rgb = 0;
  int direction = 0, content_type = 0;
  float confidence = 0;
};

struct LineBox {
  std::vector<WordBox> words;
  Rect box;
  std::string text, language;
  int block_id = 0, order_within_block = 0, direction = 0, content_type = 0, paragraph_id = 0;
  float confidence = 0;
};

// ConvertToChromeRect<goodoc::RotatedBoundingBox>: empty boxes are dropped.
std::optional<Rect> ToRect(const RotatedBox& b) {
  if (b.width == 0 || b.height == 0) return std::nullopt;
  return Rect{b.left, b.top, b.width, b.height, b.angle};
}

// GetBoundingBox: the entity polygon's rotated box, else the detector's box.
std::optional<Rect> BoundingBox(const Entity& e) {
  if (e.box) return ToRect(*e.box);
  if (e.detection) return ToRect(e.detection->detection_box);
  return std::nullopt;
}

// First language of the entity's language_mix that is not "und*" (entities carry no weights, which the original
// also requires to be > 0 when present).
std::string Language(const Entity& e) {
  for (const std::string& l : e.languages)
    if (l.compare(0, 3, "und") != 0) return l;
  return {};
}

// goodoc ContentType -> chrome_screen_ai.ContentType (ConvertContentType's table; goodoc 8 has no equivalent).
std::optional<int> ContentType(const Entity& e) {
  if (!e.content_type) return std::nullopt;
  const int t = *e.content_type;
  if (t >= 0 && t <= 7) return t;
  if (t == 9) return 8;
  return std::nullopt;
}

// goodoc.WritingDirection (LTR 0, RTL 1, TTB 2) -> chrome_screen_ai.Direction (UNSPECIFIED 0, LTR 1, RTL 2, TTB 3).
std::optional<int> Direction(const Entity& e) {
  if (!e.writing_direction) return std::nullopt;
  return *e.writing_direction + 1;
}

// Implicit presence: zero scalars, empty strings and +0.0f are not serialized.
void PutInt(pb::Writer& w, int field, int v) {
  if (v != 0) w.Int(field, v);
}
void PutFloat(pb::Writer& w, int field, float v) {
  if (std::bit_cast<uint32_t>(v) != 0) w.Float(field, v);
}
void PutString(pb::Writer& w, int field, const std::string& s) {
  if (!s.empty()) w.Bytes(field, s);
}

pb::Writer WriteRect(const Rect& r) {
  pb::Writer w;
  PutInt(w, 1, r.x);
  PutInt(w, 2, r.y);
  PutInt(w, 3, r.width);
  PutInt(w, 4, r.height);
  PutFloat(w, 5, r.angle);
  return w;
}

pb::Writer WriteSymbol(const SymbolBox& s) {
  pb::Writer w;
  w.Message(1, WriteRect(s.box));
  PutString(w, 2, s.text);
  PutFloat(w, 3, s.confidence);
  return w;
}

pb::Writer WriteWord(const WordBox& word) {
  pb::Writer w;
  for (const SymbolBox& s : word.symbols) w.Message(1, WriteSymbol(s));
  w.Message(2, WriteRect(word.box));
  PutString(w, 3, word.text);
  PutString(w, 5, word.language);
  PutInt(w, 6, word.has_space_after);
  PutInt(w, 7, word.estimate_color_success);
  PutInt(w, 8, word.foreground_gray);
  PutInt(w, 9, word.background_gray);
  PutInt(w, 10, word.foreground_rgb);
  PutInt(w, 11, word.background_rgb);
  PutInt(w, 12, word.direction);
  PutInt(w, 13, word.content_type);
  PutFloat(w, 15, word.confidence);
  PutInt(w, 16, word.estimate_gray_success);
  w.Message(17, WriteRect(word.whitespace_box));
  return w;
}

pb::Writer WriteLine(const LineBox& line) {
  pb::Writer w;
  for (const WordBox& word : line.words) w.Message(1, WriteWord(word));
  w.Message(2, WriteRect(line.box));
  PutString(w, 3, line.text);
  PutString(w, 4, line.language);
  PutInt(w, 5, line.block_id);
  PutInt(w, 6, line.order_within_block);
  PutInt(w, 7, line.direction);
  PutInt(w, 8, line.content_type);
  PutFloat(w, 10, line.confidence);
  PutInt(w, 11, line.paragraph_id);
  return w;
}

}  // namespace

std::string ConvertToVisualAnnotation(const PageLayout& layout) {
  const std::vector<Entity>& entities = layout.entities;
  std::vector<LineBox> lines;
  // entity index -> block ordinal / paragraph index within its block / line index / word index within its line
  std::map<int, int> blocks, paragraphs, line_of, word_of;
  auto parent_is = [&](const Entity& e, EntityType type) {
    return e.parent >= 0 && e.parent < static_cast<int>(entities.size()) && entities[e.parent].type == type;
  };

  for (int i = 0; i < static_cast<int>(entities.size()); i++) {
    const Entity& e = entities[i];
    switch (e.type) {
      case EntityType::kBlock: {
        const int ordinal = static_cast<int>(blocks.size());
        blocks[i] = ordinal;
        break;
      }
      case EntityType::kParagraph: {
        if (!parent_is(e, EntityType::kBlock)) break;  // CHECK failure in the original
        int index = 0;
        for (const auto& [p, unused] : paragraphs)
          if (entities[p].parent == e.parent) index++;
        paragraphs[i] = index;
        break;
      }
      case EntityType::kLine: {
        if (!parent_is(e, EntityType::kParagraph) || !paragraphs.count(e.parent)) break;
        const int block = entities[e.parent].parent;
        if (!blocks.count(block)) break;
        const std::optional<Rect> box = BoundingBox(e);
        if (!box) break;
        LineBox line;
        line.box = *box;
        line.text = e.text.value_or("");
        line.language = Language(e);
        line.block_id = blocks[block];
        for (const LineBox& l : lines) line.order_within_block += l.block_id == line.block_id;
        line.direction = Direction(e).value_or(0);
        const std::optional<int> content_type = ContentType(e);
        if (!content_type) break;
        line.content_type = *content_type;
        line.confidence = e.confidence.value_or(0);
        line.paragraph_id = paragraphs[e.parent];
        lines.push_back(std::move(line));
        line_of[i] = static_cast<int>(lines.size()) - 1;
        break;
      }
      case EntityType::kWord: {
        if (!parent_is(e, EntityType::kLine) || !line_of.count(e.parent)) break;
        const std::optional<Rect> box = BoundingBox(e);
        if (!box) break;
        LineBox& line = lines[line_of[e.parent]];
        WordBox word;
        word.box = *box;
        word.text = e.text.value_or("");
        word.language = Language(e);
        const bool space = e.word_break && e.word_break->type == 2;  // Break.SPACE
        word.has_space_after = space;
        if (e.colors) {
          const Colors& c = *e.colors;
          word.estimate_color_success = c.foreground_rgb && c.background_rgb;
          word.estimate_gray_success = true;
          word.foreground_gray = c.foreground_gray;
          word.background_gray = c.background_gray;
          word.foreground_rgb = c.foreground_rgb.value_or(0);
          word.background_rgb = c.background_rgb.value_or(0);
        }
        word.direction = Direction(e).value_or(line.direction);
        const std::optional<int> content_type = ContentType(e);
        if (!content_type) break;
        word.content_type = *content_type;
        word.confidence = e.confidence.value_or(0);
        if (space && e.word_break->box)
          if (std::optional<Rect> ws = ToRect(*e.word_break->box)) word.whitespace_box = *ws;
        word_of[i] = static_cast<int>(line.words.size());
        line.words.push_back(std::move(word));
        break;
      }
      case EntityType::kSymbol: {
        if (!parent_is(e, EntityType::kWord) || !word_of.count(e.parent)) break;
        const std::optional<Rect> box = BoundingBox(e);
        if (!box) break;
        SymbolBox symbol;
        symbol.box = *box;
        symbol.text = e.text.value_or("");
        symbol.confidence = e.confidence.value_or(0);
        lines[line_of[entities[e.parent].parent]].words[word_of[e.parent]].symbols.push_back(std::move(symbol));
        break;
      }
    }
  }

  pb::Writer w;
  for (const LineBox& line : lines)
    if (!line.words.empty()) w.Message(2, WriteLine(line));  // PruneEmptyLines
  return w.str();
}

}  // namespace screenai
