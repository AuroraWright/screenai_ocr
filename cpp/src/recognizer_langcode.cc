#include "recognizer_langcode.h"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace screenai::rec {
namespace {

#include "recognizer_langcode.inc"

bool IsAlpha(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
bool IsDigit(char c) { return c >= '0' && c <= '9'; }
char Lower(char c) { return c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : c; }
char Upper(char c) { return c >= 'a' && c <= 'z' ? static_cast<char>(c - 'a' + 'A') : c; }
bool AllAlpha(std::string_view s) { return std::all_of(s.begin(), s.end(), IsAlpha); }
bool AllDigit(std::string_view s) { return std::all_of(s.begin(), s.end(), IsDigit); }

bool ValidLanguage(std::string_view l) {
  if ((l.size() != 2 && l.size() != 3) || !AllAlpha(l)) return false;
  size_t i = 0;
  for (char c : l) i = i * 26 + static_cast<size_t>(Lower(c) - 'a');
  const uint64_t* bits = l.size() == 2 ? kValidLang2 : kValidLang3;
  return (bits[i / 64] >> (i % 64)) & 1;
}

template <size_t N>
std::string_view Find(const std::pair<std::string_view, std::string_view> (&table)[N], std::string_view key) {
  const auto it = std::lower_bound(std::begin(table), std::end(table), key,
                                   [](const auto& e, std::string_view k) { return e.first < k; });
  return it != std::end(table) && it->first == key ? it->second : std::string_view();
}

template <size_t N>
bool Contains(const std::string_view (&table)[N], std::string_view key) {
  return std::binary_search(std::begin(table), std::end(table), key);
}

}  // namespace

std::string MostLikelyScript(std::string_view tag) {
  if (tag.empty()) return "Zzzz";
  std::string lower(tag);
  for (char& c : lower) c = c == '_' ? '-' : Lower(c);
  for (const auto& [g, s] : kGrandfathered)
    if (lower == g) return std::string(s);
  std::string language, script, region;
  {
    std::vector<std::string_view> parts;
    std::string_view rest(lower);
    while (true) {
      const size_t p = rest.find('-');
      parts.push_back(rest.substr(0, p));
      if (p == std::string_view::npos) break;
      rest.remove_prefix(p + 1);
    }
    for (std::string_view p : parts)
      if (p.empty() || p.size() > 8) return "Latn";
    size_t i = 0;
    if (!ValidLanguage(parts[i])) return "Latn";
    language = parts[i++];
    if (i < parts.size() && parts[i].size() == 3 && AllAlpha(parts[i])) {  // extlang (zh-yue -> yue)
      if (!Contains(kExtlangs, language + "-" + std::string(parts[i]))) return "Latn";
      language = parts[i++];
    }
    if (i < parts.size() && parts[i].size() == 4 && AllAlpha(parts[i])) {
      script = parts[i++];
      script[0] = Upper(script[0]);
      if (!Contains(kScriptCodes, script)) {
        if (script != "Zzzz") return "Latn";
        script.clear();  // "unknown" is no script
      }
    }
    if (i < parts.size() && ((parts[i].size() == 2 && AllAlpha(parts[i])) ||
                             (parts[i].size() == 3 && AllDigit(parts[i])))) {
      region = parts[i++];
      for (char& c : region) c = Upper(c);
      if (!Contains(kRegions, region)) return "Latn";
    }
    // Variants: registered ones only, in any order and with any prefix (ru-PETR1708, zh-Hant-1994).
    while (i < parts.size() && (parts[i].size() >= 5 || (parts[i].size() == 4 && IsDigit(parts[i][0])))) {
      if (!Contains(kVariants, parts[i])) return "Latn";
      i++;
    }
    // Extensions: -t- (transformed content), -u- (locale) and -x- (private use) parse; nothing is read from them.
    if (i < parts.size()) {
      if (parts[i].size() != 1 || (parts[i] != "t" && parts[i] != "u" && parts[i] != "x") || i + 1 == parts.size())
        return "Latn";
    }
  }
  if (!script.empty()) return script;
  if (!region.empty()) {
    const std::string_view s = Find(kRegionScripts, language + "-" + region);
    if (!s.empty()) return std::string(s);
  }
  const std::string_view s = Find(kLikelyScripts, language);
  return s.empty() ? "Latn" : std::string(s);
}

}  // namespace screenai::rec
