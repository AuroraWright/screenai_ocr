// SPDX-License-Identifier: MPL-2.0
// Minimal protobuf wire-format reader/writer (no libprotobuf dependency).
#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>

namespace screenai::pb {

enum WireType { kVarint = 0, kFixed64 = 1, kLen = 2, kFixed32 = 5 };

class Writer {
 public:
  void Varint(uint64_t v) {
    while (v >= 0x80) {
      buf_.push_back(static_cast<char>(v | 0x80));
      v >>= 7;
    }
    buf_.push_back(static_cast<char>(v));
  }
  void Tag(int field, WireType wt) { Varint((static_cast<uint64_t>(field) << 3) | wt); }
  void Int(int field, int64_t v) { Tag(field, kVarint), Varint(static_cast<uint64_t>(v)); }
  void Bool(int field, bool v) { Int(field, v ? 1 : 0); }
  void Float(int field, float v) {
    Tag(field, kFixed32);
    char b[4];
    memcpy(b, &v, 4);
    buf_.append(b, 4);
  }
  void Double(int field, double v) {
    Tag(field, kFixed64);
    char b[8];
    memcpy(b, &v, 8);
    buf_.append(b, 8);
  }
  void Bytes(int field, std::string_view s) {
    Tag(field, kLen);
    Varint(s.size());
    buf_.append(s.data(), s.size());
  }
  void Message(int field, const Writer& w) { Bytes(field, w.buf_); }
  const std::string& str() const { return buf_; }
  std::string& str() { return buf_; }

 private:
  std::string buf_;
};

// Iterates the fields of one serialized message: while (r.Next()) switch (r.field()) ...
class Reader {
 public:
  explicit Reader(std::string_view s) : p_(s.data()), end_(s.data() + s.size()) {}
  bool Next() {
    if (p_ >= end_ || !ok_) return false;
    uint64_t key;
    if (!ReadVarint(&key)) return false;
    field_ = static_cast<int>(key >> 3);
    wt_ = static_cast<WireType>(key & 7);
    switch (wt_) {
      case kVarint:
        return ReadVarint(&varint_);
      case kFixed64:
        if (end_ - p_ < 8) return ok_ = false;
        memcpy(&fixed64_, p_, 8), p_ += 8;
        return true;
      case kFixed32:
        if (end_ - p_ < 4) return ok_ = false;
        memcpy(&fixed32_, p_, 4), p_ += 4;
        return true;
      case kLen: {
        uint64_t n;
        if (!ReadVarint(&n) || static_cast<uint64_t>(end_ - p_) < n) return ok_ = false;
        bytes_ = std::string_view(p_, n);
        p_ += n;
        return true;
      }
      default:
        return ok_ = false;
    }
  }
  int field() const { return field_; }
  WireType wire_type() const { return wt_; }
  uint64_t varint() const { return varint_; }
  int64_t int64() const { return static_cast<int64_t>(varint_); }
  float float32() const {
    float f;
    memcpy(&f, &fixed32_, 4);
    return f;
  }
  double float64() const {
    double d;
    memcpy(&d, &fixed64_, 8);
    return d;
  }
  std::string_view bytes() const { return bytes_; }
  bool ok() const { return ok_; }

 private:
  bool ReadVarint(uint64_t* v) {
    *v = 0;
    for (int shift = 0; shift < 64 && p_ < end_; shift += 7) {
      const uint8_t b = static_cast<uint8_t>(*p_++);
      *v |= static_cast<uint64_t>(b & 0x7F) << shift;
      if (!(b & 0x80)) return true;
    }
    return ok_ = false;
  }
  const char* p_;
  const char* end_;
  bool ok_ = true;
  int field_ = 0;
  WireType wt_ = kVarint;
  uint64_t varint_ = 0;
  uint32_t fixed32_ = 0;
  uint64_t fixed64_ = 0;
  std::string_view bytes_;
};

}  // namespace screenai::pb
