#include "libyuv_scale.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "x86_compat.h"

namespace screenai {
namespace {

inline int32_t Wrap(int64_t v) { return static_cast<int32_t>(static_cast<uint32_t>(v)); }
inline int32_t FixedDiv(int num, int div) { return static_cast<int32_t>((static_cast<int64_t>(num) << 16) / div); }
inline int32_t FixedDiv1(int num, int div) {
  return static_cast<int32_t>(((static_cast<int64_t>(num) << 16) - 0x00010001) / (div - 1));
}
inline int CenterStart(int d) { return d >= 0 ? (d >> 1) - 0x8000 : 0x8000 - ((-d) >> 1); }
inline int Min1(int v) { return v < 1 ? 1 : v; }

FilterMode ScaleFilterReduce(int sw, int sh, int dw, int dh, FilterMode f) {
  sw = std::abs(sw);
  sh = std::abs(sh);
  if (f == kFilterBox && (dw * 2 >= sw || dh * 2 >= sh)) f = kFilterBilinear;
  if (f == kFilterBilinear) {
    if (sh == 1) f = kFilterLinear;
    if (dh == sh || dh * 3 == sh) f = kFilterLinear;
    if (sw == 1) f = kFilterNone;
  }
  if (f == kFilterLinear) {
    if (sw == 1) f = kFilterNone;
    if (dw == sw || dw * 3 == sw) f = kFilterNone;
  }
  return f;
}

void ScaleSlope(int sw, int sh, int dw, int dh, FilterMode f, int* x, int* y, int* dx, int* dy) {
  if (dw == 1 && sw >= 32768) dw = sw;
  if (dh == 1 && sh >= 32768) dh = sh;
  const int asw = std::abs(sw);
  *x = *y = *dx = *dy = 0;
  if (f == kFilterBox) {
    *dx = FixedDiv(asw, dw);
    *dy = FixedDiv(sh, dh);
  } else if (f == kFilterBilinear) {
    if (dw <= asw) {
      *dx = FixedDiv(asw, dw);
      *x = CenterStart(*dx);
    } else if (asw > 1 && dw > 1) {
      *dx = FixedDiv1(asw, dw);
      *x = 0;
    }
    if (dh <= sh) {
      *dy = FixedDiv(sh, dh);
      *y = CenterStart(*dy);
    } else if (sh > 1 && dh > 1) {
      *dy = FixedDiv1(sh, dh);
      *y = 0;
    }
  } else if (f == kFilterLinear) {
    if (dw <= asw) {
      *dx = FixedDiv(asw, dw);
      *x = CenterStart(*dx);
    } else if (asw > 1 && dw > 1) {
      *dx = FixedDiv1(asw, dw);
      *x = 0;
    }
    *dy = FixedDiv(sh, dh);
    *y = *dy >> 1;
  } else {
    *dx = FixedDiv(asw, dw);
    *dy = FixedDiv(sh, dh);
    *x = *dx >= 0 ? *dx >> 1 : -((-*dx) >> 1);
    *y = *dy >= 0 ? *dy >> 1 : -((-*dy) >> 1);
  }
}

// ---- row kernels ----------------------------------------------------------------------------------

// InterpolateRow_C: f in [0,255] is the weight of r1.
void InterpolateRow(uint8_t* dst, const uint8_t* r0, const uint8_t* r1, int w, int f) {
  if (f == 0) {
    memcpy(dst, r0, w);
  } else if (f == 128) {
    for (int i = 0; i < w; i++) dst[i] = static_cast<uint8_t>((r0[i] + r1[i] + 1) >> 1);
  } else {
    for (int i = 0; i < w; i++) dst[i] = static_cast<uint8_t>((r0[i] * (256 - f) + r1[i] * f + 128) >> 8);
  }
}

// ScaleFilterCols_C (ScaleFilterCols64_C for sw >= 32768); `row` must hold sw + 1 readable bytes (last pixel
// duplicated). x86-64 builds use ScaleFilterCols_SSSE3 for sw < 32768: a 7-bit fraction (bits 9..15 of x),
// (a * (128 - f) + b * f + 64) >> 7.
void FilterCols(uint8_t* dst, const uint8_t* row, int sw, int dw, int x, int dx) {
  int64_t xx = x;
  if (SCREENAI_X86_COMPAT && sw < 32768) {
    for (int j = 0; j < dw; j++, xx += dx) {
      const int xi = static_cast<int>(xx >> 16);
      const int f = static_cast<int>(xx & 0xFFFF) >> 9;
      dst[j] = static_cast<uint8_t>((row[xi] * (128 - f) + row[xi + 1] * f + 64) >> 7);
    }
    return;
  }
  for (int j = 0; j < dw; j++, xx += dx) {
    const int xi = static_cast<int>(xx >> 16);
    const int a = row[xi], b = row[xi + 1];
    const int f = static_cast<int>(xx & 0xFFFF);
    dst[j] = static_cast<uint8_t>(a + (((b - a) * f + 0x8000) >> 16));
  }
}

struct Plane {
  const uint8_t* p;
  int stride, w, h;
  const uint8_t* row(int y) const { return p + static_cast<int64_t>(y) * stride; }
};

// Copies a source row into a buffer with the last pixel duplicated (for FilterCols).
const uint8_t* PaddedRow(const uint8_t* r, int w, std::vector<uint8_t>& buf) {
  buf.resize(w + 1);
  memcpy(buf.data(), r, w);
  buf[w] = r[w - 1];
  return buf.data();
}

// ---- plane scalers --------------------------------------------------------------------------------

void Vertical(const Plane& s, uint8_t* dst, int ds, int dh, int y, int dy, FilterMode f) {
  const int max_y = s.h > 1 ? (s.h << 16) - 0x10000 - (f == kFilterBilinear ? 1 : 0) : 0;
  for (int j = 0; j < dh; j++) {
    if (y >= max_y) y = max_y;
    const int yi = y >> 16;
    const int yf = f == kFilterBilinear ? (y >> 8) & 255 : 0;
    const uint8_t* r1 = (yf && yi + 1 < s.h) ? s.row(yi + 1) : s.row(yi);
    InterpolateRow(dst + static_cast<int64_t>(j) * ds, s.row(yi), r1, s.w, yf);
    y = Wrap(static_cast<int64_t>(y) + dy);
  }
}

void Down2(const Plane& s, uint8_t* dst, int ds, int dw, int dh, FilterMode f) {
  for (int j = 0; j < dh; j++) {
    uint8_t* d = dst + static_cast<int64_t>(j) * ds;
    const uint8_t* a = s.row(2 * j);
    const uint8_t* b = s.row(2 * j + 1);
    for (int i = 0; i < dw; i++) {
      if (f == kFilterNone) d[i] = b[2 * i + 1];
      else if (f == kFilterLinear) d[i] = static_cast<uint8_t>((a[2 * i] + a[2 * i + 1] + 1) >> 1);
      else d[i] = static_cast<uint8_t>((a[2 * i] + a[2 * i + 1] + b[2 * i] + b[2 * i + 1] + 2) >> 2);
    }
  }
}

void Down4(const Plane& s, uint8_t* dst, int ds, int dw, int dh, FilterMode f) {
  for (int j = 0; j < dh; j++) {
    uint8_t* d = dst + static_cast<int64_t>(j) * ds;
    for (int i = 0; i < dw; i++) {
      if (f == kFilterNone) {
        d[i] = s.row(4 * j + 2)[4 * i + 2];
      } else {
        uint32_t sum = 0;
        for (int r = 0; r < 4; r++)
          for (int c = 0; c < 4; c++) sum += s.row(4 * j + r)[4 * i + c];
        d[i] = static_cast<uint8_t>((sum + 8) >> 4);
      }
    }
  }
}

void Down34(const Plane& s, uint8_t* dst, int ds, int dw, int dh, FilterMode f) {
  const int stride_f = f == kFilterLinear ? 0 : 1;
  const int n = dw / 3;
  std::vector<int> ha(dw), hb(dw);
  auto hrow = [&](const uint8_t* r, std::vector<int>& o) {
    for (int k = 0; k < n; k++) {
      const int q0 = r[4 * k], q1 = r[4 * k + 1], q2 = r[4 * k + 2], q3 = r[4 * k + 3];
      o[3 * k] = (q0 * 3 + q1 + 2) >> 2;
      o[3 * k + 1] = (q1 + q2 + 1) >> 1;
      o[3 * k + 2] = (q2 + q3 * 3 + 2) >> 2;
    }
  };
  auto pick = [&](uint8_t* d, int r) {
    const uint8_t* row = s.row(r);
    for (int k = 0; k < n; k++) {
      d[3 * k] = row[4 * k];
      d[3 * k + 1] = row[4 * k + 1];
      d[3 * k + 2] = row[4 * k + 3];
    }
  };
  auto k0 = [&](uint8_t* d, int r, int off) {
    if (f == kFilterNone) return pick(d, r);
    hrow(s.row(r), ha);
    hrow(s.row(r + off * stride_f), hb);
    for (int i = 0; i < 3 * n; i++) d[i] = static_cast<uint8_t>((ha[i] * 3 + hb[i] + 2) >> 2);
  };
  auto k1 = [&](uint8_t* d, int r, int off) {
    if (f == kFilterNone) return pick(d, r);
    hrow(s.row(r), ha);
    hrow(s.row(r + off * stride_f), hb);
    for (int i = 0; i < 3 * n; i++) d[i] = static_cast<uint8_t>((ha[i] + hb[i] + 1) >> 1);
  };
  auto out = [&](int j) { return dst + static_cast<int64_t>(j) * ds; };
  int r = 0, j = 0;
  while (j < dh - 2) {
    k0(out(j), r, 1);
    k1(out(j + 1), r + 1, 1);
    k0(out(j + 2), r + 3, -1);
    r += 4;
    j += 3;
  }
  if (dh % 3 == 1) {
    k0(out(j), r, 0);
  } else if (dh % 3 == 2) {
    k0(out(j), r, 1);
    k1(out(j + 1), r + 1, 0);
  }
}

void Down38(const Plane& s, uint8_t* dst, int ds, int dw, int dh, FilterMode f) {
  const int stride_f = f == kFilterLinear ? 0 : 1;
  const int n = dw / 3;
  std::vector<int> q(8 * n);
  auto pick = [&](uint8_t* d, int r) {
    const uint8_t* row = s.row(r);
    for (int k = 0; k < n; k++) {
      d[3 * k] = row[8 * k];
      d[3 * k + 1] = row[8 * k + 3];
      d[3 * k + 2] = row[8 * k + 6];
    }
  };
  auto k3 = [&](uint8_t* d, int r, int use) {
    if (f == kFilterNone) return pick(d, r);
    const int rows[3] = {r, r + use * stride_f, r + 2 * use * stride_f};
    std::fill(q.begin(), q.end(), 0);
    for (int rr : rows)
      for (int i = 0; i < 8 * n; i++) q[i] += s.row(rr)[i];
    for (int k = 0; k < n; k++) {
      const int* p = &q[8 * k];
      d[3 * k] = static_cast<uint8_t>(((p[0] + p[1] + p[2]) * (65536 / 9)) >> 16);
      d[3 * k + 1] = static_cast<uint8_t>(((p[3] + p[4] + p[5]) * (65536 / 9)) >> 16);
      d[3 * k + 2] = static_cast<uint8_t>(((p[6] + p[7]) * (65536 / 6)) >> 16);
    }
  };
  auto k2 = [&](uint8_t* d, int r, int use) {
    if (f == kFilterNone) return pick(d, r);
    for (int i = 0; i < 8 * n; i++) q[i] = s.row(r)[i] + s.row(r + use * stride_f)[i];
    for (int k = 0; k < n; k++) {
      const int* p = &q[8 * k];
      d[3 * k] = static_cast<uint8_t>(((p[0] + p[1] + p[2]) * (65536 / 6)) >> 16);
      d[3 * k + 1] = static_cast<uint8_t>(((p[3] + p[4] + p[5]) * (65536 / 6)) >> 16);
      d[3 * k + 2] = static_cast<uint8_t>((p[6] + p[7]) >> 2);
    }
  };
  auto out = [&](int j) { return dst + static_cast<int64_t>(j) * ds; };
  int r = 0, j = 0;
  while (j < dh - 2) {
    k3(out(j), r, 1);
    k3(out(j + 1), r + 3, 1);
    k2(out(j + 2), r + 6, 1);
    r += 8;
    j += 3;
  }
  if (dh % 3 == 1) {
    k3(out(j), r, 0);
  } else if (dh % 3 == 2) {
    k3(out(j), r, 1);
    k3(out(j + 1), r + 3, 0);
  }
}

void Box(const Plane& s, uint8_t* dst, int ds, int dw, int dh) {
  int x, y, dx, dy;
  ScaleSlope(s.w, s.h, dw, dh, kFilterBox, &x, &y, &dx, &dy);
  const int max_y = s.h << 16;
  std::vector<uint16_t> row(s.w);
  const int minbox = dx >> 16;
  for (int j = 0; j < dh; j++) {
    const int iy = y >> 16;
    y = Wrap(static_cast<int64_t>(y) + dy);
    if (y >= max_y) y = max_y;
    const int bh = std::max(1, (y >> 16) - iy);
    std::fill(row.begin(), row.end(), 0);
    for (int k = 0; k < bh; k++) {  // ScaleAddRow accumulates in uint16
      const uint8_t* r = s.row(iy + k);
      for (int i = 0; i < s.w; i++) row[i] = static_cast<uint16_t>(row[i] + r[i]);
    }
    uint8_t* d = dst + static_cast<int64_t>(j) * ds;
    if (dx == 0x10000) {  // ScaleAddCols0_C
      const uint32_t sv = 65536 / bh;
      const int x0 = x >> 16;
      for (int i = 0; i < dw; i++) d[i] = static_cast<uint8_t>((row[x0 + i] * sv) >> 16);
    } else if ((dx & 0xFFFF) == 0) {  // ScaleAddCols1_C
      const int bw = Min1(minbox);
      const uint32_t sv = 65536 / (bw * bh);
      int xi = x >> 16;
      for (int i = 0; i < dw; i++, xi += bw) {
        uint32_t sum = 0;
        for (int k = 0; k < bw; k++) sum += row[xi + k];
        d[i] = static_cast<uint8_t>((sum * sv) >> 16);
      }
    } else {  // ScaleAddCols2_C
      const uint32_t tbl[2] = {static_cast<uint32_t>(65536 / (Min1(minbox) * bh)),
                               static_cast<uint32_t>(65536 / (Min1(minbox + 1) * bh))};
      int64_t xx = x;
      for (int i = 0; i < dw; i++) {
        const int ix = static_cast<int>(xx >> 16);
        xx += dx;
        const int bw = Min1(static_cast<int>(xx >> 16) - ix);
        uint32_t sum = 0;
        for (int k = 0; k < bw; k++) sum += row[ix + k];
        d[i] = static_cast<uint8_t>((sum * tbl[bw - std::max(minbox, 0)]) >> 16);
      }
    }
  }
}

void BilinearDown(const Plane& s, uint8_t* dst, int ds, int dw, int dh, FilterMode f) {
  int x, y, dx, dy;
  ScaleSlope(s.w, s.h, dw, dh, f, &x, &y, &dx, &dy);
  const int max_y = (s.h << 16) - 0x10000;
  if (y > max_y) y = max_y;
  std::vector<uint8_t> tmp(s.w + 1), pad;
  for (int j = 0; j < dh; j++) {
    const int yi = y >> 16;
    uint8_t* d = dst + static_cast<int64_t>(j) * ds;
    if (f == kFilterLinear) {
      FilterCols(d, PaddedRow(s.row(yi), s.w, pad), s.w, dw, x, dx);
    } else {
      const int yf = (y >> 8) & 255;
      const uint8_t* r1 = (yf && yi + 1 < s.h) ? s.row(yi + 1) : s.row(yi);
      InterpolateRow(tmp.data(), s.row(yi), r1, s.w, yf);
      tmp[s.w] = tmp[s.w - 1];
      FilterCols(d, tmp.data(), s.w, dw, x, dx);
    }
    y = Wrap(static_cast<int64_t>(y) + dy);
    if (y >= max_y) y = max_y;
  }
}

void BilinearUp(const Plane& s, uint8_t* dst, int ds, int dw, int dh, FilterMode f) {
  int x, y, dx, dy;
  ScaleSlope(s.w, s.h, dw, dh, f, &x, &y, &dx, &dy);
  const int max_y = (s.h << 16) - 0x10000;
  if (y > max_y) y = max_y;
  int yi = y >> 16;
  // emulate libyuv's two-row ring buffer, including the source pointer that stops advancing
  int sp = yi;
  std::vector<uint8_t> bufs[2] = {std::vector<uint8_t>(dw), std::vector<uint8_t>(dw)}, pad;
  FilterCols(bufs[0].data(), PaddedRow(s.row(sp), s.w, pad), s.w, dw, x, dx);
  if (s.h > 1) sp++;
  FilterCols(bufs[1].data(), PaddedRow(s.row(sp), s.w, pad), s.w, dw, x, dx);
  if (s.h > 2) sp++;
  int cur = 0, lasty = yi;
  for (int j = 0; j < dh; j++) {
    if (y > max_y) y = max_y;
    yi = y >> 16;
    if (yi != lasty) {
      FilterCols(bufs[cur].data(), PaddedRow(s.row(sp), s.w, pad), s.w, dw, x, dx);
      cur ^= 1;
      if (y + 0x10000 < max_y) sp++;
      lasty = yi;
    }
    uint8_t* d = dst + static_cast<int64_t>(j) * ds;
    if (f == kFilterLinear) memcpy(d, bufs[cur].data(), dw);
    else InterpolateRow(d, bufs[cur].data(), bufs[cur ^ 1].data(), dw, (y >> 8) & 255);
    y = Wrap(static_cast<int64_t>(y) + dy);
  }
}

void Up2LinearRow(uint8_t* out, const uint8_t* r, int dw) {
  out[0] = r[0];
  const int work = (dw - 1) & ~1;
  for (int k = 0; k < work / 2; k++) {
    const int a = r[k], b = r[k + 1];
    out[1 + 2 * k] = static_cast<uint8_t>((a * 3 + b + 2) >> 2);
    out[2 + 2 * k] = static_cast<uint8_t>((a + b * 3 + 2) >> 2);
  }
  out[dw - 1] = r[(dw - 1) / 2];
}

void Up2Linear(const Plane& s, uint8_t* dst, int ds, int dw, int dh) {
  if (dh == 1) return Up2LinearRow(dst, s.row((s.h - 1) / 2), dw);
  const int dy = FixedDiv(s.h - 1, dh - 1);
  int y = 0x7FFF;
  for (int j = 0; j < dh; j++) {
    Up2LinearRow(dst + static_cast<int64_t>(j) * ds, s.row(y >> 16), dw);
    y = Wrap(static_cast<int64_t>(y) + dy);
  }
}

// ScaleRowUp2_Bilinear_Any: produces two output rows from source rows a (upper) and b (lower).
void Up2BilinearRows(uint8_t* da, uint8_t* db, const uint8_t* a, const uint8_t* b, int dw) {
  da[0] = static_cast<uint8_t>((3 * a[0] + b[0] + 2) >> 2);
  if (db) db[0] = static_cast<uint8_t>((a[0] + 3 * b[0] + 2) >> 2);
  const int work = (dw - 1) & ~1;
  for (int k = 0; k < work / 2; k++) {
    const int s0 = a[k], s1 = a[k + 1], t0 = b[k], t1 = b[k + 1];
    da[1 + 2 * k] = static_cast<uint8_t>((s0 * 9 + s1 * 3 + t0 * 3 + t1 + 8) >> 4);
    da[2 + 2 * k] = static_cast<uint8_t>((s0 * 3 + s1 * 9 + t0 + t1 * 3 + 8) >> 4);
    if (db) {
      db[1 + 2 * k] = static_cast<uint8_t>((s0 * 3 + s1 + t0 * 9 + t1 * 3 + 8) >> 4);
      db[2 + 2 * k] = static_cast<uint8_t>((s0 + s1 * 3 + t0 * 3 + t1 * 9 + 8) >> 4);
    }
  }
  const int k = (dw - 1) / 2;
  da[dw - 1] = static_cast<uint8_t>((3 * a[k] + b[k] + 2) >> 2);
  if (db) db[dw - 1] = static_cast<uint8_t>((a[k] + 3 * b[k] + 2) >> 2);
}

void Up2Bilinear(const Plane& s, uint8_t* dst, int ds, int dw, int dh) {
  auto out = [&](int j) { return dst + static_cast<int64_t>(j) * ds; };
  Up2BilinearRows(out(0), nullptr, s.row(0), s.row(0), dw);
  int o = 1;
  for (int i = 1; i < s.h; i++, o += 2) Up2BilinearRows(out(o), out(o + 1), s.row(i - 1), s.row(i), dw);
  if ((dh & 1) == 0) Up2BilinearRows(out(o), nullptr, s.row(s.h - 1), s.row(s.h - 1), dw);
}

void Simple(const Plane& s, uint8_t* dst, int ds, int dw, int dh) {
  int x, y, dx, dy;
  ScaleSlope(s.w, s.h, dw, dh, kFilterNone, &x, &y, &dx, &dy);
  for (int j = 0; j < dh; j++) {
    const uint8_t* row = s.row(y >> 16);
    uint8_t* d = dst + static_cast<int64_t>(j) * ds;
    if (dx < 0x8000 && 2 * s.w == dw) {  // ScaleColsUp2_C
      for (int i = 0; i < dw; i++) d[i] = row[i >> 1];
    } else {
      int64_t xx = x;
      for (int i = 0; i < dw; i++, xx += dx) d[i] = row[xx >> 16];
    }
    y = Wrap(static_cast<int64_t>(y) + dy);
  }
}

}  // namespace

void ScalePlane(const uint8_t* src, int src_stride, int sw, int sh, uint8_t* dst, int dst_stride, int dw, int dh,
                FilterMode filtering) {
  const Plane s{src, src_stride, sw, sh};
  const FilterMode f = ScaleFilterReduce(sw, sh, dw, dh, filtering);
  if (dw == sw && dh == sh) {
    for (int j = 0; j < dh; j++) memcpy(dst + static_cast<int64_t>(j) * dst_stride, s.row(j), sw);
    return;
  }
  if (dw == sw && f != kFilterBox) {
    int y, dy;
    if (dh <= sh) {
      dy = FixedDiv(sh, dh);
      y = CenterStart(dy);
    } else if (sh > 1 && f != kFilterNone) {
      dy = FixedDiv1(sh, dh);
      y = 0;
    } else if (sh > 1) {
      dy = FixedDiv(sh, dh);
      y = dy >= 0 ? dy >> 1 : -((-dy) >> 1);
    } else {
      dy = y = 0;
    }
    return Vertical(s, dst, dst_stride, dh, y, dy, f);
  }
  if (dw <= sw && dh <= sh) {
    if (4 * dw == 3 * sw && 4 * dh == 3 * sh) return Down34(s, dst, dst_stride, dw, dh, f);
    if (2 * dw == sw && 2 * dh == sh) return Down2(s, dst, dst_stride, dw, dh, f);
    if (8 * dw == 3 * sw && 8 * dh == 3 * sh) return Down38(s, dst, dst_stride, dw, dh, f);
    if (4 * dw == sw && 4 * dh == sh && (f == kFilterBox || f == kFilterNone))
      return Down4(s, dst, dst_stride, dw, dh, f);
  }
  if (f == kFilterBox && 2 * dh < sh) return Box(s, dst, dst_stride, dw, dh);
  if (f == kFilterLinear && (dw + 1) / 2 == sw) return Up2Linear(s, dst, dst_stride, dw, dh);
  if ((f == kFilterBilinear || f == kFilterBox) && (dh + 1) / 2 == sh && (dw + 1) / 2 == sw)
    return Up2Bilinear(s, dst, dst_stride, dw, dh);
  if (f != kFilterNone && dh > sh) return BilinearUp(s, dst, dst_stride, dw, dh, f);
  if (f != kFilterNone) return BilinearDown(s, dst, dst_stride, dw, dh, f);
  Simple(s, dst, dst_stride, dw, dh);
}

}  // namespace screenai
