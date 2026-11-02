"""Bit-exact port of libyuv ScalePlane (8-bit, single channel) as compiled into libchromescreenai.

Only the code paths reachable for positive widths/heights are implemented. All arithmetic follows the C
reference kernels (which the NEON kernels in the binary match bit-for-bit).
"""
import numpy as np

NONE, LINEAR, BILINEAR, BOX = 0, 1, 2, 3


def _i32(v):
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v & 0x80000000 else v


def _cdiv(a, b):
    """C integer division (truncates toward zero)."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def fixed_div(num, div):
    return _i32(_cdiv(num << 16, div))


def fixed_div1(num, div):
    return _i32(_cdiv((num << 16) - 0x00010001, div - 1))


def _center_start(d):
    return (d >> 1) - 0x8000 if d >= 0 else 0x8000 - ((-d) >> 1)


def scale_filter_reduce(sw, sh, dw, dh, f):
    sw, sh = abs(sw), abs(sh)
    if f == BOX:
        if dw * 2 >= sw or dh * 2 >= sh:
            f = BILINEAR
    if f == BILINEAR:
        if sh == 1:
            f = LINEAR
        if dh == sh or dh * 3 == sh:
            f = LINEAR
        if sw == 1:
            f = NONE
    if f == LINEAR:
        if sw == 1:
            f = NONE
        if dw == sw or dw * 3 == sw:
            f = NONE
    return f


def scale_slope(sw, sh, dw, dh, f):
    """Returns (x, y, dx, dy) in 16.16 fixed point."""
    if dw == 1 and sw >= 32768:
        dw = sw
    if dh == 1 and sh >= 32768:
        dh = sh
    asw = abs(sw)
    x = y = dx = dy = 0
    if f == BOX:
        dx = fixed_div(asw, dw); dy = fixed_div(sh, dh)
    elif f == BILINEAR:
        if dw <= asw:
            dx = fixed_div(asw, dw); x = _center_start(dx)
        elif asw > 1 and dw > 1:
            dx = fixed_div1(asw, dw); x = 0
        if dh <= sh:
            dy = fixed_div(sh, dh); y = _center_start(dy)
        elif sh > 1 and dh > 1:
            dy = fixed_div1(sh, dh); y = 0
    elif f == LINEAR:
        if dw <= asw:
            dx = fixed_div(asw, dw); x = _center_start(dx)
        elif asw > 1 and dw > 1:
            dx = fixed_div1(asw, dw); x = 0
        dy = fixed_div(sh, dh); y = dy >> 1
    else:
        dx = fixed_div(asw, dw); dy = fixed_div(sh, dh)
        x = dx >> 1 if dx >= 0 else -((-dx) >> 1)
        y = dy >> 1 if dy >= 0 else -((-dy) >> 1)
    return x, y, dx, dy


# ---- row kernels -------------------------------------------------------------------------------

def interpolate_row(r0, r1, f):
    """InterpolateRow_C: f in [0,255] weight of r1."""
    if f == 0:
        return r0.copy()
    if f == 128:
        return ((r0.astype(np.uint16) + r1 + 1) >> 1).astype(np.uint8)
    return ((r0.astype(np.int32) * (256 - f) + r1.astype(np.int32) * f + 128) >> 8).astype(np.uint8)


def _xs(n, x, dx):
    return (x + dx * np.arange(n, dtype=np.int64)).astype(np.int64)


def filter_cols(row, dw, x, dx):
    """ScaleFilterCols_C: 16-bit fraction, (b-a)*f + 0x8000 >> 16."""
    xs = _xs(dw, x, dx)
    xi = xs >> 16
    f = xs & 0xFFFF
    src = np.concatenate([row, row[-1:]]).astype(np.int64)
    a = src[xi]; b = src[xi + 1]
    return ((a + (((b - a) * f + 0x8000) >> 16)) & 0xFF).astype(np.uint8)


def scale_cols(row, dw, x, dx):
    xs = _xs(dw, x, dx)
    return row[xs >> 16]


# ---- plane scalers -----------------------------------------------------------------------------

def _vertical(src, dh, y, dy, f):
    sh, w = src.shape
    if sh > 1:
        max_y = (sh << 16) - 0x10000 - (1 if f == BILINEAR else 0)
    else:
        max_y = 0
    out = np.empty((dh, w), np.uint8)
    for j in range(dh):
        if y >= max_y:
            y = max_y
        yi = y >> 16
        yf = (y >> 8) & 255 if f == BILINEAR else 0
        r1 = src[yi + 1] if (yf and yi + 1 < sh) else src[yi]
        out[j] = interpolate_row(src[yi], r1, yf)
        y = _i32(y + dy)
    return out


def _down2(src, dw, dh, f):
    s = src.astype(np.uint16)
    if f == NONE:
        return src[1:2 * dh:2, 1:2 * dw:2].copy()
    if f == LINEAR:
        r = s[0:2 * dh:2]
        return ((r[:, 0:2 * dw:2] + r[:, 1:2 * dw:2] + 1) >> 1).astype(np.uint8)
    a = s[0:2 * dh:2]; b = s[1:2 * dh:2]
    return ((a[:, 0:2 * dw:2] + a[:, 1:2 * dw:2] + b[:, 0:2 * dw:2] + b[:, 1:2 * dw:2] + 2) >> 2).astype(np.uint8)


def _down4(src, dw, dh, f):
    if f == NONE:
        return src[2:4 * dh:4, 2:4 * dw:4].copy()
    s = src[:4 * dh, :4 * dw].astype(np.uint32).reshape(dh, 4, dw, 4).sum((1, 3))
    return ((s + 8) >> 4).astype(np.uint8)


def _down34(src, dw, dh, f):
    stride_f = 0 if f == LINEAR else 1
    n = dw // 3

    def hrow(r):
        q = r[:4 * n].astype(np.int32).reshape(n, 4)
        return np.stack([(q[:, 0] * 3 + q[:, 1] + 2) >> 2, (q[:, 1] + q[:, 2] + 1) >> 1,
                         (q[:, 2] + q[:, 3] * 3 + 2) >> 2], 1).reshape(-1)

    def k0(r, off):
        if f == NONE:
            return src[r, :4 * n].reshape(n, 4)[:, [0, 1, 3]].reshape(-1)
        a = hrow(src[r]); b = hrow(src[r + off * stride_f])
        return ((a * 3 + b + 2) >> 2).astype(np.uint8)

    def k1(r, off):
        if f == NONE:
            return src[r, :4 * n].reshape(n, 4)[:, [0, 1, 3]].reshape(-1)
        a = hrow(src[r]); b = hrow(src[r + off * stride_f])
        return ((a + b + 1) >> 1).astype(np.uint8)

    out = np.empty((dh, dw), np.uint8)
    r = 0; j = 0
    while j < dh - 2:
        out[j] = k0(r, 1); out[j + 1] = k1(r + 1, 1); out[j + 2] = k0(r + 3, -1)
        r += 4; j += 3
    if dh % 3 == 1:
        out[j] = k0(r, 0)
    elif dh % 3 == 2:
        out[j] = k0(r, 1); out[j + 1] = k1(r + 1, 0)
    return out


def _down38(src, dw, dh, f):
    stride_f = 0 if f == LINEAR else 1
    n = dw // 3

    def k3(r, use):
        if f == NONE:
            return src[r, :8 * n].reshape(n, 8)[:, [0, 3, 6]].reshape(-1)
        rows = [r, r + use * stride_f, r + 2 * use * stride_f]
        q = sum(src[k, :8 * n].astype(np.int32) for k in rows).reshape(n, 8)
        return np.stack([((q[:, 0] + q[:, 1] + q[:, 2]) * (65536 // 9)) >> 16,
                         ((q[:, 3] + q[:, 4] + q[:, 5]) * (65536 // 9)) >> 16,
                         ((q[:, 6] + q[:, 7]) * (65536 // 6)) >> 16], 1).reshape(-1).astype(np.uint8)

    def k2(r, use):
        if f == NONE:
            return src[r, :8 * n].reshape(n, 8)[:, [0, 3, 6]].reshape(-1)
        q = (src[r, :8 * n].astype(np.int32) + src[r + use * stride_f, :8 * n]).reshape(n, 8)
        return np.stack([((q[:, 0] + q[:, 1] + q[:, 2]) * (65536 // 6)) >> 16,
                         ((q[:, 3] + q[:, 4] + q[:, 5]) * (65536 // 6)) >> 16,
                         (q[:, 6] + q[:, 7]) >> 2], 1).reshape(-1).astype(np.uint8)

    out = np.empty((dh, dw), np.uint8)
    r = 0; j = 0
    while j < dh - 2:
        out[j] = k3(r, 1); out[j + 1] = k3(r + 3, 1); out[j + 2] = k2(r + 6, 1)
        r += 8; j += 3
    if dh % 3 == 1:
        out[j] = k3(r, 0)
    elif dh % 3 == 2:
        out[j] = k3(r, 1); out[j + 1] = k3(r + 3, 0)
    return out


def _box(src, dw, dh):
    sh, sw = src.shape
    x, y, dx, dy = scale_slope(sw, sh, dw, dh, BOX)
    max_y = sh << 16
    out = np.empty((dh, dw), np.uint8)
    xs = _xs(dw + 1, x, dx) >> 16
    ix = xs[:-1]
    if dx == 0x10000:
        mode = 0
    elif dx & 0xFFFF:
        mode = 2
    else:
        mode = 1
    minbox = dx >> 16
    for j in range(dh):
        iy = y >> 16
        y = _i32(y + dy)
        if y >= max_y:
            y = max_y
        bh = max(1, (y >> 16) - iy)
        row = src[iy:iy + bh].astype(np.uint32).sum(0) & 0xFFFF  # ScaleAddRow accumulates in uint16
        if mode == 0:
            sv = 65536 // bh
            out[j] = ((row[ix] * sv) >> 16).astype(np.uint8)
        elif mode == 1:
            bw = max(1, minbox)
            sv = 65536 // (bw * bh)
            cs = np.concatenate([np.zeros(1, np.uint64), np.cumsum(row, dtype=np.uint64)])
            starts = (x >> 16) + bw * np.arange(dw)
            s = cs[starts + bw] - cs[starts]
            out[j] = ((s * sv) >> 16).astype(np.uint8)
        else:
            tbl = np.array([65536 // (max(1, minbox) * bh), 65536 // (max(1, minbox + 1) * bh)], np.uint64)
            bw = np.maximum(1, xs[1:] - xs[:-1])
            cs = np.concatenate([np.zeros(1, np.uint64), np.cumsum(row, dtype=np.uint64)])
            s = cs[ix + bw] - cs[ix]
            out[j] = ((s * tbl[bw - max(minbox, 0)]) >> 16).astype(np.uint8)
    return out


def _bilinear_down(src, dw, dh, f):
    sh, sw = src.shape
    x, y, dx, dy = scale_slope(sw, sh, dw, dh, f)
    max_y = (sh << 16) - 0x10000
    if y > max_y:
        y = max_y
    out = np.empty((dh, dw), np.uint8)
    for j in range(dh):
        yi = y >> 16
        if f == LINEAR:
            out[j] = filter_cols(src[yi], dw, x, dx)
        else:
            yf = (y >> 8) & 255
            r1 = src[yi + 1] if (yf and yi + 1 < sh) else src[yi]
            out[j] = filter_cols(interpolate_row(src[yi], r1, yf), dw, x, dx)
        y = _i32(y + dy)
        if y >= max_y:
            y = max_y
    return out


def _bilinear_up(src, dw, dh, f):
    sh, sw = src.shape
    x, y, dx, dy = scale_slope(sw, sh, dw, dh, f)
    max_y = (sh << 16) - 0x10000
    if y > max_y:
        y = max_y
    yi = y >> 16
    # emulate the two-row ring buffer exactly, including the source pointer that stops advancing
    sp = yi
    bufs = [filter_cols(src[sp], dw, x, dx), None]
    if sh > 1:
        sp += 1
    bufs[1] = filter_cols(src[sp], dw, x, dx)
    if sh > 2:
        sp += 1
    cur = 0  # index of the "rowptr" buffer; the other one is rowptr+rowstride
    lasty = yi
    out = np.empty((dh, dw), np.uint8)
    for j in range(dh):
        if y > max_y:
            y = max_y
        yi = y >> 16
        if yi != lasty:
            bufs[cur] = filter_cols(src[sp], dw, x, dx)
            cur ^= 1
            if y + 0x10000 < max_y:
                sp += 1
            lasty = yi
        if f == LINEAR:
            out[j] = bufs[cur]
        else:
            out[j] = interpolate_row(bufs[cur], bufs[cur ^ 1], (y >> 8) & 255)
        y = _i32(y + dy)
    return out


def _up2_linear_row(r, dw):
    out = np.empty(dw, np.uint8)
    r16 = r.astype(np.int32)
    out[0] = r[0]
    work = (dw - 1) & ~1
    if work > 0:
        n = work // 2
        a = r16[:n]; b = r16[1:n + 1]
        out[1:1 + work:2] = (a * 3 + b + 2) >> 2
        out[2:2 + work:2] = (a + b * 3 + 2) >> 2
    out[dw - 1] = r[(dw - 1) // 2]
    return out


def _up2_linear(src, dw, dh):
    sh, sw = src.shape
    out = np.empty((dh, dw), np.uint8)
    if dh == 1:
        out[0] = _up2_linear_row(src[(sh - 1) // 2], dw)
        return out
    dy = fixed_div(sh - 1, dh - 1)
    y = 0x7FFF
    for j in range(dh):
        out[j] = _up2_linear_row(src[y >> 16], dw)
        y = _i32(y + dy)
    return out


def _up2_bilinear_rows(sa, sb, dw):
    """Returns (da, db) as produced by ScaleRowUp2_Bilinear_Any."""
    a = sa.astype(np.int32); b = sb.astype(np.int32)
    da = np.empty(dw, np.uint8); db = np.empty(dw, np.uint8)
    da[0] = (3 * a[0] + b[0] + 2) >> 2
    db[0] = (a[0] + 3 * b[0] + 2) >> 2
    work = (dw - 1) & ~1
    if work > 0:
        n = work // 2
        s0, s1, t0, t1 = a[:n], a[1:n + 1], b[:n], b[1:n + 1]
        da[1:1 + work:2] = (s0 * 9 + s1 * 3 + t0 * 3 + t1 + 8) >> 4
        da[2:2 + work:2] = (s0 * 3 + s1 * 9 + t0 + t1 * 3 + 8) >> 4
        db[1:1 + work:2] = (s0 * 3 + s1 + t0 * 9 + t1 * 3 + 8) >> 4
        db[2:2 + work:2] = (s0 + s1 * 3 + t0 * 3 + t1 * 9 + 8) >> 4
    k = (dw - 1) // 2
    da[dw - 1] = (3 * a[k] + b[k] + 2) >> 2
    db[dw - 1] = (a[k] + 3 * b[k] + 2) >> 2
    return da, db


def _up2_bilinear(src, dw, dh):
    sh, sw = src.shape
    out = np.empty((dh, dw), np.uint8)
    da, _ = _up2_bilinear_rows(src[0], src[0], dw)
    out[0] = da
    o = 1
    for i in range(1, sh):
        da, db = _up2_bilinear_rows(src[i - 1], src[i], dw)
        out[o] = da; out[o + 1] = db
        o += 2
    if (dh & 1) == 0:
        da, _ = _up2_bilinear_rows(src[sh - 1], src[sh - 1], dw)
        out[o] = da
    return out


def _simple(src, dw, dh):
    sh, sw = src.shape
    x, y, dx, dy = scale_slope(sw, sh, dw, dh, NONE)
    out = np.empty((dh, dw), np.uint8)
    for j in range(dh):
        row = src[y >> 16]
        if dx < 0x8000 and 2 * sw == dw:  # ScaleColsUp2_C
            out[j] = np.repeat(row, 2)[:dw]
        else:
            out[j] = scale_cols(row, dw, x, dx)
        y = _i32(y + dy)
    return out


def scale_plane(src, dw, dh, filtering):
    """libyuv::ScalePlane for a uint8 HxW array."""
    src = np.ascontiguousarray(src, np.uint8)
    sh, sw = src.shape
    f = scale_filter_reduce(sw, sh, dw, dh, filtering)
    if dw == sw and dh == sh:
        return src.copy()
    if dw == sw and f != BOX:
        if dh <= sh:
            dy = fixed_div(sh, dh); y = _center_start(dy)
        elif sh > 1 and f:
            dy = fixed_div1(sh, dh); y = 0
        elif sh > 1:
            dy = fixed_div(sh, dh); y = dy >> 1 if dy >= 0 else -((-dy) >> 1)
        else:
            dy = y = 0
        return _vertical(src, dh, y, dy, f)
    if dw <= sw and dh <= sh:
        if 4 * dw == 3 * sw and 4 * dh == 3 * sh:
            return _down34(src, dw, dh, f)
        if 2 * dw == sw and 2 * dh == sh:
            return _down2(src, dw, dh, f)
        if 8 * dw == 3 * sw and 8 * dh == 3 * sh:
            return _down38(src, dw, dh, f)
        if 4 * dw == sw and 4 * dh == sh and f in (BOX, NONE):
            return _down4(src, dw, dh, f)
    if f == BOX and 2 * dh < sh:
        return _box(src, dw, dh)
    if f == LINEAR and (dw + 1) // 2 == sw:
        return _up2_linear(src, dw, dh)
    if f in (BILINEAR, BOX) and (dh + 1) // 2 == sh and (dw + 1) // 2 == sw:
        return _up2_bilinear(src, dw, dh)
    if f and dh > sh:
        return _bilinear_up(src, dw, dh, f)
    if f:
        return _bilinear_down(src, dw, dh, f)
    return _simple(src, dw, dh)
