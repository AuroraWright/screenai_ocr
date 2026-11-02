"""Write testdata/det_inject/<case>.txt: detector model output edits (SCREENAI_DET_INJECT / DD_INJECT format, see
OutputEdits in cpp/src/detector.cc) that give the detector post-processing exactly equal inputs, so that the sorts
whose keys never tie on real model outputs (line links, curve points, line order, paragraph NMS, angle estimation)
see ties. The first line of each file names the image it is applied to. Reference outputs: trace/det_inject_check.py
--ref. Tensor (0, 2) / (1, 2) are the level-1 (scale 1, 1280 px) horizontal / vertical heads, 120x160 cells of 8 px
(16 px boxes per unit of exp(log w / h)); channels are logit, dx, dy, log w, log h, cos, sin."""
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'det_inject')
LN2 = '0x1.62e43p-1'  # expf() == 2 exactly
LN3 = '0x1.193ea8p+0'  # expf() == 3 exactly
# Two sin channel values (with cos 1) whose box angles (~2.86 deg) differ but whose RobustAngleEstimation keys
# atan2f(sin, cos) of __sincosf_stret(angle) are equal. For this dominant direction (0) the key is also the
# estimate (atan2f of the median), so the tie order cannot change the result.
SN1, SN2 = '0x1.99a6acp-5', '0x1.99a6aep-5'
# The same for ~46.4 deg (dominant direction 1, key atan2f(-cos, sin)), where the estimate atan2f(sin, cos) of the
# two differs by one ulp: the tie order decides the level-1 and line angles.
SN46A, SN46B = '0x1.0cccdp+0', '0x1.0cccd2p+0'


def quant(t=2, lh='0'):
    """Every cell of horizontal tensor t an axis-aligned box of 16 x 16*exp(lh) px centred on its anchor."""
    return ['set 0 %d 0 0 0 0 %d %s' % (t, ch, v) for ch, v in ((1, '0'), (2, '0'), (3, '0'), (4, lh), (5, '1'),
                                                                 (6, '0'))]


def kill_all():
    return ['set 0 * 0 0 0 0 0 -20', 'set 1 * 0 0 0 0 0 -20']


def cell(y, x, logit='4', **ch):
    names = {'dx': 1, 'dy': 2, 'lw': 3, 'lh': 4, 'cos': 5, 'sin': 6}
    out = ['set 0 2 %d %d 1 1 0 %s' % (y, x, logit)]
    out += ['set 0 2 %d %d 1 1 %d %s' % (y, x, names[k], v) for k, v in ch.items()]
    return out


CASES = {}

# Line links: on real text, all horizontal boxes 16x16 at their anchors -> level-1 boxes on a 8 px grid whose pair
# costs (gap + centers + height ratio) repeat exactly; also every angle key ties (all angles 0).
for img in ('mirror_cols', 'repeat_word'):
    CASES['links_quant_' + img] = ('testdata/extra_ties/%s.png' % img,
                                   ['# all level-1 horizontal boxes 16x16 axis-aligned'] + quant())

# Paragraph NMS: every odd column of the vertical level-1 head a copy of the even column left of it -> pairs of
# paragraph predictions with equal scores, 8 px apart, which suppress each other (the sort order decides which
# survives).
CASES['nms_pairs_grid_8888'] = ('testdata/extra_ties/grid_8888.png',
                                ['# vertical level-1 head: column 2k+1 = column 2k'] +
                                ['copy 1 2 0 %d 0 %d 0 1' % (2 * k, 2 * k + 1) for k in range(80)])

# Paragraph NMS, decisive: per row two words W1, W2 (16 px gap) and three paragraph predictions, P around both and
# Q around W1 with the same score, R around W2 lower. P first: P suppresses Q and R, the words share P and form one
# line; Q first: Q suppresses P, R survives, the words are in different paragraphs and stay two lines. 12 rows ->
# 24 tied scores in a sort of 36.
L = ['# 12 rows of (W1, W2) words with paragraph predictions P = W1+W2 and Q = W1 (equal scores), R = W2'] + kill_all()
L += quant()
for i in range(12):
    y = 10 + 6 * i
    for x in list(range(10, 16)) + list(range(19, 25)):
        L += cell(y, x)
    for x, dx, lw, logit in ((16, '0.5', '2.1972246', '3'), (12, '0.5', '1.4469190', '3'), (21, '0', '1.3862944', '2')):
        L += ['set 1 2 %d %d 1 1 %d %s' % (y, x, ch, v) for ch, v in ((0, logit), (1, dx), (2, '0'), (3, lw), (4, LN2),
                                                                     (5, '1'), (6, '0'))]
CASES['nms_nested'] = ('testdata/extra_ties/grid_8888.png', L)

# Line links, decisive: a word A with two words B (half a height up) and C (half a height down) right of it at the
# same gap: links A-B and A-C cost exactly the same and the first one in the sorted order takes A's right side.
# 16 such forks, alternately with 8 and 16 px gaps (costs 1 and 1.5, interleaved so that the sort's partitioning
# moves the tied links).
L = ['# 16 forks: word A, words B and C right of it, 8 px up / down'] + kill_all() + quant()
for b in range(8):
    y = 10 + 12 * b
    for x, g in ((10, b % 2), (60, 1 - b % 2)):
        for k in range(3):
            L += cell(y, x + k) + cell(y - 1, x + 5 + g + k) + cell(y + 1, x + 5 + g + k)
CASES['links_fork'] = ('testdata/extra_ties/grid_8888.png', L)

# Angle estimation: horizontal level-1 angles alternate (by column) between two values with equal keys but different
# sin -> the median (and so the box angle) depends on the order of the tied keys.
for img in ('grid_8888',):
    CASES['angle_ulp_' + img] = ('testdata/extra_ties/%s.png' % img,
                                 ['# angles ~2.86 deg, two values with equal keys, alternating by column',
                                  'set 0 2 0 0 0 0 5 1', 'set 0 2 0 0 0 0 6 ' + SN1] +
                                 ['set 0 2 0 %d 0 1 6 %s' % (2 * k + 1, SN2) for k in range(80)])

# Angle estimation, decisive: three diagonal bands of 32x32 px boxes at ~46.4 deg, angles alternating by column
# between SN46A / SN46B.
L = ['# 3 diagonal bands of 32x32 boxes, angles ~46.4 deg alternating (equal keys) by column'] + kill_all()
L += quant(lh=LN2) + ['set 0 2 0 0 0 0 3 ' + LN2, 'set 0 2 0 0 0 0 6 ' + SN46A]
L += ['set 0 2 0 %d 0 1 6 %s' % (2 * k + 1, SN46B) for k in range(80)]
for x0, y0, n in ((10, 5, 60), (60, 5, 41), (100, 30, 50)):
    for k in range(n):
        for j in range(2):
            L += cell(y0 + k + j, x0 + k)
CASES['angle_diag_alt'] = ('testdata/extra_ties/grid_8888.png', list(L))  # all keys equal
# Every third row a slightly larger / smaller angle (distinct keys, as many above as below the tied ones), so the
# sort partitions the tied keys instead of finding them already in order.
L += ['set 0 2 %d 0 1 0 6 %s' % (y, '1.06' if y % 6 == 0 else '1.04') for y in range(0, 120, 3)]
CASES['angle_diag'] = ('testdata/extra_ties/grid_8888.png', L)

# Line order: a column of alternately 16 and 48 px tall boxes, each overlapping the next by exactly 3/4 of the
# smaller height (the only overlap that ConstructLines' pair checks accept for boxes above each other), chained into
# one line with a row of 12 words to the right of the top box. All 24 stacked boxes share their left edge -> 24 tied
# projections in a sort of 36 (partitioned, so reordered), and whether the first box after sorting is a 48 px one
# decides whether it is split off as its own line.
x0, y0 = 10, 10
L = ['# a stack of 12 (16 px, 48 px) box pairs at column %d, 12 three-cell words right of the top one' % x0]
L += kill_all() + quant()
for j in range(12):
    L += cell(y0 + 5 * j, x0)                       # 16 px: rows 8y-4 .. 8y+12
    L += cell(y0 + 5 * j + 2, x0, dy='0.5', lh=LN3)  # 48 px: rows 8y .. 8y+48 (y = the 16 px box's row)
for i in range(12):
    for k in range(3):
        L += cell(y0, x0 + 3 + 5 * i + k)
CASES['order_stack'] = ('testdata/extra_ties/grid_8888.png', L)

# Same stack with its words to the right of the bottom box instead.
L = ['# the order_stack column, words right of the bottom 16 px box'] + kill_all() + quant()
for j in range(12):
    L += cell(y0 + 5 * j, x0)
    L += cell(y0 + 5 * j + 2, x0, dy='0.5', lh=LN3)
yb = y0 + 5 * 12
L += cell(yb, x0)
for i in range(12):
    for k in range(3):
        L += cell(yb, x0 + 3 + 5 * i + k)
CASES['order_stack_bottom'] = ('testdata/extra_ties/grid_8888.png', L)

# A stack starting with a 48 px box, words right of it overlapping its top 12 px (so they come first in the input
# order and the 48 px box is the first stacked one).
L = ['# a stack of 12 (48 px, 16 px) box pairs at column %d, words right of the top one' % x0] + kill_all() + quant()
for j in range(12):
    L += cell(y0 + 5 * j, x0, dy='0.5', lh=LN3)  # 48 px: rows 8y-16 .. 8y+32
    L += cell(y0 + 5 * j + 3, x0)               # 16 px: rows 8y+20 .. 8y+36
for i in range(12):
    for k in range(3):
        L += cell(y0 - 2, x0 + 3 + 5 * i + k)     # rows 8y-20 .. 8y-4
CASES['order_stack_tall_first'] = ('testdata/extra_ties/grid_8888.png', L)

# Curve points: lines of two-row words (16x32 px boxes 8 px apart, which level-1 grouping joins), one box per line
# tilted ~11 deg so the line is curved (min cos < 0.99) while its robust angle stays exactly 0: the edge midpoints of
# the two rows (and of boxes two columns apart) project to the same position along the line.
L = ['# 6 lines of 6 two-row words, 16x32 px boxes, one tilted box per line'] + kill_all() + quant(lh=LN2)
for r in range(6):
    y = 10 + 16 * r
    for w in range(6):
        for k in range(6):
            for dy in (0, 1):
                L += cell(y + dy, 8 + 15 * w + k)
    L += cell(y, 8 + 15 * 2 + 3, sin='0.2')
CASES['curve_rows'] = ('testdata/extra_ties/grid_8888.png', L)

# The same on real text: all boxes 16x32 at their anchors, one tilted column.
CASES['curve_quant_mirror_cols'] = ('testdata/extra_ties/mirror_cols.png',
                                    ['# all level-1 horizontal boxes 16x32 axis-aligned, column 40 tilted'] +
                                    quant(lh=LN2) + ['set 0 2 0 40 0 1 6 0.2'])

if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    for name, (img, lines) in CASES.items():
        with open(os.path.join(OUT, name + '.txt'), 'w') as f:
            f.write('# image: %s\n' % img + '\n'.join(lines) + '\n')
    print(len(CASES), 'cases')
