# SPDX-License-Identifier: MPL-2.0
"""Differential fuzzing of the layout analysis: seeded random edits of real 002_lay_in dumps, run through the original
(trace/lay_inject.py) and cpp run_layout, compared with tools/cmp_layout.py.

  lay_fuzz.py gen DIR [--n 300] [--seed 1] [--ops 1-3] [stage dirs...]   random cases (default bases: testdata/stages*)
  lay_fuzz.py targeted DIR                                              line-merge threshold / hyphen cases
  lay_fuzz.py orig DIR [--chunk 40]                                     original's per-step dumps -> DIR/<case>/orig/
  lay_fuzz.py cmp DIR <run_layout> <model_dir> [--cases a,b]            ours vs original, failures bucketed by step

A case is DIR/<case>/{in.pb, img.png (symlink to the base image), ops.txt}. The image is the base dump's own, so word
colors and page size are consistent on both sides. Bucketing reruns each of our steps on the original's input of that
step (per-step dumps) and reports the first step whose output differs.
"""
import argparse, copy, glob, json, math, os, random, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from pool import load_pool  # noqa: E402
from google.protobuf import message_factory  # noqa: E402

_cls = None


def layout_class():
    global _cls
    if _cls is None:
        pool, _, _ = load_pool(os.path.join(ROOT, 'protos_extracted'))
        _cls = message_factory.GetMessageClass(pool.FindMessageTypeByName('goodoc.PageLayoutProto'))
    return _cls


def load(path):
    m = layout_class()(); m.ParseFromString(open(path, 'rb').read()); return m


def image_for(stage_dir):
    d = os.path.abspath(stage_dir.rstrip('/'))
    name, stage_set = os.path.basename(d), os.path.basename(os.path.dirname(d))
    sub = stage_set[len('stages_'):] if stage_set.startswith('stages_') else ''
    return os.path.join(ROOT, 'testdata', sub, name + '.png')


def default_bases():
    out = []
    for d in sorted(glob.glob(os.path.join(ROOT, 'testdata', 'stages*', '*'))):
        if 'holdout' in d or 'stages_fuzz' in d or not os.path.exists(os.path.join(d, '002_lay_in.pb')): continue
        if os.path.exists(image_for(d)): out.append(d)
    return out


# ---- entity list editing (parent ids are indices) ----

LINE, WORD, SYMBOL, PARAGRAPH = 2, 0, 5, 3


def parent(e):
    return e.parent_entity_id if e.HasField('parent_entity_id') else -1


def children(E, i):
    return [j for j, e in enumerate(E) if parent(e) == i]


def subtree(E, i):
    out = [i]
    for k in out: out.extend(children(E, k))
    return out


def rebuild(m, E, keep):
    """Keeps entities `keep` (in that order), remapping parent ids (dropped parents -> cleared)."""
    remap = {old: new for new, old in enumerate(keep)}
    out = []
    for old in keep:
        e = copy.deepcopy(E[old])
        if e.HasField('parent_entity_id'):
            p = remap.get(e.parent_entity_id)
            if p is None: e.ClearField('parent_entity_id')
            else: e.parent_entity_id = p
        out.append(e)
    del m.entities[:]
    for e in out: m.entities.add().CopyFrom(e)


def delete(m, ids):
    E = list(m.entities); gone = set()
    for i in ids: gone.update(subtree(E, i))
    rebuild(m, E, [i for i in range(len(E)) if i not in gone])


def boxes_of(e):
    """Rotated boxes of an entity (polygon and break box) that geometry edits move together."""
    out = []
    if e.polygon.HasField('rotated_box'): out.append(e.polygon.rotated_box)
    if e.polygon.HasField('curved_box'): out.append(e.polygon.curved_box.rotated_bounding_box)
    if e.word_break.HasField('box'): out.append(e.word_break.box)
    return out


def shift(e, dx, dy):
    for b in boxes_of(e): b.Left += dx; b.Top += dy
    if e.polygon.HasField('curved_box'):
        for p in e.polygon.curved_box.mid_line_curve.points: p.x += dx; p.y += dy


def copy_subtree(m, i, new_parent, dx, dy):
    """Appends a copy of entity i's subtree (children after their parents) under new_parent."""
    E = list(m.entities); ids = subtree(E, i); base = len(E); remap = {}
    for k, old in enumerate(ids): remap[old] = base + k
    for old in ids:
        e = m.entities.add(); e.CopyFrom(E[old])
        if old == i:
            if new_parent < 0: e.ClearField('parent_entity_id')
            else: e.parent_entity_id = new_parent
        else: e.parent_entity_id = remap[parent(E[old])]
        shift(e, dx, dy)
    return base


def of_type(m, t):
    return [i for i, e in enumerate(m.entities) if e.entity_type == t]


def line_box(e):
    return e.polygon.rotated_box if e.polygon.HasField('rotated_box') else e.polygon.curved_box.rotated_bounding_box


def union_box(m, ids, angle):
    bs = [line_box(m.entities[i]) for i in ids if boxes_of(m.entities[i])]
    l = min(b.Left for b in bs); t = min(b.Top for b in bs)
    r = max(b.Left + b.Width for b in bs); btm = max(b.Top + b.Height for b in bs)
    return l, t, r - l, btm - t, angle


# ---- mutations ----

def op_jitter(m, R):
    c = [i for i, e in enumerate(m.entities) if e.entity_type in (LINE, WORD, SYMBOL) and boxes_of(e)]
    if not c: return None
    i = R.choice(c); b = boxes_of(m.entities[i])[0]
    dl, dt, dw, dh = (R.randint(-3, 3) for _ in range(4))
    b.Left += dl; b.Top += dt; b.Width = max(1, b.Width + dw); b.Height = max(1, b.Height + dh)
    if R.random() < 0.4: b.Angle += R.uniform(-1.0, 1.0)
    return 'jitter %d %d,%d,%d,%d' % (i, dl, dt, dw, dh)


def op_angle(m, R):
    c = of_type(m, LINE)
    if not c: return None
    i = R.choice(c); da = R.choice([R.uniform(-4, 4), R.uniform(-0.5, 0.5), R.choice([90.0, -90.0, 180.0])])
    for k in subtree(list(m.entities), i):
        for b in boxes_of(m.entities[k]): b.Angle += da
    return 'angle %d %.3f' % (i, da)


def op_dup_line(m, R):
    c = of_type(m, LINE)
    if not c: return None
    i = R.choice(c); b = line_box(m.entities[i])
    dx = R.randint(-b.Width // 2, b.Width // 2) if R.random() < 0.5 else R.randint(-4, 4)
    dy = R.randint(-b.Height, b.Height) if R.random() < 0.5 else R.randint(-3, 3)
    copy_subtree(m, i, parent(m.entities[i]), dx, dy)
    return 'dup_line %d %d,%d' % (i, dx, dy)


def op_split_line(m, R):
    E = list(m.entities)
    c = [i for i in of_type(m, LINE) if len([j for j in children(E, i) if E[j].entity_type == WORD]) >= 2]
    if not c: return None
    i = R.choice(c); words = [j for j in children(E, i) if E[j].entity_type == WORD]
    k = R.randint(1, len(words) - 1)
    new = m.entities.add(); new.CopyFrom(E[i])
    for j in words[k:]: m.entities[j].parent_entity_id = len(m.entities) - 1
    ang = line_box(E[i]).Angle
    for line, ws in ((m.entities[i], words[:k]), (new, words[k:])):
        if line.polygon.HasField('curved_box'): continue
        l, t, w, h, a = union_box(m, ws, ang)
        rb = line.polygon.rotated_box; rb.Left, rb.Top, rb.Width, rb.Height, rb.Angle = l, t, w, h, a
    # the new line is last: reorder so its subtree follows (children index order is what matters)
    return 'split_line %d at %d' % (i, k)


def op_del_word(m, R):
    c = of_type(m, WORD)
    if not c: return None
    i = R.choice(c); delete(m, [i]); return 'del_word %d' % i


def op_del_symbol(m, R):
    c = of_type(m, SYMBOL)
    if not c: return None
    i = R.choice(c); delete(m, [i]); return 'del_symbol %d' % i


def op_break(m, R):
    c = of_type(m, WORD)
    if not c: return None
    i = R.choice(c); e = m.entities[i]; t = R.choice([0, 1, 2, 3, 4])
    e.word_break.type = t
    if R.random() < 0.3: e.word_break.ClearField('box')
    elif t == 4 and not e.word_break.HasField('box') and boxes_of(e):
        b = boxes_of(e)[0]; bb = e.word_break.box
        bb.Left, bb.Top, bb.Width, bb.Height, bb.Angle = b.Left + b.Width, b.Top, max(1, b.Height // 3), b.Height, b.Angle
    return 'break %d %d' % (i, t)


def op_hyphen_symbol(m, R):
    """Appends a '-' symbol to a word (trailing hyphen)."""
    E = list(m.entities)
    c = [i for i in of_type(m, WORD) if children(E, i)]
    if not c: return None
    i = R.choice(c); last = children(E, i)[-1]
    s = m.entities.add(); s.CopyFrom(E[last]); s.text_content = '-'
    b = s.polygon.rotated_box; b.Left += b.Width; b.Width = max(1, b.Width // 2)
    m.entities[i].text_content = m.entities[i].text_content + '-'
    return 'hyphen_symbol %d' % i


SCRIPTS = ['Latn', 'Jpan', 'Arab', 'und', 'Hani', 'Deva', 'Cyrl', 'Hebr', 'Kore', 'Thai']
LANGS = ['en', 'ja', 'ar', 'zh', 'hi', 'ru', 'he', 'de', 'und', 'ko']


def op_script(m, R):
    c = of_type(m, LINE)
    if not c: return None
    i = R.choice(c); k = R.randrange(len(SCRIPTS)); E = list(m.entities)
    m.entities[i].recognizer_script_id = SCRIPTS[k]
    for j in subtree(E, i):
        e = m.entities[j]
        if e.entity_type == WORD and R.random() < 0.7:
            del e.language_mix.languages[:]; e.language_mix.languages.add().bcp47_tag = LANGS[k]
    return 'script %d %s' % (i, SCRIPTS[k])


def op_ctype(m, R):
    c = [i for i, e in enumerate(m.entities) if e.entity_type in (LINE, WORD, SYMBOL, PARAGRAPH)]
    if not c: return None
    i = R.choice(c); t = R.choice([0, 1, 2, 3, 5, 6, 7, 100])
    m.entities[i].type = t; return 'ctype %d %d' % (i, t)


def op_conf(m, R):
    c = [i for i, e in enumerate(m.entities) if e.entity_type in (LINE, WORD, SYMBOL)]
    if not c: return None
    i = R.choice(c); e = m.entities[i]
    f = R.choice(['confidence', 'direction_confidence', 'type_confidence'])
    v = R.choice([0.0, R.random(), 1.0]); setattr(e, f, v)
    return 'conf %d %s %.4f' % (i, f, v)


def op_curve(m, R):
    c = [i for i in of_type(m, LINE) if m.entities[i].polygon.HasField('rotated_box')]
    if not c: return None
    i = R.choice(c); e = m.entities[i]; rb = copy.deepcopy(e.polygon.rotated_box)
    a = math.radians(rb.Angle); n = R.randint(2, 6); bend = R.uniform(-1, 1) * rb.Height
    e.polygon.ClearField('rotated_box'); cb = e.polygon.curved_box
    for k in range(n):
        s = k / (n - 1); x = s * rb.Width; y = rb.Height / 2 + bend * 4 * s * (1 - s)
        p = cb.mid_line_curve.points.add()
        p.x = rb.Left + x * math.cos(a) - y * math.sin(a); p.y = rb.Top + x * math.sin(a) + y * math.cos(a)
    cb.thickness = float(rb.Height); cb.top_to_bottom = False
    cb.rotated_bounding_box.CopyFrom(rb)
    return 'curve %d n=%d bend=%.1f' % (i, n, bend)


def op_orient(m, R):
    c = of_type(m, LINE)
    if not c: return None
    i = R.choice(c); E = list(m.entities)
    o, wd, tl = R.choice([(0, 0, 0), (0, 1, 0), (0, 2, 1), (1, 0, 0), (2, 0, 0), (3, 0, 0), (0, 2, 2)])
    for j in subtree(E, i):
        e = m.entities[j]
        if e.HasField('orientation'):
            e.orientation.orientation, e.orientation.writing_direction, e.orientation.textline_order = o, wd, tl
    return 'orient %d %d,%d,%d' % (i, o, wd, tl)


def op_overlap(m, R):
    """Copies a line onto another line's position (partially overlapping)."""
    c = of_type(m, LINE)
    if len(c) < 2: return None
    i, j = R.sample(c, 2); bi, bj = line_box(m.entities[i]), line_box(m.entities[j])
    dx = bj.Left - bi.Left + R.randint(-bj.Width // 3, bj.Width // 3); dy = bj.Top - bi.Top + R.randint(-bj.Height // 2, bj.Height // 2)
    copy_subtree(m, i, parent(m.entities[j]), dx, dy)
    return 'overlap %d onto %d %d,%d' % (i, j, dx, dy)


def op_dup_word(m, R):
    c = of_type(m, WORD)
    if not c: return None
    i = R.choice(c); b = boxes_of(m.entities[i])
    dx = R.randint(-b[0].Width, b[0].Width) if b else 0
    copy_subtree(m, i, parent(m.entities[i]), dx, R.randint(-2, 2))
    return 'dup_word %d %d' % (i, dx)


def op_move_line(m, R):
    """Moves a line into another (or a new) paragraph."""
    c = of_type(m, LINE); ps = of_type(m, PARAGRAPH)
    if not c: return None
    i = R.choice(c)
    if len(ps) > 1 and R.random() < 0.7: p = R.choice([q for q in ps if q != parent(m.entities[i])])
    else:
        e = m.entities.add(); e.entity_type = PARAGRAPH; p = len(m.entities) - 1
    m.entities[i].parent_entity_id = p
    return 'move_line %d -> %d' % (i, p)


OPS = [(op_dup_word, 2), (op_move_line, 1), (op_jitter, 4), (op_angle, 1), (op_dup_line, 2), (op_split_line, 2), (op_del_word, 2), (op_del_symbol, 1),
       (op_break, 3), (op_hyphen_symbol, 1), (op_script, 1), (op_ctype, 1), (op_conf, 2), (op_curve, 1),
       (op_orient, 1), (op_overlap, 2)]


def write_case(d, name, m, img, ops):
    c = os.path.join(d, name); os.makedirs(c, exist_ok=True)
    open(os.path.join(c, 'in.pb'), 'wb').write(m.SerializeToString())
    link = os.path.join(c, 'img.png')
    if os.path.lexists(link): os.remove(link)
    os.symlink(img, link)
    open(os.path.join(c, 'ops.txt'), 'w').write('\n'.join(ops) + '\n')


def cmd_gen(a):
    bases = a.bases or default_bases(); R = random.Random(a.seed)
    lo, hi = map(int, a.ops.split('-'))
    for k in range(a.n):
        b = R.choice(bases); m = load(os.path.join(b, '002_lay_in.pb'))
        ops = ['base %s' % os.path.relpath(b, ROOT)]
        for _ in range(R.randint(lo, hi)):
            fn = R.choices([f for f, _ in OPS], [w for _, w in OPS])[0]
            r = fn(m, R)
            if r: ops.append(r)
        write_case(a.dir, 'f%04d_%s' % (k + a.start, os.path.basename(b)), m, image_for(b), ops)
    print('%d cases in %s' % (a.n, a.dir))


# ---- targeted line-merge cases (base: a dump where MergeLines merges two lines) ----

def merge_pair(m):
    """The two lines of strip_text's paragraph (A then B)."""
    ls = of_type(m, LINE); return ls[0], ls[1]


def words(m, line):
    E = list(m.entities); return [j for j in children(E, line) if E[j].entity_type == WORD]


def set_subtree_angle(m, i, a):
    for k in subtree(list(m.entities), i):
        for b in boxes_of(m.entities[k]): b.Angle = a


def scale_thickness(m, i, f):
    """Scales the height of line i and its words/symbols about the line's center row."""
    lb = line_box(m.entities[i]); cy = lb.Top + lb.Height / 2.0
    for k in subtree(list(m.entities), i):
        for b in boxes_of(m.entities[k]):
            h = max(1, int(round(b.Height * f))); c = b.Top + b.Height / 2.0 - cy
            b.Top = int(round(cy + c * f - h / 2.0)); b.Height = h


def cmd_targeted(a):
    base = os.path.join(ROOT, 'testdata', 'stages_extra_det', 'strip_text')
    img = image_for(base); src = load(os.path.join(base, '002_lay_in.pb'))

    def case(name, fn, *ops):
        m = copy.deepcopy(src); fn(m); write_case(a.dir, 't_' + name, m, img, ['base strip_text'] + list(ops))

    def soft_hyphen(m, w, boxed=True):
        e = m.entities[w]; e.word_break.type = 4
        if not boxed: e.word_break.ClearField('box'); return
        b = boxes_of(e)[0]; bb = e.word_break.box
        bb.Left, bb.Top, bb.Width, bb.Height, bb.Angle = b.Left + b.Width, b.Top, max(1, b.Width // 4), b.Height, b.Angle
    def soft_hyphen_join(m):  # A's last word ends in a soft hyphen at the join
        A, B = merge_pair(m); soft_hyphen(m, words(m, A)[-1])
    def soft_hyphen_end(m):  # B's last word (the merged line's end) has a soft hyphen
        A, B = merge_pair(m); soft_hyphen(m, words(m, B)[-1])
    # A SOFT_HYPHEN break without a box: the symbol gets an empty rotated_box message (fields unset), which our
    # RotatedBox cannot represent; the original never hands layout analysis such a break.
    case('soft_hyphen_nobox', lambda m: soft_hyphen(m, words(m, merge_pair(m)[0])[-1], boxed=False),
         'A last word SOFT_HYPHEN without box (not producible upstream)')
    def trailing_dash(line_sel):
        def f(m):
            A, B = merge_pair(m); line = A if line_sel == 'A' else B
            w = words(m, line)[-1]; E = list(m.entities); last = children(E, w)[-1]
            s = m.entities.add(); s.CopyFrom(E[last]); s.text_content = '-'
            b = s.polygon.rotated_box; b.Left += b.Width; b.Width = max(1, b.Width // 2)
            m.entities[w].text_content += '-'
        return f
    def single_dash_word(m):  # B ends with a one-symbol "-" word
        A, B = merge_pair(m); w = words(m, B)[-1]
        delete(m, children(list(m.entities), w)[:-1])  # symbols follow B, so A, B and w keep their indices
        m.entities[w].text_content = '-'; m.entities[children(list(m.entities), w)[0]].text_content = '-'
    case('soft_hyphen_join', soft_hyphen_join, 'A last word break SOFT_HYPHEN')
    case('soft_hyphen_end', soft_hyphen_end, 'B last word break SOFT_HYPHEN')
    case('soft_hyphen_both', lambda m: (soft_hyphen_join(m), soft_hyphen_end(m)), 'both')
    case('dash_end', trailing_dash('B'), 'B last word + "-" symbol')
    case('dash_join', trailing_dash('A'), 'A last word + "-" symbol')
    case('dash_word_end', single_dash_word, 'B last word is a single "-"')

    # thresholds: angle difference (3.0), thickness ratio (0.6), thickness overlap (0.6), depth gap (2.5)
    for da in [2.5, 2.9, 2.99, 3.0, 3.01, 3.1, 3.5, 357.0, 183.0]:
        def f(m, da=da):
            A, B = merge_pair(m); set_subtree_angle(m, B, line_box(m.entities[A]).Angle + da)
        case('angle_%g' % da, f, 'B angle = A angle + %g' % da)
    def short_b(m, keep=2):  # B cut to its first words, so rotating it barely changes the merged thickness
        A, B = merge_pair(m); ws = words(m, B)
        delete(m, [ws[0]] + ws[1 + keep:])  # B's first word duplicates A's last one (word pruning would drop B)
        ws = words(m, B); rb = m.entities[B].polygon.rotated_box
        rb.Left, rb.Top, rb.Width, rb.Height, rb.Angle = union_box(m, ws, rb.Angle)
    def short_a(m, keep=2):  # A cut to its last words (B's words follow A's, so B keeps its index)
        A, B = merge_pair(m); ws = words(m, A)
        delete(m, ws[:-keep])
        A, B = merge_pair(m); ws = words(m, A); rb = m.entities[A].polygon.rotated_box
        rb.Left, rb.Top, rb.Width, rb.Height, rb.Angle = union_box(m, ws, rb.Angle)
    for da in [0.0, 1.0, 2.0, 2.9, 2.99, 3.0, 3.01, 3.1, 4.0, -2.9, -3.1, 357.5, 356.9]:
        def f(m, da=da):
            short_b(m, 1); short_a(m, 1); A, B = merge_pair(m)
            set_subtree_angle(m, B, line_box(m.entities[A]).Angle + da)
        case('short_angle_%g' % da, f, 'A, B cut to 1 word, B angle = A angle + %g' % da)
    for r in [0.5, 0.6, 0.7, 1.6, 1.7, 1.8]:
        def f(m, r=r):
            short_b(m); A, B = merge_pair(m); scale_thickness(m, B, r)
        case('shortb_thick_%g' % r, f, 'B cut to 2 words, thickness x %g' % r)
    for r in [0.5, 0.58, 0.6, 0.62, 0.7, 1.5, 1.66, 1.7]:
        def f(m, r=r):
            A, B = merge_pair(m); scale_thickness(m, B, r)
        case('thick_%g' % r, f, 'B thickness x %g' % r)
    for dy in [8, 9, 10, 11, 12, 14, 16, 22, -8, -9, -10, -12, -16]:
        def f(m, dy=dy):
            A, B = merge_pair(m)
            for k in subtree(list(m.entities), B): shift(m.entities[k], 0, dy)
        case('shift_y_%d' % dy, f, 'B moved by dy %d' % dy)
    for dx in [-40, 0, 20, 40, 60, 70, 72, 74, 76, 78, 80, 100]:
        def f(m, dx=dx):
            A, B = merge_pair(m)
            for k in subtree(list(m.entities), B): shift(m.entities[k], dx, 0)
        case('shift_x_%d' % dx, f, 'B moved by dx %d' % dx)
    def swap(m):  # B before A in index order
        E = list(m.entities); A, B = merge_pair(m); p = parent(E[A])
        sa, sb = subtree(E, A), subtree(E, B); rest = [i for i in range(len(E)) if i not in sa and i not in sb]
        rebuild(m, E, rest[:rest.index(p) + 1] + sb + sa + rest[rest.index(p) + 1:])
    case('swap_order', swap, 'B before A')
    def three(m):  # a third copy continuing B
        A, B = merge_pair(m); copy_subtree(m, B, parent(m.entities[B]), 700, 0)
    case('three_lines', three, 'B copied 700 px right')
    def rtl(m):
        for e in m.entities:
            if e.HasField('orientation'): e.orientation.writing_direction = 1
    case('rtl', rtl, 'all RIGHT_TO_LEFT')
    print('targeted cases in %s' % a.dir)


# ---- running ----

def case_dirs(d, sel=None):
    out = sorted(c for c in glob.glob(os.path.join(d, '*')) if os.path.exists(os.path.join(c, 'in.pb')))
    if sel: out = [c for c in out if os.path.basename(c) in sel]
    return out


def cmd_orig(a):
    cases = [c for c in case_dirs(a.dir, a.cases) if a.force or not glob.glob(os.path.join(c, 'orig', '*_final.pb'))]
    for k in range(0, len(cases), a.chunk):
        chunk = cases[k:k + a.chunk]; man = os.path.join(a.dir, 'manifest.txt')
        with open(man, 'w') as f:
            for c in chunk: f.write('%s %s\n' % (os.path.join(c, 'in.pb'), os.path.join(c, 'orig')))
        imgs = [os.path.join(c, 'img.png') for c in chunk]  # symlinks: run_ref writes img.ref.json next to them
        env = dict(os.environ, LI_CASES=man)
        log = open(os.path.join(a.dir, 'orig_%04d.log' % k), 'w')
        subprocess.run(['lldb', '-b', '-o', 'command script import %s' % os.path.join(ROOT, 'trace', 'lay_inject.py'),
                        '-o', 'run', '--', os.path.join(ROOT, '.venv', 'bin', 'python'),
                        os.path.join(ROOT, 'ref', 'run_ref.py')] + imgs, cwd=ROOT, env=env, stdout=log, stderr=log)
        done = sum(bool(glob.glob(os.path.join(c, 'orig', '*_final.pb'))) for c in chunk)
        print('orig: %d-%d: %d/%d finished' % (k, k + len(chunk), done, len(chunk)), flush=True)


STEPS = [('pre', 'Mutator_in', 'ConfidenceFilterSpec'),
         ('ConfidenceFilter', 'ConfidenceFilterSpec', 'RemoveOverlapsWordPruningStep'),
         ('WordPruning', 'RemoveOverlapsWordPruningStep', 'FillSymbolGapsSpec'),
         ('FillSymbolGaps', 'FillSymbolGapsSpec', 'RemoveOverlappingWithCurvedBoxesSpec'),
         ('CurvedOverlaps', 'RemoveOverlappingWithCurvedBoxesSpec', 'ClusterSortGcnSpec'),
         ('ClusterSort', 'ClusterSortGcnSpec', 'EstimateLinesSpec'),
         ('EstimateLines', 'EstimateLinesSpec', 'EstimateWordColorsSpec'),
         ('AnalyzerEnd', 'EstimateWordColorsSpec', 'Analyzer_out'),
         ('Colors+PageLanguage', 'Analyzer_out', 'Mutator_out')]


def dump(c, name):
    g = sorted(glob.glob(os.path.join(c, 'orig', '[0-9][0-9][0-9]_%s.pb' % name)))
    return g[0] if g else None


def compare(ref, ours):
    """(mismatch count, page fields equal, first lines of the diff)."""
    r = subprocess.run([sys.executable, os.path.join(HERE, 'cmp_layout.py'), ref, ours, '--max', '8'],
                       capture_output=True, text=True)
    lines = r.stdout.strip().splitlines()
    tot = next((int(l.split()[-1]) for l in lines if l.startswith('TOTAL mismatches')), -1)
    a, b = load(ref), load(ours)
    page = all(getattr(a, f) == getattr(b, f) for f in ('language_mix', 'creator_id'))
    return tot, page, [l for l in lines if l.startswith('  ')][:8] + [l for l in lines if 'ref' in l and 'ours' in l and not l.startswith(' ')]


def run_ours(binary, models, img, src, out, steps):
    cur = src
    for k, s in enumerate(steps):
        dst = out if k == len(steps) - 1 else out + '.%d' % k
        r = subprocess.run([binary, models, img, cur, dst] + (['--step', s] if s else []), capture_output=True, text=True)
        if r.returncode: return r.stderr.strip() or 'exit %d' % r.returncode
        cur = dst
    return None


def cmd_cmp(a):
    res = {}; buckets = {}
    for c in case_dirs(a.dir, a.cases):
        name = os.path.basename(c); final = dump(c, 'final')
        if not final:
            res[name] = {'status': 'no original output'}; buckets.setdefault('no-orig', []).append(name); continue
        img = os.path.join(c, 'img.png'); out = os.path.join(c, 'ours.pb')
        err = run_ours(a.binary, a.models, img, os.path.join(c, 'in.pb'), out, [None])
        if err:
            res[name] = {'status': 'ours failed: ' + err}; buckets.setdefault('ours-failed', []).append(name); continue
        tot, page, diff = compare(final, out)
        if tot == 0 and page:
            res[name] = {'status': 'ok'}; continue
        first = None
        for s, i, o in STEPS:
            src, exp = dump(c, i), dump(c, o)
            if not src or not exp: continue
            so = os.path.join(c, 'step_%s.pb' % s.replace('+', '_'))
            e = run_ours(a.binary, a.models, img, src, so, s.split('+'))
            if e: first = (s, -1, [e]); break
            t, p, d = compare(exp, so)
            if t != 0 or (s.startswith('Colors') and not p): first = (s, t, d); break
        b = first[0] if first else 'final-only'
        buckets.setdefault(b, []).append(name)
        res[name] = {'status': 'diff', 'mismatches': tot, 'page_ok': page, 'first_step': b,
                     'step_diff': first[2] if first else diff}
    json.dump(res, open(os.path.join(a.dir, 'results.json'), 'w'), indent=1)
    ok = sum(r['status'] == 'ok' for r in res.values())
    print('%d/%d cases match' % (ok, len(res)))
    for b, names in sorted(buckets.items(), key=lambda x: -len(x[1])):
        print('  %-22s %4d  %s' % (b, len(names), ' '.join(names[:6])))


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest='cmd', required=True)
    g = sp.add_parser('gen'); g.add_argument('dir'); g.add_argument('bases', nargs='*')
    g.add_argument('--n', type=int, default=300); g.add_argument('--seed', type=int, default=1)
    g.add_argument('--ops', default='1-3'); g.add_argument('--start', type=int, default=0)
    t = sp.add_parser('targeted'); t.add_argument('dir')
    o = sp.add_parser('orig'); o.add_argument('dir'); o.add_argument('--chunk', type=int, default=40)
    o.add_argument('--cases', type=lambda s: s.split(','), default=None); o.add_argument('--force', action='store_true')
    c = sp.add_parser('cmp'); c.add_argument('dir'); c.add_argument('binary'); c.add_argument('models')
    c.add_argument('--cases', type=lambda s: s.split(','), default=None)
    a = ap.parse_args()
    {'gen': cmd_gen, 'targeted': cmd_targeted, 'orig': cmd_orig, 'cmp': cmd_cmp}[a.cmd](a)


if __name__ == '__main__':
    main()
