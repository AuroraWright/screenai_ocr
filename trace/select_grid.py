"""Generate the recognizer-selection test grid (LanguageBasedLineRecognizerConfigSelector::Select inputs).

usage: select_grid.py OUTDIR
Writes OUTDIR/grid.txt (one case per line: content_type writing_direction recognizer_script_id languages, each
field '-' when unset, strings hex-encoded, '=' for the empty string, languages comma-separated) and
OUTDIR/grid.bin ([u32 n] then per case [u32 len][serialized goodoc.PageLayoutEntity]) for trace/select_check.py.
"""
import os, struct, sys

LANGID = ('af am ar ar-Latn az be bg bg-Latn bn bs ca ceb co cs cy da de el el-Latn en eo es et eu fa fi fil fr fy '
          'ga gd gl gu ha haw hi hi-Latn hmn hr ht hu hy id ig is it iw ja ja-Latn jv ka kk km kn ko ku ky la lb lo '
          'lt lv mg mi mk ml mn mr ms mt my ne nl no ny pa pl ps pt ro ru ru-Latn sd si sk sl sm sn so sq sr st su sv '
          'sw ta te tg th tr uk unknown ur uz vi xh yi yo zh zh-Latn zu').split()
EXTRA = ['zh-Hant', 'zh-Hans', 'zh-TW', 'zh-CN', 'zh-HK', 'zh-MO', 'zh-SG', 'zh-Hant-TW', 'zh-Hans-CN', 'zh-Hani',
         'sr-Latn', 'sr-Cyrl', 'sr-ME', 'ja-Latn', 'ja-Jpan', 'ja-JP', 'ko-KR', 'ko-Kore', 'en-US', 'iw-IL', 'he',
         'pa-Arab', 'pa-PK', 'uz-Cyrl', 'az-Cyrl', 'az-IR', 'mn-Mong', 'ms-Arab', 'yue', 'yue-Hans', 'cmn', 'nan',
         'ru-PETR1708', 'und', 'mul', '', 'xx', 'qaa', 'und-Hani', 'und-Hans', 'und-Hant', 'und-Arab', 'und-Cyrl',
         'und-Latn', 'und-TW', 'mul-Cyrl', 'Hani', 'Hans', 'Hant', 'en-Latn', 'EN', 'ZH-hant', 'zh_TW', 'zh-',
         '-Cyrl', 'x-foo', 'zh-Hant-t-i0-handwrit', 'zh-t-i0-handwrit', 'ja-t-i0-handwrit', 'en-t-i0-handwrit',
         'ru-t-i0-handwrit', 'Cyrl-t-i0-handwrit', 'und-t-i0-handwrit', 'zh-t-i0-handwrit-x', 'zh-t-i0-handwrit:vert',
         'Hani:vert', 'zh:vert', 'ar-t-i0-und']
SCRIPTS = ['Latn', 'Zyyy', 'Zzzz', 'Zinh', 'Hani', 'Hans', 'Hant', 'Jpan', 'Kore', 'Hira', 'Kana', 'Hang', 'Cyrl',
           'Arab', 'Hebr', 'Grek', 'Deva', 'Beng', 'Gujr', 'Guru', 'Knda', 'Mlym', 'Taml', 'Telu', 'Sinh', 'Thai',
           'Laoo', 'Khmr', 'Geor', 'Armn', 'Ethi', 'Mymr', 'Tibt', 'Mong', 'und']
FLAGS = [(ct, wd) for ct in (None, 0, 1, 2, 101) for wd in (None, 0, 1, 2, 3)]
FEW_FLAGS = [(None, None), (1, None), (None, 2), (1, 2), (101, None)]

def varint(v):
    out = b''
    while True:
        b = v & 127; v >>= 7
        if v: out += bytes([b | 128])
        else: return out + bytes([b])
def fld(n, wt, payload):
    return varint(n << 3 | wt) + (varint(len(payload)) + payload if wt == 2 else varint(payload))
def entity(ct, wd, sid, langs):
    b = fld(4, 0, 2)
    if ct is not None: b += fld(5, 0, ct)
    if wd is not None: b += fld(6, 2, fld(2, 0, wd))
    if langs is not None:
        b += fld(13, 2, b''.join(fld(1, 2, fld(1, 2, l.encode())) for l in langs))
    if sid is not None: b += fld(23, 2, sid.encode())
    return b
def enc(s): return '-' if s is None else ('=' if s == '' else s.encode().hex())

cases = []
tags = LANGID + EXTRA + SCRIPTS
for t in tags:
    for ct, wd in FLAGS: cases.append((ct, wd, None, [t]))
for t in tags:  # language followed by a first-pass script result (and the reverse)
    for s in SCRIPTS:
        for ct, wd in FEW_FLAGS: cases.append((ct, wd, None, [t, s]))
    for ct, wd in FEW_FLAGS: cases.append((ct, wd, None, ['Zyyy', t]))
for ct, wd in FLAGS:  # no languages / CUSTOM with recognizer_script_id
    cases.append((ct, wd, None, None)); cases.append((ct, wd, None, []))
    for sid in ('', 'Cyrl', 'Hani', 'Latn', 'und', 'zh', 'xx'):
        cases.append((ct, wd, sid, None)); cases.append((ct, wd, sid, ['ja'])); cases.append((ct, wd, sid, ['en']))

out = sys.argv[1]; os.makedirs(out, exist_ok=True)
with open(os.path.join(out, 'grid.txt'), 'w') as f:
    for ct, wd, sid, langs in cases:
        f.write('%s %s %s %s\n' % ('-' if ct is None else ct, '-' if wd is None else wd, enc(sid),
                                   '-' if langs is None else (','.join(enc(l) for l in langs) or '=none')))
with open(os.path.join(out, 'grid.bin'), 'wb') as f:
    f.write(struct.pack('<I', len(cases)))
    for c in cases:
        e = entity(*c); f.write(struct.pack('<I', len(e)) + e)
print(len(cases), 'cases')
