# SPDX-License-Identifier: MPL-2.0
"""lldb script: dump aksara::script_util::MostLikelyScript (0x100b2d148) of the original for every 2/3-letter
language code, every language x region pair, and language x script tags; script codes are named with
i18n_identifiers::ScriptCodeConverter (as LanguageBasedLineRecognizerConfigSelector::Select does).

Usage: LS_OUT=dir lldb -b -o 'command script import trace/likely_script_dump.py' -o run -- \
         .venv/bin/python ref/run_ref.py testdata/en_simple.png
Output: <LS_OUT>/names.txt ("code name"), <LS_OUT>/<phase>.txt ("tag code") for phases lang, region, script.
"""
import itertools, lldb, os, string, struct

MLS = 0x100b2d148
SCC = 0x100d3247c   # i18n_identifiers::ScriptCodeConverter::GetSingleton()
SELECT = 0x100148160
OUT = os.environ.get('LS_OUT', '/tmp/ls')
done = {'v': False}

def slide_of(t):
    mod = t.FindModule(lldb.SBFileSpec('libchromescreenai.so'))
    return mod.GetObjectFileHeaderAddress().GetLoadAddress(t) - 0x100000000

def _eval(frame, expr):
    opts = lldb.SBExpressionOptions(); opts.SetIgnoreBreakpoints(True); opts.SetTryAllThreads(False)
    opts.SetTimeoutInMicroSeconds(0)
    v = frame.EvaluateExpression(expr, opts)
    if v.GetError().Fail(): raise RuntimeError(v.GetError().GetCString())
    return v.GetValueAsUnsigned()

def run(frame, slide, tags, chunk=200000):
    proc = frame.GetThread().GetProcess(); err = lldb.SBError()
    codes = []
    for i0 in range(0, len(tags), chunk):
        part = tags[i0:i0 + chunk]
        blob = b''.join(t.encode().ljust(32, b'\0')[:32] for t in part)
        inp = _eval(frame, '(unsigned long)malloc(%d)' % len(blob))
        proc.WriteMemory(inp, blob, err)
        out = _eval(frame, '(unsigned long)calloc(%d, 4)' % len(part))
        _eval(frame, ('const char* in = (const char*)0x%x; int* out = (int*)0x%x;'
                      'for (int i = 0; i < %d; i++) { const char* s = in + 32 * i; unsigned long n = 0;'
                      '  while (n < 32 && s[n]) n++;'
                      '  out[i] = ((int(*)(const char*, unsigned long))0x%x)(s, n); } 0')
              % (inp, out, len(part), MLS + slide))
        codes += list(struct.unpack('<%di' % len(part), proc.ReadMemory(out, 4 * len(part), err)))
        _eval(frame, '(void)free((void*)0x%x), (void)free((void*)0x%x), 0' % (inp, out))
    return codes

def names(frame, slide):
    proc = frame.GetThread().GetProcess(); err = lldb.SBError()
    out = _eval(frame, '(unsigned long)calloc(256, 16)')
    _eval(frame, ('struct SV { const char* p; unsigned long n; };'
                  'void* conv = ((void*(*)())0x%x)(); SV* out = (SV*)0x%x;'
                  'for (int c = 0; c < 256; c++) out[c] = (**(SV(***)(void*, int))conv)(conv, c); 0') % (SCC + slide, out))
    res = {}
    raw = proc.ReadMemory(out, 256 * 16, err)
    for c in range(256):
        p, n = struct.unpack_from('<QQ', raw, 16 * c)
        if p and 0 < n < 64: res[c] = proc.ReadMemory(p, n, err).decode()
    return res

def write(name, tags, codes):
    with open(os.path.join(OUT, name + '.txt'), 'w') as f:
        for t, c in zip(tags, codes): f.write('%s %d\n' % (t, c))

def on_select(frame, loc, d):
    t = frame.GetThread().GetProcess().GetTarget(); slide = slide_of(t)
    if done['v'] or frame.GetPC() != SELECT + slide: return False
    done['v'] = True
    loc.GetBreakpoint().SetEnabled(False)
    nm = names(frame, slide)
    with open(os.path.join(OUT, 'names.txt'), 'w') as f:
        for c in sorted(nm): f.write('%d %s\n' % (c, nm[c]))
    lc = string.ascii_lowercase
    langs = ['und', 'mul', 'zxx', 'mis', ''] + [''.join(p) for p in itertools.product(lc, repeat=2)] + \
            [''.join(p) for p in itertools.product(lc, repeat=3)]
    codes = run(frame, slide, langs)
    write('lang', langs, codes)
    # Region sweep: und, every 2-letter code and the 3-letter codes whose likely script is not Latn or unknown.
    latn, unknown = codes[langs.index('en')], codes[langs.index('qqq')]
    keep = [l for l, c in zip(langs, codes) if l and (len(l) == 2 or l == 'und' or c not in (latn, unknown))]
    regions = [''.join(p) for p in itertools.product(string.ascii_uppercase, repeat=2)] + ['419', '150', '001', '029']
    rt = [l + '-' + r for l in keep for r in regions]
    write('region', rt, run(frame, slide, rt))
    scripts = sorted(set(nm.values()))
    st = [l + '-' + s for l in ('zh', 'en', 'und', 'ja', 'sr', 'xx', 'qqq') for s in scripts + ['Abcd', 'Zzzz']]
    write('script', st, run(frame, slide, st))
    # Which language codes parse: an explicit script is returned as is for those, Latn otherwise.
    at = [l + '-Arab' for l in langs if l]
    arab_codes = run(frame, slide, at)
    write('arab', at, arab_codes)
    names_of = lambda c: nm.get(c)
    # Languages without a likely script of their own, with a region.
    zt = [l + '-' + r for l in ('qqq', 'mul', 'zxx', 'mis', 'abc', 'zzz', 'aaa') for r in regions]
    write('region0', zt, run(frame, slide, zt))
    # Spelling variants.
    vt = ['ZH-hant', 'zh_TW', 'Zh-Tw', 'ZH', 'zh-hant-tw', 'zh-HANT', 'zh-tw', 'und-tw', 'UND-TW', 'zh-Hant-TW-x-foo',
          'zh-TW-Hant', 'zh-Hant-Hans', 'zh-foobar', 'zh-Hant-foobar', 'zh-TW-foobar', 'zh-t-i0-und', 'zh-u-nu-latn',
          'zh-x-foo', 'zh--TW', 'zh-', '-zh', ' zh', 'zh ', 'x-foo', 'i-klingon', 'en-GB-oed', 'sgn-BE-FR', 'zh-min-nan',
          'zh-guoyu', 'zh-hakka', 'zh-xiang', 'art-lojban', 'zh-cmn', 'zh-cmn-Hant', 'zh-yue', 'cmn-Hant', 'cmn-TW',
          'yue-HK', 'yue-CN', 'pa-IN', 'pa-PK', 'pa-Arab-IN', 'sr-ME', 'sr-RS', 'sr-Latn-RS', 'uz-AF', 'uz-UZ', 'ms-BN',
          'ms-ID', 'az-IR', 'az-AZ', 'ku-IQ', 'ku-IR', 'ha-SD', 'ha-NE', 'mn-CN', 'mn-MN', 'ug', 'ug-KZ', 'tg-PK',
          'iw', 'he', 'in', 'id', 'ji', 'yi', 'jw', 'jv', 'mo', 'ro', 'sh', 'tl', 'fil', 'zh-Hant-HK', 'zh-HK', 'zh-MO',
          'en-Hani', 'en-hani', 'en-HANI', 'zh-Hanb', 'zh-Jpan', 'ja-Hani', 'ko-Hani', 'ko-Hang', 'ja-Hira', 'ja-Kana',
          'zh-419', 'es-419', 'und-419', 'und-150', 'ar-001', 'zh-999', 'zh-1', 'zh-12345', 'zh-Hant-1994', 'de-1996',
          'sl-rozaj', 'sl-rozaj-biske', 'zh-a-foo', 'zh-Hant-t-i0-handwrit', 'ja-t-i0-handwrit', 'en-t-i0-handwrit',
          'abcd', 'abcde', 'abcdefgh', 'abcdefghi', 'a', 'Hani', 'Cyrl', 'Zyyy', 'Latn', 'und-Zzzz', 'zh-Zzzz-TW']
    write('variants', vt, run(frame, slide, vt))
    # IANA variant subtags (registered as of 2025) and grandfathered tags.
    variants = ('1606nict 1694acad 1901 1959acad 1994 1996 abl1943 akhmimic akuapem alalc97 aluku anpezo ao1990 '
                'aranes arcaicas arevela arevmda arkaika asante auvern baku1926 balanka barla basiceng bauddha '
                'bciav bcizbl biscayan biske blasl bohairic bohoric boont bornholm cisaup colb1945 cornu creiss '
                'dajnko ekavsk emodeng fascia fodom fonipa fonkirsh fonnapa fonupa fonxsamp gallo gascon gherd '
                'grclass grital grmistr hanoi hepburn heploc hognorsk hsistemo huett ijekavsk itihasa ivanchov '
                'jauer jyutping kkcor kleinsch kociewie kscor laukika leidentr lemosin lengadoc lipaw ltg1929 '
                'ltg2007 luna1918 lycopol mdcegyp mdctrans metelko monoton ndyuka nedis newfound nicard njiva '
                'nulik osojs oxendict pahawh2 pahawh3 pahawh4 pamaka peano pehoeji petr1708 pinyin polyton provenc '
                'puter rigik rozaj rumgr sahidic saigon scotland scouse simple solba sotav southern spanglis '
                'surmiran sursilv sutsilv synnejyl tailo tarask tongyong tunumiit uccor ucrcor ulster unifon '
                'valbadia vaidika valencia vallader vecdruka viennese vivaraup wadegile xsistemo '
                'abcde abcdefgh 1234 0abc 9999 1000 foobar').split()
    gf = ['art-lojban', 'cel-gaulish', 'en-GB-oed', 'i-ami', 'i-bnn', 'i-default', 'i-enochian', 'i-hak',
          'i-klingon', 'i-lux', 'i-mingo', 'i-navajo', 'i-pwn', 'i-tao', 'i-tay', 'i-tsu', 'no-bok', 'no-nyn',
          'sgn-BE-FR', 'sgn-BE-NL', 'sgn-CH-DE', 'zh-guoyu', 'zh-hakka', 'zh-min', 'zh-min-nan', 'zh-xiang']
    wt = ['ru-' + v for v in variants] + ['zh-Hant-' + v for v in variants] + ['ru-RU-' + v for v in variants] + \
         ['ru-petr1708-' + v for v in variants[:20]] + gf + [g + '-TW' for g in gf] + [g + '-Cyrl' for g in gf] + \
         ['ru-PETR1708-t-i0-und', 'ru-petr1708-x-a', 'ru-x-petr1708', 'ru-Cyrl-PETR1708', 'ru-RU-PETR1708']
    write('variants2', wt, run(frame, slide, wt))
    # Extended language subtags (zh-yue, zh-cmn): which 3-letter codes parse after which language.
    arab_ok = dict(zip(at, arab_codes))
    valid = [l for l in langs if l and names_of(arab_ok[l + '-Arab']) == 'Arab']
    three = [l for l in valid if len(l) == 3]
    macro = ('kok sgn mwr raj hmn luy gon kln man zap zha zza bal bik bua chm del den din doi gba grb iku jrb kpe '
             'lah mlg mon nor oji orm pus que rom srd swa syr tmh uzb yid zho ara fas est lav msa hbs aym aze chr '
             'cre ful grn kau kom kon kur sqi tgl').split()
    prefixes = [l for l in valid if len(l) == 2] + [m for m in macro if m in valid]
    et = [p + '-' + l + '-Cyrl' for p in prefixes for l in three]
    write('extlang', et, run(frame, slide, et))
    open(os.path.join(OUT, 'done'), 'w').write('ok\n')
    return False

def __lldb_init_module(debugger, d):
    t = debugger.GetSelectedTarget()
    b = t.BreakpointCreateByRegex(r'LanguageBasedLineRecognizerConfigSelector::Select\(')
    b.SetScriptCallbackFunction('likely_script_dump.on_select')
