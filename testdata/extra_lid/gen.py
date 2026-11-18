# SPDX-License-Identifier: MPL-2.0
"""Language-ID test set: multilingual sentences plus tokenizer/post-processing edge cases.

Writes texts.hex (one hex-encoded UTF-8 string per line) and carrier.png (an image with enough text lines that the
original library calls MobileLangIdV2::IdentifyLanguage at least once per text). Reference outputs come from running
the original with trace/lid_dump.py, which substitutes these strings into successive IdentifyLanguage calls:

  LID_INJECT=testdata/extra_lid/texts.hex LID_DUMP_DIR=<dump> lldb -b -o 'command script import trace/lid_dump.py' \
      -o run -- .venv/bin/python ref/run_ref.py testdata/extra_lid/carrier.png
  .venv/bin/python testdata/extra_lid/gen.py --pack <dump>     # keeps texts + softmax/logit bits in ref.jsonl

Check: .venv/bin/python tests/test_langid.py cpp/build/run_langid <model_dir> testdata/extra_lid
"""
import json, os, struct, sys

HERE = os.path.dirname(os.path.abspath(__file__))

SENTENCES = [
    'The quick brown fox jumps over the lazy dog.',
    'Le renard brun rapide saute par-dessus le chien paresseux.',
    'Der schnelle braune Fuchs springt über den faulen Hund.',
    'El rápido zorro marrón salta sobre el perro perezoso.',
    'La volpe marrone veloce salta sopra il cane pigro.',
    'A rápida raposa marrom pula sobre o cão preguiçoso.',
    'De snelle bruine vos springt over de luie hond.',
    'Szybki brązowy lis przeskakuje nad leniwym psem.',
    'Hızlı kahverengi tilki tembel köpeğin üzerinden atlar.',
    'Den snabba bruna räven hoppar över den lata hunden.',
    'Den hurtige brune ræv springer over den dovne hund.',
    'Nopea ruskea kettu hyppää laiskan koiran yli.',
    'A gyors barna róka átugrik a lusta kutyán.',
    'Rychlá hnědá liška skáče přes líného psa.',
    'Rýchla hnedá líška skáče cez lenivého psa.',
    'Vulpea maro rapidă sare peste câinele leneș.',
    'Brza smeđa lisica preskače lijenog psa.',
    'Ātrā brūnā lapsa pārlec pāri slinkajam sunim.',
    'Greita rudoji lapė peršoka tinginį šunį.',
    'Kiire pruun rebane hüppab üle laisa koera.',
    'Быстрая коричневая лиса прыгает через ленивую собаку.',
    'Швидка коричнева лисиця стрибає через ледачого пса.',
    'Бързата кафява лисица прескача мързеливото куче.',
    'Брза смеђа лисица прескаче лењог пса.',
    'Η γρήγορη καφέ αλεπού πηδάει πάνω από τον τεμπέλη σκύλο.',
    'الثعلب البني السريع يقفز فوق الكلب الكسول.',
    'השועל החום המהיר קופץ מעל הכלב העצלן.',
    'روباه قهوه ای سریع از روی سگ تنبل می پرد.',
    'तेज़ भूरी लोमड़ी आलसी कुत्ते के ऊपर कूदती है।',
    'দ্রুত বাদামী শিয়াল অলস কুকুরের উপর দিয়ে লাফ দেয়।',
    'வேகமான பழுப்பு நரி சோம்பேறி நாயின் மேல் குதிக்கிறது.',
    'สุนัขจิ้งจอกสีน้ำตาลกระโดดข้ามสุนัขขี้เกียจ',
    'Con cáo nâu nhanh nhẹn nhảy qua con chó lười.',
    'Rubah coklat yang cepat melompati anjing yang malas.',
    'Mabilis na tumalon ang kayumangging soro sa tamad na aso.',
    '敏捷的棕色狐狸跳过了懒狗。',
    '素早い茶色の狐がのろまな犬を飛び越える。',
    '빠른 갈색 여우가 게으른 개를 뛰어넘는다.',
    'სწრაფი ყავისფერი მელა ხტება ზარმაც ძაღლზე.',
    'Արագ շագանակագույն աղվեսը ցատկում է ծույլ շան վրայով։',
    'ፈጣኑ ቡናማ ቀበሮ ሰነፍ ውሻውን ዘለለ።',
    'Haraka mbweha kahawia anaruka juu ya mbwa mvivu.',
    'Kocakan kahawia mai sauri ta tsallake kare mai lalaci.',
    'Is é an sionnach donn tapa a léim thar an madra leisciúil.',
    'Mae\'r llwynog brown cyflym yn neidio dros y ci diog.',
    'Fljóti brúni refurinn stekkur yfir lata hundinn.',
    'Quisque vel lorem sed magna tincidunt dictum et non velit.',
    'mera naam Rahul hai aur main Dilli mein rehta hoon',
    'watashi wa gakusei desu, yoroshiku onegaishimasu',
    'privet, kak dela? u menya vse khorosho, spasibo',
    'Въ началѣ было Слово, и Слово было у Бога.',
    'Россiя и Сѵнодъ, объявленіе о реформѣ',
    'ИСТАМБУЛ İstanbul ılık ŞÇĞÜÖ',
    'Straße, Größe, Maß und Fußball',
    'ΣΊΣΥΦΟΣ και ΟΔΥΣΣΕΑΣ',
]

EDGE = [
    '', 'a', 'ok', 'Hi!', 'abcd', 'abcde', '12345', '3.14159', '(555) 123-4567', '!!!???...', '    ',
    'https://www.example.com/path?query=1&b=2', 'user@example.org', '#hashtag @mention',
    'Hello 👋 world 🌍 emoji', '𝔘𝔫𝔦𝔠𝔬𝔡𝔢 math letters', 'tab\tseparated\tvalues',
    'ab�cd replacement char', 'Ünïcödé àccénts', 'ALL CAPS ENGLISH SENTENCE HERE',
    'Mixed English and русский текст together', '日本語とEnglishの混在テキスト2024年',
    ('The quick brown fox jumps over the lazy dog. ' * 6).strip(),
    ('Быстрая коричневая лиса ' * 10).strip(),
    'x' * 200,
]
# Raw byte strings (invalid UTF-8 in the middle; a truncated trailing sequence would hang the original).
RAW = [b'ab\xffcd invalid byte', b'\xc3\x28 bad continuation', b'caf\xe9 latin1']


def write():
    texts = [s.encode('utf8') for s in SENTENCES + EDGE] + RAW
    with open(os.path.join(HERE, 'texts.hex'), 'w') as f:
        for t in texts: f.write(t.hex() + '\n')
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 18)
    n = len(texts) + 10  # spare lines: some recognized lines may be skipped (empty text)
    cols, rows = 3, (n + 2) // 3
    img = Image.new('RGB', (1500, 40 + rows * 30), 'white'); d = ImageDraw.Draw(img)
    for i in range(n):
        d.text((30 + (i % cols) * 500, 20 + (i // cols) * 30), 'Carrier line number %d here' % i, font=font, fill='black')
    img.save(os.path.join(HERE, 'carrier.png'))
    print(len(texts), 'texts,', n, 'carrier lines')


def pack(dump):
    """Reduce a lid_dump directory to ref.jsonl: per invoke text + softmax/logit bits; per call the result."""
    out = []
    for f in sorted(os.listdir(dump)):
        if f.endswith('.json') and f[:5].isdigit():
            meta = json.load(open(os.path.join(dump, f)))
            rd = lambda i: open(os.path.join(dump, '%s_%d.bin' % (f[:5], i)), 'rb').read()
            rec = dict(kind='invoke', text_hex=meta['text'][0].encode('utf8', 'surrogateescape').hex(), status=meta['status'])
            if meta['status'] == 0:
                rec.update(softmax=list(struct.unpack('<111I', rd(29))), logits=list(struct.unpack('<111I', rd(28))))
            out.append(rec)
    for line in open(os.path.join(dump, 'lid_results.jsonl')):
        r = json.loads(line)
        if r['status'] == 1:
            out.append(dict(kind='result', text_hex=r['text'].encode('utf8', 'surrogateescape').hex(),
                            language=r['language'], confidence_bits=r['confidence_bits']))
    with open(os.path.join(HERE, 'ref.jsonl'), 'w') as f:
        for r in out: f.write(json.dumps(r) + '\n')
    print(len(out), 'records')


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--pack': pack(sys.argv[2])
    else: write()
