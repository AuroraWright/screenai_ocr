# SPDX-License-Identifier: MPL-2.0
from PIL import Image, ImageDraw, ImageFont
import os
OUT = os.path.dirname(os.path.abspath(__file__))
F = '/System/Library/Fonts/Supplemental/Arial.ttf'
J = '/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc'
def font(p, s): return ImageFont.truetype(p, s)
def save(img, name): img.save(os.path.join(OUT, name + '.png'))

img = Image.new('RGB', (900, 300), 'white'); d = ImageDraw.Draw(img)
d.text((30, 30), 'Hello world, this is a simple OCR test.', font=font(F, 32), fill='black')
d.text((30, 100), 'Second line with numbers 12345 and symbols: $9.99 (50%)', font=font(F, 24), fill='black')
d.text((30, 200), 'Accessibility matters for everyone.', font=font(F, 40), fill=(40, 40, 160))
save(img, 'en_simple')

img = Image.new('RGB', (1000, 700), 'white'); d = ImageDraw.Draw(img)
d.text((40, 30), 'Chapter One', font=font(F, 44), fill='black')
para = ['It was a bright cold day in April, and the clocks', 'were striking thirteen. Winston Smith, his chin', 'nuzzled into his breast in an effort to escape the', 'vile wind, slipped quickly through the glass doors.']
for i, t in enumerate(para): d.text((40, 120 + i * 34), t, font=font(F, 24), fill='black')
para2 = ['The hallway smelt of boiled cabbage and old rag', 'mats. At one end of it a coloured poster, too large', 'for indoor display, had been tacked to the wall.']
for i, t in enumerate(para2): d.text((40, 300 + i * 34), t, font=font(F, 24), fill='black')
d.text((600, 600), 'Page 1', font=font(F, 20), fill='gray')
save(img, 'en_paragraphs')

img = Image.new('RGB', (900, 400), 'white'); d = ImageDraw.Draw(img)
d.text((30, 30), '吾輩は猫である。名前はまだ無い。', font=font(J, 36), fill='black')
d.text((30, 110), 'どこで生れたかとんと見当がつかぬ。', font=font(J, 30), fill='black')
d.text((30, 190), '日本語と English の混在テキスト 2024年', font=font(J, 30), fill='black')
save(img, 'ja_simple')

img = Image.new('RGB', (600, 600), (30, 30, 30)); d = ImageDraw.Draw(img)
d.text((40, 40), 'Dark mode text', font=font(F, 36), fill='white')
d.text((40, 120), 'Light on dark background', font=font(F, 26), fill=(200, 200, 200))
save(img, 'en_dark')

img = Image.new('RGB', (800, 800), 'white'); t = Image.new('RGBA', (700, 80), (255, 255, 255, 0))
ImageDraw.Draw(t).text((10, 10), 'Rotated text sample line', font=font(F, 40), fill='black')
img.paste(t.rotate(20, expand=True, resample=Image.BICUBIC), (40, 200), t.rotate(20, expand=True, resample=Image.BICUBIC))
save(img, 'en_rotated')

# ---- extended set ------------------------------------------------------------------------------------
SUP = '/System/Library/Fonts/Supplemental/'
SYS = '/System/Library/Fonts/'
img = Image.new('RGB', (1100, 620), 'white'); d = ImageDraw.Draw(img); y = 20
for fp, sz, t in [(SUP + 'Times New Roman.ttf', 30, 'Times New Roman: The quick brown fox jumps over the lazy dog.'),
                  (SUP + 'Courier New.ttf', 26, 'Courier New: int main(void) { return 0; } // code'),
                  (SUP + 'Georgia Bold.ttf', 34, 'Georgia Bold heading text'),
                  (SUP + 'Verdana Italic.ttf', 22, 'Verdana italic, with punctuation: "quotes", dashes - and; colons.'),
                  (SUP + 'Arial.ttf', 14, 'Small Arial text at fourteen pixels, often seen in user interfaces and menus.'),
                  (SUP + 'Arial.ttf', 11, 'Tiny eleven pixel text: settings, help, about, privacy policy, terms of service'),
                  (SUP + 'Arial Black.ttf', 60, 'BIG TITLE 2025'),
                  (SUP + 'Arial Narrow.ttf', 24, 'Email: someone@example.com  URL: https://www.example.org/path?q=1')]:
    d.text((30, y), t, font=font(fp, sz), fill='black'); y += int(sz * 1.6) + 6
save(img, 'en_fonts')

img = Image.new('RGB', (1200, 700), 'white'); d = ImageDraw.Draw(img)
col1 = ['Left column paragraph text that', 'continues on several lines and', 'should be kept as one block by', 'the layout analysis step.']
col2 = ['Right column has different', 'content entirely, it talks about', 'something else, and is placed', 'next to the first column.']
d.text((40, 30), 'Two Column Article Title', font=font(F, 40), fill='black')
for i, t in enumerate(col1): d.text((40, 120 + i * 32), t, font=font(F, 22), fill='black')
for i, t in enumerate(col2): d.text((640, 120 + i * 32), t, font=font(F, 22), fill='black')
d.text((40, 320), 'A full width line below both columns closes the article.', font=font(F, 22), fill='black')
d.rectangle((40, 400, 300, 460), fill=(30, 110, 220)); d.text((70, 413), 'Sign in', font=font(F, 28), fill='white')
d.rectangle((340, 400, 600, 460), outline='gray', width=2); d.text((370, 413), 'Cancel', font=font(F, 28), fill=(60, 60, 60))
d.rectangle((0, 520, 1200, 700), fill=(245, 245, 220)); d.text((40, 560), 'Footer: Copyright 2025 Example Inc. All rights reserved.', font=font(F, 18), fill=(90, 90, 90))
save(img, 'en_columns')

img = Image.new('RGB', (1000, 520), 'white'); d = ImageDraw.Draw(img); y = 20
for t in ['Le renard brun saute par-dessus le chien paresseux.', 'Der schnelle braune Fuchs springt über den faulen Hund.',
          'El rápido zorro marrón salta sobre el perro perezoso.', 'La volpe veloce salta sopra il cane pigro.',
          'A rápida raposa marrom pula sobre o cão preguiçoso.', 'De snelle bruine vos springt over de luie hond.',
          'Szybki brązowy lis przeskakuje nad leniwym psem.', 'Hızlı kahverengi tilki tembel köpeğin üzerinden atlar.']:
    d.text((30, y), t, font=font(SUP + 'Arial.ttf', 28), fill='black'); y += 60
save(img, 'latin_langs')

img = Image.new('RGB', (1000, 560), 'white'); d = ImageDraw.Draw(img); U = SUP + 'Arial Unicode.ttf'
d.text((30, 20), 'Быстрая коричневая лиса прыгает через ленивую собаку.', font=font(U, 28), fill='black')
d.text((30, 90), 'Η γρήγορη καφέ αλεπού πηδάει πάνω από τον τεμπέλη σκύλο.', font=font(U, 28), fill='black')
d.text((30, 160), '敏捷的棕色狐狸跳过了懒狗。今天天气很好。', font=font(SYS + 'Hiragino Sans GB.ttc', 32), fill='black')
d.text((30, 230), '빠른 갈색 여우가 게으른 개를 뛰어넘습니다.', font=font(SYS + 'AppleSDGothicNeo.ttc', 32), fill='black')
d.text((30, 300), 'तेज़ भूरी लोमड़ी आलसी कुत्ते के ऊपर कूदती है।', font=font(SUP + 'DevanagariMT.ttc', 32), fill='black')
d.text((30, 370), 'สุนัขจิ้งจอกสีน้ำตาลกระโดดข้ามสุนัขขี้เกียจ', font=font(SYS + 'Thonburi.ttc', 32), fill='black')
save(img, 'multiscript')

img = Image.new('RGB', (900, 300), 'white'); d = ImageDraw.Draw(img)
d.text((850, 30), 'الثعلب البني السريع يقفز فوق الكلب الكسول', font=font(SYS + 'GeezaPro.ttc', 32), fill='black', anchor='ra')
d.text((850, 120), 'השועל החום המהיר קופץ מעל הכלב העצלן', font=font(SYS + 'ArialHB.ttc', 32), fill='black', anchor='ra')
d.text((30, 210), 'Mixed: English and العربية 123', font=font(SYS + 'GeezaPro.ttc', 28), fill='black')
save(img, 'rtl')

img = Image.new('RGB', (2600, 1700), 'white'); d = ImageDraw.Draw(img)
for i in range(12):
    d.text((80, 60 + i * 130), 'Large canvas line %d with fairly big glyphs for scaling tests.' % (i + 1), font=font(F, 64), fill='black')
save(img, 'en_large')

img = Image.new('RGB', (700, 900), 'white'); t = Image.new('RGBA', (900, 70), (255, 255, 255, 0))
ImageDraw.Draw(t).text((10, 5), 'Vertical sideways text line', font=font(F, 48), fill='black')
r = t.rotate(90, expand=True); img.paste(r, (40, 0), r)
d = ImageDraw.Draw(img); d.text((200, 100), 'Normal line next to it', font=font(F, 36), fill='black')
t2 = Image.new('RGBA', (600, 70), (255, 255, 255, 0)); ImageDraw.Draw(t2).text((10, 5), 'Slightly tilted', font=font(F, 44), fill='black')
r2 = t2.rotate(-7, expand=True, resample=Image.BICUBIC); img.paste(r2, (200, 300), r2)
save(img, 'en_orient')
