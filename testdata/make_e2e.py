# SPDX-License-Identifier: MPL-2.0
"""Held-out end-to-end corpus (testdata/e2e/): not used while developing the stages, only for the final
owocr-level comparison. Deterministic (fixed seeds)."""
import os, random
from PIL import Image, ImageDraw, ImageFont, ImageFilter

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'e2e')
SUP = '/System/Library/Fonts/Supplemental/'
SYS = '/System/Library/Fonts/'
ARIAL, TIMES, COURIER, GEORGIA, VERDANA = SUP + 'Arial.ttf', SUP + 'Times New Roman.ttf', SUP + 'Courier New.ttf', SUP + 'Georgia.ttf', SUP + 'Verdana.ttf'
HIRA = SYS + 'ヒラギノ角ゴシック W3.ttc'
GOTHIC = SYS + 'AppleSDGothicNeo.ttc'
ARABIC = SYS + 'GeezaPro.ttc'
HEB = SUP + 'Arial Hebrew.ttc' if os.path.exists(SUP + 'Arial Hebrew.ttc') else SYS + 'ArialHB.ttc'
def font(p, s, index=0): return ImageFont.truetype(p, s, index=index)
def save(img, name, jpeg=None):
    if jpeg:  # round-trip through JPEG for compression artifacts, store losslessly
        import io
        b = io.BytesIO(); img.convert('RGB').save(b, 'JPEG', quality=jpeg); b.seek(0); img = Image.open(b).convert('RGB')
    img.save(os.path.join(OUT, name + '.png'))

# 1. app-like UI: toolbar, sidebar, buttons, small text
img = Image.new('RGB', (1280, 800), (246, 246, 248)); d = ImageDraw.Draw(img)
d.rectangle((0, 0, 1280, 52), fill=(52, 120, 246)); d.text((20, 12), 'Inbox — 3 unread messages', font=font(ARIAL, 22), fill='white')
d.rectangle((0, 52, 240, 800), fill=(232, 232, 236))
for i, t in enumerate(['Inbox', 'Starred', 'Sent', 'Drafts', 'Spam', 'Trash', 'Settings']):
    d.text((24, 80 + i * 40), t, font=font(VERDANA, 17), fill=(30, 30, 30))
rows = [('Alice Johnson', 'Quarterly report draft', 'Can you review the numbers before Friday?', '9:41 AM'),
        ('GitHub', '[repo] Pull request #1234 merged', 'Your changes to src/main.cc were merged.', 'Yesterday'),
        ('Bob', 'Lunch?', 'Thai place at 12:30 or the usual?', 'Mon'),
        ('Calendar', 'Reminder: Dentist appointment', 'Tuesday, November 4, 2026 at 3:00 PM', 'Oct 30')]
for i, (a, s, p, t) in enumerate(rows):
    y = 70 + i * 90
    d.rectangle((256, y, 1264, y + 80), fill='white', outline=(220, 220, 225))
    d.text((272, y + 8), a, font=font(ARIAL, 18), fill='black'); d.text((1160, y + 8), t, font=font(ARIAL, 14), fill=(120, 120, 120))
    d.text((272, y + 32), s, font=font(ARIAL, 16), fill=(20, 20, 20)); d.text((272, y + 54), p, font=font(ARIAL, 13), fill=(110, 110, 110))
for i, b in enumerate(['Reply', 'Forward', 'Archive', 'Delete']):
    x = 272 + i * 130; d.rounded_rectangle((x, 450, x + 110, 488), 8, fill=(52, 120, 246)); d.text((x + 18, 459), b, font=font(ARIAL, 17), fill='white')
d.text((272, 520), 'Tip: press ⌘K to search all mail. Storage used: 12.4 GB of 15 GB (82%)', font=font(ARIAL, 14), fill=(90, 90, 90))
save(img, 'ui_mail')

# 2. colored backgrounds, low contrast, noise
random.seed(1)
img = Image.new('RGB', (1000, 600)); px = img.load()
for y in range(600):
    for x in range(1000):
        px[x, y] = (int(200 + 40 * x / 1000) + random.randint(-12, 12), int(180 + 50 * y / 600) + random.randint(-12, 12), 150 + random.randint(-12, 12))
d = ImageDraw.Draw(img)
d.text((40, 40), 'Noisy gradient background', font=font(GEORGIA, 40), fill=(20, 20, 60))
d.text((40, 120), 'Low contrast line of text that is still readable', font=font(ARIAL, 26), fill=(150, 140, 120))
d.text((40, 200), 'Red warning: battery level critical (4%)', font=font(VERDANA, 24), fill=(200, 20, 20))
d.text((40, 280), 'Serif body text in Times New Roman, 22 px, with punctuation; quotes "like this" & dashes — ok?', font=font(TIMES, 22), fill='black')
save(img, 'noisy_gradient', jpeg=70)

# 3. code / monospace
img = Image.new('RGB', (1100, 520), (40, 42, 54)); d = ImageDraw.Draw(img)
code = ['#include <stdio.h>', '', 'int main(int argc, char** argv) {', '    for (int i = 0; i < argc; i++)', '        printf("%d: %s\\n", i, argv[i]);',
        '    return 0;  // exit status', '}', '', '$ ./a.out --verbose -o /tmp/out.txt']
cols = [(255, 121, 198), (248, 248, 242), (139, 233, 253), (248, 248, 242), (241, 250, 140), (98, 114, 164), (248, 248, 242), (248, 248, 242), (80, 250, 123)]
for i, (t, c) in enumerate(zip(code, cols)): d.text((30, 24 + i * 52), t, font=font(COURIER, 30), fill=c)
save(img, 'code_dark')

# 4. small text and a dense paragraph
img = Image.new('RGB', (900, 700), 'white'); d = ImageDraw.Draw(img)
words = ('the quick brown fox jumps over the lazy dog while seven wizards quietly judge boxing matches and '
         'pack my box with five dozen liquor jugs as sphinx of black quartz judge my vow ').split()
random.seed(2)
for i in range(22):
    line = ' '.join(random.choice(words) for _ in range(14)).capitalize() + '.'
    d.text((20, 15 + i * 30), line, font=font(ARIAL, 13 + (i % 4)), fill='black')
save(img, 'small_dense')

# 5. mixed CJK / Korean / Latin sign-like content
img = Image.new('RGB', (1000, 600), (255, 250, 235)); d = ImageDraw.Draw(img)
d.text((40, 30), '東京駅 八重洲口 → 徒歩5分', font=font(HIRA, 40), fill='black')
d.text((40, 110), '営業時間 10:00〜21:00（年中無休）', font=font(HIRA, 30), fill=(160, 0, 0))
d.text((40, 190), '서울특별시 중구 세종대로 110', font=font(GOTHIC, 34), fill='black')
d.text((40, 270), '欢迎光临 Welcome 환영합니다 ようこそ', font=font(HIRA, 34), fill=(0, 70, 140))
d.text((40, 350), '価格: ¥1,980（税込）', font=font(HIRA, 30), fill='black')
save(img, 'cjk_signs')

# 6. RTL mixed with numbers/latin
img = Image.new('RGB', (1000, 500), 'white'); d = ImageDraw.Draw(img)
d.text((960, 40), 'مرحبا بكم في موقعنا الجديد', font=font(ARABIC, 38), fill='black', anchor='ra')
d.text((960, 130), 'السعر ٢٥٠ ريال فقط', font=font(ARABIC, 34), fill='black', anchor='ra')
d.text((960, 220), 'שלום עולם, זוהי בדיקה', font=font(HEB, 36), fill='black', anchor='ra')
d.text((40, 320), 'English line between RTL text', font=font(ARIAL, 30), fill='black')
save(img, 'rtl_mixed')

# 7. blurred, slightly rotated photo-like text
img = Image.new('RGB', (1000, 700), (210, 210, 200)); t = Image.new('RGBA', (900, 300), (0, 0, 0, 0)); d = ImageDraw.Draw(t)
d.text((20, 20), 'NOTICE', font=font(ARIAL, 70), fill=(30, 30, 30)); d.text((20, 120), 'Please keep this door closed', font=font(ARIAL, 40), fill=(30, 30, 30))
d.text((20, 190), 'at all times. Thank you!', font=font(ARIAL, 40), fill=(30, 30, 30))
r = t.rotate(-6, expand=True, resample=Image.BICUBIC); img.paste(r, (40, 150), r)
save(img.filter(ImageFilter.GaussianBlur(1.2)), 'photo_notice', jpeg=60)

# 8. table with numbers
img = Image.new('RGB', (900, 500), 'white'); d = ImageDraw.Draw(img)
hdr = ['Item', 'Qty', 'Unit price', 'Total']; xs = [30, 360, 480, 700]
for x, h in zip(xs, hdr): d.text((x, 20), h, font=font(ARIAL, 22), fill='black')
d.line((20, 55, 880, 55), fill='black', width=2)
items = [('USB-C cable (2 m)', 3, 9.99), ('Wireless mouse', 1, 24.50), ('Monitor stand', 2, 39.00), ('HDMI adapter', 4, 12.75), ('Desk lamp LED', 1, 31.20)]
for i, (n, q, p) in enumerate(items):
    y = 70 + i * 50
    for x, v in zip(xs, [n, str(q), f'${p:.2f}', f'${q * p:.2f}']): d.text((x, y), v, font=font(ARIAL, 20), fill='black')
d.text((480, 340), 'Subtotal', font=font(ARIAL, 20), fill='black'); d.text((700, 340), '$231.67', font=font(ARIAL, 20), fill='black')
save(img, 'table_invoice')

# 9. large screenshot (> 2048, exercises owocr's resize)
img = Image.new('RGB', (3000, 1600), 'white'); d = ImageDraw.Draw(img)
for i in range(12):
    d.text((80, 60 + i * 120), f'Line {i + 1}: high resolution screenshots are downscaled before OCR, item #{1000 + i * 37}', font=font(ARIAL, 54), fill='black')
save(img, 'large_hidpi')

# 10. vertical Japanese
img = Image.new('RGB', (500, 800), 'white'); d = ImageDraw.Draw(img)
for c, col in enumerate(['縦書きの日本語テキスト', 'これは二列目です']):
    for i, ch in enumerate(col): d.text((400 - c * 70, 40 + i * 48), ch, font=font(HIRA, 40), fill='black')
save(img, 'ja_vertical')
print('wrote', len(os.listdir(OUT)), 'images to', OUT)
