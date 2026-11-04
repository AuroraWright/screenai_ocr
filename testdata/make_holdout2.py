# SPDX-License-Identifier: MPL-2.0
import os, random
from PIL import Image, ImageDraw, ImageFont, ImageFilter
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'holdout2')
SUP = '/System/Library/Fonts/Supplemental/'
ARIAL, TIMES, GEORGIA, VERDANA, TREB = (SUP + f for f in ['Arial.ttf', 'Times New Roman.ttf', 'Georgia.ttf', 'Verdana.ttf', 'Trebuchet MS.ttf'])
HIRA = '/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc'
def F(p, s): return ImageFont.truetype(p, s)
def save(img, n): img.save(os.path.join(OUT, n + '.png'))
random.seed(7)

# news page, 3 columns + headline + caption
img = Image.new('RGB', (1400, 900), 'white'); d = ImageDraw.Draw(img)
d.text((40, 20), 'City council approves new transit plan', font=F(GEORGIA, 46), fill='black')
d.text((40, 85), 'By Staff Reporter  |  November 1, 2026', font=F(ARIAL, 16), fill=(100, 100, 100))
txt = ('The plan adds three bus lines and extends the light rail to the airport. Officials said construction '
       'would begin next spring and take roughly four years. Residents raised concerns about noise and parking, '
       'while business owners welcomed the expected increase in foot traffic downtown.').split()
for c in range(3):
    x = 40 + c * 450; y = 130; line = ''
    random.shuffle(txt)
    for w in txt * 2:
        if d.textlength(line + ' ' + w, font=F(TIMES, 19)) > 410:
            d.text((x, y), line.strip(), font=F(TIMES, 19), fill='black'); y += 27; line = ''
        line += ' ' + w
        if y > 760: break
d.text((40, 820), 'Photo caption: The proposed station near Main Street.', font=F(ARIAL, 15), fill=(80, 80, 80))
save(img, 'news_columns')

# dark settings panel with toggles
img = Image.new('RGB', (900, 700), (28, 28, 30)); d = ImageDraw.Draw(img)
d.text((30, 25), 'Settings', font=F(ARIAL, 34), fill='white')
for i, (k, v) in enumerate([('Wi-Fi', 'HomeNetwork_5G'), ('Bluetooth', 'On'), ('Battery', '87%'), ('Display & Brightness', 'Auto'),
                            ('Notifications', '12 apps'), ('Privacy & Security', ''), ('Software Update', 'Version 26.1 available')]):
    y = 100 + i * 75; d.rounded_rectangle((20, y, 880, y + 62), 10, fill=(44, 44, 46))
    d.text((40, y + 18), k, font=F(ARIAL, 22), fill=(235, 235, 235)); d.text((860, y + 20), v, font=F(ARIAL, 20), fill=(150, 150, 155), anchor='ra')
save(img, 'dark_settings')

# chat bubbles
img = Image.new('RGB', (800, 900), (236, 229, 221)); d = ImageDraw.Draw(img)
msgs = [(0, 'Hey! Are we still on for tonight?'), (1, 'Yes, 7:30 at the usual place'), (0, 'Great, I will bring the tickets'),
        (1, 'Perfect 👍 see you there'), (0, 'Running 10 min late, sorry!'), (1, 'No worries')]
for i, (side, m) in enumerate(msgs):
    y = 40 + i * 120; w = d.textlength(m, font=F(VERDANA, 20)) + 40
    x = 30 if side == 0 else 770 - w
    d.rounded_rectangle((x, y, x + w, y + 60), 16, fill='white' if side == 0 else (220, 248, 198))
    d.text((x + 20, y + 17), m, font=F(VERDANA, 20), fill='black'); d.text((x + w - 60, y + 66), f'{18 + i}:0{i}', font=F(ARIAL, 13), fill=(120, 120, 120))
save(img, 'chat')

# steeper rotations
img = Image.new('RGB', (1000, 1000), 'white')
for ang, pos, t in [(15, (50, 100), 'Tilted fifteen degrees'), (-30, (300, 450), 'Rotated minus thirty'), (45, (80, 600), 'Diagonal 45 text')]:
    l = Image.new('RGBA', (600, 70), (0, 0, 0, 0)); ImageDraw.Draw(l).text((5, 5), t, font=F(ARIAL, 44), fill='black')
    r = l.rotate(ang, expand=True, resample=Image.BICUBIC); img.paste(r, pos, r)
save(img, 'rotations')

# low resolution: render small then upscale (blocky) + downscaled sharp
img = Image.new('RGB', (450, 150), 'white'); d = ImageDraw.Draw(img)
d.text((10, 10), 'Low resolution screenshot text', font=F(ARIAL, 16), fill='black')
d.text((10, 50), 'Upscaled 2x with nearest neighbour', font=F(ARIAL, 14), fill='black')
d.text((10, 90), 'File size: 2.4 MB  Modified: 10/31/2026', font=F(ARIAL, 13), fill=(60, 60, 60))
save(img.resize((900, 300), Image.NEAREST), 'lowres_upscaled')

# subtitles over noise photo
img = Image.new('RGB', (1280, 720)); px = img.load()
for y in range(720):
    for x in range(1280): px[x, y] = tuple(min(255, max(0, int(90 + 60 * ((x // 160 + y // 120) % 2) + random.gauss(0, 25)))) for _ in range(3))
img = img.filter(ImageFilter.GaussianBlur(3)); d = ImageDraw.Draw(img)
for i, t in enumerate(['I never thought I would see this place again.', 'Neither did I. Come on, we are late.']):
    d.text((640, 590 + i * 50), t, font=F(ARIAL, 34), fill='white', anchor='ma', stroke_width=3, stroke_fill='black')
save(img, 'subtitles')

# restaurant menu with dotted leaders and prices
img = Image.new('RGB', (900, 800), (250, 245, 230)); d = ImageDraw.Draw(img)
d.text((450, 30), 'MENU', font=F(GEORGIA, 56), fill=(90, 30, 20), anchor='ma')
items = [('Tomato soup', '6.50'), ('Caesar salad', '9.00'), ('Grilled salmon', '18.75'), ('Mushroom risotto', '15.20'), ('Tiramisu', '7.00'), ('Espresso', '2.80')]
for i, (n, p) in enumerate(items):
    y = 150 + i * 90
    d.text((80, y), n, font=F(TREB, 28), fill='black'); d.text((820, y), '€' + p, font=F(TREB, 28), fill='black', anchor='ra')
    d.text((80, y + 38), 'Fresh, seasonal ingredients', font=F(TREB, 17), fill=(110, 110, 110))
save(img, 'menu')

# Japanese UI mixed with numbers
img = Image.new('RGB', (1000, 500), 'white'); d = ImageDraw.Draw(img)
for i, t in enumerate(['ログイン', 'メールアドレスまたは電話番号', 'パスワードをお忘れですか？', '新規登録（無料）はこちら', '© 2026 サンプル株式会社']):
    d.text((50, 30 + i * 85), t, font=F(HIRA, 34 if i == 0 else 26), fill='black' if i != 2 else (0, 90, 200))
save(img, 'ja_login')
print('done', sorted(os.listdir(OUT)))
