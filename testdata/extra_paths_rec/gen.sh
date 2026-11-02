#!/bin/sh
# Recognizer paths no other test image reaches:
#   select_mls    second-pass key via MostLikelyScript: all-caps Cyrillic/Greek homoglyph words read as Latin by the
#                 und script head, langid says sr/ru -> Cyrl (LanguageBasedLineRecognizerConfigSelector::Select)
#   curve_flip    curved lines the direction head flips (d=2): arc_gentle turned by 180 degrees
#   vert_cols     vertical columns (CJK, Latin, digits), one of them detected as a top-to-bottom curved box
#   vert_cols_180 the same turned by 180 degrees: a vertical line with direction 1, curved vertical lines
#   vert_ja_180   ja_vertical turned by 180 degrees (vertical lines near +-180 degrees: word angles)
#   rtl_mix       Hebrew/Arabic/Persian/Urdu lines with numbers, punctuation, brackets, URLs, e-mail, Latin runs
# Rendered with CoreText (../extra_lm/render.swift) or Pillow (vcol.py), turned with ../extra_rot/rotate.py.
cd "$(dirname "$0")"
PY=../../.venv/bin/python
r() { swift ../extra_lm/render.swift "$1.png" "$2" "$3"; }
printf 'Helvetica\t24\t%s\n' "МОСКВА" "АПТЕКА" "КОМЕТА" "РЕСТОРАН" "ТЕАТР" "МЕТРО" "ТАКСИ" "КАРТА" "НЕВА" "ВЕРА" \
  "КЕФИР" "ХОККЕЙ" "ΤΑΞΙ" "ΚΑΦΕ" "ΜΗΤΕΡΑ" "ΑΘΗΝΑ" "12 345 678 д" "2024 г." "12.05.2024 г." "100 руб." "5 кг" \
  "10 шт." "№ 12345" "ТЕХНО МАРКЕТ 24" "ТОМ 2" "АЛМАТЫ" | r select_mls 500 1500
$PY -I ../extra_rot/rotate.py ../extra_curve/arc_gentle.png curve_flip.png 180
$PY -I vcol.py vert_cols.png 30 "縦書きの日本語テキスト" "東京都渋谷区神南一丁目" "十一口田工三王中日目" "HOTEL" "TAXI" "1234567"
$PY -I ../extra_rot/rotate.py vert_cols.png vert_cols_180.png 180
$PY -I ../extra_rot/rotate.py ../e2e/ja_vertical.png vert_ja_180.png 180
printf '%s\n' \
  "ArialHebrew	30	שלום world 123! טוב" \
  "ArialHebrew	30	מחיר: 45.90 ₪ (כולל מע״מ)" \
  "ArialHebrew	30	בקרו ב-www.example.com/path?x=1 היום" \
  "ArialHebrew	30	טל׳ 03-1234567, שלוחה 5." \
  "ArialHebrew	30	ABC-123 הוא הקוד, לא XYZ" \
  "ArialHebrew	30	״ציטוט״ - test, 2024/11/01 [סוף]" \
  "GeezaPro	30	زوروا موقعنا https://example.com اليوم" \
  "GeezaPro	30	السعر 1,250.00 USD (شامل الضريبة)" \
  "GeezaPro	30	رقم 12-34 [مثال] {test} نهاية" \
  "GeezaPro	30	Version 2.0 الإصدار الجديد!" \
  "GeezaPro	30	البريد: info@example.org شكرا" \
  "GeezaPro	30	النسبة 50% - 75% فقط." \
  "GeezaPro	30	قیمت ۱۲۳٬۴۵۶ تومان است" \
  "GeezaPro	30	قیمت 500 روپے (تقریباً)" | r rtl_mix 900 1000
