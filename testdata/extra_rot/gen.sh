#!/bin/sh
# Upside-down / tilted pages for the recognizer's direction-flip path (direction head d=2: reversed logits, later
# passes cut from the pass-1 box and turned by 180 degrees). Rendered with CoreText (../extra_lm/render.swift),
# rotated with Pillow (rotate.py).
cd "$(dirname "$0")"
PY=../../.venv/bin/python
r() { swift ../extra_lm/render.swift "$1.tmp.png" 900 "$2"; }
printf 'Helvetica\t30\tThe quick brown fox jumps over the lazy dog.\nTimes New Roman\t26\tInvoice 2024-10-31: total EUR 1,234.56 (paid)\nHelvetica\t26\tこんにちは、世界。東京都渋谷区 1-2-3\n' | r latin_cjk 260
printf 'KohinoorDevanagari-Regular\t32\tहिन्दी भाषा में पाठ पहचान\nThonburi\t30\tภาษาไทยเป็นภาษาราชการ\nGeezaPro\t30\tمرحبا بكم في القاهرة (٢٠٢٤)\n' | r scripts 260
for a in 180 175 185; do
  $PY -I rotate.py latin_cjk.tmp.png latin_cjk_$a.png $a
  $PY -I rotate.py scripts.tmp.png scripts_$a.png $a
done
rm -f latin_cjk.tmp.png scripts.tmp.png
