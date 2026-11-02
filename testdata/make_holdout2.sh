#!/bin/sh
# Second held-out set (testdata/holdout2/): only used for the final owocr-level check, never by stage development.
cd "$(dirname "$0")"
R=extra_lm/render.swift
r() { swift $R "holdout2/$1.png" "$2" "$3"; }
printf 'GeezaPro\t32\tقال المتحدث إن المشروع سيكتمل في عام ٢٠٢٧ بتكلفة ٣٫٥ مليار دولار.\nGeezaPro\t26\tيمكنكم زيارة الموقع www.example.com لمزيد من المعلومات\nArial Hebrew\t30\tהפגישה תתקיים ביום שלישי בשעה 14:30 באולם הראשי\n' | r rtl_shaped 1200 300
printf 'KohinoorDevanagari-Regular\t30\tकृपया अपना पासवर्ड दर्ज करें (Password)\nKohinoorDevanagari-Regular\t26\tकुल राशि: ₹1,250.00 — भुगतान सफल रहा\nThonburi\t30\tยินดีต้อนรับสู่ระบบ กรุณาเข้าสู่ระบบ\n' | r indic_thai_ui 1100 300
printf 'Helvetica\t30\tΗ Αθήνα είναι η πρωτεύουσα της Ελλάδας.\nHelvetica\t28\tПривет, мир! Это проверка распознавания.\nHelvetica\t28\tZażółć gęślą jaźń — Příliš žluťoučký kůň\n' | r greek_cyrl_latin 1100 300
printf 'PingFangSC-Regular\t28\t人工智能技术正在快速发展，已经广泛应用于医疗、教育和交通等领域。\nPingFangSC-Regular\t28\t专家指出，未来五年内，相关产业规模有望突破万亿元。\nPingFangTC-Regular\t28\t繁體中文測試：臺北市信義區市府路一號\n' | r zh_prose 1100 300
python3 -I make_holdout2.py
