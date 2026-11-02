#!/bin/sh
# Indic / South-East Asian / RTL test images for the recognizer's symbol reordering (pre-base vowel signs, reph,
# two-part vowels, conjuncts, mirrored brackets), rendered with CoreText shaping (../extra_lm/render.swift).
cd "$(dirname "$0")"
r() { swift ../extra_lm/render.swift "$1.png" 1100 "$2"; }
printf 'KohinoorBangla-Regular\t34\tআমি বাংলায় গান গাই।\nBanglaSangamMN\t30\tকোলকাতা শহরের বৌদ্ধ মন্দির\nKohinoorBangla-Regular\t30\tক্যামেরা ও প্রিন্টার কিনেছি\n' | r beng 330
printf 'KohinoorGujarati-Regular\t34\tગુજરાતી ભાષા કિંમત\nGujaratiSangamMN\t30\tઅમદાવાદ શહેરની સ્થિતિ\nGurmukhiMN\t32\tਪੰਜਾਬੀ ਭਾਸ਼ਾ ਕਿਤਾਬ ਦਿੱਲੀ\n' | r gujr_guru 330
printf 'KohinoorDevanagari-Regular\t34\tधर्म कर्म और संस्कृति\nKohinoorDevanagari-Regular\t30\tहिंदी किंतु पुस्तिका\nDevanagariSangamMN\t30\tप्रिय मित्र, शुभकामनाएँ\n' | r deva2 330
printf 'NotoSansKannada-Regular\t34\tಕರ್ನಾಟಕ ಸರ್ಕಾರ ಕ್ರಿಕೆಟ್\nKannadaSangamMN\t30\tಬೆಂಗಳೂರು ವಿಶ್ವವಿದ್ಯಾಲಯ ಸೂರ್ಯ\n' | r knda2 260
printf 'TamilSangamMN\t34\tகொடை கோவில் கௌரவம் வை\nTamilSangamMN\t30\tபொங்கல் ஔவையார் தெரு\n' | r taml2 260
printf 'KohinoorTelugu-Regular\t34\tక్రికెట్ ప్రపంచం శ్రీ\nTeluguSangamMN\t30\tవిజయవాడ కృష్ణా జిల్లా\n' | r telu2 260
printf 'MalayalamSangamMN\t34\tകൊച്ചി കോട്ടയം കൈരളി\nSinhalaSangamMN\t30\tශ්‍රී ලංකාව කොළඹ නගරය\n' | r mlym_sinh 260
printf 'KhmerSangamMN\t34\tភាសាខ្មែរ ប្រទេសកម្ពុជា\nLaoSangamMN\t30\tພາສາລາວ ເມືອງວຽງຈັນ ໂຮງຮຽນ\n' | r khmr_laoo 260
printf 'GeezaPro\t34\t(مرحبا) [عالم] 2024\nArialHebrew\t32\tשלום (עולם) 123 test\n' | r rtl_paren 260
