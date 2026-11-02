#!/bin/sh
# Generates the LM-decoder test images (scripts with <script>_lm.fst) with CoreText shaping.
cd "$(dirname "$0")"
r() { swift render.swift "$1.png" 1100 "$2"; }
printf 'PingFangSC-Regular\t34\t我们的语言模型测试：中文文本识别。\nPingFangSC-Regular\t28\t北京市朝阳区建国路八十八号，电话 010-12345678\nSongti SC\t30\t天下大势，分久必合，合久必分。\n' | r hani 330
printf 'AppleSDGothicNeo-Regular\t34\t대한민국의 수도는 서울입니다.\nAppleSDGothicNeo-Regular\t28\t한국어 문장 인식 시험, 2024년 10월\n' | r kore 260
printf 'Thonburi\t34\tภาษาไทยเป็นภาษาราชการของประเทศไทย\nThonburi\t28\tกรุงเทพมหานคร ๑๒๓ ถนนสุขุมวิท\n' | r thai 260
printf 'KohinoorDevanagari-Regular\t34\tभारत एक विशाल देश है।\nKohinoorDevanagari-Regular\t28\tहिन्दी भाषा में पाठ पहचान परीक्षण\n' | r deva 260
printf 'NotoSansKannada-Regular\t34\tಕನ್ನಡ ಭಾಷೆಯ ಪರೀಕ್ಷೆ\nKannadaSangamMN\t28\tಬೆಂಗಳೂರು ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ\n' | r knda 260
printf 'MalayalamSangamMN\t34\tമലയാളം ഭാഷാ പരീക്ഷണം\nMalayalamSangamMN\t28\tകേരളം ഇന്ത്യയുടെ തെക്കേ അറ്റത്താണ്\n' | r mlym 260
printf 'TamilSangamMN\t34\tதமிழ் மொழி சோதனை\nTamilSangamMN\t28\tசென்னை தமிழ்நாட்டின் தலைநகரம்\n' | r taml 260
printf 'KohinoorTelugu-Regular\t34\tతెలుగు భాష పరీక్ష\nTeluguSangamMN\t28\tహైదరాబాద్ తెలంగాణ రాజధాని\n' | r telu 260
printf 'GeezaPro\t34\tاللغة العربية جميلة جدا\nGeezaPro\t28\tمرحبا بكم في مدينة القاهرة ٢٠٢٤\n' | r arab 260
printf 'Arial\t34\tМосква — столица России.\nArial\t28\tСъешь же ещё этих мягких французских булок\n' | r cyrl 260
