// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef NEMO_SPEECH_TTS_WITH_ZH
#include "mandarin_tokenizer.h"
#endif

#ifdef NEMO_SPEECH_TTS_WITH_JA
// clang-format off
#include "mecab.h"
#include "njd.h"
#include "mecab2njd.h"
#include "njd_set_accent_phrase.h"
#include "njd_set_accent_type.h"
#include "njd_set_digit.h"
#include "njd_set_long_vowel.h"
#include "njd_set_pronunciation.h"
#include "njd_set_unvoiced_vowel.h"
#include "text2mecab.h"
// clang-format on
#endif

namespace fs = std::filesystem;

#if defined(NEMO_SPEECH_TTS_WITH_JA) && !defined(NEMO_SPEECH_OPENJTALK_DIC_DIR)
#define NEMO_SPEECH_OPENJTALK_DIC_DIR ""
#endif
#if defined(NEMO_SPEECH_TTS_WITH_JA) && !defined(NEMO_SPEECH_OPENJTALK_INSTALLED_DIC_DIR)
#define NEMO_SPEECH_OPENJTALK_INSTALLED_DIC_DIR ""
#endif

struct params {
    fs::path model;
    std::string text;
    std::string language = "en";
    std::string tokenizer_name;
    std::string locale;
    std::string grapheme_case;
    std::string grapheme_prefix;
    std::string ascii_letter_case;
    std::string phoneme_dict;
    std::string heteronyms;
    int offset = 0;
    int eos_id = 0;
    int expected_vocab_size = 0;
    bool apostrophe = true;
    bool pad_with_space = false;
    bool sentence_chunking = true;
    std::function<std::string(const std::string&, bool)> chunk_text_transform;
};

struct chunk {
    std::string text;
    std::vector<int> tokens;
};

struct tokenizer_result {
    std::string language;
    std::string tokenizer_name;
    int eos_id = 0;
    std::vector<chunk> chunks;
};

static std::string
read_file(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to read " + path.string());
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static std::vector<std::string>
split_utf8(const std::string& s) {
    std::vector<std::string> out;
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = (unsigned char)s[i];
        size_t n = 1;
        if ((c & 0xe0) == 0xc0) {
            n = 2;
        } else if ((c & 0xf0) == 0xe0) {
            n = 3;
        } else if ((c & 0xf8) == 0xf0) {
            n = 4;
        }
        if (i + n > s.size()) {
            n = 1;
        }
        out.push_back(s.substr(i, n));
        i += n;
    }
    return out;
}

static bool
is_ascii_alnum(char c) {
    return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

static std::string
ascii_upper(std::string s) {
    std::string out;
    for (const auto& ch : split_utf8(s)) {
        if (ch.size() == 1) {
            const unsigned char c = (unsigned char)ch[0];
            if (c >= 'a' && c <= 'z') {
                out.push_back((char)(c - 32));
            } else {
                out.push_back((char)c);
            }
            continue;
        }
        if (ch == "à")
            out += "À";
        else if (ch == "á")
            out += "Á";
        else if (ch == "â")
            out += "Â";
        else if (ch == "ã")
            out += "Ã";
        else if (ch == "ä")
            out += "Ä";
        else if (ch == "å")
            out += "Å";
        else if (ch == "ç")
            out += "Ç";
        else if (ch == "è")
            out += "È";
        else if (ch == "é")
            out += "É";
        else if (ch == "ê")
            out += "Ê";
        else if (ch == "ë")
            out += "Ë";
        else if (ch == "ì")
            out += "Ì";
        else if (ch == "í")
            out += "Í";
        else if (ch == "î")
            out += "Î";
        else if (ch == "ï")
            out += "Ï";
        else if (ch == "ñ")
            out += "Ñ";
        else if (ch == "ò")
            out += "Ò";
        else if (ch == "ó")
            out += "Ó";
        else if (ch == "ô")
            out += "Ô";
        else if (ch == "õ")
            out += "Õ";
        else if (ch == "ö")
            out += "Ö";
        else if (ch == "ø")
            out += "Ø";
        else if (ch == "ù")
            out += "Ù";
        else if (ch == "ú")
            out += "Ú";
        else if (ch == "û")
            out += "Û";
        else if (ch == "ü")
            out += "Ü";
        else if (ch == "ý")
            out += "Ý";
        else if (ch == "ÿ")
            out += "Ÿ";
        else if (ch == "ß")
            out += "ẞ";
        else
            out += ch;
    }
    return out;
}

static std::string
ascii_lower(std::string s) {
    std::string out;
    for (const auto& ch : split_utf8(s)) {
        if (ch.size() == 1) {
            const unsigned char c = (unsigned char)ch[0];
            if (c >= 'A' && c <= 'Z') {
                out.push_back((char)(c + 32));
            } else {
                out.push_back((char)c);
            }
            continue;
        }
        if (ch == "À")
            out += "à";
        else if (ch == "Á")
            out += "á";
        else if (ch == "Â")
            out += "â";
        else if (ch == "Ã")
            out += "ã";
        else if (ch == "Ä")
            out += "ä";
        else if (ch == "Å")
            out += "å";
        else if (ch == "Ç")
            out += "ç";
        else if (ch == "È")
            out += "è";
        else if (ch == "É")
            out += "é";
        else if (ch == "Ê")
            out += "ê";
        else if (ch == "Ë")
            out += "ë";
        else if (ch == "Ì")
            out += "ì";
        else if (ch == "Í")
            out += "í";
        else if (ch == "Î")
            out += "î";
        else if (ch == "Ï")
            out += "ï";
        else if (ch == "Ñ")
            out += "ñ";
        else if (ch == "Ò")
            out += "ò";
        else if (ch == "Ó")
            out += "ó";
        else if (ch == "Ô")
            out += "ô";
        else if (ch == "Õ")
            out += "õ";
        else if (ch == "Ö")
            out += "ö";
        else if (ch == "Ø")
            out += "ø";
        else if (ch == "Ù")
            out += "ù";
        else if (ch == "Ú")
            out += "ú";
        else if (ch == "Û")
            out += "û";
        else if (ch == "Ü")
            out += "ü";
        else if (ch == "Ý")
            out += "ý";
        else if (ch == "Ÿ")
            out += "ÿ";
        else if (ch == "ẞ")
            out += "ß";
        else
            out += ch;
    }
    return out;
}

static std::string
set_grapheme_case(const std::string& s, const std::string& mode) {
    if (mode == "upper") {
        return ascii_upper(s);
    }
    if (mode == "lower") {
        return ascii_lower(s);
    }
    return s;
}

static std::string
replace_all(std::string s, const std::string& from, const std::string& to) {
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
    return s;
}

static std::vector<std::string>
default_punct() {
    return {"!", "\"", "(", ")", ",", "-", ".", "/", ":", ";", "?", "[", "]", "{", "}"};
}

static std::vector<std::string>
hindi_chars() {
    return {"अ", "आ", "इ", "ई", "उ", "ऊ", "ऋ", "ॠ", "ए", "ऐ", "ओ", "औ", "ऍ", "ऑ", "क",
            "ख", "ग", "घ", "ङ", "च", "छ", "ज", "झ", "ञ", "ट", "ठ", "ड", "ढ", "ण", "त",
            "थ", "द", "ध", "न", "प", "फ", "ब", "भ", "म", "य", "र", "ल", "व", "श", "ष",
            "स", "ह", "ळ", "ऩ", "ऱ", "ा", "ि", "ी", "ु",  "ू",  "ृ",  "ॄ",  "े",  "ै",  "ो",
            "ौ", "ॅ",  "ॉ", "ँ",  "ं",  "ः", "्",  "़",  "ॊ", "ॢ",  "ॣ",  "ॆ",  "।"};
}

static std::vector<std::string>
ascii_lower_chars() {
    return {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
            "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"};
}

static std::vector<std::string>
hindi_tokens() {
    std::vector<std::string> tokens = {" "};
    const auto chars = hindi_chars();
    tokens.insert(tokens.end(), chars.begin(), chars.end());
    // NeMo uses case="mixed" for Hindi. Devanagari has no case, so this
    // duplicates the same symbols; its token-to-id map keeps the later IDs.
    tokens.insert(tokens.end(), chars.begin(), chars.end());
    const auto ascii = ascii_lower_chars();
    tokens.insert(tokens.end(), ascii.begin(), ascii.end());
    tokens.push_back("'");
    const auto punct = default_punct();
    tokens.insert(tokens.end(), punct.begin(), punct.end());
    tokens.push_back("<pad>");
    tokens.push_back("<oov>");
    return tokens;
}

#ifdef NEMO_SPEECH_TTS_WITH_JA
static std::vector<std::string>
japanese_tokens(bool lowercase_ascii = false) {
    std::vector<std::string> tokens = {
            " ",  "0",  "1",  "ァ", "ア",    "ィ",   "イ", "ゥ", "ウ",     "ェ", "エ", "ォ", "オ",
            "カ", "ガ", "キ", "ギ", "ク",    "グ",   "ケ", "ゲ", "コ",     "ゴ", "サ", "ザ", "シ",
            "ジ", "ス", "ズ", "セ", "ゼ",    "ソ",   "ゾ", "タ", "ダ",     "チ", "ヂ", "ッ", "ツ",
            "ヅ", "テ", "デ", "ト", "ド",    "ナ",   "ニ", "ヌ", "ネ",     "ノ", "ハ", "バ", "パ",
            "ヒ", "ビ", "ピ", "フ", "ブ",    "プ",   "ヘ", "ベ", "ペ",     "ホ", "ボ", "ポ", "マ",
            "ミ", "ム", "メ", "モ", "ャ",    "ヤ",   "ュ", "ユ", "ョ",     "ヨ", "ラ", "リ", "ル",
            "レ", "ロ", "ヮ", "ワ", "ヲ",    "ン",   "ヴ", "ヵ", "ヶ",     "ー", "A",  "B",  "C",
            "D",  "E",  "F",  "G",  "H",     "I",    "J",  "K",  "L",      "M",  "N",  "O",  "P",
            "Q",  "R",  "S",  "T",  "U",     "V",    "W",  "X",  "Y",      "Z",  "!",  "\"", "(",
            ")",  ",",  "-",  ".",  "/",     ":",    ";",  "?",  "[",      "]",  "{",  "}",  "«",
            "»",  "•",  "‥",  "…",  "‹",     "›",    "※",  "◦",  "、",     "。", "〃", "〈", "〉",
            "《", "》", "「", "」", "『",    "』",   "【", "】", "〒",     "〓", "〔", "〕", "〖",
            "〗", "〘", "〙", "〚", "〛",    "〜",   "〽", "・", "・・・", "ー", "﹅", "﹆", "！",
            "＊", "？", "｟", "｠", "<pad>", "<oov>"};
    if (lowercase_ascii) {
        for (auto& token : tokens) {
            if (token.size() == 1 && token[0] >= 'A' && token[0] <= 'Z') {
                token[0] = static_cast<char>(token[0] - 'A' + 'a');
            }
        }
    }
    return tokens;
}
#endif

static std::vector<std::string>
ipa_punct(const std::string& locale) {
    const std::vector<std::string> base = default_punct();
    std::set<std::string> p(base.begin(), base.end());
    if (locale == "de-DE" || locale == "es-ES" || locale == "it-IT" || locale == "fr-FR" ||
        locale == "ja-JP") {
        p.insert("«");
        p.insert("»");
        p.insert("‹");
        p.insert("›");
    }
    if (locale == "de-DE") {
        p.insert("„");
        p.insert("“");
        p.insert("‚");
        p.insert("‘");
        p.insert("‒");
        p.insert("–");
        p.insert("—");
    } else if (locale == "es-ES") {
        p.insert("¿");
        p.insert("¡");
    } else if (locale == "hi-IN") {
        p.insert("।");
        p.insert("॥");
    }
    return std::vector<std::string>(p.begin(), p.end());
}

static bool
is_space_char(const std::string& c) {
    return c == " " || c == "\t" || c == "\n" || c == "\r";
}

#ifdef NEMO_SPEECH_TTS_WITH_JA
static uint32_t
utf8_codepoint(const std::string& c) {
    if (c.empty()) {
        return 0;
    }
    const unsigned char b0 = (unsigned char)c[0];
    if ((b0 & 0x80) == 0) {
        return b0;
    }
    if ((b0 & 0xe0) == 0xc0 && c.size() >= 2) {
        return ((uint32_t)(b0 & 0x1f) << 6) | (uint32_t)((unsigned char)c[1] & 0x3f);
    }
    if ((b0 & 0xf0) == 0xe0 && c.size() >= 3) {
        return ((uint32_t)(b0 & 0x0f) << 12) | ((uint32_t)((unsigned char)c[1] & 0x3f) << 6) |
               (uint32_t)((unsigned char)c[2] & 0x3f);
    }
    if ((b0 & 0xf8) == 0xf0 && c.size() >= 4) {
        return ((uint32_t)(b0 & 0x07) << 18) | ((uint32_t)((unsigned char)c[1] & 0x3f) << 12) |
               ((uint32_t)((unsigned char)c[2] & 0x3f) << 6) |
               (uint32_t)((unsigned char)c[3] & 0x3f);
    }
    return b0;
}

static bool
is_japanese_mora_base(const std::string& c) {
    const uint32_t cp = utf8_codepoint(c);
    return (cp >= 0x30a2 && cp <= 0x30f3) || cp == 0x30f4;
}

static bool
is_japanese_mora_suffix(const std::string& c) {
    static const std::unordered_set<std::string> suffixes = {"ャ", "ュ", "ョ", "ァ", "ィ",
                                                             "ゥ", "ェ", "ォ", "ヮ"};
    return suffixes.count(c) != 0;
}

static bool
is_japanese_standalone_mora(const std::string& c) {
    static const std::unordered_set<std::string> standalone = {
        "ァ", "ィ", "ゥ", "ェ", "ォ", "ヵ", "ヶ", "ッ", "ャ", "ュ", "ョ", "ヮ", "ー"};
    return standalone.count(c) != 0;
}

static std::vector<std::string>
split_katakana_to_moras(const std::string& katakana) {
    const std::vector<std::string> chars = split_utf8(katakana);
    std::vector<std::string> moras;
    for (size_t i = 0; i < chars.size();) {
        if (is_japanese_mora_base(chars[i])) {
            std::string mora = chars[i++];
            if (i < chars.size() && is_japanese_mora_suffix(chars[i])) {
                mora += chars[i++];
            }
            moras.push_back(std::move(mora));
        } else if (is_japanese_standalone_mora(chars[i])) {
            moras.push_back(chars[i++]);
        } else {
            ++i;
        }
    }
    return moras;
}

static std::vector<int>
japanese_pitch_pattern(int acc, size_t total_mora) {
    if (total_mora == 0) {
        return {};
    }
    if (acc == 0 || acc >= (int)total_mora) {
        std::vector<int> pattern(total_mora, 1);
        pattern[0] = 0;
        return pattern;
    }
    if (acc == 1) {
        std::vector<int> pattern(total_mora, 0);
        pattern[0] = 1;
        return pattern;
    }
    std::vector<int> pattern;
    pattern.reserve(total_mora);
    pattern.push_back(0);
    for (int i = 1; i < acc; ++i) {
        pattern.push_back(1);
    }
    while (pattern.size() < total_mora) {
        pattern.push_back(0);
    }
    return pattern;
}

struct japanese_frontend_word {
    std::string text;
    std::string pos;
    std::string pron;
    int acc = 0;
    int mora_size = 0;
    int chain_flag = 0;
};

static std::string
openjtalk_safe_string(const char* value) {
    return value == nullptr ? std::string() : std::string(value);
}

class openjtalk_frontend {
   public:
    explicit openjtalk_frontend(const fs::path& dictionary_dir) {
        const std::string path = dictionary_dir.string();
        Mecab_initialize(&mecab_);
        NJD_initialize(&njd_);
        if (Mecab_load(&mecab_, path.c_str()) != TRUE) {
            NJD_clear(&njd_);
            Mecab_clear(&mecab_);
            throw std::runtime_error("failed to load OpenJTalk dictionary from " + path);
        }
        initialized_ = true;
    }

    openjtalk_frontend(const openjtalk_frontend&) = delete;
    openjtalk_frontend& operator=(const openjtalk_frontend&) = delete;

    ~openjtalk_frontend() {
        if (initialized_) {
            NJD_clear(&njd_);
            Mecab_clear(&mecab_);
        }
    }

    std::vector<japanese_frontend_word> run(const std::string& text) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<char> mecab_text(text.size() * 4 + 1024, 0);
        text2mecab(mecab_text.data(), text.c_str());
        if (Mecab_analysis(&mecab_, mecab_text.data()) != TRUE) {
            Mecab_refresh(&mecab_);
            throw std::runtime_error("OpenJTalk MeCab analysis failed");
        }

        mecab2njd(&njd_, Mecab_get_feature(&mecab_), Mecab_get_size(&mecab_));
        njd_set_pronunciation(&njd_);
        njd_set_digit(&njd_);
        njd_set_accent_phrase(&njd_);
        njd_set_accent_type(&njd_);
        njd_set_unvoiced_vowel(&njd_);
        njd_set_long_vowel(&njd_);

        std::vector<japanese_frontend_word> words;
        for (NJDNode* node = njd_.head; node != nullptr; node = node->next) {
            japanese_frontend_word word;
            word.text = openjtalk_safe_string(NJDNode_get_string(node));
            word.pos = openjtalk_safe_string(NJDNode_get_pos(node));
            word.pron = openjtalk_safe_string(NJDNode_get_pron(node));
            word.acc = NJDNode_get_acc(node);
            word.mora_size = NJDNode_get_mora_size(node);
            word.chain_flag = NJDNode_get_chain_flag(node);
            words.push_back(std::move(word));
        }

        NJD_refresh(&njd_);
        Mecab_refresh(&mecab_);
        return words;
    }

   private:
    Mecab mecab_{};
    NJD njd_{};
    bool initialized_ = false;
    std::mutex mutex_;
};

static bool
is_openjtalk_dictionary_dir(const fs::path& path) {
    return fs::is_regular_file(path / "sys.dic") && fs::is_regular_file(path / "unk.dic") &&
           fs::is_regular_file(path / "char.bin") && fs::is_regular_file(path / "matrix.bin");
}

static fs::path
find_openjtalk_dictionary_dir(const fs::path& model_dir) {
    std::vector<fs::path> candidates;
    if (const char* env = std::getenv("MAGPIE_OPENJTALK_DIC_DIR")) {
        if (*env != '\0') {
            candidates.emplace_back(env);
        }
    }
    candidates.push_back(model_dir / "open_jtalk_dic");
    candidates.push_back(model_dir / "open_jtalk_dic_utf_8-1.11");
    candidates.push_back(model_dir / "japanese" / "open_jtalk_dic");
    if (std::string(NEMO_SPEECH_OPENJTALK_DIC_DIR).empty() == false) {
        candidates.emplace_back(NEMO_SPEECH_OPENJTALK_DIC_DIR);
    }
    if (std::string(NEMO_SPEECH_OPENJTALK_INSTALLED_DIC_DIR).empty() == false) {
        candidates.emplace_back(NEMO_SPEECH_OPENJTALK_INSTALLED_DIC_DIR);
    }
    for (const auto& candidate : candidates) {
        if (is_openjtalk_dictionary_dir(candidate)) {
            return candidate;
        }
    }
    return {};
}

static std::shared_ptr<openjtalk_frontend>
openjtalk_frontend_for_dictionary(const fs::path& dictionary_dir) {
    static std::mutex mutex;
    static std::string cached_path;
    static std::shared_ptr<openjtalk_frontend> cached;

    std::string path;
    try {
        path = fs::weakly_canonical(dictionary_dir).string();
    }
    catch (const std::exception&) {
        path = dictionary_dir.string();
    }

    std::lock_guard<std::mutex> lock(mutex);
    if (!cached || cached_path != path) {
        cached = std::make_shared<openjtalk_frontend>(dictionary_dir);
        cached_path = path;
    }
    return cached;
}
#endif

static std::string
trim_ascii_space(const std::string& s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const auto last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

static int
count_ascii_space_words(const std::string& text) {
    int count = 0;
    bool in_word = false;
    for (const char c : text) {
        const bool is_space = c == ' ' || c == '\t' || c == '\r' || c == '\n';
        if (is_space) {
            if (in_word) {
                ++count;
                in_word = false;
            }
        } else {
            in_word = true;
        }
    }
    if (in_word) {
        ++count;
    }
    return count;
}

static std::vector<std::string>
split_words_by_limit(const std::string& text, int max_words) {
    std::vector<std::string> out;
    std::istringstream words(text);
    std::string word;
    std::string current;
    int current_words = 0;
    while (words >> word) {
        if (current_words >= max_words) {
            out.push_back(current);
            current.clear();
            current_words = 0;
        }
        if (!current.empty()) {
            current.push_back(' ');
        }
        current += word;
        ++current_words;
    }
    if (!current.empty()) {
        out.push_back(current);
    }
    for (size_t i = 0; i + 1 < out.size(); ++i) {
        const size_t last = out[i].find_last_not_of(" \t\r\n");
        if (last == std::string::npos) {
            continue;
        }
        const char marker = out[i][last];
        if (marker != '.' && marker != '?' && marker != '!' && marker != ',' && marker != ':' &&
            marker != ';') {
            out[i].push_back('.');
        }
    }
    return out;
}

static std::vector<std::string>
split_long_sentence_by_commas(const std::string& sentence) {
    constexpr int kMaxWordsPerPhrase = 35;
    if (count_ascii_space_words(sentence) <= kMaxWordsPerPhrase) {
        return {sentence};
    }

    std::vector<std::string> comma_clauses;
    size_t start = 0;
    for (size_t i = 0; i < sentence.size(); ++i) {
        if (sentence[i] != ',') {
            continue;
        }
        std::string clause = trim_ascii_space(sentence.substr(start, i + 1 - start));
        if (!clause.empty()) {
            comma_clauses.push_back(std::move(clause));
        }
        start = i + 1;
    }
    if (start < sentence.size()) {
        std::string clause = trim_ascii_space(sentence.substr(start));
        if (!clause.empty()) {
            comma_clauses.push_back(std::move(clause));
        }
    }
    if (comma_clauses.size() <= 1) {
        return split_words_by_limit(sentence, kMaxWordsPerPhrase);
    }

    std::vector<std::string> phrases;
    std::string current;
    int current_words = 0;
    for (const std::string& clause : comma_clauses) {
        std::vector<std::string> pieces;
        if (count_ascii_space_words(clause) > kMaxWordsPerPhrase) {
            pieces = split_words_by_limit(clause, kMaxWordsPerPhrase);
        } else {
            pieces.push_back(clause);
        }

        for (const std::string& piece : pieces) {
            const int piece_words = count_ascii_space_words(piece);
            if (piece_words == 0) {
                continue;
            }
            if (current.empty()) {
                current = piece;
                current_words = piece_words;
            } else if (current_words + piece_words <= kMaxWordsPerPhrase) {
                current.push_back(' ');
                current += piece;
                current_words += piece_words;
            } else {
                phrases.push_back(std::move(current));
                current = piece;
                current_words = piece_words;
            }
        }
    }
    if (!current.empty()) {
        phrases.push_back(std::move(current));
    }
    return phrases;
}

static size_t
cjk_terminal_size_at(const std::string& text, size_t position) {
    static constexpr std::array<const char*, 3> terminals = {"。", "！", "？"};
    for (const char* terminal : terminals) {
        const size_t size = std::strlen(terminal);
        if (text.compare(position, size, terminal) == 0) {
            return size;
        }
    }
    return 0;
}

static std::vector<std::string>
split_sentences(std::string paragraph) {
    paragraph = replace_all(paragraph, "-", " ");
    paragraph = replace_all(paragraph, "*", "");
    if (paragraph.find_first_not_of(" \t\r\n") == std::string::npos) {
        return {};
    }

    std::vector<std::string> sentences;
    size_t start = 0;
    for (size_t i = 0; i < paragraph.size(); ++i) {
        const char c = paragraph[i];
        const char next = i + 1 < paragraph.size() ? paragraph[i + 1] : '\0';
        const bool ascii_boundary = (c == '.' || c == '?' || c == '!') && next == ' ';
        const size_t cjk_terminal_size = cjk_terminal_size_at(paragraph, i);
        if (ascii_boundary || cjk_terminal_size != 0) {
            size_t end = i + (ascii_boundary ? 1 : cjk_terminal_size);
            if (cjk_terminal_size != 0) {
                while (const size_t next_terminal_size = cjk_terminal_size_at(paragraph, end)) {
                    end += next_terminal_size;
                }
            }
            std::string sent = paragraph.substr(start, end - start);
            start = ascii_boundary ? i + 2 : end;
            const auto first = sent.find_first_not_of(" \t\r\n");
            const auto last = sent.find_last_not_of(" \t\r\n");
            if (first != std::string::npos) {
                sentences.push_back(sent.substr(first, last - first + 1));
            }
            if (cjk_terminal_size != 0) {
                i = end - 1;
            }
        }
    }
    if (start < paragraph.size()) {
        std::string sent = paragraph.substr(start);
        sent = trim_ascii_space(sent);
        if (!sent.empty()) {
            sentences.push_back(std::move(sent));
        }
    }

    std::vector<std::string> phrases;
    for (const std::string& sent : sentences) {
        const auto split = split_long_sentence_by_commas(sent);
        phrases.insert(phrases.end(), split.begin(), split.end());
    }
    sentences = std::move(phrases);

    for (std::string& sent : sentences) {
        if (!sent.empty() && sent[0] >= 'a' && sent[0] <= 'z') {
            sent[0] = (char)(sent[0] - 32);
        }
    }
    return sentences;
}

static std::vector<std::string>
tokenizer_input_units(std::string text, bool sentence_chunking) {
    if (sentence_chunking) {
        return split_sentences(std::move(text));
    }
    return {std::move(text)};
}

static std::vector<std::string>
tokenizer_input_units(const params& p, std::string text) {
    std::vector<std::string> units = tokenizer_input_units(std::move(text), p.sentence_chunking);
    if (p.chunk_text_transform) {
        for (size_t i = 0; i < units.size(); ++i) {
            units[i] = p.chunk_text_transform(units[i], i + 1 == units.size());
        }
    }
    return units;
}

static int
token_id_for_symbol(const std::vector<std::string>& tokens, int offset, const std::string& symbol) {
    const auto it = std::find(tokens.begin(), tokens.end(), symbol);
    if (it == tokens.end()) {
        throw std::runtime_error("failed to find tokenizer symbol '" + symbol + "'");
    }
    return offset + (int)std::distance(tokens.begin(), it);
}

static void
pad_short_text_chunk_before_eos(chunk& ch, int eos_id, int pad_id) {
    constexpr size_t kMinTokensExcludingFirstAndEos = 4;
    if (ch.tokens.size() < 2 || ch.tokens.back() != eos_id) {
        return;
    }

    const size_t tokens_excluding_first_and_eos = ch.tokens.size() - 2;
    if (tokens_excluding_first_and_eos >= kMinTokensExcludingFirstAndEos) {
        return;
    }

    const size_t pad_count = kMinTokensExcludingFirstAndEos - tokens_excluding_first_and_eos;
    ch.tokens.insert(ch.tokens.begin() + 1, pad_count, pad_id);
}

struct ipa_config {
    std::string tokenizer_name;
    int offset = 0;
    std::string dict_hint;
    std::string heteronym_hint;
    std::string locale = "en-US";
    std::string grapheme_case = "upper";
    std::string grapheme_prefix;
    bool apostrophe = true;
    bool pad_with_space = false;
};

static ipa_config
ipa_config_for_params(const params& p) {
    if (p.tokenizer_name.empty() || p.phoneme_dict.empty()) {
        throw std::runtime_error("incomplete IPA tokenizer profile for language " + p.language);
    }
    return {
        p.tokenizer_name,
        p.offset,
        p.phoneme_dict,
        p.heteronyms,
        p.locale,
        p.grapheme_case,
        p.grapheme_prefix,
        p.apostrophe,
        p.pad_with_space,
    };
}

static std::vector<std::string>
exact_ipa_tokens(const std::string& tokenizer_name) {
    if (tokenizer_name == "english_phoneme") {
        return {
            "!", "\"", "'", "(", ")", ",", "-", ".", "/", "0", "1", "2", "3", "4", "5",     "6",
            "7", "8",  "9", ":", ";", "?", "A", "B", "C", "D", "E", "F", "G", "H", "I",     "J",
            "K", "L",  "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y",     "Z",
            "[", "]",  "a", "b", "d", "e", "f", "h", "i", "j", "k", "l", "m", "n", "o",     "p",
            "s", "t",  "u", "v", "w", "z", "{", "}", "À", "É", "æ", "ð", "ŋ", "ɑ", "ɔ",     "ə",
            "ɚ", "ɛ",  "ɝ", "ɡ", "ɪ", "ɹ", "ʃ", "ʊ", "ʌ", "ʒ", "ˈ", "ˌ", "θ", " ", "<pad>", "<oov>",
        };
    }
    if (tokenizer_name == "spanish_phoneme") {
        return {
            "!", "\"", "'", "(", ")", ",", "-", ".", "/", ":", ";", "?",     "A",     "B", "C",
            "D", "E",  "F", "G", "H", "I", "J", "K", "L", "M", "N", "O",     "P",     "Q", "R",
            "S", "T",  "U", "V", "W", "X", "Y", "Z", "[", "]", "a", "b",     "d",     "e", "f",
            "h", "i",  "j", "k", "l", "m", "n", "o", "p", "r", "s", "t",     "u",     "w", "x",
            "{", "}",  "¡", "«", "»", "¿", "Á", "Ç", "É", "Í", "Î", "Ñ",     "Ó",     "Ö", "Ú",
            "Ü", "Ý",  "ð", "ŋ", "ɛ", "ɟ", "ɡ", "ɣ", "ɪ", "ɲ", "ɾ", "ʃ",     "ʊ",     "ʎ", "ʒ",
            "ʝ", "ˈ",  "ˌ", "ː", "̯",  "͡",  "β", "θ", "‹", "›", " ", "<pad>", "<oov>",
        };
    }
    if (tokenizer_name == "german_phoneme") {
        return {
            "!",  "\"", "#A", "#B", "#C", "#D",    "#E",    "#F", "#G", "#H", "#I", "#J", "#K",
            "#L", "#M", "#N", "#O", "#P", "#Q",    "#R",    "#S", "#T", "#U", "#V", "#W", "#X",
            "#Y", "#Z", "#a", "#b", "#c", "#d",    "#e",    "#f", "#g", "#h", "#i", "#j", "#k",
            "#l", "#m", "#n", "#o", "#p", "#q",    "#r",    "#s", "#t", "#u", "#v", "#w", "#x",
            "#y", "#z", "#Ä", "#Å", "#Ö", "#Ü",    "#ß",    "#à", "#á", "#ä", "#å", "#ç", "#è",
            "#é", "#ê", "#ë", "#í", "#ó", "#ö",    "#ø",    "#ü", "'",  "(",  ")",  ",",  "-",
            ".",  "/",  "1",  ":",  ";",  "?",     "[",     "]",  "a",  "b",  "d",  "e",  "f",
            "h",  "i",  "j",  "k",  "l",  "m",     "n",     "o",  "p",  "r",  "s",  "t",  "u",
            "v",  "w",  "x",  "y",  "z",  "{",     "}",     "«",  "»",  "ç",  "ð",  "ø",  "ŋ",
            "œ",  "ɐ",  "ɑ",  "ɒ",  "ɔ",  "ə",     "ɛ",     "ɜ",  "ɡ",  "ɪ",  "ɹ",  "ɾ",  "ʃ",
            "ʊ",  "ʌ",  "ʒ",  "ˈ",  "ˌ",  "ː",     "̃",      "θ",  "‒",  "–",  "—",  "‘",  "‚",
            "“",  "„",  "‹",  "›",  " ",  "<pad>", "<oov>",
        };
    }
    if (tokenizer_name == "portuguese_Brazilian_phoneme") {
        return {"!",  "\"", "#A", "#B", "#C", "#D",    "#E",   "#F", "#G", "#H", "#I", "#J", "#K",
                "#L", "#M", "#N", "#O", "#P", "#Q",    "#R",   "#S", "#T", "#U", "#V", "#W", "#X",
                "#Y", "#Z", "#À", "#Á", "#Â", "#Ã",    "#Ç",   "#É", "#Ê", "#Í", "#Ó", "#Ô", "#Õ",
                "#Ú", "#Ü", "'",  "(",  ")",  ",",     "-",    ".",  "/",  ":",  ";",  "?",  "[",
                "]",  "a",  "b",  "d",  "e",  "f",     "h",    "i",  "j",  "k",  "l",  "m",  "n",
                "o",  "p",  "r",  "s",  "t",  "u",     "v",    "w",  "x",  "y",  "z",  "{",  "}",
                "ð",  "õ",  "ĩ",  "ŋ",  "ũ",  "ɐ",     "ɑ",    "ɒ",  "ɔ",  "ə",  "ɛ",  "ɜ",  "ɡ",
                "ɪ",  "ɲ",  "ɹ",  "ɾ",  "ʁ",  "ʃ",     "ʊ",    "ʌ",  "ʎ",  "ʒ",  "ʲ",  "ˈ",  "ˌ",
                "ː",  "̃",   "θ",  "ẽ",  " ",  "<pad>", "<oov>"};
    }
    if (tokenizer_name == "hindi_phoneme") {
        return {
            "!", "\"", "'", "(", ")", ",", "-", ".", "/", "0", "1", "2", "3", "4",
            "5", "6", "7", "8", "9", ":", ";", "?", "A", "B", "C", "D", "E", "F",
            "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
            "U", "V", "W", "X", "Y", "Z", "[", "]", "a", "b", "c", "d", "e", "f",
            "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u",
            "v", "w", "x", "z", "{", "}", "À", "É", "ã", "æ", "ð", "õ", "ĩ", "ŋ",
            "ũ", "ɑ", "ɔ", "ɖ", "ə", "ɚ", "ɛ", "ɝ", "ɟ", "ɡ", "ɣ", "ɪ", "ɭ", "ɲ",
            "ɳ", "ɹ", "ɾ", "ʂ", "ʃ", "ʈ", "ʊ", "ʋ", "ʌ", "ʒ", "ʰ", "ˈ", "ˌ", "ː",
            "̃", "̩", "θ", "χ", "ँ", "ं", "ः", "अ", "आ", "इ", "ई", "उ", "ऊ", "ऋ",
            "ऌ", "ऍ", "ऎ", "ए", "ऐ", "ऑ", "ओ", "औ", "क", "ख", "ग", "घ", "ङ", "च",
            "छ", "ज", "झ", "ञ", "ट", "ठ", "ड", "ढ", "ण", "त", "थ", "द", "ध", "न",
            "ऩ", "प", "फ", "ब", "भ", "म", "य", "र", "ऱ", "ल", "ळ", "ऴ", "व", "श",
            "ष", "स", "ह", "ऺ", "़", "ऽ", "ा", "ि", "ी", "ु", "ू", "ृ", "ॅ", "ॆ",
            "े", "ै", "ॉ", "ॊ", "ो", "ौ", "्", "ॐ", "॓", "ॠ", "ॡ", "ॢ", "।", "॥",
            "॰", "ẽ", " ", "<pad>", "<oov>",
        };
    }
    return {};
}

class ipa_tokenizer {
   public:
    ipa_tokenizer(const fs::path& root, ipa_config cfg) : cfg_(std::move(cfg)) { load(root); }

    int vocab_size() const { return static_cast<int>(tokens_.size()); }
    int pad_id() const { return cfg_.offset + token_to_id_.at("<pad>"); }

    std::vector<int> encode(const std::string& raw_text) const {
        std::string text = raw_text;
        if (cfg_.locale == "en-US") {
            text = replace_all(text, "’", "'");
        } else {
            text = replace_all(text, "’", "'");
        }

        const std::vector<std::string> pieces = lexical_pieces(text);
        std::vector<std::string> g2p;
        for (size_t i = 0; i < pieces.size(); ++i) {
            const auto& piece = pieces[i];
            if (piece == " ") {
                g2p.push_back(piece);
            } else if (is_word_piece(piece)) {
                const bool phonemize_single_char =
                    is_single_character_word_piece(piece) &&
                    !has_adjacent_single_character_word_piece(pieces, i);
                const auto parsed = parse_word(piece, phonemize_single_char);
                g2p.insert(g2p.end(), parsed.begin(), parsed.end());
            } else {
                const auto chars = split_utf8(piece);
                g2p.insert(g2p.end(), chars.begin(), chars.end());
            }
        }

        std::vector<std::string> symbols;
        std::unordered_set<std::string> token_set(tokens_.begin(), tokens_.end());
        const std::string space = " ";
        for (const auto& p : g2p) {
            if (p == space && !symbols.empty() && symbols.back() != space) {
                symbols.push_back(p);
            } else if ((p == "'" || token_set.count(p) != 0) && token_set.count(p) != 0) {
                symbols.push_back(p);
            } else if (punct_.count(p) != 0) {
                symbols.push_back(p);
            }
        }
        while (!symbols.empty() && symbols.back() == space) {
            symbols.pop_back();
        }
        if (cfg_.pad_with_space) {
            symbols.insert(symbols.begin(), space);
            symbols.push_back(space);
        }

        std::vector<int> ids;
        ids.reserve(symbols.size());
        for (const auto& s : symbols) {
            const auto it = token_to_id_.find(s);
            if (it != token_to_id_.end()) {
                ids.push_back(cfg_.offset + it->second);
            }
        }
        return ids;
    }

   private:
    ipa_config cfg_;
    std::map<std::string, std::vector<std::vector<std::string>>> dict_;
    std::unordered_set<std::string> heteronyms_;
    std::vector<std::string> tokens_;
    std::unordered_map<std::string, int> token_to_id_;
    std::unordered_set<std::string> punct_;

    void load(const fs::path& root) {
        const fs::path dict_path = root / cfg_.dict_hint;
        if (!fs::is_regular_file(dict_path)) {
            throw std::runtime_error("failed to find tokenizer dictionary '" + dict_path.string() + "'");
        }
        for (const auto& p : ipa_punct(cfg_.locale)) {
            punct_.insert(p);
        }
        if (!cfg_.heteronym_hint.empty()) {
            const fs::path heteronym_path = root / cfg_.heteronym_hint;
            if (fs::is_regular_file(heteronym_path)) {
                std::istringstream in(read_file(heteronym_path));
                std::string line;
                while (std::getline(in, line)) {
                    if (!line.empty() && line.back() == '\r') {
                        line.pop_back();
                    }
                    if (!line.empty()) {
                        heteronyms_.insert(set_grapheme_case(line, cfg_.grapheme_case));
                    }
                }
            }
        }

        std::set<std::string> symbols;
        std::istringstream in(read_file(dict_path));
        std::string line;
        const std::regex alt_re("\\([0-9]+\\)");
        while (std::getline(in, line)) {
            if (line.empty()) {
                continue;
            }
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            const unsigned char first = (unsigned char)line[0];
            if (!(first == '\'' || first >= 0x80 || is_ascii_alnum((char)first))) {
                continue;
            }
            std::istringstream ls(line);
            std::string word;
            ls >> word;
            std::string pron;
            std::getline(ls, pron);
            if (word.empty() || pron.empty()) {
                continue;
            }
            word = std::regex_replace(word, alt_re, "");
            word = set_grapheme_case(word, cfg_.grapheme_case);
            pron.erase(
                std::remove_if(
                    pron.begin(), pron.end(),
                    [](unsigned char c) { return c == ' ' || c == '\t'; }),
                pron.end());
            std::vector<std::string> pron_chars = split_utf8(pron);
            dict_[word].push_back(pron_chars);
            if (cfg_.grapheme_case == "mixed") {
                const std::string upper = ascii_upper(word);
                if (upper != word) {
                    dict_[upper].push_back(pron_chars);
                }
            }
            symbols.insert(pron_chars.begin(), pron_chars.end());

            std::string word_no_punct;
            for (const auto& c : split_utf8(word)) {
                if (c != "'" && punct_.count(c) == 0) {
                    word_no_punct += c;
                }
            }
            for (const auto& c : split_utf8(word_no_punct)) {
                symbols.insert(cfg_.grapheme_prefix + c);
            }
        }

        for (const auto& p : punct_) {
            symbols.insert(p);
        }
        if (cfg_.apostrophe) {
            symbols.insert("'");
        }
        symbols.insert(" ");

        tokens_ = exact_ipa_tokens(cfg_.tokenizer_name);
        if (tokens_.empty()) {
            tokens_ = std::vector<std::string>(symbols.begin(), symbols.end());
            if (std::find(tokens_.begin(), tokens_.end(), " ") == tokens_.end()) {
                tokens_.push_back(" ");
            }
            tokens_.push_back("<pad>");
            tokens_.push_back("<oov>");
        }

        for (size_t i = 0; i < tokens_.size(); ++i) {
            token_to_id_[tokens_[i]] = (int)i;
        }
    }

    static bool is_word_char(const std::string& c) {
        if (c.size() == 1) {
            const char ch = c[0];
            return is_ascii_alnum(ch) || ch == '\'' || ch == '-';
        }
        static const std::unordered_set<std::string> utf8_punct = {
            "¡", "«", "»", "¿", "‘", "’", "‚", "“", "”", "„", "‹", "›", "‒", "–", "—",
        };
        return c != " " && c != "\t" && c != "\n" && c != "\r" && utf8_punct.count(c) == 0;
    }

    static bool is_word_piece(const std::string& piece) {
        for (const auto& c : split_utf8(piece)) {
            if (is_word_char(c) && c != "-") {
                return true;
            }
        }
        return false;
    }

    static bool is_single_character_word_piece(const std::string& piece) {
        const auto chars = split_utf8(piece);
        return chars.size() == 1 && is_word_char(chars.front()) && chars.front() != "'" &&
               chars.front() != "-";
    }

    static bool has_adjacent_single_character_word_piece(
        const std::vector<std::string>& pieces, size_t index) {
        if (index >= 2 && pieces[index - 1] == " " &&
            is_single_character_word_piece(pieces[index - 2])) {
            return true;
        }
        if (index + 2 < pieces.size() && pieces[index + 1] == " " &&
            is_single_character_word_piece(pieces[index + 2])) {
            return true;
        }
        return false;
    }

    static std::vector<std::string> lexical_pieces(const std::string& text) {
        std::vector<std::string> pieces;
        std::string cur;
        enum class kind { none, word, space, other };
        kind cur_kind = kind::none;
        for (const auto& c : split_utf8(text)) {
            kind k = kind::other;
            if (is_space_char(c)) {
                k = kind::space;
            } else if (is_word_char(c)) {
                k = kind::word;
            }
            if (k == kind::space) {
                if (!cur.empty()) {
                    pieces.push_back(cur);
                    cur.clear();
                }
                if (pieces.empty() || pieces.back() != " ") {
                    pieces.push_back(" ");
                }
                cur_kind = kind::none;
            } else if (k != cur_kind && !cur.empty()) {
                pieces.push_back(cur);
                cur = c;
                cur_kind = k;
            } else {
                cur += c;
                cur_kind = k;
            }
        }
        if (!cur.empty()) {
            pieces.push_back(cur);
        }
        return pieces;
    }

    std::vector<std::string> prepend_grapheme_prefix(const std::string& word) const {
        std::vector<std::string> out;
        for (const auto& c : split_utf8(word)) {
            out.push_back(cfg_.grapheme_prefix + c);
        }
        return out;
    }

    bool dict_has_unique(const std::string& word) const {
        const auto it = dict_.find(word);
        return it != dict_.end() && it->second.size() == 1;
    }

    std::vector<std::string> parse_word_direct(
        const std::string& cased_word, bool& handled, bool phonemize_single_char) const {
        handled = true;
        const auto chars = split_utf8(cased_word);
        bool has_alnum = false;
        for (const auto& c : chars) {
            if (c.size() != 1 || is_ascii_alnum(c[0])) {
                has_alnum = true;
                break;
            }
        }
        if (!has_alnum) {
            return chars;
        }
        if (chars.size() == 1 && is_word_char(chars.front()) && !phonemize_single_char) {
            return prepend_grapheme_prefix(cased_word);
        }
        if (heteronyms_.count(cased_word) != 0) {
            return prepend_grapheme_prefix(cased_word);
        }

        if (cfg_.locale == "en-US") {
            auto suffix_lookup = [&](const std::string& suffix,
                                     std::vector<std::string> append) -> std::vector<std::string> {
                if (cased_word.size() <= suffix.size()) {
                    return {};
                }
                if (cased_word.rfind(suffix) != cased_word.size() - suffix.size()) {
                    return {};
                }
                if (dict_.count(cased_word) != 0) {
                    return {};
                }
                const std::string base = cased_word.substr(0, cased_word.size() - suffix.size());
                if (!dict_has_unique(base) && dict_.count(base) == 0) {
                    return {};
                }
                const auto it = dict_.find(base);
                if (it == dict_.end()) {
                    return {};
                }
                auto out = it->second.front();
                out.insert(out.end(), append.begin(), append.end());
                return out;
            };
            auto out = suffix_lookup("'S", {"z"});
            if (!out.empty())
                return out;
            out = suffix_lookup("S", {"z"});
            if (!out.empty())
                return out;
        }

        auto it = dict_.find(cased_word);
        if (it != dict_.end()) {
            return it->second.front();
        }
        if (cfg_.grapheme_case == "mixed") {
            const std::string upper = ascii_upper(cased_word);
            it = dict_.find(upper);
            if (it != dict_.end()) {
                return it->second.front();
            }
        }
        handled = false;
        return prepend_grapheme_prefix(cased_word);
    }

    std::vector<std::string> parse_word(
        const std::string& word, bool phonemize_single_char = false) const {
        const std::string cased = set_grapheme_case(word, cfg_.grapheme_case);
        bool handled = false;
        auto pron = parse_word_direct(cased, handled, phonemize_single_char);
        if (!handled && cased.find('-') != std::string::npos) {
            std::vector<std::string> out;
            std::stringstream ss(cased);
            std::string sub;
            bool first = true;
            while (std::getline(ss, sub, '-')) {
                if (!first) {
                    out.push_back("-");
                }
                bool sub_handled = false;
                auto p = parse_word_direct(sub, sub_handled, phonemize_single_char);
                out.insert(out.end(), p.begin(), p.end());
                first = false;
            }
            return out;
        }
        return pron;
    }
};

static tokenizer_result
run_byt5_native(const params& p) {
    tokenizer_result result;
    result.language = p.language;
    result.tokenizer_name = p.tokenizer_name;
    result.eos_id = p.eos_id;
    for (const std::string& sentence : tokenizer_input_units(p, p.text)) {
        chunk ch;
        ch.text = sentence;
        for (unsigned char b : sentence) {
            ch.tokens.push_back(p.offset + (int)b + 3);
        }
        ch.tokens.push_back(p.offset + 1);
        ch.tokens.push_back(result.eos_id);
        pad_short_text_chunk_before_eos(ch, result.eos_id, p.offset);
        result.chunks.push_back(std::move(ch));
    }
    return result;
}

static tokenizer_result
run_hindi_native(const params& p) {
    const std::vector<std::string> tokens = hindi_tokens();
    if (static_cast<int>(tokens.size()) != p.expected_vocab_size) {
        throw std::runtime_error("Hindi character vocabulary does not match tokenizer profile");
    }
    std::unordered_map<std::string, int> token_to_id;
    for (size_t i = 0; i < tokens.size(); ++i) {
        token_to_id[tokens[i]] = (int)i;
    }
    const int pad_id = token_id_for_symbol(tokens, p.offset, "<pad>");

    tokenizer_result result;
    result.language = p.language;
    result.tokenizer_name = p.tokenizer_name;
    result.eos_id = p.eos_id;
    for (const std::string& sentence : tokenizer_input_units(p, replace_all(p.text, "’", "'"))) {
        chunk ch;
        ch.text = sentence;
        std::vector<std::string> chars;
        const std::string space = " ";
        for (const auto& c : split_utf8(sentence)) {
            if (c == space) {
                if (!chars.empty() && chars.back() != space) {
                    chars.push_back(c);
                }
            } else if (token_to_id.count(c) != 0) {
                chars.push_back(c);
            }
        }
        while (!chars.empty() && chars.back() == space) {
            chars.pop_back();
        }
        chars.insert(chars.begin(), space);
        chars.push_back(space);
        ch.tokens.reserve(chars.size() + 1);
        for (const auto& c : chars) {
            const auto it = token_to_id.find(c);
            if (it != token_to_id.end()) {
                ch.tokens.push_back(p.offset + it->second);
            }
        }
        ch.tokens.push_back(result.eos_id);
        pad_short_text_chunk_before_eos(ch, result.eos_id, pad_id);
        result.chunks.push_back(std::move(ch));
    }
    return result;
}

static std::vector<std::string>
arabic_tokens() {
    std::vector<std::string> tokens = {" ", "ء", "آ", "أ", "إ", "ؤ", "ئ", "ا", "ب", "ة", "ت", "ث",
                                       "ج", "ح", "خ", "د", "ذ", "ر", "ز", "س", "ش", "ص", "ض", "ط",
                                       "ظ", "ع", "غ", "ف", "ق", "ك", "ل", "م", "ن", "ه", "و", "ى",
                                       "ي", "ً",  "ٌ",  "ٍ",  "َ",  "ُ",  "ِ",  "ّ",  "ٰ",  "ْ"};
    // v2607 pins NeMo ArabicCharsTokenizer charset_version=1, whose mixed-case
    // Arabic character set deliberately contains this second copy.
    const std::vector<std::string> arabic_chars(tokens.begin() + 1, tokens.end());
    tokens.insert(tokens.end(), arabic_chars.begin(), arabic_chars.end());
    for (char c = 'a'; c <= 'z'; ++c) tokens.emplace_back(1, c);
    for (char c = 'A'; c <= 'Z'; ++c) tokens.emplace_back(1, c);
    tokens.push_back("'");
    const auto punct = default_punct();
    tokens.insert(tokens.end(), punct.begin(), punct.end());
    tokens.insert(tokens.end(), {"،", "؛", "؟", "<pad>", "<oov>"});
    return tokens;
}

static tokenizer_result
run_arabic_native(const params& p) {
    const auto tokens = arabic_tokens();
    if (static_cast<int>(tokens.size()) != p.expected_vocab_size) {
        throw std::runtime_error("Arabic vocabulary does not match tokenizer profile");
    }
    std::unordered_map<std::string, int> ids;
    for (size_t i = 0; i < tokens.size(); ++i) ids[tokens[i]] = (int)i;
    tokenizer_result result;
    result.language = p.language;
    result.tokenizer_name = p.tokenizer_name;
    result.eos_id = p.eos_id;
    const int pad_id = token_id_for_symbol(tokens, p.offset, "<pad>");
    for (const std::string& sentence : tokenizer_input_units(p, p.text)) {
        chunk ch;
        ch.text = sentence;
        std::vector<std::string> symbols;
        for (const auto& symbol : split_utf8(sentence)) {
            if (symbol == " ") {
                if (!symbols.empty() && symbols.back() != symbol)
                    symbols.push_back(symbol);
            } else if (ids.count(symbol) != 0) {
                symbols.push_back(symbol);
            }
        }
        while (!symbols.empty() && symbols.back() == " ") symbols.pop_back();
        symbols.insert(symbols.begin(), " ");
        symbols.push_back(" ");
        for (const auto& symbol : symbols) ch.tokens.push_back(p.offset + ids[symbol]);
        ch.tokens.push_back(result.eos_id);
        pad_short_text_chunk_before_eos(ch, result.eos_id, pad_id);
        result.chunks.push_back(std::move(ch));
    }
    return result;
}

#ifdef NEMO_SPEECH_TTS_WITH_ZH
static tokenizer_result
run_mandarin_native(const params& p, const mandarin_tokenizer& tok) {
    tokenizer_result result;
    result.language = p.language;
    result.tokenizer_name = p.tokenizer_name;
    result.eos_id = p.eos_id;
    for (const std::string& sentence : tokenizer_input_units(p, p.text)) {
        chunk ch;
        ch.text = sentence;
        ch.tokens = tok.encode(sentence);
        ch.tokens.push_back(result.eos_id);
        pad_short_text_chunk_before_eos(ch, result.eos_id, tok.pad_id());
        result.chunks.push_back(std::move(ch));
    }
    return result;
}
#endif

#ifdef NEMO_SPEECH_TTS_WITH_JA
static bool
is_ascii_letter_word(const std::string& text, bool lowercase) {
    if (text.empty()) {
        return false;
    }
    for (const unsigned char c : text) {
        const unsigned char first = lowercase ? 'a' : 'A';
        const unsigned char last = lowercase ? 'z' : 'Z';
        if (c < first || c > last) {
            return false;
        }
    }
    return true;
}

static void
process_japanese_chain(
    const std::vector<japanese_frontend_word>& chain, std::vector<std::string>& result) {
    if (chain.empty()) {
        return;
    }

    size_t chain_starter = 0;
    for (size_t i = 0; i < chain.size(); ++i) {
        if (chain[i].chain_flag != 1) {
            chain_starter = i;
            break;
        }
    }
    const int acc = chain[chain_starter].acc;

    std::vector<std::string> moras;
    for (const auto& word : chain) {
        const auto word_moras = split_katakana_to_moras(word.pron);
        moras.insert(moras.end(), word_moras.begin(), word_moras.end());
    }

    const auto pitches = japanese_pitch_pattern(acc, moras.size());
    for (size_t i = 0; i < moras.size() && i < pitches.size(); ++i) {
        result.push_back(std::to_string(pitches[i]));
        const auto mora_chars = split_utf8(moras[i]);
        result.insert(result.end(), mora_chars.begin(), mora_chars.end());
    }
}

static std::vector<std::string>
japanese_g2p(openjtalk_frontend& frontend, const std::string& text, bool lowercase_ascii) {
    const std::vector<japanese_frontend_word> words = frontend.run(text);
    std::vector<std::string> result;
    std::vector<japanese_frontend_word> current_chain;
    const std::unordered_map<std::string, int> token_to_id = [lowercase_ascii] {
        std::unordered_map<std::string, int> ids;
        const auto tokens = japanese_tokens(lowercase_ascii);
        for (size_t i = 0; i < tokens.size(); ++i) {
            ids[tokens[i]] = (int)i;
        }
        return ids;
    }();

    for (size_t idx = 0; idx < words.size(); ++idx) {
        const auto& word = words[idx];
        if (is_ascii_letter_word(word.text, lowercase_ascii)) {
            process_japanese_chain(current_chain, result);
            current_chain.clear();
            const auto chars = split_utf8(word.text);
            result.insert(result.end(), chars.begin(), chars.end());
            continue;
        }

        if (word.pos == "記号" || word.pos == "補助記号") {
            process_japanese_chain(current_chain, result);
            current_chain.clear();
            if (word.text.find_first_not_of(" \t\r\n") == std::string::npos) {
                result.push_back(" ");
            } else if (token_to_id.count(word.text) != 0) {
                result.push_back(word.text);
            }
            continue;
        }

        if (word.pron.empty() || word.mora_size == 0) {
            continue;
        }

        current_chain.push_back(word);
        const bool next_has_chain = idx + 1 < words.size() && words[idx + 1].chain_flag == 1;
        if (!next_has_chain) {
            process_japanese_chain(current_chain, result);
            current_chain.clear();
        }
    }

    process_japanese_chain(current_chain, result);
    return result;
}

static tokenizer_result
run_japanese_native(const params& p) {
    const bool lowercase_ascii = p.ascii_letter_case == "lower";
    const fs::path dictionary_dir = find_openjtalk_dictionary_dir(p.model);
    if (dictionary_dir.empty()) {
        throw std::runtime_error(
            "failed to find OpenJTalk dictionary; set MAGPIE_OPENJTALK_DIC_DIR or build "
            "open_jtalk_dic");
    }
    const auto frontend = openjtalk_frontend_for_dictionary(dictionary_dir);
    const std::vector<std::string> tokens = japanese_tokens(lowercase_ascii);
    if (static_cast<int>(tokens.size()) != p.expected_vocab_size) {
        throw std::runtime_error("Japanese vocabulary does not match tokenizer profile");
    }
    std::unordered_map<std::string, int> token_to_id;
    for (size_t i = 0; i < tokens.size(); ++i) {
        token_to_id[tokens[i]] = (int)i;
    }
    const int pad_id = token_id_for_symbol(tokens, p.offset, "<pad>");

    tokenizer_result result;
    result.language = p.language;
    result.tokenizer_name = p.tokenizer_name;
    result.eos_id = p.eos_id;
    const std::string cased_text = lowercase_ascii ? ascii_lower(p.text) : ascii_upper(p.text);
    for (const std::string& sentence : tokenizer_input_units(p, cased_text)) {
        chunk ch;
        ch.text = sentence;
        std::vector<std::string> symbols;
        const std::string space = " ";
        for (const auto& symbol : japanese_g2p(*frontend, sentence, lowercase_ascii)) {
            if (symbol == space) {
                if (!symbols.empty() && symbols.back() != space) {
                    symbols.push_back(symbol);
                }
            } else if (token_to_id.count(symbol) != 0) {
                symbols.push_back(symbol);
            }
        }
        while (!symbols.empty() && symbols.back() == space) {
            symbols.pop_back();
        }
        symbols.insert(symbols.begin(), space);
        symbols.push_back(space);
        ch.tokens.reserve(symbols.size() + 1);
        for (const auto& symbol : symbols) {
            const auto it = token_to_id.find(symbol);
            if (it != token_to_id.end()) {
                ch.tokens.push_back(p.offset + it->second);
            }
        }
        ch.tokens.push_back(result.eos_id);
        pad_short_text_chunk_before_eos(ch, result.eos_id, pad_id);
        result.chunks.push_back(std::move(ch));
    }
    return result;
}
#endif
