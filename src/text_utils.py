"""Tiện ích tiền xử lý văn bản tiếng Việt cho bài toán phân loại cảm xúc bình luận."""
import re
import sys
import unicodedata
from pathlib import Path

from pyvi import ViTokenizer

ROOT = Path(__file__).resolve().parents[1]
# --vit5: chạy pipeline trên dữ liệu đã sửa chính tả bằng ViT5 (cả train/validation/test), kết quả ghi vào outputs/vit5/
VIT5 = "--vit5" in sys.argv
OUT_ORIG = ROOT / "outputs"
OUT_VIT5 = OUT_ORIG / "vit5"
OUT = OUT_VIT5 if VIT5 else OUT_ORIG
STOPWORD_FILE = ROOT / "resources" / "vietnamese_stopwords.txt"

# Chuẩn hóa kiểu bỏ dấu cũ -> mới (hoà -> hòa, thuý -> thúy, ...)
_TONE_MAP = {
    "òa": "oà", "óa": "oá", "ỏa": "oả", "õa": "oã", "ọa": "oạ",
    "òe": "oè", "óe": "oé", "ỏe": "oẻ", "õe": "oẽ", "ọe": "oẹ",
    "ùy": "uỳ", "úy": "uý", "ủy": "uỷ", "ũy": "uỹ", "ụy": "uỵ",
}
_TONE_MAP = {old: new for new, old in _TONE_MAP.items()}  # đưa về kiểu mới: oà -> òa
_TONE_RE = re.compile("|".join(map(re.escape, _TONE_MAP)))

# Từ viết tắt / teencode phổ biến trong bình luận thương mại điện tử
TEENCODE = {
    "ko": "không", "k": "không", "kg": "không", "kh": "không", "khg": "không",
    "hok": "không", "hông": "không", "hem": "không", "khong": "không", "kô": "không",
    "đc": "được", "dc": "được", "đk": "được", "dk": "được", "duoc": "được", "đượt": "được",
    "vs": "với", "z": "vậy", "v": "vậy", "zậy": "vậy", "dzậy": "vậy",
    "j": "gì", "gi": "gì", "r": "rồi", "rùi": "rồi", "ròi": "rồi",
    "cx": "cũng", "cũg": "cũng", "củng": "cũng", "cug": "cũng",
    "wá": "quá", "qá": "quá", "qua'": "quá", "lém": "lắm", "nhìu": "nhiều",
    "ntn": "như thế nào", "sp": "sản phẩm", "nv": "nhân viên", "nvien": "nhân viên",
    "mn": "mọi người", "mng": "mọi người", "bt": "bình thường", "bth": "bình thường",
    "dt": "điện thoại", "đt": "điện thoại", "dmx": "điện máy xanh", "đmx": "điện máy xanh",
    "oke": "ok", "okie": "ok", "oki": "ok", "okê": "ok", "okay": "ok", "okela": "ok",
    "tks": "cảm ơn", "thanks": "cảm ơn", "thank": "cảm ơn", "cám": "cảm",
    "mik": "mình", "mk": "mình", "mh": "mình",
    "bik": "biết", "bít": "biết", "trc": "trước", "sd": "sử dụng",
    "tot": "tốt", "tôt": "tốt", "đep": "đẹp", "dep": "đẹp", "lòg": "lòng",
    "bh": "bảo hành", "ship": "giao hàng", "shipper": "người giao hàng",
}

POS_EMOJI = set("😀😁😂🤣😃😄😅😆😉😊😋😎😍😘🥰😗😙😚🙂🤗🤩👍👌💯❤️💕💖💗💓💞🌹🎉✌️🥳😻")
NEG_EMOJI = set("😞😔😟😕🙁☹️😣😖😫😩😢😭😤😠😡🤬👎💔🤮🤢😒🙄😑")

_URL_RE = re.compile(r"<url>|https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_NUM_RE = re.compile(r"<number>|<phone>|\d+([.,]\d+)*[a-zđ]*")  # gồm đơn vị: 500k, 188lit, 2tr
_REPEAT_RE = re.compile(r"(\w)\1{2,}")       # tốtttt -> tốt
_PUNCT_SPACE_RE = re.compile(r"([.,!?;:()\[\]\"/…-])")
_NON_WORD_RE = re.compile(r"[^\w\s_]")
_SPACE_RE = re.compile(r"\s+")


def load_stopwords(path=STOPWORD_FILE):
    words = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            words.add(line.replace(" ", "_"))
    return words


def normalize(text: str) -> str:
    """Chuẩn hóa: Unicode NFC, kiểu bỏ dấu, chữ thường, URL/số, emoji, teencode, ký tự lặp."""
    text = unicodedata.normalize("NFC", str(text)).lower()
    text = _TONE_RE.sub(lambda m: _TONE_MAP[m.group()], text)
    text = _URL_RE.sub(" url ", text)
    text = _EMAIL_RE.sub(" email ", text)
    text = _NUM_RE.sub(" num ", text)
    text = "".join(
        " emopos " if ch in POS_EMOJI else " emoneg " if ch in NEG_EMOJI else ch
        for ch in text
    )
    text = _REPEAT_RE.sub(r"\1", text)
    text = _PUNCT_SPACE_RE.sub(r" \1 ", text)  # tách dấu câu dính liền: "nhanh.nhiệt" -> "nhanh . nhiệt"
    tokens = [TEENCODE.get(tok, tok) for tok in text.split()]
    return _SPACE_RE.sub(" ", " ".join(tokens)).strip()


def segment(text: str) -> str:
    """Tách từ tiếng Việt bằng pyvi: 'máy giặt chạy êm' -> 'máy_giặt chạy êm'."""
    return ViTokenizer.tokenize(text) if text else ""


def clean_tokens(segmented: str, stopwords=None):
    """Bỏ dấu câu/ký tự đặc biệt và (tùy chọn) stopword; trả về danh sách token."""
    text = _NON_WORD_RE.sub(" ", segmented)
    toks = [t.strip("_") for t in text.split()]
    toks = [t for t in toks if t]
    if stopwords:
        toks = [t for t in toks if t not in stopwords]
    return toks
