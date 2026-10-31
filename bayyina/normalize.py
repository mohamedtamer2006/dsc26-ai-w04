"""Arabic-aware normalisation and tokenisation."""
from __future__ import annotations

import re

# Translation table for Arabic normalization:
# - Remove U+064B–U+0652 (tanween, harakat, sukun, shaddah) and U+0670 (superscript alef)
# - Remove tatweel U+0640
# - Map أ إ آ ٱ -> ا
# - Map ى -> ي, ة -> ه
# - Map Arabic-Indic digits ٠–٩ and Eastern Arabic-Indic digits ۰–۹ to ASCII 0–9
_NORM_TRANS = str.maketrans({
    "\u064b": None,
    "\u064c": None,
    "\u064d": None,
    "\u064e": None,
    "\u064f": None,
    "\u0650": None,
    "\u0651": None,
    "\u0652": None,
    "\u0670": None,
    "\u0640": None,
    "أ": "ا",
    "إ": "ا",
    "آ": "ا",
    "ٱ": "ا",
    "ى": "ي",
    "ة": "ه",
    "٠": "0",
    "١": "1",
    "٢": "2",
    "٣": "3",
    "٤": "4",
    "٥": "5",
    "٦": "6",
    "٧": "7",
    "٨": "8",
    "٩": "9",
    "۰": "0",
    "۱": "1",
    "۲": "2",
    "۳": "3",
    "۴": "4",
    "۵": "5",
    "۶": "6",
    "۷": "7",
    "۸": "8",
    "۹": "9",
})

_TOKEN_RE = re.compile(r"\w+")


def normalize_ar(text: str) -> str:
    """Remove U+064B-U+0652 and U+0670; remove tatweel U+0640; map أ إ آ ٱ -> ا, ى -> ي, ة -> ه;
    map ٠-٩ and ۰-۹ to 0-9; lowercase; collapse runs of whitespace to one space and strip.
    """
    translated = text.translate(_NORM_TRANS).lower()
    return " ".join(translated.split())


def tokenize(text: str) -> list[str]:
    """normalize_ar, then every match of the regex \\w+ (Unicode)."""
    norm = normalize_ar(text)
    return _TOKEN_RE.findall(norm)
