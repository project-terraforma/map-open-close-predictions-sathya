"""
Name and address normalization utilities.
"""

import re
import unicodedata


# Common street abbreviations
_STREET_ABBREVS = {
    r"\bst\b": "street",
    r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\brd\b": "road",
    r"\bct\b": "court",
    r"\bpl\b": "place",
    r"\bpkwy\b": "parkway",
    r"\bhwy\b": "highway",
    r"\bfl\b": "floor",
    r"\bste\b": "suite",
    r"\bapt\b": "apartment",
}


def normalize_name(name: str | None) -> str | None:
    if not name:
        return None
    text = name.strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[^\w\s]", "", text)  # strip punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def normalize_address(address: str | None) -> str | None:
    if not address:
        return None
    text = address.strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[^\w\s]", " ", text)
    for abbrev, full in _STREET_ABBREVS.items():
        text = re.sub(abbrev, full, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None
