"""
Language detection functions for Asian languages.
"""

import unicodedata


def is_japanese_char(char: str) -> bool:
    """Check if a character is Japanese (Hiragana, Katakana, CJK Unified Ideographs)."""
    # CJK Unified Ideographs (U+4E00 to U+9FFF)
    # Hiragana (U+3040 to U+309F)
    # Katakana (U+30A0 to U+30FF)
    # CJK Symbols and Punctuation (U+3000 to U+303F)
    # Halfwidth and Fullwidth Forms (U+FF00 to U+FFEF) - includes fullwidth chars
    if any(
        [
            "\u4e00" <= char <= "\u9fff",  # CJK Ideographs
            "\u3040" <= char <= "\u309f",  # Hiragana
            "\u30a0" <= char <= "\u30ff",  # Katakana
            "\u3000" <= char <= "\u303f",  # CJK Symbols/Punctuation
            "\uff00" <= char <= "\uffef",  # Halfwidth/Fullwidth Forms
        ]
    ):
        return True
    # Check via unicodedata name as a fallback
    name = unicodedata.name(char, "").upper()
    return "CJK" in name or "HIRAGANA" in name or "KATAKANA" in name


def is_chinese_char(char: str) -> bool:
    """Check if a character is Chinese (CJK Unified Ideographs)."""
    # Primarily relies on the CJK Unified Ideographs block
    if "\u4e00" <= char <= "\u9fff":
        return True
    # Include some common Chinese punctuation in Fullwidth forms
    if "\uff00" <= char <= "\uffef":
        return True
    # Check via unicodedata name
    name = unicodedata.name(char, "").upper()
    return "CJK" in name


def is_korean_char(char: str) -> bool:
    """Check if a character is Korean (Hangul Syllables, Jamo)."""
    # Hangul Syllables (U+AC00 to U+D7AF)
    # Hangul Jamo (U+1100 to U+11FF)
    # Hangul Compatibility Jamo (U+3130 to U+318F)
    # Hangul Jamo Extended-A (U+A960 to U+A97F)
    # Hangul Jamo Extended-B (U+D7B0 to U+D7FF)
    if any(
        [
            "\uac00" <= char <= "\ud7af",  # Hangul Syllables
            "\u1100" <= char <= "\u11ff",  # Hangul Jamo
            "\u3130" <= char <= "\u318f",  # Hangul Compatibility Jamo
            "\ua960" <= char <= "\ua97f",  # Hangul Jamo Extended-A
            "\ud7b0" <= char <= "\ud7ff",  # Hangul Jamo Extended-B
        ]
    ):
        return True
    # Check via unicodedata name
    name = unicodedata.name(char, "").upper()
    return "HANGUL" in name
