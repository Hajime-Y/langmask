"""
Language detection functions for Asian languages.
"""

import unicodedata
from typing import Union


def is_japanese_char(text: Union[str, chr]) -> bool:
    """Check if a text contains Japanese characters."""
    if len(text) == 0:
        return False

    def _is_japanese_char(char: str) -> bool:
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
        try:
            name = unicodedata.name(char, "").upper()
            return "CJK" in name or "HIRAGANA" in name or "KATAKANA" in name
        except TypeError:
            return False

    # 1文字の場合は直接チェック
    if len(text) == 1:
        return _is_japanese_char(text)
    
    # 複数文字の場合は1文字ずつチェック
    return any(_is_japanese_char(char) for char in text)


def is_chinese_char(text: Union[str, chr]) -> bool:
    """Check if a text contains Chinese characters."""
    if len(text) == 0:
        return False

    def _is_chinese_char(char: str) -> bool:
        if "\u4e00" <= char <= "\u9fff":  # CJK Unified Ideographs
            return True
        if "\uff00" <= char <= "\uffef":  # Fullwidth forms
            return True
        try:
            name = unicodedata.name(char, "").upper()
            return "CJK" in name
        except TypeError:
            return False

    # 1文字の場合は直接チェック
    if len(text) == 1:
        return _is_chinese_char(text)
    
    # 複数文字の場合は1文字ずつチェック
    return any(_is_chinese_char(char) for char in text)


def is_korean_char(text: Union[str, chr]) -> bool:
    """Check if a text contains Korean characters."""
    if len(text) == 0:
        return False

    def _is_korean_char(char: str) -> bool:
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
        try:
            name = unicodedata.name(char, "").upper()
            return "HANGUL" in name
        except TypeError:
            return False

    # 1文字の場合は直接チェック
    if len(text) == 1:
        return _is_korean_char(text)
    
    # 複数文字の場合は1文字ずつチェック
    return any(_is_korean_char(char) for char in text)
