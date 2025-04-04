"""
Language detectors package.
"""

from .asian import is_chinese_char, is_japanese_char, is_korean_char
from .european import (
    is_english_char,
    is_french_char,
    is_german_char,
    is_italian_char,
    is_portuguese_char,
    is_russian_char,
    is_spanish_char,
)
from .utils import get_language_char_detector, is_language_token

__all__ = [
    "is_language_token",
    "get_language_char_detector",
    "is_japanese_char",
    "is_chinese_char",
    "is_korean_char",
    "is_english_char",
    "is_french_char",
    "is_german_char",
    "is_spanish_char",
    "is_italian_char",
    "is_russian_char",
    "is_portuguese_char",
]
