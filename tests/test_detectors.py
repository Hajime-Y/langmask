import pytest

from langmask.detectors.asian import is_chinese_char, is_japanese_char, is_korean_char
from langmask.detectors.european import is_english_char


def test_japanese_detection() -> None:
    assert is_japanese_char("こんにちは") is True
    assert is_japanese_char("Hello") is False
    assert is_japanese_char("こんにちは、World!") is True  # 混合テキスト


def test_english_detection() -> None:
    assert all(is_english_char(c) for c in "Hello World")
    assert not any(is_english_char(c) for c in "こんにちは")
    assert is_english_char("Hello、世界！") is True  # 混合テキスト


def test_mixed_language_detection() -> None:
    text = "Hello こんにちは 你好"
    assert is_japanese_char(text) and is_english_char(text)  # 両方検出される
