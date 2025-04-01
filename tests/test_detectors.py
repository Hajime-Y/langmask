import pytest

from langmask.detectors.asian import is_chinese, is_japanese, is_korean
from langmask.detectors.european import is_english


def test_japanese_detection():
    assert is_japanese("こんにちは") is True
    assert is_japanese("Hello") is False
    assert is_japanese("こんにちは、World!") is True  # 混合テキスト


def test_english_detection():
    assert is_english("Hello World") is True
    assert is_english("こんにちは") is False
    assert is_english("Hello、世界！") is True  # 混合テキスト


def test_mixed_language_detection():
    text = "Hello こんにちは 你好"
    assert is_japanese(text) and is_english(text)  # 両方検出される
