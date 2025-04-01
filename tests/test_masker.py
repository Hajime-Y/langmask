import pytest
from langmask.masker import MultilingualTokenMasker


def test_masker_initialization():
    masker = MultilingualTokenMasker(allowed_languages=["JA", "EN"])
    assert masker.allowed_languages == ["JA", "EN"]
    assert masker.mask_strength == 0.9  # デフォルト値


def test_mask_strength_adjustment():
    masker = MultilingualTokenMasker(allowed_languages=["JA"])
    masker.set_mask_strength(0.5)
    assert masker.mask_strength == 0.5

    # 範囲外の値をテスト
    with pytest.raises(ValueError):
        masker.set_mask_strength(1.5)
    with pytest.raises(ValueError):
        masker.set_mask_strength(-0.1)


def test_language_switching():
    masker = MultilingualTokenMasker(allowed_languages=["JA"])
    masker.set_languages(["EN"])
    assert masker.allowed_languages == ["EN"]

    masker.set_languages(["JA", "EN"])
    assert set(masker.allowed_languages) == {"JA", "EN"}