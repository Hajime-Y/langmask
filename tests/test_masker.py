import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from langmask.masker import MultilingualTokenMasker


@pytest.fixture
def masker() -> MultilingualTokenMasker:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat")
    return MultilingualTokenMasker(tokenizer=tokenizer, model=model, allowed_languages=["JA", "EN"])


def test_masker_initialization(masker: MultilingualTokenMasker) -> None:
    assert masker.allowed_languages == ["JA", "EN"]
    assert masker.mask_strength == 0.9  # デフォルト値


def test_mask_strength_adjustment(masker: MultilingualTokenMasker) -> None:
    masker.set_mask_strength(0.5)
    assert masker.mask_strength == 0.5

    # 範囲外の値をテスト
    with pytest.raises(ValueError):
        masker.set_mask_strength(1.5)
    with pytest.raises(ValueError):
        masker.set_mask_strength(-0.1)


def test_language_switching(masker: MultilingualTokenMasker) -> None:
    masker.set_allowed_languages(["EN"])
    assert masker.allowed_languages == ["EN"]

    masker.set_allowed_languages(["JA", "EN"])
    assert set(masker.allowed_languages) == {"JA", "EN"}
