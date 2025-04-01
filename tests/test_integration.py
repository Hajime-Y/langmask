import pytest

from langmask import MultilingualLanguageModel


@pytest.fixture
def model() -> MultilingualLanguageModel:
    return MultilingualLanguageModel(
        model_name="Qwen/Qwen-7B-Chat",
        allowed_languages=["JA"],
    )


def test_japanese_generation(model: MultilingualLanguageModel) -> None:
    response = model.generate("AIについて説明してください")
    # 日本語文字が含まれていることを確認
    assert any("\u3040" <= c <= "\u309f" or "\u4e00" <= c <= "\u9fff" for c in response)


def test_language_switching(model: MultilingualLanguageModel) -> None:
    # 日本語生成
    ja_response = model.generate("こんにちは")

    # 英語に切り替え
    model.set_languages(["EN"])
    en_response = model.generate("Hello")

    # 英語のみが含まれていることを確認
    assert all(ord(c) < 128 for c in en_response if not c.isspace())


def test_mask_strength_effect(model: MultilingualLanguageModel) -> None:
    # 強いマスキング
    model.set_mask_strength(1.0)
    strict_response = model.generate("AIについて説明してください")

    # 弱いマスキング
    model.set_mask_strength(0.5)
    soft_response = model.generate("AIについて説明してください")

    # 応答が異なることを確認
    assert strict_response != soft_response
