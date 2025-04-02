import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from langmask import MultilingualLanguageModel


@pytest.fixture
def base_model() -> AutoModelForCausalLM:
    return AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        trust_remote_code=True,
    )


@pytest.fixture
def tokenizer() -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        trust_remote_code=True,
    )


@pytest.fixture
def model(
    base_model: AutoModelForCausalLM, tokenizer: AutoTokenizer
) -> MultilingualLanguageModel:
    return MultilingualLanguageModel(
        model=base_model,
        tokenizer=tokenizer,
        allowed_languages=["JA"],
    )


def test_japanese_generation(
    model: MultilingualLanguageModel, tokenizer: AutoTokenizer
) -> None:
    # 日本語のプロンプトを使用
    inputs = tokenizer(["こんにちは、私の名前は田中です。"], return_tensors="pt").to(
        model.device
    )
    outputs = model.generate(
        **inputs,
        # Qwenモデル用の生成設定
        num_beams=4,
        repetition_penalty=1.1,
        top_k=40,
        length_penalty=1.0,
        return_dict_in_generate=True,
        output_scores=True,
    )

    # 生成部分のみを取得
    if isinstance(outputs, dict):
        generated_ids = outputs["sequences"][:, inputs.input_ids.shape[1] :]
    else:
        generated_ids = outputs[:, inputs.input_ids.shape[1] :]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # 日本語文字（ひらがな、カタカナ、漢字）が含まれていることを確認
    has_japanese = any(
        ("\u3040" <= c <= "\u309f")  # ひらがな
        or ("\u30a0" <= c <= "\u30ff")  # カタカナ
        or ("\u4e00" <= c <= "\u9fff")  # 漢字
        for c in response
    )
    assert has_japanese, f"Response does not contain Japanese characters: {response}"


def test_language_switching(
    model: MultilingualLanguageModel, tokenizer: AutoTokenizer
) -> None:
    # 日本語生成
    ja_inputs = tokenizer(["こんにちは"], return_tensors="pt").to(model.device)
    ja_outputs = model.generate(**ja_inputs)
    if isinstance(ja_outputs, dict):
        ja_response = tokenizer.batch_decode(
            ja_outputs["sequences"][:, ja_inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        )[0]
    else:
        ja_response = tokenizer.batch_decode(
            ja_outputs[:, ja_inputs.input_ids.shape[1] :], skip_special_tokens=True
        )[0]

    # 英語に切り替え
    model.set_languages(["EN"])
    en_inputs = tokenizer(["Hello"], return_tensors="pt").to(model.device)
    en_outputs = model.generate(**en_inputs)
    if isinstance(en_outputs, dict):
        en_response = tokenizer.batch_decode(
            en_outputs["sequences"][:, en_inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        )[0]
    else:
        en_response = tokenizer.batch_decode(
            en_outputs[:, en_inputs.input_ids.shape[1] :], skip_special_tokens=True
        )[0]

    # 英語のみが含まれていることを確認（空白とASCII文字のみ）
    assert all(ord(c) < 128 for c in en_response if not c.isspace())


def test_mask_strength_effect(
    model: MultilingualLanguageModel, tokenizer: AutoTokenizer
) -> None:
    prompt = "AIについて説明してください"
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    # 強いマスキング
    model.set_mask_strength(1.0)
    strict_outputs = model.generate(**inputs)
    if isinstance(strict_outputs, dict):
        strict_response = tokenizer.batch_decode(
            strict_outputs["sequences"][:, inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        )[0]
    else:
        strict_response = tokenizer.batch_decode(
            strict_outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )[0]

    # 弱いマスキング
    model.set_mask_strength(0.5)
    soft_outputs = model.generate(**inputs)
    if isinstance(soft_outputs, dict):
        soft_response = tokenizer.batch_decode(
            soft_outputs["sequences"][:, inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        )[0]
    else:
        soft_response = tokenizer.batch_decode(
            soft_outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )[0]

    # 応答が異なることを確認
    assert strict_response != soft_response
