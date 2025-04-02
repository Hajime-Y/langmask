import unicodedata

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from langmask.masker import MultilingualTokenMasker


@pytest.fixture
def tokenizer() -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True
    )


@pytest.fixture
def model() -> AutoModelForCausalLM:
    return AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True
    )


@pytest.fixture
def masker(
    model: AutoModelForCausalLM, tokenizer: AutoTokenizer
) -> MultilingualTokenMasker:
    return MultilingualTokenMasker(
        tokenizer=tokenizer, model=model, allowed_languages=["JA", "EN"]
    )


def test_masker_initialization(masker: MultilingualTokenMasker) -> None:
    assert masker.allowed_languages == ["JA", "EN"]
    assert masker.mask_strength == 0.8  # デフォルト値


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


def test_logits_processor(
    masker: MultilingualTokenMasker, tokenizer: AutoTokenizer
) -> None:
    # テスト用のlogitsを作成
    vocab_size = len(tokenizer)
    batch_size = 1
    logits = torch.randn(batch_size, vocab_size, device=masker.device)

    # logits_processorを取得
    processor = masker.logits_processor()

    # プロセッサを適用
    modified_logits = processor(None, logits)

    # 形状が保持されていることを確認
    assert modified_logits.shape == logits.shape

    # マスキングが適用されていることを確認（値が変更されている）
    assert not torch.allclose(modified_logits, logits)


def test_token_classification(masker: MultilingualTokenMasker) -> None:
    # 日本語と英語のテキストで分類をテスト
    text = "Hello こんにちは"

    # トークナイザーの出力を確認
    tokens = masker.tokenizer.tokenize(text)
    token_ids = masker.tokenizer.encode(text)
    print("\nTokenizer Debug:")
    print(f"Text: {text}")
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}")

    stats = masker.debug_token_classification(text, verbose=True)

    # 分類結果の詳細を出力
    print("\nToken Classification Results:")
    print(f"Text: {text}")
    print(f"Statistics: {stats}")

    # 日本語と英語のトークンが検出されていることを確認
    assert stats["JA"] > 0, f"日本語トークンが検出されませんでした。統計: {stats}"
    assert stats["EN"] > 0, f"英語トークンが検出されませんでした。統計: {stats}"


def test_token_unicode_analysis(masker: MultilingualTokenMasker) -> None:
    """個々のトークンのUnicode情報を分析するテスト"""
    text = "Hello こんにちは"
    tokens = masker.tokenizer.tokenize(text)

    print("\nToken Unicode Analysis:")
    print(f"Original text: {text}")
    print(f"Tokenized: {tokens}")

    for token in tokens:
        # トークンを単独でデコード
        token_id = masker.tokenizer.convert_tokens_to_ids([token])[0]
        decoded = masker.tokenizer.decode([token_id])

        print(f"\nToken: {token}")
        print(f"Token ID: {token_id}")
        print(f"Decoded: {decoded}")
        print("Unicode points:")
        for char in decoded:
            print(
                f"  {char!r}: U+{ord(char):04X} - {unicodedata.name(char, 'Unknown')}"
            )

    # 全体をデコード
    all_ids = masker.tokenizer.convert_tokens_to_ids(tokens)
    full_decoded = masker.tokenizer.decode(all_ids)
    print(f"\nFull decoded text: {full_decoded}")
    print("Full text Unicode points:")
    for char in full_decoded:
        print(f"  {char!r}: U+{ord(char):04X} - {unicodedata.name(char, 'Unknown')}")
