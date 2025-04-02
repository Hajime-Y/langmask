# LangMask

> **注意**: このプロジェクトは失敗プロジェクトとして終了しました。
> 
> 失敗の主な理由：
> 1. テストを重視しすぎたことでコードが必要以上に複雑化してしまいました。
> 2. `model`と`masker`の設計が適切ではありませんでした。HuggingFaceの`LogitsProcessor`を使用することで、より簡潔に同様の機能を実装できることが判明しました。
>
> このリポジトリは今後アーカイブされます。

[![PyPI version](https://badge.fury.io/py/langmask.svg)](https://badge.fury.io/py/langmask)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Documentation Status](https://readthedocs.org/projects/langmask/badge/?version=latest)](https://langmask.readthedocs.io/en/latest/?badge=latest)

LangMask（ラングマスク）は大規模言語モデル（LLM）の出力言語を制御するためのPythonライブラリです。QwenやLlamaなどの多言語モデルでも、特定の言語だけで応答するよう制御できます。

## 主な機能

- 🌍 複数言語のサポート（日本語、英語、中国語、韓国語、フランス語など）
- 🔄 動的な言語切り替え（実行時に出力言語を変更可能）
- 🎛️ 調整可能なマスキング強度（ソフトからハードまで）
- 📊 詳細なトークン分類と可視化
- 🚀 Hugging Faceモデルとの簡単な統合
- 🔌 標準的なHugging Faceインターフェースとの完全な互換性

## インストール

```bash
# uvのインストール（初回のみ）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 方法1: PyPIからの安定版インストール
uv add langmask

# 方法2: GitHubからの最新版インストール
uv add git+https://github.com/Hajime-Y/langmask.git

# 方法3: 開発用インストール（コントリビューター向け）
git clone https://github.com/Hajime-Y/langmask.git
cd langmask
uv sync --extra dev
```

## 使用例

### 基本的な使い方

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from langmask import MultilingualLanguageModel

# モデルとトークナイザーの準備
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat")

# 言語制御モデルの初期化（日本語のみを許可）
model = MultilingualLanguageModel(
    model=base_model,
    tokenizer=tokenizer,
    allowed_languages=["JA"],
    mask_strength=0.9
)

# テキスト生成
inputs = tokenizer(["AIについて説明してください"], return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200)

# 入力を除いた生成部分のみを取得
generated_ids = [
    output_ids[len(input_ids):] 
    for input_ids, output_ids in zip(inputs.input_ids, outputs)
]

# デコード
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```

### 複数言語の同時許可

```python
# 日本語と英語の両方を許可
model = MultilingualLanguageModel(
    model=base_model,
    tokenizer=tokenizer,
    allowed_languages=["JA", "EN"]
)

# 英語混じりの日本語プロンプトでも、日本語と英語の応答が可能
inputs = tokenizer(["AIの未来についてexplainしてください。"], return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200)
generated_ids = [
    output_ids[len(input_ids):] 
    for input_ids, output_ids in zip(inputs.input_ids, outputs)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```

### 動的な言語切り替え

```python
# モデルの初期化（最初は日本語のみ）
model = MultilingualLanguageModel(
    model=base_model,
    tokenizer=tokenizer,
    allowed_languages=["JA"]
)

# 日本語での応答
ja_inputs = tokenizer(["AIについて説明してください"], return_tensors="pt").to(model.device)
ja_outputs = model.generate(**ja_inputs, max_new_tokens=200)

# 英語に切り替え
model.set_languages(["EN"])
en_inputs = tokenizer(["Explain about AI"], return_tensors="pt").to(model.device)
en_outputs = model.generate(**en_inputs, max_new_tokens=200)

# 日本語と英語の両方を許可
model.set_languages(["JA", "EN"])
mixed_inputs = tokenizer(["AIについてexplainしてください"], return_tensors="pt").to(model.device)
mixed_outputs = model.generate(**mixed_inputs, max_new_tokens=200)
```

### マスキング強度の調整

```python
# ソフトマスキング（他の言語も少し混ざる可能性あり）
model.set_mask_strength(0.7)

# 強めのソフトマスキング（ほとんど指定言語のみ）
model.set_mask_strength(0.95)

# ハードマスキング（指定言語のみを強制）
model.set_mask_strength(1.0)
```

### トークン分類の確認

```python
# テキスト内のトークンを言語ごとに分類
stats = model.debug_token_classification(
    "こんにちは、世界！Hello, World! 你好，世界！",
    tokenizer=tokenizer,
    verbose=True  # 詳細出力
)

# 結果（例）:
# JA (Japanese) トークン: 4 (40.0%)
# EN (English) トークン: 4 (40.0%)
# ZH (Chinese) トークン: 2 (20.0%)
```

## 対応言語

現在、以下の言語に対応しています：

| 言語コード | 言語名 |
|------------|--------|
| JA | 日本語 (Japanese) |
| EN | 英語 (English) |
| ZH | 中国語 (Chinese) |
| KO | 韓国語 (Korean) |
| FR | フランス語 (French) |
| DE | ドイツ語 (German) |
| ES | スペイン語 (Spanish) |
| IT | イタリア語 (Italian) |
| RU | ロシア語 (Russian) |
| PT | ポルトガル語 (Portuguese) |

## 仕組み

LangMaskは、LLMの生成過程でlogitsと呼ばれる次トークンの確率分布を操作します。

1. テキスト生成時、モデルは各トークンの選択確率を計算します（logits）
2. LangMaskは指定された言語以外のトークンにペナルティを適用します
3. ペナルティの強さはmask_strengthパラメータで調整できます
4. 言語判定は事前に解析されたトークン分類に基づいて行われます

この方法により、モデルの内部を修正することなく、出力言語を制御できます。

## 貢献方法

バグ報告、機能リクエスト、プルリクエストを歓迎します！

1. このリポジトリをフォークします
2. 新しいブランチを作成します (`git checkout -b feature/amazing-feature`)
3. 開発環境をセットアップします:
   ```bash
   git clone https://github.com/Hajime-Y/langmask.git
   cd langmask
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   curl -LsSf https://astral.sh/uv/install.sh | sh  # uvのインストール
   uv sync --extra dev  # 開発用依存関係のインストール
   ```
4. 変更をコミットします (`git commit -m 'Add some amazing feature'`)
5. ブランチをプッシュします (`git push origin feature/amazing-feature`)
6. プルリクエストを作成します

## ライセンス

Apache License 2.0の下で配布されています。詳細は[LICENSE](LICENSE)ファイルをご覧ください。

## 引用

このライブラリを研究で使用する場合は、以下の形式で引用してください：

```
@software{langmask2025,
  author = {Hajime Yagihara},
  title = {LangMask: A Library for Controlling Output Languages in Large Language Models},
  year = {2025},
  url = {https://github.com/Hajime-Y/langmask}
}
```

## 今後の予定

- より多くの言語のサポート
- より高度な言語判定アルゴリズム
- 特定の専門分野向けの単語リスト
- WebUIデモの作成
- トークン可視化ツールの強化

---

**免責事項**: このライブラリは実験的なツールであり、すべてのモデルやトークナイザーで完璧に動作することを保証するものではありません。特に専門用語や固有名詞などでは想定通りの結果にならない場合があります。