#!/bin/bash

# エラー発生時にスクリプトを停止
set -e

# 現在のディレクトリをプロジェクトルートに設定
cd "$(dirname "$0")/.."

# コミットメッセージを取得（デフォルト: "update: format code"）
commit_message="${1:-"update: format code"}"

# ステージングされた変更があるかチェック
if ! git diff --cached --quiet; then
    echo "Error: コミットのためにステージングされた変更があります。"
    echo "先にコミットするか、変更を取り消してください。"
    exit 1
fi

# blackとisortを実行
echo "🎨 コードのフォーマットを実行中..."
uv run black .
uv run isort .

# 変更があるかチェック
if ! git diff --quiet; then
    echo "✨ 変更をコミットします..."
    git add .
    git commit -m "$commit_message"
    echo "🚀 変更をプッシュします..."
    git push origin main
    echo "✅ 完了しました！"
else
    echo "💡 フォーマットによる変更はありませんでした。"
fi 