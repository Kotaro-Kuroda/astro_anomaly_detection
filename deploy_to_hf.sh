#!/bin/bash

# Hugging Face Spacesにデプロイするスクリプト

echo "🚀 Hugging Face Spacesへのデプロイを開始します"

# 変数設定（あなたのHugging Faceユーザー名を入力してください）
read -p "Hugging Faceユーザー名を入力してください: " HF_USERNAME
SPACE_NAME="dinov2-anomaly-detection"

echo ""
echo "📦 Space名: $SPACE_NAME"
echo "👤 ユーザー名: $HF_USERNAME"
echo ""

# Hugging Face CLIがインストールされているか確認
if ! command -v huggingface-cli &> /dev/null; then
    echo "❌ Hugging Face CLIがインストールされていません"
    echo "インストールコマンド: pip install huggingface_hub"
    exit 1
fi

# ログイン確認
echo "🔐 Hugging Faceにログインしています..."
huggingface-cli whoami &> /dev/null
if [ $? -ne 0 ]; then
    echo "ログインが必要です"
    huggingface-cli login
fi

# Spaceを作成（既に存在する場合はスキップ）
echo "📁 Spaceを作成しています..."
huggingface-cli repo create $SPACE_NAME --type space --space_sdk gradio 2>/dev/null || echo "Space already exists, continuing..."

# 一時ディレクトリを作成
TEMP_DIR="/tmp/hf_space_deploy"
rm -rf $TEMP_DIR
mkdir -p $TEMP_DIR

echo "📄 ファイルをコピーしています..."
cp app.py $TEMP_DIR/
cp vit.py $TEMP_DIR/
cp requirements.txt $TEMP_DIR/
cp HF_README.md $TEMP_DIR/README.md

# Spaceをクローン
cd $TEMP_DIR
git clone https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME
cd $SPACE_NAME

# ファイルをコピー
cp ../app.py .
cp ../vit.py .
cp ../requirements.txt .
cp ../README.md .

# Gitにコミット＆プッシュ
git add .
git commit -m "Deploy anomaly detection backend" || echo "No changes to commit"
git push

echo ""
echo "✅ デプロイ完了！"
echo "🌐 SpaceのURL: https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME"
echo ""
echo "⏳ ビルドには数分かかります。上記のURLで進捗を確認してください。"
echo ""
echo "次のステップ:"
echo "1. SpaceのURLをコピーしてください"
echo "2. Astroアプリの設定でこのURLを使用します"
