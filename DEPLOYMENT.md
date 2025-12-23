# DINOv2 異常検出システム - デプロイガイド

このプロジェクトはGitHub Pages（フロントエンド）とHugging Face Spaces（バックエンド）を使用した完全無料のデプロイ構成です。

## 📋 デプロイ手順

### 1. Hugging Face Spacesにバックエンドをデプロイ

```bash
# Hugging Face CLIをインストール（未インストールの場合）
pip install huggingface_hub

# デプロイスクリプトを実行
cd /home/localuser/Documents/astro_test
./deploy_to_hf.sh
```

スクリプトが自動的に以下を実行します：
- Hugging Faceにログイン
- Spaceを作成
- 必要なファイル（app.py、vit.py、requirements.txt）をアップロード
- デプロイ

デプロイ後、SpaceのURLをメモしてください（例: `https://YOUR_USERNAME-dinov2-anomaly-detection.hf.space`）

### 2. 環境変数を設定

```bash
# .envファイルを作成
cp .env.example .env

# .envファイルを編集してHugging Face SpaceのURLを設定
nano .env
```

`.env`ファイルの内容:
```env
PUBLIC_HF_SPACE_URL=https://YOUR_USERNAME-dinov2-anomaly-detection.hf.space
```

### 3. GitHub Pagesの設定

GitHubリポジトリで以下を設定：

1. リポジトリの**Settings** → **Pages**に移動
2. **Source**を「GitHub Actions」に設定
3. 保存

### 4. デプロイ

```bash
# 変更をコミット
git add .
git commit -m "Setup deployment for GitHub Pages and HF Spaces"

# GitHubにプッシュ
git push origin main
```

GitHub Actionsが自動的にビルド＆デプロイを開始します。

## 🌐 アクセスURL

- **フロントエンド（GitHub Pages）**: `https://Kotaro-Kuroda.github.io/astro_anomaly_detection/`
- **バックエンド（Hugging Face Spaces）**: デプロイ時に取得したURL

## 💰 コスト

- **GitHub Pages**: 完全無料
- **Hugging Face Spaces**: 完全無料（GPUも無料で利用可能）
- **合計月額コスト**: **¥0**

## 🔧 ローカル開発

```bash
# 依存関係をインストール
npm install

# 開発サーバーを起動
npm run dev

# ビルド
npm run build

# プレビュー
npm run preview
```

## 📝 注意事項

1. **Hugging Face Spacesの起動**:
   - 初回アクセス時やアイドル状態からの復帰時は数秒～数十秒かかります
   - 無料版では同時リクエスト数に制限があります

2. **GPU利用**:
   - Hugging Face Spacesの設定で**Hardware: T4 (small)**を選択してGPUを有効化できます
   - 無料プランでも利用可能です

3. **CORS設定**:
   - Gradio APIは自動的にCORSを許可するため、追加設定は不要です

## 🚀 更新方法

### バックエンドの更新

```bash
./deploy_to_hf.sh
```

### フロントエンドの更新

```bash
git add .
git commit -m "Update frontend"
git push origin main
```

GitHub Actionsが自動的にデプロイします。

## 🐛 トラブルシューティング

### Hugging Face Spacesに接続できない

1. SpaceのURLが正しく設定されているか確認
2. Spaceが正常に起動しているか確認（Hugging FaceのSpaceページで確認）
3. ブラウザのコンソールでエラーメッセージを確認

### GitHub Pagesにアクセスできない

1. GitHub Actionsのワークフローが正常に完了しているか確認
2. リポジトリのSettings → Pagesで設定を確認
3. 数分待ってから再度アクセス

## 📚 参考リンク

- [Astro Documentation](https://docs.astro.build/)
- [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces)
- [GitHub Pages](https://docs.github.com/pages)
