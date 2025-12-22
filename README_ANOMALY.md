# DINOv2 異常検出 Webアプリケーション

このプロジェクトは、Meta AIのDINOv2モデルを使用したone-shot異常検出システムのWebインターフェースです。

## 概要

- **フロントエンド**: Astro + React + Tailwind CSS
- **バックエンド**: FastAPI + PyTorch
- **モデル**: DINOv2 (dinov2_vits14_reg)

## セットアップ手順

### 1. 依存関係のインストール

#### Node.js依存関係
```bash
npm install
```

#### Python依存関係
```bash
pip install -e .
```

または

```bash
pip install fastapi uvicorn python-multipart pillow torch torchvision matplotlib
```

### 2. APIサーバーの起動

```bash
python api_server.py
```

または

```bash
npm run api
```

APIサーバーは `http://localhost:8000` で起動します。

### 3. Astro開発サーバーの起動

別のターミナルで:

```bash
npm run dev
```

フロントエンドは `http://localhost:4321` で起動します。

## 使い方

1. ブラウザで `http://localhost:4321` にアクセス
2. 参照画像（正常な画像）をアップロード
3. ターゲット画像（検査したい画像）をアップロード
4. 「異常検出を実行」ボタンをクリック
5. 異常マップと統計情報が表示されます

## APIエンドポイント

### POST `/detect`

異常検出を実行します。

**リクエスト**:
- `reference_image`: 参照画像ファイル (multipart/form-data)
- `target_image`: ターゲット画像ファイル (multipart/form-data)

**レスポンス**:
```json
{
  "success": true,
  "heatmap": "data:image/png;base64,...",
  "statistics": {
    "mean": 0.1234,
    "max": 0.5678,
    "min": 0.0123,
    "std": 0.0987
  }
}
```

### GET `/health`

APIサーバーのヘルスチェック。

**レスポンス**:
```json
{
  "status": "ok",
  "device": "cuda"
}
```

## プロジェクト構造

```
.
├── api_server.py           # FastAPI バックエンドサーバー
├── demo.py                 # オリジナルのデモスクリプト
├── vit.py                  # DINOv2モデルクラス
├── src/
│   ├── components/
│   │   └── AnomalyDetector.tsx  # 異常検出UIコンポーネント
│   ├── pages/
│   │   └── index.astro     # メインページ
│   └── layouts/
│       └── Layout.astro    # レイアウト
├── package.json            # Node.js依存関係
└── pyproject.toml          # Python依存関係
```

## 技術詳細

### DINOv2モデル

このシステムはFacebookResearchのDINOv2モデル（`dinov2_vits14_reg`）を使用します。モデルは初回実行時に自動的にダウンロードされます。

### 異常検出アルゴリズム

1. 参照画像とターゲット画像を518x518にリサイズ
2. 各画像からDINOv2で特徴量を抽出
3. コサイン距離を計算（Top-20の平均）
4. 異常スコアマップを生成
5. ヒートマップとして可視化

## 注意事項

- 初回実行時はDINOv2モデルのダウンロードに時間がかかります
- GPUが利用可能な場合は自動的に使用されます
- 大きな画像は自動的にリサイズされます

## ライセンス

MIT
