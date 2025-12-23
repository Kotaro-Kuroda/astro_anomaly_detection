---
title: DINOv2 Anomaly Detection API
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# DINOv2 Anomaly Detection API

DINOv2を使用したone-shot異常検出システムのバックエンドAPI。

## 使い方

このSpaceはフロントエンド（GitHub Pages）からAPIとして呼び出されます。

### API Endpoint

- Gradio APIエンドポイント: `/api/predict`
- 入力: 参照画像、ターゲット画像
- 出力: ヒートマップ画像、統計情報、クリック情報、ターゲット画像

## フロントエンド

フロントエンドは以下で公開予定:
https://kotaro-kuroda.github.io/astro_anomaly_detection/

## 技術スタック

- DINOv2 ViT-S/14 with registers
- PyTorch
- Gradio
