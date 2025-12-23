import io

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms

from vit import ViT

matplotlib.use('Agg')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# モデルを事前にロード
print(f"Using device: {device}")
model = ViT('dinov2_vits14_reg')
model = model.to(device)
model.eval()

# グローバル変数で異常マップを保存
current_anomaly_map = None
current_target_image = None
current_heatmap_image = None


def cos_distance(feat, ref_feat):
    feat = torch.nn.functional.normalize(feat, dim=-1)
    ref_feat = torch.nn.functional.normalize(ref_feat, dim=-1)
    cos = torch.einsum('bnc, bmc -> bnm', feat, ref_feat)
    topk, _ = torch.topk(cos, k=20, dim=-1)
    ano = 1 - topk.mean(dim=-1)
    return ano


def preprocess_image(image):
    """PIL Imageを前処理"""
    transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)


def create_heatmap(anomaly_map):
    """異常マップをヒートマップ画像に変換"""
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(anomaly_map, cmap='jet')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis('off')
    ax.set_title('Anomaly Heatmap', fontsize=14, pad=20)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)

    return img


def inference(ref_image, target_image):
    """異常検出のメイン関数"""
    global current_anomaly_map, current_target_image, current_heatmap_image

    if ref_image is None or target_image is None:
        return None, "参照画像とターゲット画像の両方を選択してください。", "", target_image

    try:
        # PIL Imageに変換
        ref_img = ref_image.convert('RGB')
        target_img = target_image.convert('RGB')
        original_size = target_img.size

        # ターゲット画像を保存
        current_target_image = target_img.copy()

        # 画像を前処理
        ref_tensor = preprocess_image(ref_img)
        target_tensor = preprocess_image(target_img)

        # 推論実行
        with torch.inference_mode():
            target_feat = model(target_tensor)
            ref_feat = model(ref_tensor)

        # コサイン距離を計算
        cosine_distance = cos_distance(target_feat, ref_feat)
        B, L = cosine_distance.shape
        H = W = int(L ** 0.5)
        anomaly_map = cosine_distance.reshape(B, H, W)
        anomaly_map = torch.nn.functional.interpolate(
            anomaly_map.unsqueeze(1),
            size=(original_size[1], original_size[0]),
            mode='bilinear',
        )
        anomaly_map = anomaly_map.squeeze().cpu().numpy()

        # グローバル変数に保存
        current_anomaly_map = anomaly_map

        # ヒートマップを生成
        heatmap_img = create_heatmap(anomaly_map)
        current_heatmap_image = heatmap_img.copy()

        # 統計情報を計算
        stats_text = f"""
## 統計情報

- **平均値**: {np.mean(anomaly_map):.4f}
- **最大値**: {np.max(anomaly_map):.4f}
- **最小値**: {np.min(anomaly_map):.4f}
- **標準偏差**: {np.std(anomaly_map):.4f}

---

**デバイス**: {device}
**画像サイズ**: 518x518
"""

        return heatmap_img, stats_text, "ヒートマップまたはターゲット画像をクリックすると、その位置のスコアが表示されます。", target_image

    except Exception as e:
        current_anomaly_map = None
        current_target_image = None
        current_heatmap_image = None
        return None, f"❌ エラーが発生しました: {str(e)}", "", target_image


def draw_marker(image, x, y, size=15):
    """画像に十字マーカーを描画"""
    if image is None:
        return None

    img = image.copy()
    draw = ImageDraw.Draw(img)

    # 十字マーカーを描画（白と赤の二重線で見やすく）
    # 外側の白い線
    draw.line([(x - size, y), (x + size, y)], fill='white', width=5)
    draw.line([(x, y - size), (x, y + size)], fill='white', width=5)

    # 内側の赤い線
    draw.line([(x - size, y), (x + size, y)], fill='red', width=3)
    draw.line([(x, y - size), (x, y + size)], fill='red', width=3)

    # 中心の円
    circle_size = 8
    draw.ellipse([(x - circle_size, y - circle_size),
                  (x + circle_size, y + circle_size)],
                 outline='white', width=3)
    draw.ellipse([(x - circle_size, y - circle_size),
                  (x + circle_size, y + circle_size)],
                 outline='red', width=2)

    return img


def get_pixel_score(evt: gr.SelectData):
    """クリックされた位置のスコアを取得して、マーカー付き画像を返す"""
    global current_anomaly_map, current_target_image, current_heatmap_image

    if current_anomaly_map is None:
        return "先に異常検出を実行してください。", None, None

    try:
        # クリック座標を取得
        x, y = evt.index

        # ヒートマップのサイズ
        h, w = current_anomaly_map.shape

        # 座標をヒートマップのサイズにスケーリング
        # (Gradioの画像表示サイズとヒートマップのサイズが異なる可能性があるため)
        map_x = int(x * w / 800)  # 800はおおよその表示幅
        map_y = int(y * h / 800)

        # 範囲チェック
        map_x = max(0, min(map_x, w - 1))
        map_y = max(0, min(map_y, h - 1))

        score = current_anomaly_map[map_y, map_x]

        # マーカー付き画像を生成
        marked_heatmap = draw_marker(current_heatmap_image, x, y)
        marked_target = draw_marker(current_target_image, x, y)

        info_text = f"""
### 📍 クリック位置のスコア

- **位置**: ({x}, {y})
- **ヒートマップ座標**: ({map_x}, {map_y})
- **異常スコア**: **{score:.4f}**

{'🔴 **異常の可能性が高い**' if score > 0.3 else '🔵 **正常範囲**'}
"""
        return info_text, marked_heatmap, marked_target
    except Exception as e:
        return f"エラー: {str(e)}", None, None


# Gradio インターフェース
with gr.Blocks(title="DINOv2 Anomaly Detection", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔍 DINOv2 異常検出システム

    DINOv2を使用したone-shot異常検出モデルです。参照画像とターゲット画像をアップロードして、異常部分を検出します。

    ### 使い方
    1. **参照画像**: 正常な状態の画像をアップロード
    2. **ターゲット画像**: 検査対象の画像をアップロード
    3. **異常検出を実行**ボタンをクリック
    """)

    with gr.Row():
        with gr.Column():
            reference_input = gr.Image(
                label="📷 参照画像（正常画像）",
                type="pil",
                height=400
            )
        with gr.Column():
            target_input = gr.Image(
                label="🎯 ターゲット画像（検査対象）",
                type="pil",
                height=400
            )

    detect_btn = gr.Button("🚀 異常検出を実行", variant="primary", size="lg")

    target_with_click = gr.Image(label="🎯 ターゲット画像（クリックしてスコア表示）", height=400, visible=False)

    with gr.Row():
        with gr.Column():
            heatmap_output = gr.Image(label="🌡️ 異常ヒートマップ（クリックしてスコア表示）", height=400)
        with gr.Column():
            stats_output = gr.Markdown(label="📊 統計情報")

    click_info = gr.Markdown(label="📍 クリック位置の情報")

    detect_btn.click(
        fn=inference,
        inputs=[reference_input, target_input],
        outputs=[heatmap_output, stats_output, click_info, target_with_click]
    )

    # ヒートマップクリック時のイベント
    heatmap_output.select(
        fn=get_pixel_score,
        outputs=[click_info, heatmap_output, target_with_click]
    )

    # ターゲット画像クリック時のイベント
    target_input.select(
        fn=get_pixel_score,
        outputs=[click_info, heatmap_output, target_with_click]
    )

    # 検出後のターゲット画像クリック時のイベント
    target_with_click.select(
        fn=get_pixel_score,
        outputs=[click_info, heatmap_output, target_with_click]
    )

    gr.Markdown(f"""
    ---
    ### 📊 技術詳細
    - **モデル**: DINOv2 ViT-S/14 with registers
    - **デバイス**: {device}
    - **画像サイズ**: 518x518
    - **特徴マッチング**: コサイン類似度（Top-20平均）
    """)

    gr.Markdown("""
    ### 💡 ヒント
    - 参照画像には正常な状態の画像を使用してください
    - 異常領域は赤色（高値）で表示されます
    - 正常領域は青色（低値）で表示されます
    """)

if __name__ == "__main__":
    demo.launch()
