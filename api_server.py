import base64
import io

import matplotlib
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from vit import ViT

app = FastAPI()

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では適切に制限してください
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# モデルを事前にロード
model = None


def get_model():
    global model
    if model is None:
        model = ViT('dinov2_vits14_reg')
        model = model.to(device)
        model.eval()
    return model


def cos_distance(feat, ref_feat):
    feat = torch.nn.functional.normalize(feat, dim=-1)
    ref_feat = torch.nn.functional.normalize(ref_feat, dim=-1)
    cos = torch.einsum('bnc, bmc -> bnm', feat, ref_feat)
    topk, _ = torch.topk(cos, k=20, dim=-1)
    ano = 1 - topk.mean(dim=-1)
    return ano


def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0).to(device)


def create_heatmap(anomaly_map):
    """異常マップをヒートマップ画像に変換してBase64エンコード"""
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(anomaly_map, cmap='jet')
    plt.colorbar(im, ax=ax)
    ax.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return img_base64


@app.get("/")
async def root():
    return {"message": "DINOv2 Anomaly Detection API"}


@app.get("/health")
async def health():
    return {"status": "ok", "device": str(device)}


@app.post("/detect")
async def detect_anomaly(
    reference_image: UploadFile = File(...),
    target_image: UploadFile = File(...)
):
    try:
        # モデルを取得
        model = get_model()

        # 画像を読み込み
        ref_bytes = await reference_image.read()
        target_bytes = await target_image.read()

        # 画像を前処理
        ref_tensor = preprocess_image(ref_bytes)
        target_tensor = preprocess_image(target_bytes)

        # 推論実行
        with torch.inference_mode():
            target_feat = model(target_tensor)
            ref_feat = model(ref_tensor)

        # コサイン距離を計算
        cosine_distance = cos_distance(target_feat, ref_feat)
        B, L = cosine_distance.shape
        H = W = int(L ** 0.5)
        anomaly_map = cosine_distance.reshape(B, H, W).cpu().numpy()[0]

        # ヒートマップを生成
        heatmap_base64 = create_heatmap(anomaly_map)

        # 統計情報を計算
        stats = {
            "mean": float(np.mean(anomaly_map)),
            "max": float(np.max(anomaly_map)),
            "min": float(np.min(anomaly_map)),
            "std": float(np.std(anomaly_map))
        }

        return JSONResponse(content={
            "success": True,
            "heatmap": f"data:image/png;base64,{heatmap_base64}",
            "statistics": stats
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
