import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

from vit import ViT

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def cos_distance(feat, ref_feat):
    feat = torch.nn.functional.normalize(feat, dim=-1)
    ref_feat = torch.nn.functional.normalize(ref_feat, dim=-1)
    cos = torch.einsum('bnc, bmc -> bnm', feat, ref_feat)
    topk, _ = torch.topk(cos, k=20, dim=-1)
    ano = 1 - topk.mean(dim=-1)
    return ano


def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0).to(device)


def inference(ref_image, target_image, model):
    ref_tensor = preprocess_image(ref_image)
    target_tensor = preprocess_image(target_image)
    with torch.inference_mode():
        target_feat = model(target_tensor)
        ref_feat = model(ref_tensor)
    cosine_distance = cos_distance(target_feat, ref_feat)
    B, L = cosine_distance.shape
    H = W = int(L ** 0.5)
    anomaly_map = cosine_distance.reshape(B, H, W).cpu().numpy()
    return anomaly_map[0]


def main():

    model = ViT('dinov2_vits14_reg')
    model = model.to(device)
    model.eval()

    ref_image = '/home/localuser/Documents/dinov2-anomaly-detection/images/reference/212802XP0S_-20_-21.jpg'
    target_image = '/home/localuser/Documents/dinov2-anomaly-detection/images/targets/207S02OR0S_12_-36.jpg'

    output = inference(ref_image, target_image, model)
    plt.imshow(output, cmap='jet')
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    main()
