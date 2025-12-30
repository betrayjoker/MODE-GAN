import os
import torch
import torch.nn as nn
import numpy as np
import rasterio
from skimage.transform import resize

# ----------------- SRCNN 模型定义 -----------------
class SRCNN(nn.Module):
    def __init__(self, num_channels=4):
        super(SRCNN, self).__init__()
        self.layer1 = nn.Conv2d(num_channels, 64, 9, padding=4)
        self.layer2 = nn.Conv2d(64, 32, 5, padding=2)
        self.layer3 = nn.Conv2d(32, num_channels, 5, padding=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# ----------------- 推理函数 -----------------
def inference_srcnn(model, input_path, output_dir, scale=3):
    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        img = src.read().astype(np.float32)  # [C,H,W]

    if img.shape[0] != 4:
        raise ValueError("输入影像非四波段！")

    img = img.transpose(1,2,0)  # [H,W,C]

    # 1. 上采样到目标尺寸
    H, W, C = img.shape
    H_hr, W_hr = H * scale, W * scale
    img_up = resize(img, (H_hr, W_hr, C), order=3, mode='reflect', anti_aliasing=True)

    # 2. 归一化
    min_val = img_up.min(axis=(0,1), keepdims=True)
    max_val = img_up.max(axis=(0,1), keepdims=True)
    img_norm = (img_up - min_val) / (max_val - min_val + 1e-8)
    img_tensor = torch.from_numpy(img_norm.transpose(2,0,1)).unsqueeze(0)  # [B,C,H,W]

    # 3. SRCNN 推理
    model.eval()
    with torch.no_grad():
        sr_tensor = model(img_tensor)

    output = sr_tensor.squeeze(0).cpu().numpy().transpose(1,2,0)  # [H,W,C]
    output_img = output * (max_val - min_val) + min_val
    output_img = output_img.transpose(2,0,1)  # [C,H,W]

    # 4. 更新 profile
    profile.update(dtype=rasterio.float32,
                   count=output_img.shape[0],
                   height=output_img.shape[1],
                   width=output_img.shape[2],
                   transform=rasterio.Affine(profile['transform'].a/scale,
                                             profile['transform'].b,
                                             profile['transform'].c,
                                             profile['transform'].d,
                                             profile['transform'].e/scale,
                                             profile['transform'].f))

    # 5. 保存
    input_stem = os.path.splitext(os.path.basename(input_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{input_stem}_SRCNN.tif")
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(output_img)
    print(f"{input_stem} SRCNN 推理完成，保存至 {output_path}")

# ----------------- 主函数 -----------------
def main():
    input_root = "data/test/LR"
    output_root = "results"
    scale = 3

    model = SRCNN(num_channels=4)
    # 如果有训练好的 SRCNN 权重，可以在这里加载：
    # model.load_state_dict(torch.load("path_to_srcnn_weights.pth"))

    inputs = [os.path.join(input_root, f) for f in sorted(os.listdir(input_root)) if f.endswith('.tif')]
    if not inputs:
        print("没有找到tif文件，请检查路径！")
        return

    for input_path in inputs:
        save_dir = os.path.join(output_root, os.path.splitext(os.path.basename(input_path))[0])
        os.makedirs(save_dir, exist_ok=True)
        inference_srcnn(model, input_path, save_dir, scale=scale)

if __name__ == "__main__":
    main()
