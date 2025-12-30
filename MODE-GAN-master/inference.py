import torch
import numpy as np
import tifffile
import rasterio
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# ---------------- 参数设置 ----------------
# model_path = 'experiments/RealESRGAN_8.25night/models/net_g_latest.pth'  # 你的模型路径
model_path = 'experiments/RealESRGANx3plus_9.12day_casaea/models/net_g_latest.pth'
input_path = 'data/test/LR/tile_0_1.tif'   # 输入四波段tif
output_path = 'results/try.tif'  # 输出tif
scale = 3   # 放大倍数

# ---------------- 读取四波段tif ----------------
with rasterio.open(input_path) as src:
    profile = src.profile.copy()
    img = src.read().astype(np.float32)   # shape: [C, H, W]


if img.shape[0] != 4:
    raise ValueError("Input image must有 4 个波段")

# [C, H, W] -> [H, W, C]
img = np.transpose(img, (1, 2, 0))

# 归一化到 0-1
min = img.min(axis=(0,1), keepdims=True)
max = img.max(axis=(0,1), keepdims=True)
print(min, max)
img = (img - min) / (max - min + 1e-8)

# if img.max() > 1.0:
#     img = img / 65535.0 if img.dtype == np.uint16 else img / 255.0

print(f"输入图像 shape: {img.shape}", img.dtype, img.min(), img.max(), img.shape)

# ---------------- 加载模型 ----------------
model = RRDBNet(
    num_in_ch=4,   # 输入4波段
    num_out_ch=4,  # 输出4波段
    num_feat=64,
    num_block=23,
    num_grow_ch=32,
    scale=3
)

upsampler = RealESRGANer(
    scale=scale,
    model_path=model_path,
    model=model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=False
)

# ---------------- 推理 ----------------
output, _ = upsampler.enhance(img, outscale=scale)

print("output",output.dtype, output.min(), output.max(), output.shape)

# 反归一化
output_img = output.astype(np.float32)
# print("dddddd", max,min)
output_img = output_img * (max - min) + min
# output_img = np.clip(output_img, min, max)

# [H, W, C] -> [C, H, W]
output_img = np.transpose(output_img, (2, 0, 1))
print("output_img", output_img.dtype, output_img.min(), output_img.max(), output_img.shape)

# ---------------- 保存结果 ----------------
profile.update(
    dtype=rasterio.float32,
    count=output_img.shape[0],  # 波段数
    height=output_img.shape[1],
    width=output_img.shape[2],
    # compress='lzw'          # 可选压缩
)

# 同时更新 transform，让像元大小缩小 1/scale，覆盖范围保持不变
transform = profile['transform']
profile.update(
    transform=rasterio.Affine(
        transform.a / scale, transform.b, transform.c,
        transform.d, transform.e / scale, transform.f
    )
)

with rasterio.open(output_path, 'w', **profile) as dst:
    dst.write(output_img)

print("✅ 推理完成，结果保存到:", output_path)

