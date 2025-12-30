import os
import rasterio
import numpy as np
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import tifffile as tiff
import pandas as pd

# 测试集路径
test_lr_dir = 'data/test/LR'
test_hr_dir = 'data/test/HR'

results = []

for fname in os.listdir(test_lr_dir):
    if not fname.endswith('.tif'):
        continue

    lr_path = os.path.join(test_lr_dir, fname)
    hr_path = os.path.join(test_hr_dir, fname)

    with rasterio.open(lr_path) as lr_src, rasterio.open(hr_path) as hr_src:
        lr = lr_src.read().astype(np.float32)  # shape: [C,H,W]
        hr = hr_src.read().astype(np.float32)

        # 将 [C,H,W] -> [H,W,C] 方便 resize
        lr = np.transpose(lr, (1,2,0))
        hr = np.transpose(hr, (1,2,0))

        # Bicubic 插值到 HR 大小
        bicubic = resize(lr, hr.shape, order=3, mode='reflect', anti_aliasing=True)

        # 计算指标
        psnr_val = psnr(hr, bicubic, data_range=hr.max()-hr.min())
        ssim_val = ssim(hr, bicubic, multichannel=True, data_range=hr.max()-hr.min())

        results.append({
            'filename': fname,
            'PSNR': psnr_val,
            'SSIM': ssim_val
        })

df = pd.DataFrame(results)
print(df.mean())  # 平均指标
df.to_csv('bicubic_results.csv', index=False)
