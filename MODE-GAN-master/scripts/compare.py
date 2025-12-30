import os
import glob
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import tifffile as tiff
import lpips
from tqdm import tqdm
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

# ===================== 参数 =====================
results_root = 'results'   # SR 结果文件夹
hr_root = 'data/test/HR'  # HR 文件夹
device = 'cuda' if torch.cuda.is_available() else 'cpu'
psnr_threshold = 15
ssim_threshold = 0.5

# 初始化 LPIPS (默认 Alex)
lpips_model = lpips.LPIPS(net='vgg').to(device)

# ===================== 工具函数 =====================
def load_tif(path):
    img = tiff.imread(path).astype(np.float32)
    return img

def normalize_img(img):
    min_val = img.min()
    max_val = img.max()
    return (img - min_val) / (max_val - min_val + 1e-8)

def compute_psnr(img1, img2):
    return psnr(img1, img2, data_range=1.0)

def compute_ssim(img1, img2):
    # 只用前三通道
    img1 = img1[...,:3]
    img2 = img2[...,:3]
    return ssim(img1, img2, channel_axis=-1, win_size=7, data_range=1.0)

def compute_lpips(img1, img2):
    # img1,img2: C,H,W, normalized to [0,1]
    tensor1 = torch.from_numpy(img1[:3]).unsqueeze(0).to(device)
    tensor2 = torch.from_numpy(img2[:3]).unsqueeze(0).to(device)
    tensor1 = tensor1 * 2 - 1
    tensor2 = tensor2 * 2 - 1
    with torch.no_grad():
        val = lpips_model(tensor1, tensor2)
    return val.item()

def compute_sam(img1, img2):
    """
    计算光谱角映射 (Spectral Angle Mapper).
    img1, img2 形状为 (C, H, W) 或 (H, W, C)
    返回单位为度 (degree)
    """
    # 确保通道在最后一条轴上 (H, W, C)
    if img1.shape[0] <= 4:
        img1 = np.transpose(img1, (1, 2, 0))
        img2 = np.transpose(img2, (1, 2, 0))

    # 展开为 (N, C) 其中 N = H*W
    vec1 = img1.reshape(-1, img1.shape[-1])
    vec2 = img2.reshape(-1, img2.shape[-1])

    # 计算点积和模长
    dot_product = np.sum(vec1 * vec2, axis=1)
    norm1 = np.linalg.norm(vec1, axis=1)
    norm2 = np.linalg.norm(vec2, axis=1)

    # 防止除以零
    similarity = dot_product / (norm1 * norm2 + 1e-8)
    # 剪裁范围防止 arccos 超域
    similarity = np.clip(similarity, -1.0, 1.0)

    # 计算角度 (弧度 -> 角度)
    angle = np.arccos(similarity)
    sam_val = np.mean(angle) * (180.0 / np.pi)

    return sam_val


# ===================== 批量计算 =====================
all_results = []

hr_folders = sorted([d for d in os.listdir(results_root)
                     if os.path.isdir(os.path.join(results_root,d))])

for hr_folder in tqdm(hr_folders, desc="Processing HR folders"):
    hr_folder_path = os.path.join(results_root, hr_folder)

    # 对应 HR
    hr_path = os.path.join(hr_root, f"{hr_folder}.tif")
    if not os.path.exists(hr_path):
        print(f"HR not found for {hr_folder}")
        continue
    hr_img = load_tif(hr_path)
    hr_img_norm = normalize_img(hr_img)

    # 遍历 SR tif 文件
    sr_files = sorted([f for f in glob.glob(os.path.join(hr_folder_path, "*.tif"))
                       if os.path.isfile(f)])
    if len(sr_files) == 0:
        continue

    for sr_path in tqdm(sr_files, desc=f"Processing {hr_folder}", leave=False):
        sr_img = load_tif(sr_path)
        sr_img_norm = normalize_img(sr_img)

        if sr_img.shape != hr_img.shape:
            print(f"Shape mismatch: {sr_path}")
            continue

        # H,W,C
        if sr_img_norm.ndim == 3 and sr_img_norm.shape[0] <= 4:
            sr_norm_HW = np.transpose(sr_img_norm, (1,2,0))
            hr_norm_HW = np.transpose(hr_img_norm, (1,2,0))
        else:
            sr_norm_HW = sr_img_norm
            hr_norm_HW = hr_img_norm

        # 计算指标
        # 1. PSNR (全波段)
        val_psnr = compute_psnr(sr_img_norm, hr_img_norm)

        # 2. SSIM (全波段平均，不要切片 [...,:3])
        # 修改：channel_axis 改为对应你数据格式的轴
        val_ssim = ssim(sr_norm_HW, hr_norm_HW, channel_axis=-1,
                        win_size=7, data_range=1.0)

        # 3. LPIPS (由于模型限制，通常只用 RGB 三通道)
        val_lpips = compute_lpips(sr_norm_HW.transpose(2,0,1),
                                  hr_norm_HW.transpose(2,0,1))

        # 4. SAM (新增：反映光谱保真度)
        val_sam = compute_sam(sr_norm_HW, hr_norm_HW)



        # 提取模型名称：去掉 tile 前缀和 tif 后缀
        fname = os.path.basename(sr_path)
        model_name = "_".join(fname.split("_")[3:]).replace(".tif", "")
        import re
        model_name = re.sub(r'^\d+_', '', model_name)

       # 5. 存储结果
        all_results.append({
            'hr_folder': hr_folder,
            'model': model_name,
            'psnr': val_psnr,
            'ssim': val_ssim,
            'lpips': val_lpips,
            'sam': val_sam  # 存入列表
        })

# ===================== 保存 CSV =====================
df = pd.DataFrame(all_results)
df.to_csv("sr_metrics.csv", index=False)

df_filtered = df[(df['psnr'] >= psnr_threshold) & (df['ssim'] >= ssim_threshold)].copy()

# ========== 提取区域 ID (hr_folder 的第一个字符) ==========
# 假设格式为 "1_tile_xxx", 则提取出 "1"
df_filtered['region'] = df_filtered['hr_folder'].str.split('_').str[0]

# ========== 计算指标 ==========

# A. 计算每个模型在每个区域的平均值 (Regional Average)
regional_avg = df_filtered.groupby(['model', 'region'])[['psnr', 'ssim', 'lpips', 'sam']].mean().reset_index()

# B. 计算每个模型在全球/整体的平均值 (Overall Average)
overall_avg = df_filtered.groupby('model')[['psnr', 'ssim', 'lpips', 'sam']].mean().reset_index()
overall_avg['region'] = 'Overall'  # 标记为整体平均

# ========== 合并并保存结果 ==========
# 将分区域和整体平均合并在一起，方便查阅
final_report = pd.concat([regional_avg, overall_avg], axis=0).sort_values(by=['model', 'region'])

# 保存 CSV
df_filtered.to_csv("sr_metrics_filtered.csv", index=False)
final_report.to_csv("sr_metrics_by_region_report.csv", index=False)

print("处理完成！")
print(f"详细过滤数据已保存至: sr_metrics_filtered.csv")
print(f"分区域及整体平均报告已保存至: sr_metrics_by_region_report.csv")

# 预览一下结果
print("\n部分模型结果预览:")
print(final_report.head(10))




