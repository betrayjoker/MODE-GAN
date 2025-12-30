import cv2
import numpy as np
import tifffile as tiff
from scipy.ndimage import gaussian_laplace
import os


# ================= 文件路径 =================
input_tif = "results/4_tile_5_21/HR.tif"   # 输入影像路径（四波段）
output_tif = "results/4_tile_5_21/HR_mslog.tif" # 输出影像路径
operator = "mslog"   # 可选："sobel" | "scharr" | "mslog" | "scharr+mslog"
scales = [0.8, 1.6, 3.2]    # mslog多尺度

# ================= 读取多波段影像 =================
img = tiff.imread(input_tif)   # shape: (H, W, 4)
img = img.transpose(1,2,0) #[H,W,C]
if img.ndim == 2:
    raise ValueError("输入影像只有一个波段，请检查文件。")
if img.shape[2] != 4:
    raise ValueError(f"输入影像波段数为 {img.shape[2]}，应为4。")

# ================= 定义算子函数 =================
def sobel_edge(band):
    sobelx = cv2.Sobel(band, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(band, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    return mag

def scharr_edge(band):
    scharrx = cv2.Scharr(band, cv2.CV_32F, 1, 0)
    scharry = cv2.Scharr(band, cv2.CV_32F, 0, 1)
    mag = np.sqrt(scharrx ** 2 + scharry ** 2)
    return mag

def mslog_edge(band, scales=[0.8, 1.6, 3.2]):
    """多尺度 Laplacian of Gaussian (LoG)"""
    edges = []
    for sigma in scales:
        log_resp = gaussian_laplace(band, sigma=sigma)
        edges.append(np.abs(log_resp))
    return np.mean(edges, axis=0)

def scharr_mslog_edge(band, scales=[0.8, 1.6, 3.2]):
    """融合 Scharr + 多尺度LoG"""
    scharr_mag = scharr_edge(band)
    log_mag = mslog_edge(band, scales)
    fused = cv2.addWeighted(scharr_mag, 0.5, log_mag, 0.5, 0)
    return fused

# ================= 按算子处理每个波段 =================
edge_bands = []
for i in range(4):
    band = img[:, :, i].astype(np.float32)

    if operator.lower() == "sobel":
        edge = sobel_edge(band)
    elif operator.lower() == "scharr":
        edge = scharr_edge(band)
    elif operator.lower() == "mslog":
        edge = mslog_edge(band, scales)
    elif operator.lower() == "scharr+mslog":
        edge = scharr_mslog_edge(band, scales)
    else:
        raise ValueError(f"未知算子类型: {operator}")

    # 归一化到 0–255
    edge_norm = cv2.normalize(edge, None, 0, 255, cv2.NORM_MINMAX)
    edge_bands.append(edge_norm.astype(np.uint8))

# ================= 合并为四波段 =================
edge_4band = np.stack(edge_bands, axis=-1)  # shape: (H, W, 4)

# ================= 保存为 TIF =================
os.makedirs(os.path.dirname(output_tif), exist_ok=True)
tiff.imwrite(output_tif, edge_4band)

print(f"✅ 边缘检测完成 ({operator})，结果已保存：{output_tif}")
