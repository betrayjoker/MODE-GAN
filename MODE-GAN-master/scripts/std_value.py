import os
import rasterio
import numpy as np
from scipy.ndimage import generic_filter

# ---------------- 参数 ----------------
input_dir = "results/1_tile_0_18"  # 指定要处理的子文件夹
output_dir = "results_std/1_tile_0_18"  # 输出保存路径
window_size = 3
os.makedirs(output_dir, exist_ok=True)

# ---------------- 局部标准差函数 ----------------
def local_std(arr, size=3):
    return generic_filter(arr, np.std, size=(size, size))

# ---------------- 批量处理 ----------------
for tif_file in sorted(os.listdir(input_dir)):
    if not tif_file.lower().endswith(".tif"):
        continue
    input_path = os.path.join(input_dir, tif_file)
    with rasterio.open(input_path) as src:
        img = src.read().astype(np.float32)
        profile = src.profile.copy()

    std_img = np.stack([local_std(band, window_size) for band in img], axis=0)

    profile.update(dtype=rasterio.float32)
    output_path = os.path.join(output_dir, tif_file.replace(".tif", f"_std.tif"))
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(std_img)

    print(f"{tif_file} 局部标准差生成完成，保存至 {output_path}")
