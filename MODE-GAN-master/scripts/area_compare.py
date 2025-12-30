import pandas as pd

# 读取文件
file_path = "sr_metrics_filtered_all.csv"
df = pd.read_csv(file_path)

# 清理无用列并提取区域编号
df = df[['hr_folder', 'model', 'psnr', 'ssim', 'lpips']].copy()
df['region'] = df['hr_folder'].str.extract(r'(\d+)_').astype(int)

# 计算每个模型在每个区域的平均值
model_region_avg = (
    df.groupby(['model', 'region'])[['psnr', 'ssim', 'lpips']]
    .mean()
    .reset_index()
)

# 计算区域0（即1–4区域整体平均）
region0 = (
    model_region_avg.groupby('model')[['psnr', 'ssim', 'lpips']]
    .mean()
    .reset_index()
)
region0['region'] = 0  # 添加区域编号0

# 合并区域1–4与区域0
final_df = pd.concat([model_region_avg, region0], ignore_index=True)
final_df = final_df.sort_values(['model', 'region']).reset_index(drop=True)

# 查看结果
print(final_df)

# 可选：保存为 CSV
final_df.to_csv("model_region_avg_all.csv", index=False)
