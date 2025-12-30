import pandas as pd

# ========== 参数 ==========
input_csv = "sr_metrics.csv"   # 原始全部值 CSV
psnr_threshold = 1
ssim_threshold = 0

# ========== 1. 读取原始数据 ==========
df = pd.read_csv(input_csv)

# ========== 2. 过滤低质量结果 ==========
df_filtered = df[(df['psnr'] >= psnr_threshold) & (df['ssim'] >= ssim_threshold)].copy()

# ========== 3. 提取区域 ID (hr_folder 的第一个字符) ==========
# 假设格式为 "1_tile_xxx", 则提取出 "1"
df_filtered['region'] = df_filtered['hr_folder'].str.split('_').str[0]

# ========== 4. 计算指标 ==========

# A. 计算每个模型在每个区域的平均值 (Regional Average)
regional_avg = df_filtered.groupby(['model', 'region'])[['psnr', 'ssim', 'lpips', 'sam']].mean().reset_index()

# B. 计算每个模型在全球/整体的平均值 (Overall Average)
overall_avg = df_filtered.groupby('model')[['psnr', 'ssim', 'lpips', 'sam']].mean().reset_index()
overall_avg['region'] = 'Overall'  # 标记为整体平均

# ========== 5. 合并并保存结果 ==========
# 将分区域和整体平均合并在一起，方便查阅
final_report = pd.concat([regional_avg, overall_avg], axis=0).sort_values(by=['model', 'region'])

# 保存 CSV
df_filtered.to_csv("sr_metrics_filtered1.csv", index=False)
final_report.to_csv("sr_metrics_by_region_report1.csv", index=False)

print("处理完成！")
print(f"详细过滤数据已保存至: sr_metrics_filtered.csv")
print(f"分区域及整体平均报告已保存至: sr_metrics_by_region_report.csv")

# 预览一下结果
print("\n部分模型结果预览:")
print(final_report.head(10))
