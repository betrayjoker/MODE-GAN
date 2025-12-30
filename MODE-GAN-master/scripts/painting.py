import matplotlib.pyplot as plt
import numpy as np

# ====================== 数据 ======================
region_names = ['North China Plain', 'Qinghai–Tibet Plateau', 'Sichuan Basin', 'Pearl River Delta']
regions_short = ['Region1', 'Region2', 'Region3', 'Region4']

models = ['Bicubic', 'SRCNN', 'SRResNet', 'EDSR', 'ESRGAN', 'Real-ESRGAN', 'RCAN', 'SwinIR', 'HAT', 'Swin2SR', 'MODE-GAN']

psnr = np.array([
    [17.22, 18.14, 18.03, 18.23],
    [21.17, 23.09, 22.48, 22.52],
    [22.23, 24.08, 23.61, 23.88],
    [24.41, 25.47, 25.42, 24.67],
    [24.58, 25.51, 25.03, 24.73],
    [25.29, 26.84, 25.36, 25.64],
    [25.01, 26.49, 25.50, 25.65],
    [24.90, 26.88, 25.23, 25.19],
    [25.02, 25.63, 25.12, 25.39],
    [25.08, 25.81, 25.20, 25.19],
    [26.02, 26.94, 26.58, 26.71],
])

ssim = np.array([
    [0.55, 0.64, 0.57, 0.60],
    [0.62, 0.68, 0.64, 0.67],
    [0.66, 0.76, 0.64, 0.69],
    [0.70, 0.80, 0.66, 0.68],
    [0.70, 0.81, 0.64, 0.69],
    [0.77, 0.86, 0.67, 0.78],
    [0.76, 0.86, 0.67, 0.76],
    [0.73, 0.84, 0.65, 0.72],
    [0.68, 0.70, 0.67, 0.69],
    [0.71, 0.73, 0.68, 0.72],
    [0.81, 0.91, 0.73, 0.83],
])

lpips = np.array([
    [0.51, 0.40, 0.57, 0.52],
    [0.44, 0.38, 0.43, 0.43],
    [0.38, 0.30, 0.33, 0.32],
    [0.28, 0.24, 0.33, 0.31],
    [0.26, 0.23, 0.33, 0.27],
    [0.20, 0.17, 0.30, 0.21],
    [0.21, 0.18, 0.30, 0.23],
    [0.24, 0.20, 0.32, 0.26],
    [0.27, 0.27, 0.29, 0.29],
    [0.24, 0.23, 0.24, 0.25],
    [0.20, 0.17, 0.28, 0.21],
])

# ====================== 绘图 ======================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
colors = plt.cm.tab20(np.linspace(0, 1, len(models)))

metrics = {'PSNR↑': psnr, 'SSIM↑': ssim, 'LPIPS↓': lpips}

fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharex=True)

for i, (name, data) in enumerate(metrics.items()):
    ax = axes[i]
    for j, model in enumerate(models):
        ax.plot(regions_short, data[j], marker='o', linewidth=2, color=colors[j], label=model)
    ax.set_title(name, fontsize=14, fontweight='bold')
    if i == 0:
        ax.set_ylabel('Value', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.tick_params(axis='x', rotation=0)

# ====================== 单一图例 ======================
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels,
           loc='lower center',          # 底部正中
           bbox_to_anchor=(0.5, 0.0), # x=0.5居中，y=-0.02稍微下移到子图外
           ncol=len(models),            # 横向排列，每个模型一列
           fontsize=10,
           frameon=True,
           edgecolor='black',
           facecolor='white')


# ====================== 调整子图间距 ======================
plt.subplots_adjust(wspace=0.1, bottom=0.15)  # wspace增大图间距，bottom留出空间放图例

# # ====================== 区域说明 ======================
# region_mapping_text = ' | '.join([f'{short}: {full}' for short, full in zip(regions_short, region_names)])
# plt.figtext(0.5, 0.01, region_mapping_text, ha='center', fontsize=10, fontstyle='normal')

plt.show()
