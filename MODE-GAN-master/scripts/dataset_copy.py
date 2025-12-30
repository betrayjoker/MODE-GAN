import os
import shutil

# 路径设置
test_dir = "test"
hr_dir = os.path.join(test_dir, "HR")
lr_dir = os.path.join(test_dir, "LR")
results_dir = "results"

# 遍历 results 下的所有子文件夹
for folder_name in os.listdir(results_dir):
    folder_path = os.path.join(results_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue  # 忽略非文件夹

    # 文件夹名字对应的 HR 和 LR 文件
    hr_file = os.path.join(hr_dir, f"{folder_name}.tif")
    lr_file = os.path.join(lr_dir, f"{folder_name}.tif")

    # 检查文件是否存在并复制
    if os.path.exists(hr_file):
        shutil.copy(hr_file, os.path.join(folder_path, "HR.tif"))
        print(f"Copied HR: {hr_file} -> {folder_path}/HR.tif")
    else:
        print(f"HR file not found: {hr_file}")

    if os.path.exists(lr_file):
        shutil.copy(lr_file, os.path.join(folder_path, "LR.tif"))
        print(f"Copied LR: {lr_file} -> {folder_path}/LR.tif")
    else:
        print(f"LR file not found: {lr_file}")
