import os
import shutil
import random

HR_path = 'data/HR'
LR_path = 'data/LR'
output_path = 'data'

splits = ['train', 'val', 'test']
split_ratio = [0.8, 0.1, 0.1]

# 创建输出目录
for split in splits:
    for modality in ['HR', 'LR']:
        os.makedirs(os.path.join(output_path, split, modality), exist_ok=True)

# 获取HR和LR子文件夹列表，并排序
HR_subfolders = sorted(os.listdir(HR_path))
LR_subfolders = sorted(os.listdir(LR_path))

# 检查数量一致
assert len(HR_subfolders) == len(LR_subfolders), "HR和LR子文件夹数量不一致"

all_files = []

for idx, (hr_folder, lr_folder) in enumerate(zip(HR_subfolders, LR_subfolders)):
    folder_prefix = f"{idx+1}_"
    HR_subfolder = os.path.join(HR_path, hr_folder)
    LR_subfolder = os.path.join(LR_path, lr_folder)

    hr_files = sorted(os.listdir(HR_subfolder))
    lr_files = sorted(os.listdir(LR_subfolder))

    assert len(hr_files) == len(lr_files), f"{hr_folder} 和 {lr_folder} 文件数量不一致"

    for hr_file, lr_file in zip(hr_files, lr_files):
        hr_file_path = os.path.join(HR_subfolder, hr_file)
        lr_file_path = os.path.join(LR_subfolder, lr_file)
        new_name = folder_prefix + hr_file  # 输出文件名加前缀
        all_files.append((hr_file_path, lr_file_path, new_name))

# 打乱
random.shuffle(all_files)

# 划分数量
n_total = len(all_files)
n_train = int(n_total * split_ratio[0])
n_val = int(n_total * split_ratio[1])
n_test = n_total - n_train - n_val

splits_counts = {'train': n_train, 'val': n_val, 'test': n_test}

start_idx = 0
for split in splits:
    count = splits_counts[split]
    selected = all_files[start_idx:start_idx + count]
    for hr_file, lr_file, new_name in selected:
        shutil.copy2(hr_file, os.path.join(output_path, split, 'HR', new_name))
        shutil.copy2(lr_file, os.path.join(output_path, split, 'LR', new_name))
    start_idx += count

print("数据集划分完成！")
