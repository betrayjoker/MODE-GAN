import os
from glob import glob

def generate_meta_info(root_dir, subset_name, save_path):
    """
    root_dir: 数据根目录，例如 data/train
    subset_name: "train" / "val" / "test"
    save_path: 保存 txt 文件路径
    """
    hr_dir = os.path.join(root_dir, "HR")
    lr_dir = os.path.join(root_dir, "LR")

    hr_files = sorted(glob(os.path.join(hr_dir, "*.tif")))
    lr_files = sorted(glob(os.path.join(lr_dir, "*.tif")))

    assert len(hr_files) == len(lr_files), f"{subset_name}: HR/LR 数量不一致！"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        for hr, lr in zip(hr_files, lr_files):
            # f.write(f"{os.path.relpath(hr, root_dir)} {os.path.relpath(lr, root_dir)}\n")
            f.write(f"{os.path.basename(hr)} {os.path.basename(lr)}\n")

    print(f"✅ {subset_name} meta info 已保存到 {save_path}, 共 {len(hr_files)} 对样本")

if __name__ == "__main__":
    generate_meta_info("data/train", "train", "data/meta_info/meta_info_train.txt")
    generate_meta_info("data/val", "val", "data/meta_info/meta_info_val.txt")
    generate_meta_info("data/test", "test", "data/meta_info/meta_info_test.txt")