import os

# 根目录
root_dir = "results"

# 你自己设定的旧后缀（yyyy）和新后缀（xxxx），不需要加 .tif
old_suffix = "Zscharr"      # ← 原文件名里要替换掉的部分
new_suffix = "a_scharr"      # ← 你想改成的部分

for folder_name in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue

    for filename in os.listdir(folder_path):
        old_path = os.path.join(folder_path, filename)

        if not filename.lower().endswith(".tif"):
            continue

        # 找出匹配的文件名，例如 1_tile_0_18_SRGAN.tif
        if f"_{old_suffix}.tif" in filename:
            new_name = filename.replace(f"_{old_suffix}.tif", f"_{new_suffix}.tif")
            new_path = os.path.join(folder_path, new_name)

            if os.path.exists(new_path):
                print(f"⚠️ 已存在同名文件，跳过: {new_path}")
                continue

            os.rename(old_path, new_path)
            print(f"✅ {filename} → {new_name}")

print("🎉 全部重命名完成！")
