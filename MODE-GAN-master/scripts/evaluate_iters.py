def estimate_iters(num_samples, batch_size, target_epoch):
    """
    估算训练迭代次数（iters）

    参数：
        num_samples: int, 数据集样本数量
        batch_size: int, 每个 batch 的大小
        target_epoch: float, 目标训练轮数（epoch）

    返回：
        int, 推荐 iters
    """
    iters = (target_epoch * num_samples) / batch_size
    return int(iters)

# 示例：
num_samples = 10396   # 训练数据量
batch_size = 16        # batch size
target_epoch = 53.85  # 按你之前经验的 epoch

recommended_iters = estimate_iters(num_samples, batch_size, target_epoch)
print(f"推荐 iters: {recommended_iters}")
