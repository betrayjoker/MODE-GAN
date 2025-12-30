import torch
import time
from fvcore.nn import FlopCountAnalysis
from basicsr.archs.rrdbnet_arch import RRDBNet

# ==================== 配置 ====================
scale = 3  # 超分倍率
device = 'cuda' if torch.cuda.is_available() else 'cpu'
C, H, W = 4, 64, 64  # 输入通道和尺寸

# ==================== 初始化模型 ====================
model = RRDBNet(
    num_in_ch=4,
    num_out_ch=4,
    num_feat=64,
    num_block=23,
    num_grow_ch=32,
    scale=scale
).to(device)
model.eval()

# ==================== 计算参数量 ====================
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameters: {num_params/1e6:.2f} M")

# ==================== 计算 FLOPs ====================
x = torch.randn(1, C, H, W).to(device)
flops = FlopCountAnalysis(model, x)
print(f"FLOPs: {flops.total()/1e9:.2f} G")

# ==================== 测量推理时间 ====================
# GPU 预热
for _ in range(10):
    _ = model(x)

# 测量平均推理时间
torch.cuda.synchronize()
start = time.time()
n = 100  # 测 100 次取平均
for _ in range(n):
    _ = model(x)
torch.cuda.synchronize()
end = time.time()

avg_time_ms = (end - start)/n * 1000
print(f"Inference Time: {avg_time_ms:.2f} ms")

# ==================== 格式化输出（论文表格可用） ====================
print(f"\nModel: RRDBNet | Parameters: {num_params/1e6:.2f} M | FLOPs: {flops.total()/1e9:.2f} G | Inference Time: {avg_time_ms:.2f} ms")
