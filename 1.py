# from basicsr.utils.registry import ARCH_REGISTRY
# import basicsr.archs.rrdbnet_arch

# # 打印所有注册的网络名字
# print(list(ARCH_REGISTRY.keys()))

from basicsr.utils.registry import ARCH_REGISTRY
import basicsr.archs.rrdbnet_arch  # 确保注册执行

print('RRDBNet_new' in ARCH_REGISTRY.keys())
