import numpy as np

loadData = np.load(f"./results/OriginDDPG_Hopper-v2_0.npy")
# 只存储了（每10个episodes的均值）

print(loadData)
print(len(loadData))