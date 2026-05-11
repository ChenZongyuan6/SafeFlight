import torch

data = torch.load("collected_data/dataset_20260316_0028/dataset_20260316_0028.pt")
actions = data["inputs"][:, :, 15:19] # 提取 u_ctrl (4维动作)

# 计算每个通道的全局平均值
action_means = actions.mean(dim=(0, 1))
print("动作 4 个通道的平均值:", action_means)