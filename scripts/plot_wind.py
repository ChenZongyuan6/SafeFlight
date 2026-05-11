import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_wind_data(path="./pinn_dataset.pt"):
    print(f"Loading {path}...")
    try:
        data = torch.load(path)
    except FileNotFoundError:
        print("未找到数据文件！")
        return

    # 获取 Labels (gt_disturbance)
    # Shape: [Num_Envs, Time_Steps, 3]
    labels = data["labels"]
    
    # 随机选一个环境（比如第 10 个）
    env_idx = 10
    time_steps = np.arange(labels.shape[1]) * 0.02 # 假设 dt=0.02s
    
    # 提取该环境下的 xyz 扰动
    disturbance = labels[env_idx].cpu().numpy() # [Time, 3]
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, disturbance[:, 0], label='Disturbance X (Body)', alpha=0.8)
    plt.plot(time_steps, disturbance[:, 1], label='Disturbance Y (Body)', alpha=0.8)
    plt.plot(time_steps, disturbance[:, 2], label='Disturbance Z (Body)', alpha=0.8)
    
    plt.title(f"Ground Truth Disturbance in Environment #{env_idx}")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s^2)")
    plt.legend()
    plt.grid(True)
    
    # 保存图片
    save_name = "wind_verification.png"
    plt.savefig(save_name)
    print(f"验证图片已保存为: {save_name}")
    print(f"数据统计 (Env {env_idx}):")
    print(f"  Max Disturbance: {np.max(np.abs(disturbance)):.4f} m/s^2")
    print(f"  Mean Disturbance: {np.mean(np.abs(disturbance)):.4f} m/s^2")

    # 全局统计
    all_mean = torch.abs(labels).mean().item()
    if all_mean < 1e-4:
        print("\n[严重警告] 全局风场平均值为 0！")
        print("可能原因：track.yaml 中 wind: false 或者代码未正确保存 gt_disturbance。")
    else:
        print(f"\n[成功] 全局风场正常，平均强度: {all_mean:.4f}")

if __name__ == "__main__":
    plot_wind_data()