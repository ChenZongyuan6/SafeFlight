import torch
import numpy as np

def check_dataset(path="./pinn_dataset.pt"):
    print(f"Loading {path} ...")
    try:
        data = torch.load(path)
    except FileNotFoundError:
        print("Error: 文件未找到，请先运行采集脚本！")
        return

    # 1. 检查数据结构
    inputs = data["inputs"] # [Envs, Time, 19]
    labels = data["labels"] # [Envs, Time, 3]
    
    print("-" * 30)
    print(f"数据形状: {inputs.shape}")
    print(f"  - Environments: {inputs.shape[0]}")
    print(f"  - Time Steps:   {inputs.shape[1]}")
    print("-" * 30)

    # 2. 验证风场 (Wind) 是否生效
    # 检查 Label (gt_disturbance) 的绝对值平均数
    # 如果 wind=False，这里应该是全 0
    wind_energy = labels.abs().mean().item()
    print(f"[检查 1] 风场标签 (Label Y):")
    if wind_energy > 1e-5:
        print(f"  ✅ 风场已生效！平均扰动强度: {wind_energy:.4f} m/s²")
        print(f"  Max Disturbance: {labels.max().item():.4f}")
    else:
        print(f"  ❌ 风场未生效！数据全为 0。请检查 yaml 中的 wind: true")

    # 3. 验证阻力 (Drag) 是否生效
    # 检查 Input 中的机体速度 (前3维)
    # 如果阻力过大(如 0.2)，速度会很小；如果阻力为0，速度可能会发散
    v_body = inputs[..., 0:3]
    v_mean = v_body.norm(dim=-1).mean().item()
    v_max = v_body.max().item()
    
    print(f"[检查 2] 飞行速度 (Input X):")
    print(f"  平均速度: {v_mean:.2f} m/s")
    print(f"  最大速度: {v_max:.2f} m/s")
    
    if v_mean > 0.1 and v_max < 50.0:
        print("  ✅ 动力学正常。无人机在飞且没有炸机。")
    elif v_mean < 0.01:
        print("  ⚠️ 警告：无人机似乎没动？(速度接近0)")
    else:
        print("  ⚠️ 警告：速度异常巨大，可能物理参数发散！")

    # 4. 验证随机化 (Randomization)
    # 既然 drag 固定，我们很难直接从 output 验证 drag。
    # 但我们可以看看数据分布是否过于单一。
    print("-" * 30)
    print("验证完成。")

if __name__ == "__main__":
    check_dataset()