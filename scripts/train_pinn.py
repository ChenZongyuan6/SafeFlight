import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import wandb
import datetime

# ==========================================
# 1. 配置参数 (Configuration)
# ==========================================
class Config:
    # --- WandB 配置 [新增 2] ---
    WANDB_PROJECT = "PINN-Drone-Dynamics"  # 项目名称 (在 wandb 网页上显示的项目名)
    # 注意：WANDB_RUN_NAME 会在运行时动态添加时间戳，这里只定义前缀
    WANDB_RUN_NAME = "TCN_Physics_Run_01"  # 本次实验的名称 (可选，若不填 wandb 会随机生成)
    # --- 数据相关 ---
    # 历史窗口长度 (e.g., 20 steps = 0.4s @ 50Hz)
    # TCN 将利用这 20 帧历史来预测当前的扰动
    WINDOW_SIZE = 20        
    
    # 仿真步长 (必须与 collect_data.py 设置一致)
    DT = 0.02               
    
    # 数据集划分比例 (前 90% 环境用于训练，后 10% 用于验证)
    TRAIN_RATIO = 0.9       
    
    # --- 无人机物理参数 ---
    # 请根据你的 SimpleFlight/Isaac Sim 实际配置修改
    MASS =  0.93     #1.0              # kg
    GRAVITY = 9.81          # m/s^2
    
    # --- 训练超参数 ---
    BATCH_SIZE = 256        # 训练 Batch 大小
    VAL_BATCH_SIZE = 512    # 验证 Batch 大小 (不存梯度，可更大)
    LR = 1e-3               # 学习率
    EPOCHS = 50             # 总训练轮数
    
    # Loss 权重系数 (PINN 的核心)
    LAMBDA_PHY = 1.0        # 物理约束 Loss 权重
    LAMBDA_DATA = 1.0       # 数据回归 Loss 权重
    
    # --- 网络维度 ---
    # 输入特征: 19维 = v(3) + R(9) + w(3) + u(4)
    INPUT_DIM = 19
    # 输出特征: 3维 = 扰动加速度 (d_x, d_y, d_z) 在机体系下
    OUTPUT_DIM = 3          

# ==========================================
# 2. 数据集: 动态切片与划分 (Dataset with Slicing & Splitting)
# ==========================================
class DroneDisturbanceDataset(Dataset):
    def __init__(self, data_path, window_size=20, mode='train', split_ratio=0.9):
        """
        data_path: collect_data.py 保存的 .pt 文件路径
        window_size: TCN 需要的历史长度
        mode: 'train' (训练集) 或 'val' (验证集)
        split_ratio: 训练集所占比例 (0.0 ~ 1.0)
        """
        # 1. 加载数据
        if os.path.exists(data_path):
            loaded_data = torch.load(data_path)
            # 原始数据形状: [Num_Envs, Total_Steps, Features]
            self.inputs = loaded_data["inputs"].float() 
            self.labels = loaded_data["labels"].float() 
            print(f"[{mode.upper()}] Loaded raw data from {data_path}, Shape: {self.inputs.shape}")
        else:
            # 仅用于测试的假数据生成 (Dummy Data Generation)
            print(f"[{mode.upper()}] Warning: Data path {data_path} not found. Generating Dummy Data...")
            self.inputs = torch.randn(100, 1000, 19) 
            self.labels = torch.randn(100, 1000, 3)

        total_envs, self.total_len, self.feat_dim = self.inputs.shape
        self.window_size = window_size
        
        # 2. 计算每个环境能产生的有效样本数
        # 有效样本数 = 总长度 - 窗口长度 - 1 (预留一帧给 Next State 用于物理验证)
        self.samples_per_env = self.total_len - self.window_size - 1
        
        # 3. 划分训练集与验证集 (按 Environment ID 划分，防止数据泄露)
        split_idx = int(total_envs * split_ratio)
        
        if mode == 'train':
            # 训练集: 使用前 split_idx 个环境
            self.env_start_idx = 0
            self.num_envs = split_idx
        elif mode == 'val':
            # 验证集: 使用后 (total - split_idx) 个环境
            self.env_start_idx = split_idx
            self.num_envs = total_envs - split_idx
        else:
            raise ValueError("Mode must be 'train' or 'val'")
            
        # 计算该模式下的总样本数 (用于 __len__)
        self.total_samples = self.num_envs * self.samples_per_env
        print(f"[{mode.upper()}] Effective Envs: {self.num_envs}, Total Samples: {self.total_samples}")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # --- 核心切片逻辑 (Real-time Slicing) ---
        
        # 1. 将全局索引 idx 映射为 (局部环境ID, 时间偏移)
        local_env_id = idx // self.samples_per_env
        time_offset = idx % self.samples_per_env
        
        # 2. 转换为全局环境ID (加上偏移量)
        global_env_id = self.env_start_idx + local_env_id
        
        # 3. 计算时间窗口范围
        # start_t: 窗口开始
        # end_t: 窗口结束 (也是我们要预测的时刻 t)
        start_t = time_offset
        end_t = start_t + self.window_size 
        
        # 4. 提取数据
        # (A) 历史窗口输入 X: [Window, 19] -> t-W 到 t-1
        input_window = self.inputs[global_env_id, start_t:end_t, :]
        
        # (B) 真实扰动标签 Y (Ground Truth): [3] -> t-1 时刻受到的扰动
        # 注意: 我们用过去的信息预测当前帧(end_t-1)所受的扰动
        label_dist = self.labels[global_env_id, end_t-1, :]
        
        # (C) 物理验证数据
        # 当前状态 x_t (用于代入动力学方程): 窗口最后一帧
        state_curr = self.inputs[global_env_id, end_t-1, :]
        # 下一时刻真实状态 x_{t+1} (用于计算 Physics Loss): 窗口后一帧
        v_next_real = self.inputs[global_env_id, end_t, :3] # 取速度分量

        return input_window, label_dist, state_curr, v_next_real

# ==========================================
# 3. 网络模型: 因果 TCN (Causal PI-WAN)
# ==========================================
class CausalConv1d(nn.Module):
    """ 因果卷积层：确保 t 时刻的输出只依赖于 t 及其之前的输入，严禁利用未来信息 """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        # 计算 Padding: (K-1) * D
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size, 
            padding=self.padding, 
            dilation=dilation
        )

    def forward(self, x):
        # Conv1d 会输出 [Batch, Out_Ch, Length + Padding]
        # 我们必须切掉后面多余的 Padding，以保持时间维度对齐
        x = self.conv(x)
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x

class PI_WAN(nn.Module):
    def __init__(self, input_dim=19, output_dim=3, hidden_dim=64):
        super().__init__()
        
        # --- TCN Backbone (时序特征提取) ---
        # 使用膨胀卷积 (Dilation) 扩大感受野
        self.net = nn.Sequential(
            # Layer 1: 基础特征提取
            CausalConv1d(input_dim, hidden_dim, kernel_size=3, dilation=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            
            # Layer 2: 感受野扩大
            CausalConv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            
            # Layer 3: 进一步扩大
            CausalConv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            
            # Layer 4: 覆盖长时依赖
            CausalConv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=8),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
        )
        
        # --- Readout Head (回归预测) ---
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim) # 输出: [Fx_dist, Fy_dist, Fz_dist]
        )
        
        # [Trick] 初始化最后一层为极小值
        # 让网络初始预测接近 0，防止 Physics Loss 在训练初期爆炸
        nn.init.uniform_(self.head[-1].weight, -0.01, 0.01)
        nn.init.constant_(self.head[-1].bias, 0)

    def forward(self, x):
        # x input shape: [Batch, Window, Feat]
        # Conv1d 需要: [Batch, Feat, Window] -> Permute
        x = x.permute(0, 2, 1)
        
        # 提取时序特征 -> [Batch, Hidden, Window]
        feat = self.net(x)
        
        # 我们只关心“当前时刻”(窗口最后一步) 的预测结果
        feat_last = feat[:, :, -1] 
        
        # 回归预测
        out = self.head(feat_last)
        return out

# ==========================================
# 4. 物理引擎: 可微标称动力学 (Differentiable Nominal Dynamics)
# ==========================================
class QuadrotorDynamics(nn.Module):
    def __init__(self, mass, gravity, dt):
        super().__init__()
        self.mass = mass
        self.dt = dt
        self.max_thrust = max_thrust  
        # 注册重力向量 (假设世界系 Z 轴向上，重力为 -g)
        self.register_buffer('gravity_vec', torch.tensor([0., 0., -gravity]))

    def forward(self, state, disturbance_pred):
        """
        利用动力学方程计算下一时刻的预测速度 (用于 Physics Loss)
        
        Args:
            state: 当前状态 [Batch, 19] -> v(3), R(9), w(3), u(4)
            disturbance_pred: 预测的扰动加速度 [Batch, 3] (机体坐标系)
        Returns:
            v_next_pred: 预测的下一时刻机体速度 [Batch, 3]
        """
        batch_size = state.shape[0]
        
        # 1. 解包状态变量
        v_body = state[:, 0:3]     # 机体速度
        R_flat = state[:, 3:12]    # 旋转矩阵 (扁平化)
        w_body = state[:, 12:15]   # 机体角速度
        u_ctrl = state[:, 15:19]   # 控制输入 [Thrust_Norm, w_x, w_y, w_z]
        
        # 2. 恢复旋转矩阵形状 [Batch, 3, 3]
        # R 矩阵表示从 机体系 到 世界系 的旋转 (R_body_to_world)
        R = R_flat.view(batch_size, 3, 3)
        
        # # 3. 计算推力加速度 (Thrust Acceleration)
        # # 假设 u_ctrl[0] 是推力大小 (Force in Newtons)
        # # 如果 collect_data 中采集的是归一化油门，需在此处乘最大推力系数
        # thrust_mag = u_ctrl[:, 0] 
        # thrust_vec_body = torch.zeros(batch_size, 3, device=state.device)
        # thrust_vec_body[:, 2] = thrust_mag # 推力沿机体 Z 轴
        # a_thrust = thrust_vec_body / self.mass
        # g_body = R^T * g_world
        # R.transpose(1,2) 即 R^T
        # 4. 计算重力分量 (Gravity in Body Frame)
        # g_world_batch = self.gravity_vec.view(1, 3, 1).expand(batch_size, -1, -1)
        # g_body = torch.bmm(R.transpose(1, 2), g_world_batch).squeeze(-1)
        # 5. 计算科里奥利力项 (Coriolis Term: w x v)
        # 这一项在高机动飞行中不可忽略
        # a_coriolis = torch.cross(w_body, v_body, dim=1)
        
        # # 6. 动力学方程汇总 (机体系)
        # # dv/dt = (F_thrust/m) + g_body - (w x v) + d_pred
        # acc_total = a_thrust + g_body - a_coriolis + disturbance_pred
        
        # # 7. 欧拉积分 (Euler Integration)
        # v_next_pred = v_body + acc_total * self.dt

        # 1. 计算真实的物理推力 (Newtons)
        throttle_normalized = (u_ctrl[:, 0] + 1.0) / 2.0  # 映射到 [0, 1]
        thrust_mag = throttle_normalized * self.max_thrust # 真实推力 (N)
        
        thrust_vec_body = torch.zeros(batch_size, 3, device=state.device)
        thrust_vec_body[:, 2] = thrust_mag 
        a_thrust = thrust_vec_body / self.mass  # 推力加速度
        
        # 2. 计算重力 (机体系下)
        g_world_batch = self.gravity_vec.view(1, 3, 1).expand(batch_size, -1, -1)
        g_body = torch.bmm(R.transpose(1, 2), g_world_batch).squeeze(-1)
        
        # [修改点 2] 彻底删除了 a_drag 的计算。
        # 我们默认标称模型在“真空中”飞行，所有的气动阻力误差全被挤压给 lumped_disturbance_pred 去弥补
        # 3. 计算科里奥利力
        a_coriolis = torch.cross(w_body, v_body, dim=1)
        
        # 4. 总动力学方程: dv/dt = a_thrust + g_body - (w x v) + d_pred
        # 此时的 lumped_disturbance_pred 会自动逼近真实环境中的 (Wind + Drag)
        acc_total = a_thrust + g_body - a_coriolis + lumped_disturbance_pred
        v_next_pred = v_body + acc_total * self.dt

        
        return v_next_pred

# ==========================================
# 5. 训练主循环 (Training Pipeline)
# ==========================================
def train_pipeline():
    # --- 设备配置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. 生成时间戳 & 准备路径 [修改点] ---
    # 获取当前时间，格式: 月_日_小时_分钟 (例如: 03_08_15_30)
    current_time_str = datetime.datetime.now().strftime("%m_%d_%H_%M")
    # 构造 WandB 的 Run Name，方便云端对应
    run_name = f"{Config.WANDB_NAME_PREFIX}_{current_time_str}"
    
    # 构造保存目录: 当前脚本目录/pinncheckpoint/月_日_小时_分钟/
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_script_dir, "pinncheckpoint", current_time_str)
    
    # 创建文件夹
    os.makedirs(save_dir, exist_ok=True)
    print(f">>> [INFO] Checkpoints will be saved to: {save_dir}")
    # --- 2. WandB 初始化 ---
    config_dict = {k: v for k, v in Config.__dict__.items() if not k.startswith('__')}
    config_dict['save_dir'] = save_dir  # 把保存路径也记录到 wandb config 里
    
    wandb.init(
        project=Config.WANDB_PROJECT,
        name=run_name,   # 使用带时间戳的名字
        config=config_dict
    )


    # # ==========================
    # # 0. 准备保存路径 (新增/修改部分)
    # # ==========================
    # # 获取当前脚本文件的绝对路径目录
    # current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # # 拼接出 pinncheckpoint 文件夹的绝对路径
    # save_dir = os.path.join(current_script_dir, "pinncheckpoint")
    # # 如果文件夹不存在，则创建
    # os.makedirs(save_dir, exist_ok=True)
    # print(f">>> Checkpoints will be saved to: {save_dir}")

    # --- 1. 准备数据 ---
    # 使用 mode 参数分别加载训练集和验证集
    # 请确保路径指向 collect_data.py 生成的真实 .pt 文件
    data_file = "collected_data/pinn_dataset.pt" 
    
    train_dataset = DroneDisturbanceDataset(
        data_file, window_size=Config.WINDOW_SIZE, mode='train', split_ratio=Config.TRAIN_RATIO
    )
    val_dataset = DroneDisturbanceDataset(
        data_file, window_size=Config.WINDOW_SIZE, mode='val', split_ratio=Config.TRAIN_RATIO
    )
    
    # DataLoader 配置
    # Train: Shuffle=True (全局打乱，打破时间相关性)
    # Val: Shuffle=False (顺序验证即可)
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.VAL_BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )
    
    # --- 2. 初始化模型与组件 ---
    model = PI_WAN(input_dim=Config.INPUT_DIM, output_dim=Config.OUTPUT_DIM).to(device)
    dynamics = QuadrotorDynamics(
        mass=Config.MASS, 
        gravity=Config.GRAVITY, 
        dt=Config.DT,
        max_thrust=38.00861358642578  # [注意] 请填入你在终端打印出来的那个真实最大推力数值
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    mse_criterion = nn.MSELoss()
    
    # 用于保存最佳模型
    best_val_loss = float('inf')
    os.makedirs("checkpoints", exist_ok=True)
    # --- WandB 初始化 [新增 3] ---
    # 这里我们将 Config 的属性转为字典，传给 wandb 用于记录超参数
    config_dict = {k: v for k, v in Config.__dict__.items() if not k.startswith('__')}
    
    wandb.init(
        project=Config.WANDB_PROJECT,
        name=Config.WANDB_RUN_NAME,
        config=config_dict
    )
    
    # --- 设备配置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- 准备保存路径 ---
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_script_dir, "pinncheckpoint")
    os.makedirs(save_dir, exist_ok=True)
    print(f">>> Checkpoints will be saved to: {save_dir}")

    # --- 准备数据 ---
    data_file = "collected_data/pinn_dataset.pt" 
    train_dataset = DroneDisturbanceDataset(data_file, window_size=Config.WINDOW_SIZE, mode='train', split_ratio=Config.TRAIN_RATIO)
    val_dataset = DroneDisturbanceDataset(data_file, window_size=Config.WINDOW_SIZE, mode='val', split_ratio=Config.TRAIN_RATIO)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.VAL_BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    # --- 初始化模型 ---
    model = PI_WAN(input_dim=Config.INPUT_DIM, output_dim=Config.OUTPUT_DIM).to(device)
    dynamics = QuadrotorDynamics(Config.MASS, Config.GRAVITY, Config.DT).to(device)
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    mse_criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    # 也可以让 wandb 监控模型结构 (可选)
    wandb.watch(model, log="all")

    print(">>> Start Training Pipeline...")
    
    for epoch in range(Config.EPOCHS):
        # ==========================
        # Training Phase
        # ==========================
        model.train()
        train_loss_accum = 0.0
        
        for batch_idx, (x_window, label_dist, state_curr, v_next_real) in enumerate(train_loader):
            # 移动数据到 GPU
            x_window = x_window.to(device)
            label_dist = label_dist.to(device)
            state_curr = state_curr.to(device)
            v_next_real = v_next_real.to(device)
            
            # 1. 前向传播
            d_pred = model(x_window)
            
            # 2. 计算 Loss
            # (A) Data Loss: 监督学习，拟合 GT 扰动
            loss_data = mse_criterion(d_pred, label_dist)
            
            # (B) Physics Loss: 动力学约束验证
            # 预测速度 = 标称动力学(当前状态) + 预测扰动
            v_next_pred = dynamics(state_curr, d_pred)
            loss_phy = mse_criterion(v_next_pred, v_next_real)
            
            # (C) Total Loss
            loss = Config.LAMBDA_DATA * loss_data + Config.LAMBDA_PHY * loss_phy
            
            # 3. 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪 (防止 TCN 在训练初期梯度爆炸)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss_accum += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"[Epoch {epoch}][Batch {batch_idx}] Loss: {loss.item():.4f} "
                      f"(Data: {loss_data.item():.4f}, Phy: {loss_phy.item():.4f})")
        
        avg_train_loss = train_loss_accum / len(train_loader)
        # [新增 4] 记录训练 Log 到 WandB
        wandb.log({
            "Train/Total_Loss": avg_train_loss,
            "Train/Data_Loss": avg_train_data,
            "Train/Phy_Loss": avg_train_phy,
            "epoch": epoch
        })

        # ==========================
        # Validation Phase
        # ==========================
        model.eval()
        val_loss_accum = 0.0
        
        with torch.no_grad(): # 验证时不计算梯度
            for x_window, label_dist, state_curr, v_next_real in val_loader:
                x_window = x_window.to(device)
                label_dist = label_dist.to(device)
                state_curr = state_curr.to(device)
                v_next_real = v_next_real.to(device)
                
                # 前向传播
                d_pred = model(x_window)
                
                # 计算 Loss (仅用于评估)
                loss_data = mse_criterion(d_pred, label_dist)
                v_next_pred = dynamics(state_curr, d_pred)
                loss_phy = mse_criterion(v_next_pred, v_next_real)
                
                loss = Config.LAMBDA_DATA * loss_data + Config.LAMBDA_PHY * loss_phy
                val_loss_accum += loss.item()
        
        avg_val_loss = val_loss_accum / len(val_loader)
        
        # [新增 5] 记录验证 Log 到 WandB
        wandb.log({
            "Val/Total_Loss": avg_val_loss,
            "Val/Data_Loss": avg_val_data,
            "Val/Phy_Loss": avg_val_phy,
            "epoch": epoch
        })

        print(f"=== Epoch {epoch} Result ===")
        print(f"    Train Loss: {avg_train_loss:.5f}")
        print(f"    Val Loss  : {avg_val_loss:.5f}")
        
        # ==========================
        # Checkpoint Saving
        # ==========================
        # 只有当验证集 Loss 创新低时，才保存为最佳模型
        # 1. 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # 文件名带上 best 前缀
            best_model_path = os.path.join(save_dir, "best_pi_wan_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f">>> New Best Model Saved to: {best_model_path}")
            # torch.save(model.state_dict(), "checkpoints/best_pi_wan_model.pth")
            # print(">>> New Best Model Saved!")
        
        # 2. 定期保存 (每10个Epoch)
        if (epoch + 1) % 10 == 0:
            epoch_model_path = os.path.join(save_dir, f"pi_wan_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), epoch_model_path)
            print(f">>> Checkpoint saved: {epoch_model_path}")
            # torch.save(model.state_dict(), f"checkpoints/pi_wan_epoch_{epoch+1}.pth")
        # if avg_val_loss < best_val_loss:
        #     best_val_loss = avg_val_loss
        #     # 拼接完整文件路径
        #     best_model_path = os.path.join(save_dir, "best_pi_wan_model.pth")
        #     torch.save(model.state_dict(), best_model_path)
        #     print(f">>> New Best Model Saved to: {best_model_path}")
        #     # torch.save(model.state_dict(), "checkpoints/best_pi_wan_model.pth")
        #     # print(">>> New Best Model Saved!")
        
        # # 定期保存普通 Checkpoint
        # if (epoch + 1) % 10 == 0:
        #     epoch_model_path = os.path.join(save_dir, f"pi_wan_epoch_{epoch+1}.pth")
        #     torch.save(model.state_dict(), epoch_model_path)
        #     print(f">>> Checkpoint saved: {epoch_model_path}")
        #     # torch.save(model.state_dict(), f"checkpoints/pi_wan_epoch_{epoch+1}.pth")
        
    # [新增 6] 结束 WandB Run
        wandb.finish()
        print("Training Complete.")
if __name__ == "__main__":
    # 入口函数
    # 提示: 请先运行 collect_data.py 生成数据，再运行此脚本
    train_pipeline()