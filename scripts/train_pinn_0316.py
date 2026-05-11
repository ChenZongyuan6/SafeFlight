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
    # --- WandB 配置 ---
    WANDB_PROJECT = "PINN_DOB"  
    WANDB_NAME_PREFIX = "TCN_win5"  
    
    # --- 数据相关 ---
    # [建议] 缩短窗口到 5-10 之间，避免网络记住风的周期波形变成“算命机器”
    WINDOW_SIZE = 5        
    DT = 0.02               
    TRAIN_RATIO = 0.9       
    
    # --- 无人机物理参数 ---
    # [修改] 使用你打印出来的真实值
    MASS =  0.93          # kg
    GRAVITY = 9.81          # m/s^2
    MAX_THRUST = 38.0086    # [新增] 你的最大真实推力
    
    # --- 训练超参数 ---
    BATCH_SIZE = 256        
    VAL_BATCH_SIZE = 512    
    LR = 1e-3               
    EPOCHS = 50             
    
    # Loss 权重系数
    LAMBDA_PHY = 1.0        
    LAMBDA_DATA = 1.0       
    
    INPUT_DIM = 19
    OUTPUT_DIM = 3          

# ==========================================
# 2. 数据集: 动态切片与归一化 
# ==========================================
class DroneDisturbanceDataset(Dataset):
    def __init__(self, data_path, window_size=20, mode='train', split_ratio=0.9, stats=None):
        if os.path.exists(data_path):
            loaded_data = torch.load(data_path)
            self.inputs = loaded_data["inputs"].float() 
            self.labels = loaded_data["labels"].float() 
            print(f"[{mode.upper()}] Loaded raw data from {data_path}, Shape: {self.inputs.shape}")
        else:
            raise FileNotFoundError(f"找不到数据文件: {data_path}")

        total_envs, self.total_len, self.feat_dim = self.inputs.shape
        self.window_size = window_size
        self.samples_per_env = self.total_len - self.window_size - 1
        
        split_idx = int(total_envs * split_ratio)
        
        if mode == 'train':
            self.env_start_idx = 0
            self.num_envs = split_idx
            
            # [关键修复: 归一化] 计算训练集的均值和方差
            flat_inputs = self.inputs[0:split_idx].reshape(-1, self.feat_dim)
            self.mean = flat_inputs.mean(dim=0)
            self.std = flat_inputs.std(dim=0) + 1e-6 # 加 1e-6 防止除以 0
            print(f"[{mode.upper()}] Computed normalization stats.")
        elif mode == 'val':
            self.env_start_idx = split_idx
            self.num_envs = total_envs - split_idx
            
            # [关键修复: 归一化] 验证集必须使用训练集的均值和方差！
            assert stats is not None, "Validation set needs stats from train set!"
            self.mean = stats['mean']
            self.std = stats['std']
        else:
            raise ValueError("Mode must be 'train' or 'val'")
            
        # 执行归一化 (Z-Score)
        self.inputs = (self.inputs - self.mean) / self.std
        
        self.total_samples = self.num_envs * self.samples_per_env
        print(f"[{mode.upper()}] Effective Envs: {self.num_envs}, Total Samples: {self.total_samples}")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        local_env_id = idx // self.samples_per_env
        time_offset = idx % self.samples_per_env
        global_env_id = self.env_start_idx + local_env_id
        
        start_t = time_offset
        end_t = start_t + self.window_size 
        
        input_window = self.inputs[global_env_id, start_t:end_t, :]
        label_dist = self.labels[global_env_id, end_t-1, :]
        
        # 注意：用于物理引擎计算的 state_curr 必须是**未归一化**的原始物理值！
        # 所以我们要用均值和方差把它还原回去
        state_curr_normalized = self.inputs[global_env_id, end_t-1, :]
        state_curr_raw = state_curr_normalized * self.std + self.mean
        
        # 物理验证的目标速度也要还原回原始值
        v_next_normalized = self.inputs[global_env_id, end_t, :3]
        v_next_real_raw = v_next_normalized * self.std[:3] + self.mean[:3]

        return input_window, label_dist, state_curr_raw, v_next_real_raw

# ==========================================
# 3. 网络模型: (保持不变)
# ==========================================
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)

    def forward(self, x):
        x = self.conv(x)
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x

class PI_WAN(nn.Module):
    def __init__(self, input_dim=19, output_dim=3, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(input_dim, hidden_dim, kernel_size=3, dilation=1),
            nn.ReLU(), nn.BatchNorm1d(hidden_dim),
            CausalConv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=2),
            nn.ReLU(), nn.BatchNorm1d(hidden_dim),
            CausalConv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=4),
            nn.ReLU(), nn.BatchNorm1d(hidden_dim),
            CausalConv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=8),
            nn.ReLU(), nn.BatchNorm1d(hidden_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim) 
        )
        nn.init.uniform_(self.head[-1].weight, -0.01, 0.01)
        nn.init.constant_(self.head[-1].bias, 0)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        feat = self.net(x)
        feat_last = feat[:, :, -1] 
        out = self.head(feat_last)
        return out

# ==========================================
# 4. 物理引擎: 可微标称动力学 
# ==========================================
class QuadrotorDynamics(nn.Module):
    def __init__(self, mass, gravity, dt, max_thrust):
        super().__init__()
        self.mass = mass
        self.dt = dt
        self.max_thrust = max_thrust  
        self.register_buffer('gravity_vec', torch.tensor([0., 0., -gravity]))

    def forward(self, state, disturbance_pred):
        batch_size = state.shape[0]
        v_body = state[:, 0:3]     
        R_flat = state[:, 3:12]    
        w_body = state[:, 12:15]   
        u_ctrl = state[:, 15:19]   
        
        R = R_flat.view(batch_size, 3, 3)

        # 1. 计算真实的物理推力 (Newtons)
        throttle_normalized = (u_ctrl[:, 3] + 1.0) / 2.0  
        thrust_mag = throttle_normalized * self.max_thrust 
        
        thrust_vec_body = torch.zeros(batch_size, 3, device=state.device)
        thrust_vec_body[:, 2] = thrust_mag 
        a_thrust = thrust_vec_body / self.mass  
        
        # 2. 计算重力 (机体系下)
        g_world_batch = self.gravity_vec.view(1, 3, 1).expand(batch_size, -1, -1)
        g_body = torch.bmm(R.transpose(1, 2), g_world_batch).squeeze(-1)
        
        # 3. 计算科里奥利力
        a_coriolis = torch.cross(w_body, v_body, dim=1)
        
        # 4. 总动力学方程: dv/dt = a_thrust + g_body - (w x v) + d_pred
        acc_total = a_thrust + g_body - a_coriolis + disturbance_pred
        v_next_pred = v_body + acc_total * self.dt
        
        return v_next_pred

# ==========================================
# 5. 训练主循环
# ==========================================
def train_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. 生成时间戳 & 准备路径 ---
    current_time_str = datetime.datetime.now().strftime("%m_%d_%H_%M")
    run_name = f"{Config.WANDB_NAME_PREFIX}_{current_time_str}"
    
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_script_dir, "pinncheckpoint", current_time_str)
    os.makedirs(save_dir, exist_ok=True)
    print(f">>> [INFO] Checkpoints will be saved to: {save_dir}")
    
    # --- 2. WandB 初始化 ---
    config_dict = {k: v for k, v in Config.__dict__.items() if not k.startswith('__')}
    config_dict['save_dir'] = save_dir  
    
    # wandb.init(project=Config.WANDB_PROJECT, name=run_name, config=config_dict)
    # [强烈建议的改进方案]
    wandb.init(
        project=Config.WANDB_PROJECT, 
        name=run_name,               # 依然使用带时间戳的名称保证唯一性
        group="TCN",   # 【新增】实验分组：比如这批实验都在测窗口长度为5的基线
        tags=["TCN", "Window_5"], # 【新增】标签：方便以后在网页端用过滤器筛选
        notes="修复了物理公式质量的Bug，将窗口从20缩短为5，测试收敛性。20260320发现202603160056效果最好，重新尝试复现", # 【新增】一句话备忘录
        config=config_dict
    )

    # --- 3. 准备数据 ---
    # [注意] 请确保这里的路径是你最新采集的数据集路径！
    data_file = "collected_data/dataset_20260316_0028/dataset_20260316_0028.pt" 
    
    
    train_dataset = DroneDisturbanceDataset(data_file, window_size=Config.WINDOW_SIZE, mode='train', split_ratio=Config.TRAIN_RATIO)
    
    # 获取训练集的统计数据，传给验证集
    train_stats = {'mean': train_dataset.mean, 'std': train_dataset.std}
    val_dataset = DroneDisturbanceDataset(data_file, window_size=Config.WINDOW_SIZE, mode='val', split_ratio=Config.TRAIN_RATIO, stats=train_stats)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.VAL_BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    # --- 4. 初始化模型 ---
    model = PI_WAN(input_dim=Config.INPUT_DIM, output_dim=Config.OUTPUT_DIM).to(device)
    dynamics = QuadrotorDynamics(mass=Config.MASS, gravity=Config.GRAVITY, dt=Config.DT, max_thrust=Config.MAX_THRUST).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    mse_criterion = nn.MSELoss()
    
    best_val_loss = float('inf')

    print(">>> Start Training Pipeline...")
    
    for epoch in range(Config.EPOCHS):
        # ==========================
        # Training Phase
        # ==========================
        model.train()
        train_loss_total = 0.0
        train_loss_data = 0.0
        train_loss_phy = 0.0
        
        for batch_idx, (x_window, label_dist, state_curr, v_next_real) in enumerate(train_loader):
            x_window = x_window.to(device)
            label_dist = label_dist.to(device)
            state_curr = state_curr.to(device)
            v_next_real = v_next_real.to(device)
            
            d_pred = model(x_window)
            
            loss_data = mse_criterion(d_pred, label_dist)
            v_next_pred = dynamics(state_curr, d_pred)
            loss_phy = mse_criterion(v_next_pred, v_next_real)
            
            loss = Config.LAMBDA_DATA * loss_data + Config.LAMBDA_PHY * loss_phy
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # [修复 2] 正确累加分类 Loss
            train_loss_total += loss.item()
            train_loss_data += loss_data.item()
            train_loss_phy += loss_phy.item()
            
            if batch_idx % 100 == 0:
                print(f"[Epoch {epoch}][Batch {batch_idx}] Loss: {loss.item():.4f} "
                      f"(Data: {loss_data.item():.4f}, Phy: {loss_phy.item():.4f})")
        
        avg_train_loss = train_loss_total / len(train_loader)
        avg_train_data = train_loss_data / len(train_loader)
        avg_train_phy = train_loss_phy / len(train_loader)
        
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
        val_loss_total = 0.0
        val_loss_data = 0.0
        val_loss_phy = 0.0
        
        with torch.no_grad():
            for x_window, label_dist, state_curr, v_next_real in val_loader:
                x_window = x_window.to(device)
                label_dist = label_dist.to(device)
                state_curr = state_curr.to(device)
                v_next_real = v_next_real.to(device)
                
                d_pred = model(x_window)
                
                loss_data = mse_criterion(d_pred, label_dist)
                v_next_pred = dynamics(state_curr, d_pred)
                loss_phy = mse_criterion(v_next_pred, v_next_real)
                
                loss = Config.LAMBDA_DATA * loss_data + Config.LAMBDA_PHY * loss_phy
                
                val_loss_total += loss.item()
                val_loss_data += loss_data.item()
                val_loss_phy += loss_phy.item()
        
        avg_val_loss = val_loss_total / len(val_loader)
        avg_val_data = val_loss_data / len(val_loader)
        avg_val_phy = val_loss_phy / len(val_loader)
        
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
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(save_dir, "best_pi_wan_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f">>> New Best Model Saved to: {best_model_path}")
        
        if (epoch + 1) % 10 == 0:
            epoch_model_path = os.path.join(save_dir, f"pi_wan_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), epoch_model_path)
            print(f">>> Checkpoint saved: {epoch_model_path}")
            
    # [修复 1] 将 finish 移出 for 循环，确保在全部训练结束后执行
    wandb.finish()
    print("Training Complete.")

if __name__ == "__main__":
    train_pipeline()