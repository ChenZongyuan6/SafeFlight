"""
eval_residual_6.py
==================
验证"冻结基础策略 + 冻结 PINN 扰动观测器 + 残差补偿策略"的闭环表现。
与 eval_residual.py 的区别：一次性评估 6 种轨迹（slow/normal/fast/poly/zigzag/pentagram），
并将各轨迹结果分别上报 wandb，对齐 eval_datt.py 的风格。
num_envs 固定为 64（避免显存 OOM）。

使用方式：
    cd /root/SimpleFlight/scripts
    python eval_residual_6.py headless=true task=TrackResidual \
        task.base_model_dir=... task.pinn_model_dir=... residual_model_dir=...
"""

import logging
import os
import copy
from typing import Sequence, Optional

import hydra
import torch
import torch.nn as nn
import numpy as np
import wandb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from omegaconf import OmegaConf
from tensordict import TensorDict, TensorDictBase

from omni_drones import CONFIG_PATH, init_simulation_app
from omni_drones.utils.torchrl import SyncDataCollector, AgentSpec
from omni_drones.utils.torchrl.transforms import (
    FromMultiDiscreteAction,
    FromDiscreteAction,
    ravel_composite,
    History,
)
from omni_drones.utils.wandb import init_wandb
from omni_drones.learning import (
    MAPPOPolicy,
    HAPPOPolicy,
    QMIXPolicy,
    DQNPolicy,
    SACPolicy,
    TD3Policy,
    MATD3Policy,
    TDMPCPolicy,
    Policy,
    PPOPolicy,
    PPOAdaptivePolicy,
    PPORNNPolicy,
)

from setproctitle import setproctitle
from torchrl.envs.transforms import (
    TransformedEnv,
    InitTracker,
    Compose,
)

from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec
from tqdm import tqdm


# ============================================================
# 所有支持的轨迹类型
# ============================================================
ALL_TRAJ_TYPES = ["slow", "normal", "fast", "poly", "zigzag", "pentagram"]


# ============================================================
# 1. 通用工具（和 train_residual.py 保持一致）
# ============================================================

class Every:
    def __init__(self, func, steps):
        self.func = func
        self.steps = steps
        self.i = 0

    def __call__(self, *args, **kwargs):
        if self.i % self.steps == 0:
            self.func(*args, **kwargs)
        self.i += 1


class EpisodeStats:
    def __init__(self, in_keys: Sequence[str] = None):
        self.in_keys = in_keys
        self._stats = []
        self._episodes = 0

    def __call__(self, tensordict: TensorDictBase) -> TensorDictBase:
        done = tensordict.get(("next", "done"))
        truncated = tensordict.get(("next", "truncated"), None)
        done_or_truncated = (done | truncated) if truncated is not None else done.clone()
        if done_or_truncated.any():
            done_or_truncated = done_or_truncated.squeeze(-1)
            self._episodes += done_or_truncated.sum().item()
            self._stats.extend(
                tensordict.select(*self.in_keys)[:, 1:][done_or_truncated[:, :-1]].clone().unbind(0)
            )

    def pop(self):
        stats: TensorDictBase = torch.stack(self._stats).to_tensordict()
        self._stats.clear()
        return stats

    def __len__(self):
        return len(self._stats)


# ============================================================
# 2. PINN 网络定义（保持与 train_residual.py 一致）
# ============================================================

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation,
        )

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
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        feat = self.net(x)
        feat_last = feat[:, :, -1]
        out = self.head(feat_last)
        return out


# ============================================================
# 3. 读取 PINN 归一化统计量
# ============================================================

def load_pinn_stats_from_dataset(data_path: str, train_ratio: float = 0.9, device: str = "cuda"):
    loaded = torch.load(data_path)
    inputs = loaded["inputs"].float()
    split_idx = int(inputs.shape[0] * train_ratio)
    flat_train_inputs = inputs[0:split_idx].reshape(-1, inputs.shape[-1])
    mean = flat_train_inputs.mean(dim=0).to(device)
    std = flat_train_inputs.std(dim=0).to(device) + 1e-6
    return mean, std


# ============================================================
# 4. 构造虚拟 base policy spec（和 train_residual.py 一致）
# ============================================================

class _VirtualEnvForAgentSpec:
    def __init__(self, observation_spec, action_spec, reward_spec):
        self.observation_spec = observation_spec
        self.action_spec = action_spec
        self.reward_spec = reward_spec


def build_virtual_base_agent_spec(
    num_envs: int,
    obs_dim: int,
    state_dim: int,
    action_dim: int,
    device: str = "cuda",
):
    observation_spec = CompositeSpec({
        "agents": {
            "observation": UnboundedContinuousTensorSpec((1, obs_dim)),
            "state": UnboundedContinuousTensorSpec((state_dim,)),
        }
    }).expand(num_envs).to(device)

    action_spec = CompositeSpec({
        "agents": {
            "action": UnboundedContinuousTensorSpec((1, action_dim)),
        }
    }).expand(num_envs).to(device)

    reward_spec = CompositeSpec({
        "agents": {
            "reward": UnboundedContinuousTensorSpec((1, 1)),
        }
    }).expand(num_envs).to(device)

    agent_spec = AgentSpec(
        "drone", 1,
        observation_key=("agents", "observation"),
        action_key=("agents", "action"),
        reward_key=("agents", "reward"),
        state_key=("agents", "state"),
    )

    virtual_env = _VirtualEnvForAgentSpec(
        observation_spec=observation_spec,
        action_spec=action_spec,
        reward_spec=reward_spec,
    )
    agent_spec._env = virtual_env
    agent_spec.env = virtual_env

    return agent_spec


# ============================================================
# 5. FrozenBaseAndPinnResidualPolicy（和 train_residual.py 一致）
# ============================================================

class FrozenBaseAndPinnResidualPolicy:
    def __init__(
        self,
        base_policy,
        residual_policy,
        pinn_model,
        pinn_mean: torch.Tensor,
        pinn_std: torch.Tensor,
        num_envs: int,
        pinn_window_size: int,
        device: str = "cuda",
        disturbance_input: str = "pinn",
    ):
        self.base_policy = base_policy
        self.residual_policy = residual_policy
        self.pinn_model = pinn_model
        self.pinn_mean = pinn_mean
        self.pinn_std = pinn_std
        self.disturbance_input = disturbance_input
        self.num_envs = num_envs
        self.pinn_window_size = pinn_window_size
        self.device = device

        if pinn_mean is not None:
            self.pinn_feature_history = torch.zeros(
                num_envs, pinn_window_size, pinn_mean.numel(), device=device
            )
        else:
            self.pinn_feature_history = None
        self.history_initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)

    @torch.no_grad()
    def reset_history(self, env_ids: Optional[torch.Tensor] = None):
        if self.pinn_feature_history is None:
            if env_ids is None:
                self.history_initialized[:] = False
            else:
                self.history_initialized[env_ids] = False
            return
        if env_ids is None:
            self.pinn_feature_history.zero_()
            self.history_initialized[:] = False
        else:
            self.pinn_feature_history[env_ids] = 0.0
            self.history_initialized[env_ids] = False

    def _get_is_init_mask(self, td: TensorDictBase):
        is_init = None
        if "is_init" in td.keys():
            is_init = td.get("is_init")
        elif ("is_init",) in td.keys(True):
            is_init = td.get(("is_init",))
        elif ("next", "is_init") in td.keys(True):
            is_init = td.get(("next", "is_init"))
        if is_init is None:
            return None
        while is_init.ndim > 1:
            is_init = is_init.squeeze(-1)
        return is_init.bool()

    def _extract_base_obs_and_state(self, td: TensorDictBase):
        info = td["info"]
        if "base_obs" not in info.keys() or "base_state" not in info.keys():
            raise KeyError(
                "[ERROR] info 中缺少 'base_obs' / 'base_state'。"
            )
        return info["base_obs"], info["base_state"]

    def _build_base_policy_td(self, td: TensorDictBase):
        base_obs, base_state = self._extract_base_obs_and_state(td)
        base_td = TensorDict(
            {
                "agents": {
                    "observation": base_obs,
                    "state": base_state,
                }
            },
            batch_size=td.batch_size,
            device=self.device,
        )
        return base_td

    def _build_current_pinn_feature(self, v_body, w_body, R_flat, u_base):
        # [WARNING] 分布不匹配: PINN 训练数据集的 action 槽存的是电机转速指令 (motor_cmd),
        # 但此处 u_base = tanh(base_policy) 是 CTBR 输出 [rate(3), thrust(1)], 两者分布完全不同。
        # 若要使 disturbance_input="pinn" 模式正确工作, 需重采集数据保存 ("info","policy_action")
        # 并重新训练 PINN。当前使用 disturbance_input="gt" 模式, 此分支不会被调用。
        u_base_flat = u_base.squeeze(1)
        feat = torch.cat([v_body, w_body, R_flat, u_base_flat], dim=-1)
        return feat

    def _update_pinn_window(self, current_feat, is_init_mask):
        num_envs = current_feat.shape[0]
        never_init_mask = ~self.history_initialized
        if never_init_mask.any():
            self.pinn_feature_history[never_init_mask] = current_feat[never_init_mask].unsqueeze(1).repeat(
                1, self.pinn_window_size, 1
            )
            self.history_initialized[never_init_mask] = True

        if is_init_mask is not None and is_init_mask.any():
            self.pinn_feature_history[is_init_mask] = current_feat[is_init_mask].unsqueeze(1).repeat(
                1, self.pinn_window_size, 1
            )
            self.history_initialized[is_init_mask] = True

        normal_mask = torch.ones(num_envs, dtype=torch.bool, device=self.device)
        if is_init_mask is not None:
            normal_mask = ~is_init_mask

        if normal_mask.any():
            self.pinn_feature_history[normal_mask, :-1] = self.pinn_feature_history[normal_mask, 1:].clone()
            self.pinn_feature_history[normal_mask, -1] = current_feat[normal_mask]

    def _predict_disturbance(self, current_feat, is_init_mask):
        self._update_pinn_window(current_feat, is_init_mask)
        x_window = self.pinn_feature_history
        x_window_norm = (x_window - self.pinn_mean.view(1, 1, -1)) / self.pinn_std.view(1, 1, -1)
        a_hat = self.pinn_model(x_window_norm)
        return a_hat

    def _build_comp_obs(self, a_hat, u_base, v_body, w_body, R_flat, rpos_body, prev_rpos0_body, vel_error_body):
        """输出46维: [a_hat(3), u_base(4), v_body(3), w_body(3), R_flat(9), rpos_flat(S*3), prev_rpos0(3), vel_error(3)]"""
        u_base_flat = u_base.squeeze(1)
        rpos_flat = rpos_body.flatten(1)  # [N, S*3]
        comp_obs = torch.cat([
            a_hat, u_base_flat, v_body, w_body, R_flat,
            rpos_flat, prev_rpos0_body, vel_error_body
        ], dim=-1)
        return comp_obs

    @torch.no_grad()
    def __call__(self, td: TensorDictBase, deterministic: bool = False):
        # (1) 主策略推理，tanh 对齐训练时的输出范围
        base_td = self._build_base_policy_td(td)
        base_out = self.base_policy(base_td, deterministic=True)
        u_base = torch.tanh(base_out[("agents", "action")]).detach()  # [N,1,4] CTBR [-1,1]

        # (2) 从环境 info 里取所需量
        info = td["info"]
        v_body          = info["v_body"]           # [N, 3]
        w_body          = info["w_body"]           # [N, 3]
        R_flat          = info["R_flat"]           # [N, 9]
        rpos_steps      = info["rpos_steps"]       # [N, S, 3]
        prev_rpos0      = info["prev_rpos0"]       # [N, 3]
        vel_error_body  = info["vel_error_body"]   # [N, 3]

        # (3) 扰动估计：gt 模式直接读真值，pinn 模式走 PINN
        is_init_mask = self._get_is_init_mask(td)
        if self.disturbance_input == "gt":
            a_hat = info["gt_disturbance"].detach()  # [N, 3]
        else:
            current_feat = self._build_current_pinn_feature(v_body, w_body, R_flat, u_base)
            a_hat = self._predict_disturbance(current_feat, is_init_mask)  # [N, 3]

        # (4) 构造 46 维补偿策略 observation
        comp_obs = self._build_comp_obs(
            a_hat, u_base, v_body, w_body, R_flat,
            rpos_steps, prev_rpos0, vel_error_body
        )

        td[("agents", "observation")] = comp_obs.unsqueeze(1)  # [N, 1, 46]
        td[("agents", "state")] = comp_obs                     # [N, 46]
        td[("info", "base_action")] = u_base
        td[("info", "pred_disturbance")] = a_hat

        # (5) 调 residual policy
        td_out = self.residual_policy(td, deterministic=deterministic)
        return td_out


# ============================================================
# 6. 主流程
# ============================================================

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train_residual")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    # 将 num_envs 固定为 64，避免 eval 时显存 OOM（与 eval_datt.py 保持一致）
    eval_num_envs = cfg.get("eval_num_envs", 64)
    if cfg.task.env.num_envs != eval_num_envs:
        print(f"[eval_residual_6] num_envs: {cfg.task.env.num_envs} -> {eval_num_envs} (set eval_num_envs=N to change)")
        cfg.task.env.num_envs = eval_num_envs
        # OmegaConf.resolve 后 cfg.env 是 ${task.env} 展开的独立节点，必须同步修改
        if hasattr(cfg, "env") and hasattr(cfg.env, "num_envs"):
            cfg.env.num_envs = eval_num_envs

    simulation_app = init_simulation_app(cfg)
    run = init_wandb(cfg)
    setproctitle(run.name)
    print(OmegaConf.to_yaml(cfg))

    algos = {
        "ppo": PPOPolicy,
        "ppo_adaptive": PPOAdaptivePolicy,
        "ppo_rnn": PPORNNPolicy,
        "mappo": MAPPOPolicy,
        "happo": HAPPOPolicy,
        "qmix": QMIXPolicy,
        "dqn": DQNPolicy,
        "sac": SACPolicy,
        "td3": TD3Policy,
        "matd3": MATD3Policy,
        "tdmpc": TDMPCPolicy,
        "test": Policy,
    }

    from omni_drones.envs.isaac_env import IsaacEnv

    # --------------------------------------------------------
    # 创建 TrackResidual 环境
    # --------------------------------------------------------
    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)

    transforms = [InitTracker()]

    if cfg.task.get("flatten_obs", False):
        transforms.append(ravel_composite(base_env.observation_spec, ("agents", "observation")))
    if cfg.task.get("flatten_state", False):
        transforms.append(ravel_composite(base_env.observation_spec, ("agents", "state")))
    if (
        cfg.task.get("flatten_intrinsics", True)
        and ("agents", "intrinsics") in base_env.observation_spec.keys(True)
    ):
        transforms.append(ravel_composite(base_env.observation_spec, ("agents", "intrinsics"), start_dim=-1))

    if cfg.task.get("history", False):
        transforms.append(History([("agents", "observation")], steps=4))

    env = TransformedEnv(base_env, Compose(*transforms)).train()
    env.set_seed(cfg.seed)

    # --------------------------------------------------------
    # 创建 residual policy（自动从 checkpoint 推断网络大小）
    # --------------------------------------------------------
    residual_agent_spec: AgentSpec = env.agent_spec["drone"]

    residual_model_dir = cfg.get("residual_model_dir", None) or cfg.get("model_dir", None)
    if residual_model_dir is None:
        residual_model_dir = "/root/SimpleFlight/scripts/outputs/track_residual/04-16_01-48/wandb/latest-run/files/checkpoint_629407744.pt"

    # 从 residual_model_dir 向上找 track_residual/日期 这一级，作为可视化输出目录
    _parts = residual_model_dir.replace("\\", "/").split("/")
    vis_dir = os.path.dirname(residual_model_dir)  # fallback
    for _i, _p in enumerate(_parts):
        if _p == "track_residual" and _i + 1 < len(_parts):
            vis_dir = "/".join(_parts[:_i + 2])
            break
    os.makedirs(vis_dir, exist_ok=True)
    print(f"[vis] Visualization output dir: {vis_dir}")

    # 从 checkpoint 自动推断 hidden_units
    # MAPPO 保存的是 actor_params(TensorDictParams) + critic(state_dict)
    # 用 critic state_dict 的字符串 key 来推断更可靠
    from omegaconf import open_dict
    _ckpt_residual = torch.load(residual_model_dir, map_location="cuda")
    _critic_sd = _ckpt_residual.get("critic", {})
    # layers.0 是 Linear（2D），layers.2/5/8 是 LayerNorm（1D）
    # 必须只数 2D 权重，否则 6 个有 .weight 的层 → max(6-1,2)=5 层（错误）
    _first_w = next(
        (v for k, v in _critic_sd.items()
         if "base.1" in k and "layers.0.weight" in k
         and hasattr(v, "shape") and v.ndim == 2),
        None,
    )
    if _first_w is not None:
        # 只统计 2D (Linear) 权重，跳过 1D LayerNorm 权重
        _n_layers = sum(
            1 for k, v in _critic_sd.items()
            if k.endswith(".weight") and "layers" in k and "base.1" in k
            and hasattr(v, "shape") and v.ndim == 2
        )
        # _n_layers 就是 hidden 层数，直接用（不减 1，因为 v_out 是独立的）
        _residual_hidden = [_first_w.shape[0]] * _n_layers
    else:
        _residual_hidden = list(cfg.algo.actor.get("hidden_units", [256, 256, 256]))

    with open_dict(cfg):
        _saved_actor_h = cfg.algo.actor.get("hidden_units", None)
        _saved_critic_h = cfg.algo.critic.get("hidden_units", None)
        cfg.algo.actor.hidden_units = _residual_hidden
        cfg.algo.critic.hidden_units = _residual_hidden
        # 训练时 train_residual.py 显式设 tanh=True（使用 TanhIndependentNormalModule）
        # eval 时必须保持一致，否则 actor 结构不同（fc_mean vs operator）
        cfg.algo.actor.tanh = True
    residual_policy = algos[cfg.algo.name.lower()](cfg.algo, agent_spec=residual_agent_spec, device="cuda")
    with open_dict(cfg):
        if _saved_actor_h is not None:
            cfg.algo.actor.hidden_units = _saved_actor_h
        if _saved_critic_h is not None:
            cfg.algo.critic.hidden_units = _saved_critic_h
        cfg.algo.actor.tanh = False  # 恢复，base_policy 使用 tanh=False

    print("===== Residual policy spec =====")
    print(residual_agent_spec.observation_spec)
    print(residual_agent_spec.action_spec)
    print(f"  [auto-detected hidden_units: {_residual_hidden}]")
    print("================================")

    print(f"[INFO] Loading residual policy from: {residual_model_dir}")
    residual_policy.load_state_dict(_ckpt_residual)

    if hasattr(residual_policy, "actor"):
        residual_policy.actor.eval()
    if hasattr(residual_policy, "critic"):
        residual_policy.critic.eval()

    # --------------------------------------------------------
    # 创建冻结主策略
    # --------------------------------------------------------
    if not hasattr(base_env, "base_obs_dim") or not hasattr(base_env, "base_state_dim"):
        raise RuntimeError(
            "当前 TrackResidual 环境中没有 base_obs_dim / base_state_dim。"
        )

    base_obs_dim = int(base_env.base_obs_dim)
    base_state_dim = int(base_env.base_state_dim)
    base_action_dim = 4

    print("===== Base policy virtual spec dims =====")
    print("base_obs_dim:", base_obs_dim)
    print("base_state_dim:", base_state_dim)
    print("base_action_dim:", base_action_dim)
    print("=========================================")

    base_agent_spec = build_virtual_base_agent_spec(
        num_envs=env.num_envs,
        obs_dim=base_obs_dim,
        state_dim=base_state_dim,
        action_dim=base_action_dim,
        device="cuda",
    )

    with open_dict(cfg):
        _saved_actor_h2 = cfg.algo.actor.get("hidden_units", None)
        _saved_critic_h2 = cfg.algo.critic.get("hidden_units", None)
        cfg.algo.actor.hidden_units = [256, 256, 256]
        cfg.algo.critic.hidden_units = [256, 256, 256]
    base_policy = algos[cfg.algo.name.lower()](cfg.algo, agent_spec=base_agent_spec, device="cuda")
    with open_dict(cfg):
        if _saved_actor_h2 is not None:
            cfg.algo.actor.hidden_units = _saved_actor_h2
        if _saved_critic_h2 is not None:
            cfg.algo.critic.hidden_units = _saved_critic_h2

    if cfg.task.get("base_model_dir", None) is None:
        raise ValueError("cfg.task.base_model_dir 不能为空。")

    print(f"[INFO] Loading frozen base policy from: {cfg.task.base_model_dir}")
    base_policy.load_state_dict(torch.load(cfg.task.base_model_dir, map_location="cuda"))

    if hasattr(base_policy, "actor"):
        base_policy.actor.eval()
        for p in base_policy.actor.parameters():
            p.requires_grad = False
    if hasattr(base_policy, "critic"):
        base_policy.critic.eval()
        for p in base_policy.critic.parameters():
            p.requires_grad = False

    # --------------------------------------------------------
    # 加载冻结 PINN
    # --------------------------------------------------------
    if cfg.task.get("pinn_model_dir", None) is None:
        raise ValueError("cfg.task.pinn_model_dir 不能为空。")

    pinn_model = PI_WAN(
        input_dim=int(cfg.task.pinn_input_dim),
        output_dim=int(cfg.task.pinn_output_dim),
        hidden_dim=int(cfg.task.pinn_hidden_dim),
    ).to("cuda")

    print(f"[INFO] Loading frozen PINN from: {cfg.task.pinn_model_dir}")
    pinn_model.load_state_dict(torch.load(cfg.task.pinn_model_dir, map_location="cuda"))
    pinn_model.eval()
    for p in pinn_model.parameters():
        p.requires_grad = False

    # --------------------------------------------------------
    # 读取 PINN 归一化统计量
    # --------------------------------------------------------
    if cfg.task.get("pinn_stats_dataset", None) is None:
        raise ValueError("cfg.task.pinn_stats_dataset 不能为空。")

    pinn_mean, pinn_std = load_pinn_stats_from_dataset(
        data_path=cfg.task.pinn_stats_dataset,
        train_ratio=float(cfg.task.get("pinn_train_ratio", 0.9)),
        device="cuda",
    )

    # --------------------------------------------------------
    # 构造"冻结主策略 + 冻结 PINN + 残差补偿策略"编排器
    # --------------------------------------------------------
    _disturbance_input = str(cfg.task.get("disturbance_input", "pinn")).lower()
    # gt 模式：让环境暴露真实扰动，并跳过 PINN 窗口
    if _disturbance_input == "gt":
        with open_dict(cfg):
            cfg.task.expose_gt_disturbance = True
        _composed_pinn_mean = None
        _composed_pinn_std = None
    else:
        _composed_pinn_mean = pinn_mean
        _composed_pinn_std = pinn_std
    composed_policy = FrozenBaseAndPinnResidualPolicy(
        base_policy=base_policy,
        residual_policy=residual_policy,
        pinn_model=pinn_model,
        pinn_mean=_composed_pinn_mean,
        pinn_std=_composed_pinn_std,
        num_envs=env.num_envs,
        pinn_window_size=int(cfg.task.pinn_window_size),
        device="cuda",
        disturbance_input=_disturbance_input,
    )

    # --------------------------------------------------------
    # 评估函数：每次评估一条轨迹（对齐 eval_datt.py 风格）
    # --------------------------------------------------------
    @torch.no_grad()
    def evaluate(traj_type: str, seed: int = 0):
        """Run one deterministic episode on the given trajectory type."""
        base_env.eval_traj = traj_type
        base_env._apply_eval_traj()
        # [20260506] 强制覆盖，避免 hydra struct 模式导致 cfg.task.get 读不到 eval_no_reset=true
        base_env.eval_no_reset = True

        frames = []
        base_env.enable_render(True)
        base_env.eval()
        env.eval()
        env.set_seed(seed)

        composed_policy.reset_history()

        t = tqdm(total=base_env.max_episode_length, desc=traj_type)

        def record_frame(*args, **kwargs):
            frame = env.base_env.render(mode="rgb_array")
            frames.append(frame)
            t.update(2)

        trajs = env.rollout(
            max_steps=base_env.max_episode_length,
            policy=lambda x: composed_policy(x, deterministic=True),
            callback=Every(record_frame, 2),
            auto_reset=True,
            break_when_any_done=False,
            return_contiguous=False,
        ).clone()
        t.close()

        base_env.enable_render(not cfg.headless)
        env.reset()

        done = trajs.get(("next", "done"))
        first_done = torch.argmax(done.long(), dim=1).cpu()

        def take_first_episode(tensor: torch.Tensor):
            indices = first_done.reshape(first_done.shape + (1,) * (tensor.ndim - 2))
            return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)

        traj_stats = {
            k: take_first_episode(v)
            for k, v in trajs[("next", "stats")].cpu().items()
        }
        # ---- Disturbance vs Compensation visualization ----
        try:
            _ep_len = int(first_done[0].item()) + 1
            _gt_dist = (
                trajs[("next", "info", "gt_disturbance")][0, :_ep_len]
                .float().cpu().numpy().reshape(_ep_len, 3)
            )  # [T, 3] body-frame disturbance
            _comp_act = (
                trajs[("next", "info", "comp_action")][0, :_ep_len]
                .float().cpu().numpy().reshape(_ep_len, 4)
            )  # [T, 4]  CTBR compensation

            def _norm_peak(arr):
                m = float(np.abs(arr).max())
                return arr / (m + 1e-8)

            T_steps = np.arange(_ep_len)
            fig, axes = plt.subplots(3, 1, figsize=(14, 9))
            fig.suptitle(
                f"Disturbance vs Compensation  [{traj_type}]  (env 0, ep 0)",
                fontsize=13
            )

            _channels = [
                ("Z-axis: dist_body[2]  vs  comp_action[3] (thrust)",
                 _gt_dist[:, 2], _comp_act[:, 3]),
                ("X-axis: dist_body[0]  vs  comp_action[1] (pitch_rate)",
                 _gt_dist[:, 0], _comp_act[:, 1]),
                ("Y-axis: dist_body[1]  vs  comp_action[0] (roll_rate)",
                 _gt_dist[:, 1], _comp_act[:, 0]),
            ]

            for ax, (title, dist_ch, comp_ch) in zip(axes, _channels):
                ax.plot(T_steps, _norm_peak(dist_ch),
                        label="gt_disturbance (norm)",
                        color="tab:blue", linewidth=1.2)
                ax.plot(T_steps, _norm_peak(comp_ch),
                        label="comp_action (norm)",
                        color="tab:orange", linewidth=1.2, alpha=0.85)
                ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
                ax.set_title(title, fontsize=10)
                ax.legend(fontsize=8, loc="upper right")
                ax.set_xlabel("step", fontsize=8)
                ax.set_ylabel("normalized amplitude", fontsize=8)
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            _save_path = os.path.join(vis_dir, f"disturbance_comp_{traj_type}.png")
            fig.savefig(_save_path, dpi=120)
            plt.close(fig)
            print(f"[vis] Saved: {_save_path}")
        except Exception as _vis_err:
            print(f"[vis] Warning: visualization failed for {traj_type}: {_vis_err}")
        # ---- end visualization ----

        # traj_stats 已全部移到 CPU，立即释放 GPU 上的大型 rollout tensor
        del trajs, done, first_done
        torch.cuda.empty_cache()

        info = {
            f"eval/{traj_type}/stats.{k}": torch.nanmean(v.float()).item()
            for k, v in traj_stats.items()
        }

        if len(frames):
            video_array = np.stack(frames).transpose(0, 3, 1, 2)
            frames.clear()
            info[f"eval/{traj_type}/recording"] = wandb.Video(
                video_array, fps=0.5 / cfg.sim.dt, format="mp4"
            )

        return info

    # --------------------------------------------------------
    # 遍历 6 种轨迹执行评估
    # --------------------------------------------------------
    traj_types = cfg.get("traj_types", ALL_TRAJ_TYPES)
    if isinstance(traj_types, str):
        traj_types = [t.strip() for t in traj_types.split(",")]

    all_info = {}
    summary_rows = []

    for i, traj in enumerate(traj_types):
        print(f"\n[{i+1}/{len(traj_types)}] Evaluating trajectory: {traj}")
        info = evaluate(traj_type=traj, seed=i)
        all_info.update(info)
        tracking_err = info.get(f"eval/{traj}/stats.tracking_error", float("nan"))
        err_max = info.get(f"eval/{traj}/stats.tracking_error_max", float("nan"))
        ret = info.get(f"eval/{traj}/stats.return", float("nan"))
        ep_len = info.get(f"eval/{traj}/stats.episode_len", float("nan"))
        summary_rows.append((traj, tracking_err, err_max, ret, ep_len))

    # 一次性上报所有轨迹结果
    run.log(all_info)

    # 打印汇总表格
    print("\n" + "=" * 85)
    print(f"{'Trajectory':<15}  {'tracking_error':>15}  {'error_max':>12}  {'return':>12}  {'episode_len':>12}")
    print("-" * 85)
    for traj, err, err_max, ret, ep_len in summary_rows:
        print(f"{traj:<15}  {err:>15.4f}  {err_max:>12.4f}  {ret:>12.2f}  {ep_len:>12.1f}")
    print("=" * 85)

    wandb.finish()
    simulation_app.close()


if __name__ == "__main__":
    main()
