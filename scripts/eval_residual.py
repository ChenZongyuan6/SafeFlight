"""
eval_residual.py
================
验证"冻结基础策略 + 冻结 PINN 扰动观测器 + 残差补偿策略"的闭环表现。

使用方式：
    cd /root/SimpleFlight/scripts
    python eval_residual.py headless=true task=TrackResidual task.use_eval=true task.eval_traj=fast

该脚本：
1. 创建 TrackResidual 环境（与 train_residual.py 相同）
2. 加载冻结基础 Track 策略
3. 加载冻结 PINN 扰动观测器 + 归一化统计量
4. 加载训练好的残差补偿策略
5. 用 FrozenBaseAndPinnResidualPolicy 编排，执行 rollout 评估
6. 打印评估统计量，并录制视频（如果 wandb 在线）
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
    ):
        self.base_policy = base_policy
        self.residual_policy = residual_policy
        self.pinn_model = pinn_model
        self.pinn_mean = pinn_mean
        self.pinn_std = pinn_std
        self.num_envs = num_envs
        self.pinn_window_size = pinn_window_size
        self.device = device

        self.pinn_feature_history = torch.zeros(
            num_envs, pinn_window_size, pinn_mean.numel(), device=device
        )
        self.history_initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)

    @torch.no_grad()
    def reset_history(self, env_ids: Optional[torch.Tensor] = None):
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
        """
        rpos_body: [N, S, 3] 体坐标系下的未来S步轨迹误差
        prev_rpos0_body: [N, 3] 体坐标系下的上一步位置误差
        vel_error_body: [N, 3] 体坐标系下的速度误差
        """
        u_base_flat = u_base.squeeze(1)
        rpos_flat = rpos_body.flatten(1)  # [N, S*3]
        comp_obs = torch.cat([
            a_hat, u_base_flat, v_body, w_body, R_flat,
            rpos_flat, prev_rpos0_body, vel_error_body
        ], dim=-1)
        return comp_obs

    @torch.no_grad()
    def __call__(self, td: TensorDictBase, deterministic: bool = False):
        # (1) 主策略推理
        base_td = self._build_base_policy_td(td)
        base_out = self.base_policy(base_td, deterministic=True)
        u_base = torch.tanh(base_out[("agents", "action")]).detach()  # CTBR [-1,1]

        # (2) 从环境 info 里取 PINN 所需量
        info = td["info"]
        v_body = info["v_body"]
        w_body = info["w_body"]
        R_flat = info["R_flat"]
        rpos0 = info["rpos0"]                # [N, 3] 当前位置误差（用于PINN特征）
        rpos_steps = info["rpos_steps"]       # [N, S, 3] 体坐标系下的未来S步轨迹误差
        prev_rpos0 = info["prev_rpos0"]       # [N, 3] 体坐标系下的上一步位置误差
        vel_error_body = info["vel_error_body"]  # [N, 3] 体坐标系下的速度误差

        current_feat = self._build_current_pinn_feature(v_body, w_body, R_flat, u_base)
        is_init_mask = self._get_is_init_mask(td)
        a_hat = self._predict_disturbance(current_feat, is_init_mask)

        comp_obs = self._build_comp_obs(
            a_hat, u_base, v_body, w_body, R_flat,
            rpos_steps, prev_rpos0, vel_error_body
        )

        td[("agents", "observation")] = comp_obs.unsqueeze(1)
        td[("agents", "state")] = comp_obs
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

    # 强制设置评估模式相关参数
    cfg.task.use_eval = True
    # 如果未通过命令行指定 eval_traj，默认用 fast（和 eval.py 一致）
    if not cfg.task.get("eval_traj"):
        cfg.task.eval_traj = "fast"
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

    residual_model_dir = cfg.get("residual_model_dir", None)
    if residual_model_dir is None:
        residual_model_dir = "/root/SimpleFlight/scripts/outputs/track_residual/03-30_09-36/wandb/latest-run/files/checkpoint_26476544.pt"

    # 从 checkpoint 推断网络大小，避免 cfg.algo hidden_units 与 checkpoint 不匹配
    from omegaconf import open_dict
    _ckpt_residual = torch.load(residual_model_dir, map_location="cpu")
    # 从 critic 权重推断网络大小（checkpoint key = "critic"，不是 "actor"）
    _critic_w = _ckpt_residual.get("critic", {})
    _first_layer = _critic_w.get("module.base.1.layers.0.weight", None)
    if _first_layer is not None:
        # 只统计 2D（Linear）权重，跳过 1D 的 LayerNorm 权重
        n_linear = sum(1 for k, v in _critic_w.items()
                       if "base.1.layers" in k and k.endswith(".weight") and len(v.shape) == 2)
        _residual_hidden = [int(_first_layer.shape[0])] * n_linear
    else:
        _residual_hidden = [256, 256, 256]  # 安全默认值
    print(f"[INFO] Residual checkpoint hidden_units auto-detected: {_residual_hidden}")

    with open_dict(cfg):
        cfg.algo.actor.hidden_units = _residual_hidden
        cfg.algo.critic.hidden_units = _residual_hidden
    residual_policy = algos[cfg.algo.name.lower()](cfg.algo, agent_spec=residual_agent_spec, device="cuda")
    with open_dict(cfg):
        del cfg.algo.actor.hidden_units
        del cfg.algo.critic.hidden_units

    print("===== Residual policy spec =====")
    print(residual_agent_spec.observation_spec)
    print(residual_agent_spec.action_spec)
    print("================================")

    print(f"[INFO] Loading residual policy from: {residual_model_dir}")
    residual_policy.load_state_dict(_ckpt_residual)

    # 设为 eval 模式
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

    # base policy 使用 mappo.yaml 默认的 [256,256,256]
    base_policy = algos[cfg.algo.name.lower()](cfg.algo, agent_spec=base_agent_spec, device="cuda")

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
    composed_policy = FrozenBaseAndPinnResidualPolicy(
        base_policy=base_policy,
        residual_policy=residual_policy,
        pinn_model=pinn_model,
        pinn_mean=pinn_mean,
        pinn_std=pinn_std,
        num_envs=env.num_envs,
        pinn_window_size=int(cfg.task.pinn_window_size),
        device="cuda",
    )

    # --------------------------------------------------------
    # 评估函数（和 eval.py / train_residual.py 中的 evaluate 一致）
    # --------------------------------------------------------
    @torch.no_grad()
    def evaluate(seed: int = 0):
        frames = []

        base_env.enable_render(True)
        base_env.eval()
        env.eval()
        env.set_seed(seed)

        composed_policy.reset_history()

        tbar = tqdm(total=base_env.max_episode_length)

        def record_frame(*args, **kwargs):
            frame = env.base_env.render(mode="rgb_array")
            frames.append(frame)
            tbar.update(2)

        trajs = env.rollout(
            max_steps=base_env.max_episode_length,
            policy=lambda x: composed_policy(x, deterministic=True),
            callback=Every(record_frame, 2),
            auto_reset=True,
            break_when_any_done=False,
            return_contiguous=False,
        ).clone()

        base_env.enable_render(not cfg.headless)

        done = trajs.get(("next", "done"))
        first_done = torch.argmax(done.long(), dim=1).cpu()

        def take_first_episode(tensor: torch.Tensor):
            indices = first_done.reshape(first_done.shape + (1,) * (tensor.ndim - 2))
            return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)

        traj_stats = {
            k: take_first_episode(v)
            for k, v in trajs[("next", "stats")].cpu().items()
        }

        info = {
            "eval/stats." + k: torch.nanmean(v.float()).item()
            for k, v in traj_stats.items()
        }

        # 打印关键评估指标
        print("\n" + "=" * 60)
        print("  Residual Policy Evaluation Results")
        print("=" * 60)
        for k, v in sorted(info.items()):
            if "recording" not in k:
                print(f"  {k}: {v:.6f}")
        print("=" * 60 + "\n")

        if len(frames):
            video_array = np.stack(frames).transpose(0, 3, 1, 2)
            frames.clear()
            info["recording"] = wandb.Video(
                video_array, fps=0.5 / cfg.sim.dt, format="mp4"
            )

        return info

    # --------------------------------------------------------
    # 执行评估
    # --------------------------------------------------------
    print("\n[INFO] Starting evaluation of residual policy...")
    print(f"  Eval trajectory: {cfg.task.eval_traj}")
    print(f"  Num envs: {env.num_envs}")
    print(f"  Max episode length: {base_env.max_episode_length}")
    print(f"  Residual alpha: {cfg.task.get('residual_alpha', 0.3)}")
    print(f"  Wind: {cfg.task.wind}")
    print()

    info = {}
    info.update(evaluate())
    run.log(info)
    wandb.finish()

    simulation_app.close()


if __name__ == "__main__":
    main()
