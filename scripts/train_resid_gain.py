"""
train_resid_gain.py -- structured gain compensation training script

Physics:
    a_hat_world = R @ a_hat_body
    delta_CT = -a_hat_body[2] / 7.5
    delta_theta_des = -a_hat_world[0] / g
    delta_phi_des =  a_hat_world[1] / g
    delta_p = K_phi * delta_phi_des
    delta_q = K_theta * delta_theta_des
    delta_r = 0
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


GRAVITY = 9.81
# CTBR thrust mapping: thrust_cmd = (CT + 1) / 2 * 15.0
# => delta_CT = delta_thrust / 7.5
CTBR_THRUST_SCALE = 7.5


# ============================================================
# 1. Utilities
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
# 2. PINN
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
        return self.head(feat_last)


# ============================================================
# 3. PINN stats
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
# 4. Virtual base policy spec
# ============================================================

class _VirtualEnvForAgentSpec:
    def __init__(self, observation_spec, action_spec, reward_spec):
        self.observation_spec = observation_spec
        self.action_spec = action_spec
        self.reward_spec = reward_spec


def build_virtual_base_agent_spec(
    num_envs: int, obs_dim: int, state_dim: int, action_dim: int, device: str = "cuda",
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
# 5. FrozenBaseAndPinnGainPolicy -- core orchestrator
# ============================================================

class FrozenBaseAndPinnGainPolicy:
    def __init__(
        self,
        base_policy,
        gain_policy,
        pinn_model,
        pinn_mean: torch.Tensor,
        pinn_std: torch.Tensor,
        num_envs: int,
        pinn_window_size: int,
        device: str = "cuda",
    ):
        self.base_policy = base_policy
        self.gain_policy = gain_policy
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

    def _build_base_policy_td(self, td: TensorDictBase):
        info = td["info"]
        base_obs = info["base_obs"]
        base_state = info["base_state"]
        return TensorDict(
            {"agents": {"observation": base_obs, "state": base_state}},
            batch_size=td.batch_size, device=self.device,
        )

    def _build_pinn_feature(self, v_body, w_body, R_flat, u_base):
        u_base_flat = u_base.squeeze(1)
        return torch.cat([v_body, w_body, R_flat, u_base_flat], dim=-1)

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
        return self.pinn_model(x_window_norm)  # [N, 3]

    @torch.no_grad()
    def __call__(self, td: TensorDictBase, deterministic: bool = False):
        """
        Orchestration:
        1. frozen base policy -> u_base
        2. frozen PINN -> a_hat_body
        3. analytic: delta_CT, delta_phi_des, delta_theta_des
        4. build gain_obs = delta_analytical(3) + u_base(4) + v_body(3) + w_body(3) + R_flat(9) + rpos_body(18) + prev_rpos0(3) + vel_error(3) = 46-dim
        5. inject analytic quantities into info
        6. gain_policy -> [K_phi, K_theta]
        """
        # (1) base policy
        base_td = self._build_base_policy_td(td)
        base_out = self.base_policy(base_td, deterministic=True)
        u_base = torch.tanh(base_out[("agents", "action")]).detach()  # [N, 1, 4]

        # (2) PINN
        info = td["info"]
        v_body = info["v_body"]      # [N, 3]
        w_body = info["w_body"]      # [N, 3]
        R_flat = info["R_flat"]      # [N, 9]

        current_feat = self._build_pinn_feature(v_body, w_body, R_flat, u_base)
        is_init_mask = self._get_is_init_mask(td)
        a_hat_body = self._predict_disturbance(current_feat, is_init_mask)  # [N, 3]

        # (3) analytic compensation
        # 3a. thrust compensation (body z-axis disturbance)
        delta_ct = -a_hat_body[:, 2] / CTBR_THRUST_SCALE  # [N]

        # 3b. angle compensation (body -> world frame)
        R = R_flat.reshape(-1, 3, 3)  # [N, 3, 3]
        a_hat_world = torch.bmm(R, a_hat_body.unsqueeze(-1)).squeeze(-1)  # [N, 3]

        # x world disturbance -> pitch: delta_theta_des = -a_x^w / g
        # y world disturbance -> roll:  delta_phi_des = a_y^w / g
        delta_theta_des = -a_hat_world[:, 0] / GRAVITY  # [N]
        delta_phi_des = a_hat_world[:, 1] / GRAVITY      # [N]

        # (4) build rich gain_obs (46-dim)
        #     = delta_analytical(3) + u_base(4) + v_body(3) + w_body(3) + R_flat(9)
        #       + rpos_body(comp_rpos_steps*3) + prev_rpos0_body(3) + vel_error_body(3)
        rpos_steps = info["rpos_steps"]          # [N, S, 3] body frame
        prev_rpos0 = info["prev_rpos0"]          # [N, 3] body frame
        vel_error_body = info["vel_error_body"]  # [N, 3]
        rpos_flat = rpos_steps.flatten(1)          # [N, S*3]

        gain_obs = torch.cat([
            delta_phi_des.unsqueeze(-1),    # [N, 1]
            delta_theta_des.unsqueeze(-1),  # [N, 1]
            delta_ct.unsqueeze(-1),         # [N, 1]
            u_base.squeeze(1),              # [N, 4]
            v_body,                         # [N, 3]
            w_body,                         # [N, 3]
            R_flat,                         # [N, 9]
            rpos_flat,                      # [N, S*3=18]
            prev_rpos0,                     # [N, 3]
            vel_error_body,                 # [N, 3]
        ], dim=-1)  # [N, 46]

        td[("agents", "observation")] = gain_obs.unsqueeze(1)  # [N, 1, 46]
        td[("agents", "state")] = gain_obs                     # [N, 46]

        # (5) inject into info for env._pre_sim_step
        td[("info", "base_action")] = u_base
        td[("info", "pred_disturbance")] = a_hat_body
        td[("info", "delta_ct_analytic")] = delta_ct.unsqueeze(-1)       # [N, 1]
        td[("info", "delta_phi_des")] = delta_phi_des.unsqueeze(-1)      # [N, 1]
        td[("info", "delta_theta_des")] = delta_theta_des.unsqueeze(-1)  # [N, 1]

        # (6) gain policy
        td_out = self.gain_policy(td, deterministic=deterministic)
        return td_out


# ============================================================
# 6. Main
# ============================================================

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train_resid_gain")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

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
    # Create TrackResidGain env
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
    # Create gain policy (the only trainable policy)
    # --------------------------------------------------------
    gain_agent_spec: AgentSpec = env.agent_spec["drone"]

    # Deep-copy algo config for gain policy so that later changes to cfg.algo
    # (restoring tanh=False for base policy) do NOT affect gain_policy.cfg.
    # If gain_policy.cfg were the same object as cfg.algo, restoring tanh=False
    # would silently break the TanhNormal branch check inside MAPPO.train_op.
    gain_algo_cfg = OmegaConf.create(OmegaConf.to_container(cfg.algo, resolve=True))

    _gain_hu = cfg.task.get("gain_hidden_units", None)
    from omegaconf import open_dict
    with open_dict(gain_algo_cfg):
        gain_algo_cfg.actor.tanh = True   # TanhNormal keeps action in (-1, 1)
        if _gain_hu is not None:
            gain_algo_cfg.actor.hidden_units = list(_gain_hu)
            gain_algo_cfg.critic.hidden_units = list(_gain_hu)

    gain_policy = algos[cfg.algo.name.lower()](gain_algo_cfg, agent_spec=gain_agent_spec, device="cuda")

    print("===== Gain policy spec =====")
    print(gain_agent_spec.observation_spec)
    print(gain_agent_spec.action_spec)
    print("============================")

    # --------------------------------------------------------
    # Create frozen base policy
    # --------------------------------------------------------
    if not hasattr(base_env, "base_obs_dim") or not hasattr(base_env, "base_state_dim"):
        raise RuntimeError(
            "TrackResidGain env missing base_obs_dim / base_state_dim."
        )

    base_obs_dim = int(base_env.base_obs_dim)
    base_state_dim = int(base_env.base_state_dim)
    base_action_dim = 4

    print(f"===== Base policy virtual spec: obs={base_obs_dim}, state={base_state_dim}, act={base_action_dim} =====")

    base_agent_spec = build_virtual_base_agent_spec(
        num_envs=env.num_envs,
        obs_dim=base_obs_dim,
        state_dim=base_state_dim,
        action_dim=base_action_dim,
        device="cuda",
    )

    base_policy = algos[cfg.algo.name.lower()](cfg.algo, agent_spec=base_agent_spec, device="cuda")

    if cfg.task.get("base_model_dir", None) is None:
        raise ValueError("cfg.task.base_model_dir must not be empty.")

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
    # Create frozen PINN
    # --------------------------------------------------------
    if cfg.task.get("pinn_model_dir", None) is None:
        raise ValueError("cfg.task.pinn_model_dir must not be empty.")

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
    # PINN normalization stats
    # --------------------------------------------------------
    if cfg.task.get("pinn_stats_dataset", None) is None:
        raise ValueError("cfg.task.pinn_stats_dataset must not be empty.")

    pinn_mean, pinn_std = load_pinn_stats_from_dataset(
        data_path=cfg.task.pinn_stats_dataset,
        train_ratio=float(cfg.task.get("pinn_train_ratio", 0.9)),
        device="cuda",
    )

    # --------------------------------------------------------
    # Build orchestrator
    # --------------------------------------------------------
    composed_policy = FrozenBaseAndPinnGainPolicy(
        base_policy=base_policy,
        gain_policy=gain_policy,
        pinn_model=pinn_model,
        pinn_mean=pinn_mean,
        pinn_std=pinn_std,
        num_envs=env.num_envs,
        pinn_window_size=int(cfg.task.pinn_window_size),
        device="cuda",
    )

    # --------------------------------------------------------
    # Collector: uses composed_policy, trains gain_policy
    # --------------------------------------------------------
    frames_per_batch = env.num_envs * int(cfg.algo.train_every)
    total_frames = cfg.get("total_frames", -1) // frames_per_batch * frames_per_batch
    max_iters = cfg.get("max_iters", -1)
    eval_interval = cfg.get("eval_interval", -1)
    save_interval = cfg.get("save_interval", -1)

    stats_keys = [
        k for k in base_env.observation_spec.keys(True, True)
        if isinstance(k, tuple) and k[0] == "stats"
    ]
    episode_stats = EpisodeStats(stats_keys)

    collector = SyncDataCollector(
        env,
        policy=composed_policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=cfg.sim.device,
        return_same_td=True,
    )

    # --------------------------------------------------------
    # Evaluate function
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

        info = {
            "eval/stats." + k: torch.nanmean(v.float()).item()
            for k, v in traj_stats.items()
        }

        if len(frames):
            video_array = np.stack(frames).transpose(0, 3, 1, 2)
            frames.clear()
            info["recording"] = wandb.Video(video_array, fps=0.5 / cfg.sim.dt, format="mp4")

        return info

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------
    pbar = tqdm(collector)
    env.train()

    for i, data in enumerate(pbar):
        base_env.count = i
        info = {"env_frames": collector._frames, "rollout_fps": collector._fps}

        episode_stats(data.to_tensordict())

        if len(episode_stats) >= base_env.num_envs:
            stats = {
                "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v).item()
                for k, v in episode_stats.pop().items(True, True)
            }
            info.update(stats)

        info.update(gain_policy.train_op(data.to_tensordict()))

        if eval_interval > 0 and i % eval_interval == 0:
            logging.info(f"Eval at {collector._frames} steps.")
            info.update(evaluate())
            env.train()

        if save_interval > 0 and i % save_interval == 0:
            if hasattr(gain_policy, "state_dict"):
                ckpt_path = os.path.join(run.dir, f"checkpoint_{collector._frames}.pt")
                logging.info(f"Save checkpoint to {str(ckpt_path)}")
                torch.save(gain_policy.state_dict(), ckpt_path)

        run.log(info)

        printable_info = {k: v for k, v in info.items() if isinstance(v, float)}
        print(OmegaConf.to_yaml(printable_info))

        pbar.set_postfix({
            "rollout_fps": collector._fps,
            "frames": collector._frames,
        })

        if max_iters > 0 and i >= max_iters - 1:
            break

    # --------------------------------------------------------
    # Final eval and save
    # --------------------------------------------------------
    logging.info(f"Final Eval at {collector._frames} steps.")
    info = {"env_frames": collector._frames}
    info.update(evaluate())
    run.log(info)

    if hasattr(gain_policy, "state_dict"):
        ckpt_path = os.path.join(run.dir, "checkpoint_final.pt")
        logging.info(f"Save gain checkpoint to {str(ckpt_path)}")
        torch.save(gain_policy.state_dict(), ckpt_path)

    wandb.save(os.path.join(run.dir, "checkpoint*"))
    wandb.finish()

    simulation_app.close()


if __name__ == "__main__":
    main()
