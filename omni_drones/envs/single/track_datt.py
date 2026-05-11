"""
TrackDATT  ---  DATT adaptive baseline environment
====================================================
Supports two wind modes, switchable via cfg.task.wind_mode:

  wind_mode: "sinusoidal"  (default)
      Identical to Track — time-varying wind using 8-frequency sinusoidal
      model (wind_i * sin(t * wind_w).sum(-1)).  Matches the standard
      SimpleFlight training setup for fair comparison.

  wind_mode: "constant"
      Episode-constant wind, sampled once per reset from
      Uniform([-wind_max, wind_max]^3) [m/s^2].  This is the original
      DATT paper setting.

In both modes, gt_wind exposes the current wind acceleration [N,3] in
world frame via info["gt_wind"].  When include_gt_wind=True, gt_wind is
appended to obs so the policy can condition on it (the "oracle" input).
"""

import torch

from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec

from .track import Track


class TrackDATT(Track):
    """DATT baseline environment.  Inherits :class:`Track`.

    Adds gt_wind exposure and optional obs augmentation.
    Wind model is controlled by wind_mode (see module docstring).
    """

    def __init__(self, cfg, headless):
        # Must be set BEFORE super().__init__() because super() calls
        # _set_specs() which reads these attributes.
        self.include_gt_wind: bool = bool(cfg.task.get("include_gt_wind", True))
        self.wind_mode: str = str(cfg.task.get("wind_mode", "sinusoidal"))
        self.wind_max: float = float(cfg.task.get("wind_max", 1.0))
        self.e_dim: int = 3

        super().__init__(cfg, headless)

        # Episode-constant wind buffer (used when wind_mode="constant")
        self.episode_wind = torch.zeros(self.num_envs, self.e_dim, device=self.device)

    # ------------------------------------------------------------------
    def _set_specs(self):
        super()._set_specs()

        if self.include_gt_wind:
            agents_spec   = self.observation_spec["agents"]
            old_obs_dim   = agents_spec["observation"].shape[-1]
            old_state_dim = agents_spec["state"].shape[-1]

            self.observation_spec["agents"] = CompositeSpec({
                "observation": UnboundedContinuousTensorSpec(
                    (1, old_obs_dim + self.e_dim)),
                "state": UnboundedContinuousTensorSpec(
                    (old_state_dim + self.e_dim,)),
            }).expand(self.num_envs).to(self.device)

        # Rebuild info_spec to include gt_wind
        info_spec = CompositeSpec({
            "drone_state": UnboundedContinuousTensorSpec(
                (self.drone.n, 13), device=self.device),
            "prev_action": torch.stack(
                [self.drone.action_spec] * self.drone.n, 0).to(self.device),
            "policy_action": torch.stack(
                [self.drone.action_spec] * self.drone.n, 0).to(self.device),
            "gt_wind": UnboundedContinuousTensorSpec(
                (self.e_dim,), device=self.device),
        }).expand(self.num_envs).to(self.device)

        self.observation_spec["info"] = info_spec
        self.info = info_spec.zero()

    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        if not hasattr(self, "episode_wind"):
            return
        if self.wind_mode == "constant":
            n = len(env_ids)
            self.episode_wind[env_ids] = (
                torch.rand(n, self.e_dim, device=self.device) * 2.0 - 1.0
            ) * self.wind_max

    # ------------------------------------------------------------------
    def _pre_sim_step(self, tensordict: TensorDictBase):
        if self.wind_mode == "sinusoidal":
            # Identical to Track._pre_sim_step — sinusoidal wind handled by super()
            super()._pre_sim_step(tensordict)
            return

        # wind_mode == "constant": episode-constant wind, override wind part
        actions = tensordict[("agents", "action")]
        self.info["prev_action"]   = tensordict[("info", "prev_action")]
        self.info["policy_action"] = tensordict[("info", "policy_action")]
        self.policy_actions = tensordict[("info", "policy_action")].clone()
        self.prev_actions   = self.info["prev_action"].clone()

        self.action_error_order1 = tensordict[("stats", "action_error_order1")].clone()
        self.stats["action_error_order1_mean"].add_(
            self.action_error_order1.mean(dim=-1).unsqueeze(-1))
        self.stats["action_error_order1_max"].set_(torch.max(
            self.stats["action_error_order1_max"],
            self.action_error_order1.mean(dim=-1).unsqueeze(-1)))

        self.effort = self.drone.apply_action(actions)

        if self.wind:
            wind_forces = (
                self.total_mass.reshape(self.num_envs, 1, 1)
                * self.episode_wind.unsqueeze(1)   # [N, 1, 3]
            )
            self.drone.base_link.apply_forces(wind_forces, is_global=True)

    # ------------------------------------------------------------------
    def _compute_state_and_obs(self):
        td = super()._compute_state_and_obs()

        # Compute gt_wind for info
        if self.wind_mode == "sinusoidal":
            # wind_force is set by Track._pre_sim_step each step.
            # On the very first reset() it doesn't exist yet -> keep zeros.
            if self.wind and hasattr(self, "wind_force"):
                self.episode_wind[:] = self.wind_force
        # constant mode: episode_wind already holds the sampled value

        self.info["gt_wind"] = self.episode_wind.clone()

        if self.include_gt_wind:
            obs   = td["agents"]["observation"]        # [N, 1, obs_dim]
            state = td["agents"]["state"]              # [N, state_dim]
            gt_obs   = self.episode_wind.unsqueeze(1)  # [N, 1, 3]
            gt_state = self.episode_wind               # [N, 3]
            td["agents"]["observation"] = torch.cat([obs,   gt_obs],   dim=-1)
            td["agents"]["state"]       = torch.cat([state, gt_state], dim=-1)

        return td
