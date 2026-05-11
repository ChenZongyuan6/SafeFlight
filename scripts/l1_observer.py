"""
L1 Adaptive Control Disturbance Observer for Quadrotor Translational Dynamics.

Coordinate convention (same as DATT and NeuralIMC):
    v_dot_world = g_world + R_bw @ [0, 0, thrust_acc] + a_disturbance_world

Reference implementations:
    DATT:      https://github.com/KevinHuang8/DATT/blob/main/learning/adaptation_module.py
    NeuralIMC: https://github.com/thu-uav/NeuralIMC/blob/main/torch_control/controllers/l1ac.py

L1 adaptation law (piecewise constant, per-axis scalar Am=As):
    d_hat[k] = -As * exp(As*dt) / (exp(As*dt) - 1) * (v_hat[k] - v[k])

Correct update order at each step k:
    1. v_error   = v_hat[k] - v[k]                    (pre-propagation error)
    2. a_new     = coeff * v_error                     (raw L1 adaptation)
    3. a_hat[k]  = alpha * a_hat[k-1] + (1-alpha) * a_new   (low-pass)
    4. v_hat[k+1] = v_hat[k] + (g + R*u*e3 + a_hat[k] + As*v_error) * dt
"""

import math
import torch
from typing import Optional, Sequence


class L1AccelerationObserver:
    """
    Batch L1 disturbance observer for quadrotor dynamics.

    All quantities are in the **world frame** (Z-axis up).
    Output a_hat_world must be rotated to body frame if ground-truth labels
    are expressed in body frame (use world_to_body()).

    Args:
        num_envs:    Number of parallel environments (batch size).
        dt:          Simulation time step in seconds.
        gravity:     Gravity vector in world frame, shape (3,).
        As_value:    Scalar L1 bandwidth parameter (negative for stability).
                     Larger magnitude → faster adaptation.
                     Recommended: -6.0 to -20.0.
        alpha:       EMA smoothing factor (0 < alpha < 1).
                     Higher → smoother but slower response.
                     alpha=0.9 gives ~0.2s time constant at 50 Hz.
        clip_value:  Clip |a_hat| to this value in m/s². 0 = disabled.
        device:      Torch device string.
    """

    def __init__(
        self,
        num_envs: int,
        dt: float = 0.02,
        gravity: Sequence[float] = (0.0, 0.0, -9.81),
        As_value: float = -0.01,   #-10.0
        alpha: float = 0.99,  #0.90
        clip_value: float = 5.0,
        device: str = "cuda",
    ):
        self.num_envs = num_envs
        self.dt = dt
        self.As_value = As_value
        self.alpha = alpha
        self.clip_value = clip_value
        self.device = device

        self.g_vec = torch.tensor(gravity, dtype=torch.float32, device=device)  # [3]

        # Precompute L1 coefficient: coeff = -As * exp(As*dt) / (exp(As*dt) - 1)
        expA = math.exp(As_value * dt)
        self.coeff = -(As_value * expA) / (expA - 1.0)  # positive when As < 0

        # Internal state
        self.vel_hat = torch.zeros(num_envs, 3, device=device)  # predicted world velocity
        self.a_hat = torch.zeros(num_envs, 3, device=device)    # disturbance estimate (world, m/s²)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(
        self,
        env_ids: Optional[torch.Tensor] = None,
        v_world: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Reset observer state for selected environments.

        Args:
            env_ids:  1-D index tensor. None = reset all.
            v_world:  Initial world-frame velocity, shape [len(env_ids), 3]. None → zero.
        """
        if env_ids is None:
            self.vel_hat.zero_()
            self.a_hat.zero_()
            if v_world is not None:
                self.vel_hat[:] = v_world
        else:
            self.vel_hat[env_ids] = 0.0 if v_world is None else v_world
            self.a_hat[env_ids] = 0.0

    def step(
        self,
        v_world: torch.Tensor,
        R_bw: torch.Tensor,
        thrust_acc: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process one time step and return the disturbance estimate (world frame).

        Args:
            v_world:    Measured world-frame velocity, shape [N, 3].
            R_bw:       Body-to-world rotation matrix, shape [N, 3, 3].
                        R_bw[:, :, 2] = body Z-axis in world (thrust direction).
            thrust_acc: Mass-normalised thrust scalar (m/s²), shape [N] or [N, 1].
                        This should be the thrust that will drive the state FORWARD
                        from the current v_world measurement.

        Returns:
            a_hat_world: Estimated disturbance in world frame, shape [N, 3].
        """
        if thrust_acc.dim() == 1:
            thrust_acc = thrust_acc.unsqueeze(-1)   # [N, 1]

        # Thrust vector in world frame
        thrust_world = R_bw[:, :, 2] * thrust_acc          # [N, 3]
        a_nominal = self.g_vec.unsqueeze(0) + thrust_world  # [N, 3]

        # ---- Step 1: compute adaptation from PRE-propagation error ----
        v_error = self.vel_hat - v_world                    # v_hat[k] - v[k]
        a_new = self.coeff * v_error                        # raw L1 law

        # ---- Step 2: apply low-pass filter to get d_hat[k] ----
        self.a_hat = self.alpha * self.a_hat + (1.0 - self.alpha) * a_new

        if self.clip_value > 0.0:
            self.a_hat = self.a_hat.clamp(-self.clip_value, self.clip_value)

        # ---- Step 3: propagate predictor using updated d_hat[k] ----
        v_hat_dot = a_nominal + self.a_hat + self.As_value * v_error
        self.vel_hat = self.vel_hat + v_hat_dot * self.dt

        return self.a_hat.clone()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def world_to_body(a_world: torch.Tensor, R_bw: torch.Tensor) -> torch.Tensor:
        """
        Rotate world-frame disturbance to body frame: a_body = R_bw^T @ a_world.

        Args:
            a_world:  [N, 3]
            R_bw:     [N, 3, 3]
        Returns:
            a_body:   [N, 3]
        """
        return torch.bmm(R_bw.transpose(-1, -2), a_world.unsqueeze(-1)).squeeze(-1)
