# MIT License
#
# Copyright (c) 2025 ReinFlow Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Multi-step base-anchored bridge flow for frozen ReinFlow actors."""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from model.flow.ft_flac.flac_bridge_flow import FLACBridgeFlow


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if dim <= 0:
            raise ValueError("time embedding dim must be positive.")
        self.dim = int(dim)

    def forward(self, t: Tensor) -> Tensor:
        if t.ndim == 1:
            t = t[:, None]
        if self.dim == 1:
            return t
        half_dim = self.dim // 2
        freq = torch.exp(
            torch.arange(half_dim, device=t.device, dtype=t.dtype)
            * (-math.log(10000.0) / max(half_dim - 1, 1))
        )
        args = t * freq[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, t], dim=-1)
        return emb


class BridgeVelocityMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        velocity_scale: float = 0.2,
    ):
        super().__init__()
        self.velocity_scale = float(velocity_scale)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, output_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: Tensor) -> Tensor:
        raw = self.net(x)
        if self.velocity_scale > 0:
            return self.velocity_scale * torch.tanh(raw)
        return raw


class FLACMultiStepBridgeFlow(FLACBridgeFlow):
    """A true flow bridge solver anchored at the frozen base action.

    The frozen base flow actor samples ``a_base``.  The trainable bridge starts
    from ``X_0 = a_base`` and integrates a zero-initialized velocity field:

    ``X_{k+1} = X_k + dt * u_theta(obs_feat, a_base, X_k, z, t_k)``.

    With zero final-layer initialization this is exactly the frozen base policy
    for any number of bridge steps.  The FLAC energy is only the trainable bridge
    control energy, not the frozen base actor's flow kinetic energy.
    """

    def __init__(
        self,
        *args,
        bridge_steps: int = 4,
        bridge_hidden_dim: int = 256,
        bridge_time_dim: int = 32,
        bridge_obs_dim: Optional[int] = None,
        bridge_velocity_scale: float = 0.2,
        bridge_noise_condition_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            *args,
            bridge_hidden_dim=bridge_hidden_dim,
            bridge_obs_dim=bridge_obs_dim,
            **kwargs,
        )
        self.bridge_steps = int(bridge_steps)
        if self.bridge_steps <= 0:
            raise ValueError("bridge_steps must be >= 1.")
        self.bridge_time_dim = int(bridge_time_dim)
        self.bridge_velocity_scale = float(bridge_velocity_scale)
        self.bridge_noise_condition_scale = float(bridge_noise_condition_scale)
        self.time_embedding = SinusoidalTimeEmbedding(self.bridge_time_dim).to(self.device)

        bridge_input_dim = (
            self.bridge_obs_dim
            + self.bridge_action_dim  # base action
            + self.bridge_action_dim  # current solver state
            + self.bridge_action_dim  # stochastic condition
            + self.bridge_time_dim
        )
        self.bridge_net = BridgeVelocityMLP(
            input_dim=bridge_input_dim,
            output_dim=self.bridge_action_dim,
            hidden_dim=bridge_hidden_dim,
            velocity_scale=self.bridge_velocity_scale,
        ).to(self.device)

        import logging

        logging.getLogger(__name__).info(
            "FLACMultiStepBridge initialized with frozen base actor %.2fM params, "
            "bridge %.2fM params, steps=%d, velocity_scale=%.3f",
            sum(p.numel() for p in self.actor.parameters()) / 1e6,
            sum(p.numel() for p in self.bridge_net.parameters()) / 1e6,
            self.bridge_steps,
            self.bridge_velocity_scale,
        )

    def _sample_bridge_condition(
        self,
        a_base: Tensor,
        deterministic: bool,
        z: Optional[Tensor],
    ) -> Tensor:
        if z is not None:
            return z.to(device=a_base.device, dtype=a_base.dtype)
        if deterministic or self.bridge_noise_condition_scale <= 0:
            return torch.zeros_like(a_base)
        bridge_z = torch.randn_like(a_base) * self.bridge_noise_condition_scale
        if self.randn_clip_value is not None:
            clip_value = self.randn_clip_value * max(self.bridge_noise_condition_scale, 1e-8)
            bridge_z = bridge_z.clamp(-clip_value, clip_value)
        return bridge_z

    def sample(
        self,
        cond: Dict[str, Tensor],
        deterministic: bool = False,
        z: Optional[Tensor] = None,
        return_info: bool = False,
    ):
        with torch.no_grad():
            a_base, _ = self.sample_with_actor(self.actor, cond, deterministic=deterministic)
            obs_feat = self._encode_bridge_obs(cond).detach()
            a_base = a_base.detach()

        bridge_z = self._sample_bridge_condition(a_base, deterministic=deterministic, z=z)
        batch_size = a_base.shape[0]
        dtype = a_base.dtype
        device = a_base.device
        dt = 1.0 / self.bridge_steps
        dt_tensor = torch.tensor(dt, device=device, dtype=dtype)

        x = a_base
        energy = torch.zeros(batch_size, device=device, dtype=dtype)
        velocity_norm_sum = torch.zeros(batch_size, device=device, dtype=dtype)

        a_base_flat = a_base.flatten(1)
        z_flat = bridge_z.flatten(1)
        for step in range(self.bridge_steps):
            t = torch.full((batch_size,), step * dt, device=device, dtype=dtype)
            t_emb = self.time_embedding(t)
            bridge_input = torch.cat(
                [
                    obs_feat,
                    a_base_flat,
                    x.flatten(1),
                    z_flat,
                    t_emb,
                ],
                dim=-1,
            )
            velocity = self.bridge_net(bridge_input).view_as(a_base)
            velocity_flat = velocity.flatten(1)
            energy = energy + 0.5 * velocity_flat.pow(2).sum(dim=1) * dt_tensor
            velocity_norm_sum = velocity_norm_sum + velocity_flat.norm(dim=1)

            x = x + dt * velocity
            if self.clip_intermediate_actions:
                x = x.clamp(self.act_min, self.act_max)

        action = x.clamp(self.act_min, self.act_max) if self.final_squash == "clamp" else x
        residual = action - a_base

        if return_info:
            return action, {
                "a_base": a_base,
                "u": residual,
                "bridge_energy": energy,
                "bridge_velocity_norm_mean_path": velocity_norm_sum / self.bridge_steps,
                "bridge_steps": torch.full_like(energy, float(self.bridge_steps)),
            }
        return action, energy
