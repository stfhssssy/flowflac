"""DSRL-style visual bridge policy anchored on frozen Pi0 actions."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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


class DSRLVisualEncoder(nn.Module):
    """64x64 image encoder with a 50-dim DSRL-style bottleneck."""

    def __init__(self, image_latent_dim: int = 50):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, image_latent_dim),
            nn.LayerNorm(image_latent_dim),
            nn.Tanh(),
        )

    def forward(self, pixels: Tensor) -> Tensor:
        if pixels.ndim != 4:
            raise ValueError(f"Expected pixels [B,H,W,C] or [B,C,H,W], got {pixels.shape}.")
        if pixels.shape[-1] == 3:
            pixels = pixels.permute(0, 3, 1, 2)
        pixels = pixels.float()
        if pixels.max() > 2.0:
            pixels = pixels / 255.0
        return self.proj(self.conv(pixels))


class BridgeVelocityMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, velocity_scale: float):
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


class DoubleQCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class Pi0LiberoDSRLBridgeFlow(nn.Module):
    """Trainable flow bridge over frozen Pi0 LIBERO action chunks.

    The model follows the DSRL observation interface: 64x64 RGB image plus an
    8-dim robot state.  Pi0 itself is not part of this module; rollout code
    supplies cached Pi0 base actions from replay.
    """

    def __init__(
        self,
        device: str = "cuda:0",
        image_latent_dim: int = 50,
        state_dim: int = 8,
        horizon_steps: int = 50,
        action_dim: int = 7,
        bridge_steps: int = 4,
        bridge_hidden_dim: int = 256,
        bridge_time_dim: int = 32,
        bridge_velocity_scale: float = 0.2,
        bridge_noise_condition_scale: float = 1.0,
        critic_hidden_dim: int = 256,
        act_min: float = -1.0,
        act_max: float = 1.0,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.image_latent_dim = int(image_latent_dim)
        self.state_dim = int(state_dim)
        self.obs_feat_dim = self.image_latent_dim + self.state_dim
        self.horizon_steps = int(horizon_steps)
        self.action_dim = int(action_dim)
        self.action_flat_dim = self.horizon_steps * self.action_dim
        self.bridge_steps = int(bridge_steps)
        self.bridge_noise_condition_scale = float(bridge_noise_condition_scale)
        self.act_min = float(act_min)
        self.act_max = float(act_max)

        if self.bridge_steps <= 0:
            raise ValueError("bridge_steps must be >= 1.")

        self.actor_encoder = DSRLVisualEncoder(self.image_latent_dim)
        self.critic_encoder = DSRLVisualEncoder(self.image_latent_dim)
        self.target_critic_encoder = DSRLVisualEncoder(self.image_latent_dim)
        self.time_embedding = SinusoidalTimeEmbedding(bridge_time_dim)

        bridge_input_dim = (
            self.obs_feat_dim
            + self.action_flat_dim  # base action
            + self.action_flat_dim  # current solver state
            + self.action_flat_dim  # stochastic condition
            + bridge_time_dim
        )
        self.bridge_net = BridgeVelocityMLP(
            bridge_input_dim,
            self.action_flat_dim,
            bridge_hidden_dim,
            bridge_velocity_scale,
        )

        critic_input_dim = self.obs_feat_dim + self.action_flat_dim
        self.critic = DoubleQCritic(critic_input_dim, critic_hidden_dim)
        self.target_critic = DoubleQCritic(critic_input_dim, critic_hidden_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic_encoder.load_state_dict(self.critic_encoder.state_dict())
        self.to(self.device)

    def actor_parameters(self):
        return list(self.actor_encoder.parameters()) + list(self.bridge_net.parameters())

    def critic_parameters(self):
        return list(self.critic_encoder.parameters()) + list(self.critic.parameters())

    def _obs_feat(self, obs: Dict[str, Tensor], *, encoder: nn.Module) -> Tensor:
        pixels = obs["pixels"].to(self.device)
        state = obs["state"].to(self.device).float()
        if state.ndim > 2:
            state = state.flatten(1)
        img_feat = encoder(pixels)
        return torch.cat([img_feat, state], dim=-1)

    def _format_action(self, action: Tensor) -> Tensor:
        action = action.to(self.device).float()
        if action.shape[-2:] != (self.horizon_steps, self.action_dim):
            raise ValueError(
                f"Expected action shape [...,{self.horizon_steps},{self.action_dim}], got {action.shape}."
            )
        return action

    def _sample_bridge_condition(self, base_action: Tensor, deterministic: bool) -> Tensor:
        if deterministic or self.bridge_noise_condition_scale <= 0:
            return torch.zeros_like(base_action)
        return torch.randn_like(base_action) * self.bridge_noise_condition_scale

    def sample_from_base(
        self,
        obs: Dict[str, Tensor],
        base_action: Tensor,
        *,
        deterministic: bool = False,
        return_info: bool = False,
    ):
        base_action = self._format_action(base_action)
        obs_feat = self._obs_feat(obs, encoder=self.actor_encoder)
        z = self._sample_bridge_condition(base_action, deterministic)
        batch_size = base_action.shape[0]
        dtype = base_action.dtype
        device = base_action.device
        dt = 1.0 / self.bridge_steps
        dt_tensor = torch.tensor(dt, device=device, dtype=dtype)

        x = base_action
        energy = torch.zeros(batch_size, device=device, dtype=dtype)
        velocity_norm_sum = torch.zeros(batch_size, device=device, dtype=dtype)
        base_flat = base_action.flatten(1)
        z_flat = z.flatten(1)

        for step in range(self.bridge_steps):
            t = torch.full((batch_size,), step * dt, device=device, dtype=dtype)
            bridge_input = torch.cat(
                [
                    obs_feat,
                    base_flat,
                    x.flatten(1),
                    z_flat,
                    self.time_embedding(t),
                ],
                dim=-1,
            )
            velocity = self.bridge_net(bridge_input).view_as(base_action)
            velocity_flat = velocity.flatten(1)
            energy = energy + 0.5 * velocity_flat.pow(2).sum(dim=1) * dt_tensor
            velocity_norm_sum = velocity_norm_sum + velocity_flat.norm(dim=1)
            x = (x + dt * velocity).clamp(self.act_min, self.act_max)

        action = x
        residual = action - base_action
        if return_info:
            residual_norm = residual.flatten(1).norm(dim=1)
            return action, {
                "bridge_energy": energy,
                "u": residual,
                "residual_norm_mean": residual_norm.mean().item(),
                "residual_norm_p90": torch.quantile(residual_norm.detach(), 0.90).item(),
                "base_action_norm": base_action.flatten(1).norm(dim=1).mean().item(),
                "final_action_norm": action.flatten(1).norm(dim=1).mean().item(),
                "bridge_velocity_norm_mean_path": (velocity_norm_sum / self.bridge_steps).mean().item(),
            }
        return action, energy

    def _critic_value(
        self,
        obs: Dict[str, Tensor],
        action: Tensor,
        *,
        target: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        action = self._format_action(action)
        encoder = self.target_critic_encoder if target else self.critic_encoder
        critic = self.target_critic if target else self.critic
        obs_feat = self._obs_feat(obs, encoder=encoder)
        critic_input = torch.cat([obs_feat, action.flatten(1)], dim=-1)
        return critic(critic_input)

    def loss_critic(
        self,
        obs: Dict[str, Tensor],
        actions: Tensor,
        rewards: Tensor,
        next_obs: Dict[str, Tensor],
        terminated: Tensor,
        base_actions: Tensor,
        next_base_actions: Tensor,
        gamma: float,
        alpha: Tensor,
    ):
        del base_actions  # current base action is not needed for fitted Q(s, a).
        actions = self._format_action(actions)
        rewards = rewards.to(self.device).float()
        terminated = terminated.to(self.device).float()
        with torch.no_grad():
            next_actions, next_energy = self.sample_from_base(
                next_obs,
                next_base_actions,
                deterministic=False,
            )
            next_q1, next_q2 = self._critic_value(next_obs, next_actions, target=True)
            next_q = torch.min(next_q1, next_q2) - alpha.detach() * next_energy
            target_q = rewards + gamma * (1.0 - terminated) * next_q

        q1, q2 = self._critic_value(obs, actions, target=False)
        loss_q1 = F.mse_loss(q1, target_q)
        loss_q2 = F.mse_loss(q2, target_q)
        loss = loss_q1 + loss_q2
        return loss, {
            "loss_critic": loss.item(),
            "loss_q1": loss_q1.item(),
            "loss_q2": loss_q2.item(),
            "q1": q1.mean().item(),
            "q2": q2.mean().item(),
            "target_q": target_q.mean().item(),
            "next_bridge_energy": next_energy.mean().item(),
            "next_kinetic": next_energy.mean().item(),
        }

    def loss_actor(
        self,
        obs: Dict[str, Tensor],
        base_actions: Tensor,
        alpha: Tensor,
        actor_q_coef: float = 1.0,
    ):
        actions, bridge_info = self.sample_from_base(
            obs,
            base_actions,
            deterministic=False,
            return_info=True,
        )
        bridge_energy = bridge_info["bridge_energy"]
        q1, q2 = self._critic_value(obs, actions, target=False)
        q = torch.min(q1, q2)
        policy_loss = (-q + alpha.detach() * bridge_energy).mean()
        loss = actor_q_coef * policy_loss
        info = {
            "loss_actor": loss.item(),
            "actor_policy_loss": policy_loss.item(),
            "actor_q": q.mean().item(),
            "actor_q_coef": float(actor_q_coef),
            "kinetic": bridge_energy.mean().item(),
            "bridge_energy": bridge_energy.mean().item(),
            "bridge_energy_p75": torch.quantile(bridge_energy.detach(), 0.75).item(),
            "bridge_energy_p90": torch.quantile(bridge_energy.detach(), 0.90).item(),
            **{k: v for k, v in bridge_info.items() if k != "bridge_energy" and not torch.is_tensor(v)},
        }
        return loss, bridge_energy, info

    def loss_alpha(self, log_alpha: Tensor, energy: Tensor, target_energy: float) -> Tensor:
        return log_alpha * (target_energy - energy.detach().mean())

    def update_target_critic(self, tau: float) -> None:
        for target, source in zip(self.target_critic.parameters(), self.critic.parameters()):
            target.data.copy_(target.data * (1.0 - tau) + source.data * tau)
        for target, source in zip(self.target_critic_encoder.parameters(), self.critic_encoder.parameters()):
            target.data.copy_(target.data * (1.0 - tau) + source.data * tau)

