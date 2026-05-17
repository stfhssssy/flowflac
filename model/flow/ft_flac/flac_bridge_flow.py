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
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""One-step residual bridge policy for frozen ReinFlow actors.

The base flow policy is loaded from a checkpoint and frozen.  A small MLP learns
only a zero-initialized residual action correction.  This keeps the initial
policy exactly equal to the frozen base policy while allowing FLAC-style Q
optimization with a bridge energy penalty on the correction.
"""

from typing import Dict, Optional, Tuple

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.common.modules import SpatialEmb
from model.flow.ft_flac.flac_flow import FLACFlow


class BridgeMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class FLACBridgeFlow(FLACFlow):
    """Frozen base flow actor followed by a one-step learnable bridge.

    ``self.actor`` remains the frozen base actor for compatibility with the
    existing FLAC checkpoint loader.  The only trainable policy module is
    ``self.bridge_net``.
    """

    def __init__(
        self,
        *args,
        bridge_hidden_dim: int = 256,
        bridge_obs_dim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        for param in self.actor.parameters():
            param.requires_grad_(False)
        self.actor.eval()

        obs_feat_dim = bridge_obs_dim
        if obs_feat_dim is None:
            obs_feat_dim = getattr(self.actor, "cond_enc_dim", None)
        if obs_feat_dim is None:
            obs_feat_dim = getattr(self.actor, "cond_dim", None)
        if obs_feat_dim is None:
            raise ValueError(
                "Could not infer bridge observation feature dimension. "
                "Set model.bridge_obs_dim explicitly."
            )

        self.bridge_obs_dim = int(obs_feat_dim)
        self.bridge_action_dim = self.horizon_steps * self.action_dim
        bridge_input_dim = self.bridge_obs_dim + self.bridge_action_dim + self.bridge_action_dim + 1
        self.bridge_net = BridgeMLP(
            input_dim=bridge_input_dim,
            output_dim=self.bridge_action_dim,
            hidden_dim=bridge_hidden_dim,
        ).to(self.device)

        log_suffix = (
            f"FLACBridge initialized with frozen base actor "
            f"{sum(p.numel() for p in self.actor.parameters()) / 1e6:.2f}M params "
            f"and bridge {sum(p.numel() for p in self.bridge_net.parameters()) / 1e6:.2f}M params"
        )
        import logging

        logging.getLogger(__name__).info(log_suffix)

    def train(self, mode: bool = True):
        super().train(mode)
        self.actor.eval()
        return self

    def _encode_bridge_obs(self, cond: Dict[str, Tensor]) -> Tensor:
        """Reuse the frozen actor's conditioning encoder without gradients."""
        if "rgb" not in cond:
            batch_size = cond["state"].shape[0]
            state = cond["state"].view(batch_size, -1)
            if hasattr(self.actor, "cond_mlp"):
                return self.actor.cond_mlp(state)
            return state

        actor = self.actor
        batch_size, _, _, height, width = cond["rgb"].shape
        state = cond["state"].view(batch_size, -1)
        rgb = cond["rgb"][:, -actor.img_cond_steps :]
        t_rgb = rgb.shape[1]

        if actor.num_img > 1:
            rgb = rgb.reshape(batch_size, t_rgb, actor.num_img, 3, height, width)
            rgb = einops.rearrange(rgb, "b t n c h w -> b n (t c) h w")
        elif actor.num_img == 1:
            rgb = einops.rearrange(rgb, "b t c h w -> b (t c) h w")
        else:
            raise ValueError(f"actor.num_img={actor.num_img} must be >= 1.")

        rgb = rgb.float()
        if actor.num_img == 2:
            rgb1 = rgb[:, 0]
            rgb2 = rgb[:, 1]
            if actor.augment:
                rgb1 = actor.aug(rgb1)
                rgb2 = actor.aug(rgb2)
            feat1 = actor.backbone.forward(rgb1)
            feat1 = actor.compress1.forward(feat1, state)
            feat2 = actor.backbone.forward(rgb2)
            feat2 = actor.compress2.forward(feat2, state)
            feat = torch.cat([feat1, feat2], dim=-1)
        else:
            if actor.augment:
                rgb = actor.aug(rgb)
            feat = actor.backbone.forward(rgb)
            if isinstance(actor.compress, SpatialEmb):
                feat = actor.compress.forward(feat, state)
            else:
                feat = feat.flatten(1, -1)
                feat = actor.compress(feat)

        return torch.cat([feat, state], dim=-1)

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

        if z is None:
            bridge_z = torch.zeros_like(a_base) if deterministic else torch.randn_like(a_base)
            if self.randn_clip_value is not None:
                bridge_z = bridge_z.clamp(-self.randn_clip_value, self.randn_clip_value)
        else:
            bridge_z = z.to(device=a_base.device, dtype=a_base.dtype)

        batch_size = a_base.shape[0]
        t = torch.zeros(batch_size, 1, device=a_base.device, dtype=a_base.dtype)
        bridge_input = torch.cat(
            [
                obs_feat,
                a_base.flatten(1),
                bridge_z.flatten(1),
                t,
            ],
            dim=-1,
        )
        u = self.bridge_net(bridge_input).view_as(a_base)
        action = (a_base + u).clamp(self.act_min, self.act_max)
        bridge_energy = 0.5 * u.flatten(1).pow(2).sum(dim=1)

        if return_info:
            return action, {
                "a_base": a_base,
                "u": u,
                "bridge_energy": bridge_energy,
            }
        return action, bridge_energy

    def forward(self, cond: Dict[str, Tensor], deterministic: bool = False) -> Tensor:
        actions, _ = self.sample(cond, deterministic=deterministic)
        return actions

    def loss_critic(
        self,
        obs: Dict[str, Tensor],
        actions: Tensor,
        rewards: Tensor,
        next_obs: Dict[str, Tensor],
        terminated: Tensor,
        gamma: float,
        alpha: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        with torch.no_grad():
            next_actions, next_bridge_energy = self.sample(next_obs, deterministic=False)
            if self.distributional_critic:
                next_logits1, next_logits2 = self.target_critic(next_obs, next_actions)
                next_probs1 = F.softmax(next_logits1.float(), dim=-1)
                next_probs2 = F.softmax(next_logits2.float(), dim=-1)
                next_q1 = self._critic_expectation(next_logits1)
                next_q2 = self._critic_expectation(next_logits2)
                use_q1 = next_q1 <= next_q2
                next_probs = torch.where(use_q1.unsqueeze(-1), next_probs1, next_probs2)
                atoms = self.critic_atoms.to(device=rewards.device, dtype=rewards.dtype)
                target_atoms = rewards.unsqueeze(-1) + gamma * (1.0 - terminated).unsqueeze(-1) * (
                    atoms.unsqueeze(0) - alpha.detach().to(dtype=rewards.dtype) * next_bridge_energy.unsqueeze(-1)
                )
                target_dist = self._project_c51_target(target_atoms, next_probs)
                target_q = (target_dist * atoms.unsqueeze(0)).sum(dim=-1)
            else:
                next_q1, next_q2 = self.target_critic(next_obs, next_actions)
                next_q = torch.min(next_q1, next_q2) - alpha.detach() * next_bridge_energy
                target_q = rewards + gamma * (1.0 - terminated) * next_q

        critic_out1, critic_out2 = self.critic(obs, actions)
        if self.distributional_critic:
            log_prob1 = F.log_softmax(critic_out1.float(), dim=-1)
            log_prob2 = F.log_softmax(critic_out2.float(), dim=-1)
            loss_q1 = -(target_dist * log_prob1).sum(dim=-1).mean()
            loss_q2 = -(target_dist * log_prob2).sum(dim=-1).mean()
            q1 = self._critic_expectation(critic_out1)
            q2 = self._critic_expectation(critic_out2)
        else:
            q1, q2 = critic_out1, critic_out2
            loss_q1 = F.mse_loss(q1, target_q)
            loss_q2 = F.mse_loss(q2, target_q)

        loss_critic = loss_q1 + loss_q2
        info = {
            "loss_critic": loss_critic.item(),
            "loss_q1": loss_q1.item(),
            "loss_q2": loss_q2.item(),
            "q1": q1.mean().item(),
            "q2": q2.mean().item(),
            "target_q": target_q.mean().item(),
            "next_kinetic": next_bridge_energy.mean().item(),
            "next_bridge_energy": next_bridge_energy.mean().item(),
        }
        return loss_critic, info

    def loss_actor(
        self,
        obs: Dict[str, Tensor],
        alpha: Tensor,
        actor_q_coef: float = 1.0,
        bc_anchor_coef: float = 0.0,
        reference_actor: Optional[nn.Module] = None,
    ) -> Tuple[Tensor, Tensor, Dict[str, float]]:
        actions, bridge_info = self.sample(obs, deterministic=False, return_info=True)
        bridge_energy = bridge_info["bridge_energy"]
        u = bridge_info["u"]
        a_base = bridge_info["a_base"]

        critic_out1, critic_out2 = self.critic(obs, actions)
        if self.distributional_critic:
            q1 = self._critic_expectation(critic_out1)
            q2 = self._critic_expectation(critic_out2)
        else:
            q1, q2 = critic_out1, critic_out2
        q = torch.min(q1, q2)

        policy_loss = (-q + alpha.detach() * bridge_energy).mean()
        loss_actor = actor_q_coef * policy_loss
        residual_norm = u.flatten(1).norm(dim=1)
        base_action_norm = a_base.flatten(1).norm(dim=1)
        final_action_norm = actions.flatten(1).norm(dim=1)

        info = {
            "loss_actor": loss_actor.item(),
            "actor_policy_loss": policy_loss.item(),
            "actor_q": q.mean().item(),
            "actor_q_coef": float(actor_q_coef),
            "kinetic": bridge_energy.mean().item(),
            "bridge_energy": bridge_energy.mean().item(),
            "bridge_energy_p75": torch.quantile(bridge_energy.detach(), 0.75).item(),
            "bridge_energy_p90": torch.quantile(bridge_energy.detach(), 0.90).item(),
            "residual_norm_mean": residual_norm.mean().item(),
            "residual_norm_p90": torch.quantile(residual_norm.detach(), 0.90).item(),
            "base_action_norm": base_action_norm.mean().item(),
            "final_action_norm": final_action_norm.mean().item(),
            "bc_anchor": 0.0,
            "bc_anchor_coef": float(bc_anchor_coef),
        }
        return loss_actor, bridge_energy, info
