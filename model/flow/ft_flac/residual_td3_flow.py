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

"""Residual TD3 wrapper for frozen ReinFlow base policies.

This module follows the residual-offpolicy-rl structure in the ReinFlow codebase:
the pretrained flow actor is frozen and a zero-initialized residual actor learns
small corrections around the base action.  The critic is trained on the final
executed action, while TD3 targets use a target residual actor plus target critic.
"""

import copy
import logging
from typing import Dict, Optional, Tuple

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.common.modules import SpatialEmb
from model.flow.ft_flac.flac_flow import FLACFlow

log = logging.getLogger(__name__)


class ResidualActorMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        action_scale: float = 0.1,
    ):
        super().__init__()
        self.action_scale = float(action_scale)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.action_scale * torch.tanh(self.net(x))


class ResidualTD3Flow(FLACFlow):
    """Frozen base ReinFlow actor with a TD3 residual policy head."""

    def __init__(
        self,
        *args,
        residual_hidden_dim: int = 256,
        residual_obs_dim: Optional[int] = None,
        residual_action_scale: float = 0.1,
        policy_gradient_type: str = "mean",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.distributional_critic:
            raise ValueError("ResidualTD3Flow currently expects a scalar double-Q critic.")

        for param in self.actor.parameters():
            param.requires_grad_(False)
        self.actor.eval()

        obs_feat_dim = residual_obs_dim
        if obs_feat_dim is None:
            obs_feat_dim = getattr(self.actor, "cond_enc_dim", None)
        if obs_feat_dim is None:
            obs_feat_dim = getattr(self.actor, "cond_dim", None)
        if obs_feat_dim is None:
            raise ValueError(
                "Could not infer residual observation feature dimension. "
                "Set model.residual_obs_dim explicitly."
            )

        self.residual_obs_dim = int(obs_feat_dim)
        self.residual_action_dim = self.horizon_steps * self.action_dim
        self.residual_action_scale = float(residual_action_scale)
        self.policy_gradient_type = str(policy_gradient_type)
        residual_input_dim = self.residual_obs_dim + self.residual_action_dim
        self.residual_net = ResidualActorMLP(
            input_dim=residual_input_dim,
            output_dim=self.residual_action_dim,
            hidden_dim=residual_hidden_dim,
            action_scale=self.residual_action_scale,
        ).to(self.device)
        self.target_residual_net = copy.deepcopy(self.residual_net).to(self.device)
        for param in self.target_residual_net.parameters():
            param.requires_grad_(False)

        log.info(
            "ResidualTD3Flow initialized with frozen base actor %.2fM params, "
            "residual actor %.2fM params, critic %.2fM params, action_scale %.3f",
            sum(p.numel() for p in self.actor.parameters()) / 1e6,
            sum(p.numel() for p in self.residual_net.parameters()) / 1e6,
            sum(p.numel() for p in self.critic.parameters()) / 1e6,
            self.residual_action_scale,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.actor.eval()
        self.target_residual_net.eval()
        return self

    def _encode_residual_obs(self, cond: Dict[str, Tensor]) -> Tensor:
        """Use the frozen base actor's visual encoder as the residual feature."""
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

    def sample_base_action(self, cond: Dict[str, Tensor], deterministic: bool = False) -> Tensor:
        with torch.no_grad():
            action, _ = self.sample_with_actor(self.actor, cond, deterministic=deterministic)
        return action.detach()

    def sample_residual_action(
        self,
        cond: Dict[str, Tensor],
        deterministic: bool = False,
        use_target: bool = False,
        stddev: float = 0.0,
        stddev_clip: float = 0.3,
        return_info: bool = False,
    ):
        with torch.no_grad():
            a_base = self.sample_base_action(cond, deterministic=deterministic)
            obs_feat = self._encode_residual_obs(cond).detach()

        net = self.target_residual_net if use_target else self.residual_net
        residual_input = torch.cat([obs_feat, a_base.flatten(1)], dim=-1)
        residual_mean = net(residual_input).view_as(a_base)
        residual = residual_mean
        if not deterministic and stddev > 0:
            noise = torch.randn_like(residual_mean) * float(stddev)
            if stddev_clip is not None and stddev_clip > 0:
                noise = noise.clamp(-float(stddev_clip), float(stddev_clip))
            residual = residual_mean + noise
            residual = residual.clamp(-self.residual_action_scale, self.residual_action_scale)

        action = (a_base + residual).clamp(self.act_min, self.act_max)

        if return_info:
            residual_flat = residual.flatten(1)
            residual_mean_flat = residual_mean.flatten(1)
            base_flat = a_base.flatten(1)
            action_flat = action.flatten(1)
            return action, {
                "a_base": a_base,
                "residual": residual,
                "residual_mean": residual_mean,
                "residual_l2": residual_flat.pow(2).sum(dim=1),
                "residual_norm": residual_flat.norm(dim=1),
                "residual_mean_norm": residual_mean_flat.norm(dim=1),
                "base_action_norm": base_flat.norm(dim=1),
                "final_action_norm": action_flat.norm(dim=1),
            }
        return action

    def sample_base_plus_uniform_noise(
        self,
        cond: Dict[str, Tensor],
        noise_scale: float,
        deterministic_base: bool = False,
    ) -> Tensor:
        a_base = self.sample_base_action(cond, deterministic=deterministic_base)
        if noise_scale > 0:
            residual = (torch.rand_like(a_base) * 2.0 - 1.0) * float(noise_scale)
            action = (a_base + residual).clamp(self.act_min, self.act_max)
        else:
            action = a_base
        return action

    def forward(self, cond: Dict[str, Tensor], deterministic: bool = False, stddev: float = 0.0) -> Tensor:
        return self.sample_residual_action(
            cond,
            deterministic=deterministic,
            use_target=False,
            stddev=stddev,
        )

    def _actor_q_for_loss(self, q1: Tensor, q2: Tensor) -> Tensor:
        if self.policy_gradient_type == "q1":
            return q1
        if self.policy_gradient_type == "min":
            return torch.min(q1, q2)
        if self.policy_gradient_type == "mean":
            return 0.5 * (q1 + q2)
        raise ValueError(f"Unknown policy_gradient_type={self.policy_gradient_type}.")

    def loss_critic_td3(
        self,
        obs: Dict[str, Tensor],
        actions: Tensor,
        rewards: Tensor,
        next_obs: Dict[str, Tensor],
        terminated: Tensor,
        gamma: float,
        target_stddev: float = 0.05,
        target_stddev_clip: float = 0.3,
    ) -> Tuple[Tensor, Dict[str, float]]:
        with torch.no_grad():
            next_actions, next_info = self.sample_residual_action(
                next_obs,
                deterministic=False,
                use_target=True,
                stddev=target_stddev,
                stddev_clip=target_stddev_clip,
                return_info=True,
            )
            next_q1, next_q2 = self.target_critic(next_obs, next_actions)
            next_q = torch.min(next_q1, next_q2)
            target_q = rewards + gamma * (1.0 - terminated) * next_q

        q1, q2 = self.critic(obs, actions)
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
            "next_q": next_q.mean().item(),
            "next_residual_norm": next_info["residual_norm"].mean().item(),
        }
        return loss_critic, info

    def loss_actor_td3(
        self,
        obs: Dict[str, Tensor],
        action_l2_weight: float = 0.0,
    ) -> Tuple[Tensor, Dict[str, float]]:
        actions, policy_info = self.sample_residual_action(
            obs,
            deterministic=True,
            use_target=False,
            stddev=0.0,
            return_info=True,
        )
        q1, q2 = self.critic(obs, actions)
        q = self._actor_q_for_loss(q1, q2)
        residual_l2 = policy_info["residual_l2"]
        action_l2 = actions.flatten(1).pow(2).sum(dim=1)
        loss_actor = -q.mean()
        if action_l2_weight > 0:
            loss_actor = loss_actor + float(action_l2_weight) * action_l2.mean()

        residual_norm = policy_info["residual_norm"]
        residual_mean_norm = policy_info["residual_mean_norm"]
        info = {
            "loss_actor": loss_actor.item(),
            "actor_q": q.mean().item(),
            "actor_q1": q1.mean().item(),
            "actor_q2": q2.mean().item(),
            "residual_l2": residual_l2.mean().item(),
            "residual_norm_mean": residual_norm.mean().item(),
            "residual_norm_p90": torch.quantile(residual_norm.detach(), 0.90).item(),
            "residual_mean_norm": residual_mean_norm.mean().item(),
            "base_action_norm": policy_info["base_action_norm"].mean().item(),
            "final_action_norm": policy_info["final_action_norm"].mean().item(),
            "action_l2": action_l2.mean().item(),
        }
        return loss_actor, info

    def update_targets(self, tau: float):
        self.update_target_critic(tau)
        for target_param, source_param in zip(
            self.target_residual_net.parameters(),
            self.residual_net.parameters(),
        ):
            target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)
