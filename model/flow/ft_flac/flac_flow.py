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

"""FLAC objective for ReinFlow FlowMLP policies.

This first integration targets low-dimensional observations and action chunks.
It keeps the ReinFlow model interface but replaces PPO log-probability training
with FLAC's off-policy actor-critic objective.
"""

import copy
import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.common.critic import CriticObsAct
from model.flow.mlp_flow import FlowMLP

log = logging.getLogger(__name__)


class FLACFlow(nn.Module):
    def __init__(
        self,
        actor: FlowMLP,
        critic: CriticObsAct,
        device,
        inference_steps: int,
        horizon_steps: int,
        action_dim: int,
        act_min: float = -1.0,
        act_max: float = 1.0,
        actor_policy_path: Optional[str] = None,
        randn_clip_value: Optional[float] = 1.0,
        clip_intermediate_actions: bool = True,
        final_squash: str = "clamp",
        integration_method: str = "euler",
        use_zero_noise_for_deterministic: bool = False,
        distributional_critic: bool = False,
        critic_num_atoms: int = 101,
        critic_v_min: float = -150.0,
        critic_v_max: float = 150.0,
    ):
        super().__init__()

        self.device = torch.device(device)
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)

        self.inference_steps = inference_steps
        self.horizon_steps = horizon_steps
        self.action_dim = action_dim
        self.act_min = act_min
        self.act_max = act_max
        self.randn_clip_value = randn_clip_value
        self.clip_intermediate_actions = clip_intermediate_actions
        self.final_squash = final_squash
        self.integration_method = integration_method
        self.use_zero_noise_for_deterministic = use_zero_noise_for_deterministic
        self.distributional_critic = distributional_critic
        self.critic_num_atoms = int(critic_num_atoms)
        self.critic_v_min = float(critic_v_min)
        self.critic_v_max = float(critic_v_max)
        if self.distributional_critic:
            critic_output_dim = getattr(self.critic, "output_dim", None)
            if critic_output_dim != self.critic_num_atoms:
                raise ValueError(
                    "distributional_critic=True requires critic output_dim to match "
                    f"critic_num_atoms ({self.critic_num_atoms}); got {critic_output_dim}. "
                    "Use model.critic.output_dim=<critic_num_atoms>."
                )
            if self.critic_num_atoms < 2:
                raise ValueError("critic_num_atoms must be at least 2 for distributional critics.")
            if self.critic_v_min >= self.critic_v_max:
                raise ValueError("critic_v_min must be smaller than critic_v_max.")
            self.register_buffer(
                "critic_atoms",
                torch.linspace(self.critic_v_min, self.critic_v_max, self.critic_num_atoms),
            )
            self.critic_delta = (self.critic_v_max - self.critic_v_min) / (self.critic_num_atoms - 1)
        if self.integration_method not in ("euler", "midpoint"):
            raise ValueError(f"Unknown integration_method={integration_method}. Expected euler or midpoint.")
        if self.final_squash not in ("clamp", "none"):
            raise ValueError(f"Unknown final_squash={final_squash}. Expected clamp or none.")

        if actor_policy_path:
            self.load_actor(actor_policy_path)

        log.info(
            "FLACFlow initialized with %.2fM actor params and %.2fM critic params%s",
            sum(p.numel() for p in self.actor.parameters()) / 1e6,
            sum(p.numel() for p in self.critic.parameters()) / 1e6,
            (
                f" | C51 atoms={self.critic_num_atoms} support=[{self.critic_v_min}, {self.critic_v_max}]"
                if self.distributional_critic
                else ""
            ),
        )

    def _critic_expectation(self, logits: Tensor) -> Tensor:
        if logits.ndim != 2 or logits.shape[-1] != self.critic_num_atoms:
            raise ValueError(
                "distributional_critic=True requires critic outputs shaped "
                f"(B, {self.critic_num_atoms}); got {tuple(logits.shape)}. "
                "Set model.critic.output_dim to model.critic_num_atoms."
            )
        atoms = self.critic_atoms.to(device=logits.device, dtype=logits.dtype)
        probs = F.softmax(logits.float(), dim=-1).to(dtype=logits.dtype)
        return (probs * atoms).sum(dim=-1)

    def _project_c51_target(self, target_atoms: Tensor, target_probs: Tensor) -> Tensor:
        batch_size = target_atoms.shape[0]
        target_atoms = target_atoms.clamp(self.critic_v_min, self.critic_v_max)
        b = (target_atoms - self.critic_v_min) / self.critic_delta
        lower = b.floor().long().clamp(0, self.critic_num_atoms - 1)
        upper = b.ceil().long().clamp(0, self.critic_num_atoms - 1)

        lower_weight = upper.to(b.dtype) - b
        upper_weight = b - lower.to(b.dtype)
        same_bin = upper == lower
        lower_weight = torch.where(same_bin, torch.ones_like(lower_weight), lower_weight)
        upper_weight = torch.where(same_bin, torch.zeros_like(upper_weight), upper_weight)

        projected = torch.zeros(
            batch_size,
            self.critic_num_atoms,
            device=target_probs.device,
            dtype=target_probs.dtype,
        )
        offset = (
            torch.arange(batch_size, device=target_probs.device).unsqueeze(1) * self.critic_num_atoms
        )
        projected.view(-1).scatter_add_(0, (lower + offset).view(-1), (target_probs * lower_weight).view(-1))
        projected.view(-1).scatter_add_(0, (upper + offset).view(-1), (target_probs * upper_weight).view(-1))
        return projected

    def load_actor(self, actor_policy_path: str):
        data = torch.load(actor_policy_path, map_location=self.device, weights_only=True)
        state = data.get("ema", data.get("model", data))
        state = {k.replace("network.", ""): v for k, v in state.items()}
        self.actor.load_state_dict(state, strict=True)
        log.info("Loaded FLAC actor from %s", actor_policy_path)

    def forward(self, cond: Dict[str, Tensor], deterministic: bool = False) -> Tensor:
        actions, _ = self.sample(cond, deterministic=deterministic)
        return actions

    def sample_init_noise(self, cond: Dict[str, Tensor], deterministic: bool = False) -> Tensor:
        batch_size = cond["state"].shape[0]
        dtype = cond["state"].dtype
        device = cond["state"].device

        if deterministic and self.use_zero_noise_for_deterministic:
            action = torch.zeros(batch_size, self.horizon_steps, self.action_dim, device=device, dtype=dtype)
        else:
            action = torch.randn(batch_size, self.horizon_steps, self.action_dim, device=device, dtype=dtype)
            if self.randn_clip_value is not None:
                action = action.clamp(-self.randn_clip_value, self.randn_clip_value)
        return action

    def sample(
        self,
        cond: Dict[str, Tensor],
        deterministic: bool = False,
        z: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        return self.sample_with_actor(self.actor, cond, deterministic=deterministic, z=z)

    def sample_with_actor(
        self,
        actor: nn.Module,
        cond: Dict[str, Tensor],
        deterministic: bool = False,
        z: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Sample an action chunk and return its path kinetic energy.

        Args:
            cond: observation dictionary with ``state`` shaped ``(B, To, Do)``.
            deterministic: use zero initial noise for evaluation.
            z: optional initial noise shaped ``(B, Ta, Da)``.

        Returns:
            actions: ``(B, horizon_steps, action_dim)``.
            kinetic: ``(B,)`` accumulated as ``0.5 * ||v||^2 * dt``.
        """
        batch_size = cond["state"].shape[0]
        dtype = cond["state"].dtype
        device = cond["state"].device

        if z is None:
            action = self.sample_init_noise(cond, deterministic=deterministic)
        else:
            action = z.to(device=device, dtype=dtype)

        kinetic = torch.zeros(batch_size, device=device, dtype=dtype)
        dt = 1.0 / self.inference_steps
        dt_tensor = torch.tensor(dt, device=device, dtype=dtype)

        for i in range(self.inference_steps):
            t_start = torch.full((batch_size,), i * dt, device=device, dtype=dtype)
            v_start = actor(action, t_start, cond)
            if self.integration_method == "midpoint":
                action_mid = action + 0.5 * dt * v_start
                t_mid = torch.full((batch_size,), i * dt + 0.5 * dt, device=device, dtype=dtype)
                velocity = actor(action_mid, t_mid, cond)
            else:
                velocity = v_start

            action = action + dt * velocity
            kinetic = kinetic + 0.5 * velocity.flatten(1).pow(2).sum(dim=1) * dt_tensor

            is_last_step = i == self.inference_steps - 1
            if self.clip_intermediate_actions or (is_last_step and self.final_squash == "clamp"):
                action = action.clamp(self.act_min, self.act_max)

        return action, kinetic

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
            next_actions, next_kinetic = self.sample(next_obs, deterministic=False)
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
                    atoms.unsqueeze(0) - alpha.detach().to(dtype=rewards.dtype) * next_kinetic.unsqueeze(-1)
                )
                target_dist = self._project_c51_target(target_atoms, next_probs)
                target_q = (target_dist * atoms.unsqueeze(0)).sum(dim=-1)
            else:
                next_q1, next_q2 = self.target_critic(next_obs, next_actions)
                next_q = torch.min(next_q1, next_q2) - alpha.detach() * next_kinetic
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
            "next_kinetic": next_kinetic.mean().item(),
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
        init_noise = self.sample_init_noise(obs, deterministic=False)
        actions, kinetic = self.sample(obs, deterministic=False, z=init_noise)
        critic_out1, critic_out2 = self.critic(obs, actions)
        if self.distributional_critic:
            q1 = self._critic_expectation(critic_out1)
            q2 = self._critic_expectation(critic_out2)
        else:
            q1, q2 = critic_out1, critic_out2
        q = torch.min(q1, q2)
        bc_anchor = actions.new_tensor(0.0)
        if bc_anchor_coef > 0:
            if reference_actor is None:
                raise ValueError("bc_anchor_coef > 0 requires a frozen reference_actor.")
            reference_actor.eval()
            with torch.no_grad():
                ref_actions, _ = self.sample_with_actor(
                    reference_actor,
                    obs,
                    deterministic=False,
                    z=init_noise,
                )
            bc_anchor = F.mse_loss(actions, ref_actions)

        policy_loss = (-q + alpha.detach() * kinetic).mean()
        loss_actor = actor_q_coef * policy_loss + bc_anchor_coef * bc_anchor
        info = {
            "loss_actor": loss_actor.item(),
            "actor_policy_loss": policy_loss.item(),
            "actor_q": q.mean().item(),
            "actor_q_coef": float(actor_q_coef),
            "kinetic": kinetic.mean().item(),
            "bc_anchor": bc_anchor.item(),
            "bc_anchor_coef": float(bc_anchor_coef),
        }
        return loss_actor, kinetic, info

    def loss_alpha(self, log_alpha: Tensor, kinetic: Tensor, target_kinetic: float) -> Tensor:
        return log_alpha * (target_kinetic - kinetic.detach().mean())

    def update_target_critic(self, tau: float):
        for target_param, source_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)
