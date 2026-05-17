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

"""Image-conditioned FLAC runner for frozen-base one-step bridge policies."""

from typing import Dict, Optional

import numpy as np
import torch

from agent.finetune.reinflow.train_flac_flow_img_agent import TrainFLACImgFlowAgent


class TrainFLACBridgeImgFlowAgent(TrainFLACImgFlowAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        if not hasattr(self.model, "bridge_net"):
            raise ValueError("TrainFLACBridgeImgFlowAgent requires model.bridge_net.")

        self.target_bridge_energy = float(cfg.train.get("target_bridge_energy", self.target_kinetic))
        self.target_kinetic = self.target_bridge_energy
        self.exploration_noise = float(cfg.train.get("exploration_noise", 0.0))
        self.base_policy_warmup_steps = int(cfg.train.get("base_policy_warmup_steps", 0))
        self.base_policy_warmup_noise = float(
            cfg.train.get("base_policy_warmup_noise", self.exploration_noise)
        )
        if self.base_policy_warmup_steps > 0:
            self.actor_update_after_steps = max(
                self.actor_update_after_steps,
                self.base_policy_warmup_steps,
            )

        self.actor_optimizer = torch.optim.AdamW(
            self.model.bridge_net.parameters(),
            lr=cfg.train.actor_lr,
            weight_decay=cfg.train.get("actor_weight_decay", 0.0),
        )

        base_param_ids = {id(param) for param in self.model.actor.parameters()}
        opt_param_ids = {
            id(param)
            for group in self.actor_optimizer.param_groups
            for param in group["params"]
        }
        if base_param_ids & opt_param_ids:
            raise RuntimeError("Bridge actor optimizer unexpectedly contains frozen base actor parameters.")
        for param in self.model.actor.parameters():
            if param.requires_grad:
                raise RuntimeError("Frozen base actor has trainable parameters.")

    def _sample_policy_action(self, obs_venv: Dict[str, np.ndarray], deterministic: bool) -> np.ndarray:
        with torch.no_grad():
            cond = self._obs_venv_to_torch(obs_venv)
            if not deterministic and self.cnt_train_step < self.base_policy_warmup_steps:
                samples, _ = self.model.sample_with_actor(
                    self.model.actor,
                    cond,
                    deterministic=False,
                )
                noise_scale = self.base_policy_warmup_noise
            else:
                samples = self.model(cond=cond, deterministic=deterministic)
                noise_scale = self.exploration_noise

            if not deterministic and noise_scale > 0:
                samples = samples + noise_scale * torch.randn_like(samples)
                samples = samples.clamp(self.model.act_min, self.model.act_max)
        return samples.cpu().numpy()

    def agent_update(self) -> Optional[Dict[str, float]]:
        batch = self._sample_batch()
        if batch is None:
            return None

        obs_b, actions_b, rewards_b, next_obs_b, terminated_b = batch
        alpha = self.log_alpha.exp()

        loss_critic, critic_info = self.model.loss_critic(
            obs_b,
            actions_b,
            rewards_b,
            next_obs_b,
            terminated_b,
            self.gamma,
            alpha,
        )
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        if self.critic_max_grad_norm:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.critic.parameters(),
                self.critic_max_grad_norm,
            )
        else:
            critic_grad_norm = None
        self.critic_optimizer.step()

        info = dict(critic_info)
        if critic_grad_norm is not None:
            info["critic_grad_norm"] = float(critic_grad_norm)

        self.update_count += 1
        do_actor_update = (
            self.cnt_train_step >= self.actor_update_after_steps
            and self.update_count % self.actor_update_interval == 0
        )
        if do_actor_update:
            loss_actor, bridge_energy, actor_info = self.model.loss_actor(
                obs_b,
                alpha,
                actor_q_coef=self.actor_q_coef,
                bc_anchor_coef=0.0,
                reference_actor=None,
            )
            self.actor_optimizer.zero_grad()
            loss_actor.backward()
            if self.actor_max_grad_norm:
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.bridge_net.parameters(),
                    self.actor_max_grad_norm,
                )
            else:
                actor_grad_norm = None
            self.actor_optimizer.step()

            info.update(actor_info)
            if actor_grad_norm is not None:
                info["actor_grad_norm"] = float(actor_grad_norm)
            if self.auto_alpha:
                loss_alpha = self.model.loss_alpha(
                    self.log_alpha,
                    bridge_energy,
                    self.target_bridge_energy,
                )
                self.alpha_optimizer.zero_grad()
                loss_alpha.backward()
                self.alpha_optimizer.step()
                info["loss_alpha"] = loss_alpha.item()

        if self.update_count % self.target_update_interval == 0:
            self.model.update_target_critic(self.tau)

        info["alpha"] = self.log_alpha.exp().item()
        info["log_alpha"] = self.log_alpha.item()
        info["energy_alpha"] = info["alpha"]
        info["log_energy_alpha"] = info["log_alpha"]
        info["target_kinetic"] = self.target_bridge_energy
        info["target_bridge_energy"] = self.target_bridge_energy
        if "kinetic" in info:
            info["kinetic_minus_target"] = info["kinetic"] - self.target_bridge_energy
            info["kinetic_ratio"] = info["kinetic"] / max(self.target_bridge_energy, 1e-8)
        info["bc_anchor_coef"] = 0.0
        info["replay_size"] = len(self.obs_buffer)
        info["actor_update_enabled"] = float(do_actor_update)
        info["base_policy_warmup_enabled"] = float(self.cnt_train_step < self.base_policy_warmup_steps)
        info["base_policy_warmup_steps"] = float(self.base_policy_warmup_steps)
        info["base_policy_warmup_noise"] = float(self.base_policy_warmup_noise)
        return info
