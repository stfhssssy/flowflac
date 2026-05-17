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

"""Image-conditioned residual TD3 runner for frozen ReinFlow base policies."""

import logging
from typing import Dict, Optional

import numpy as np
import torch

from agent.finetune.reinflow.train_flac_flow_img_agent import TrainFLACImgFlowAgent

log = logging.getLogger(__name__)


class TrainResidualTD3ImgFlowAgent(TrainFLACImgFlowAgent):
    """Residual-offpolicy-style TD3 fine-tuning around a frozen ReinFlow actor."""

    def __init__(self, cfg):
        super().__init__(cfg)
        if not hasattr(self.model, "residual_net"):
            raise ValueError("TrainResidualTD3ImgFlowAgent requires model.residual_net.")

        self.learning_starts = int(cfg.train.get("learning_starts", 10000))
        self.critic_warmup_updates = int(cfg.train.get("critic_warmup_updates", 10000))
        self.random_action_noise_scale = float(cfg.train.get("random_action_noise_scale", 0.2))
        self.use_base_policy_for_warmup = bool(cfg.train.get("use_base_policy_for_warmup", True))
        self.stddev_max = float(cfg.train.get("stddev_max", 0.05))
        self.stddev_min = float(cfg.train.get("stddev_min", 0.05))
        self.stddev_step = int(cfg.train.get("stddev_step", 300000))
        self.stddev_clip = float(cfg.train.get("stddev_clip", 0.3))
        self.target_stddev = float(cfg.train.get("target_stddev", self.stddev_max))
        self.target_stddev_clip = float(cfg.train.get("target_stddev_clip", self.stddev_clip))
        self.action_l2_weight = float(cfg.train.get("action_l2_weight", 0.0))
        self.critic_warmup_done = False

        self.actor_optimizer = torch.optim.AdamW(
            self.model.residual_net.parameters(),
            lr=cfg.train.actor_lr,
            weight_decay=cfg.train.get("actor_weight_decay", 0.0),
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.model.critic.parameters(),
            lr=cfg.train.critic_lr,
            weight_decay=cfg.train.get("critic_weight_decay", 0.0),
        )

        base_param_ids = {id(param) for param in self.model.actor.parameters()}
        residual_opt_param_ids = {
            id(param)
            for group in self.actor_optimizer.param_groups
            for param in group["params"]
        }
        if base_param_ids & residual_opt_param_ids:
            raise RuntimeError("Residual optimizer unexpectedly contains frozen base actor parameters.")
        for param in self.model.actor.parameters():
            if param.requires_grad:
                raise RuntimeError("Frozen base actor has trainable parameters.")

        log.info(
            "Residual TD3 setup | learning_starts=%d transitions | critic_warmup=%d updates | "
            "utd=%d | actor_interval=%d | random_noise=%.3f | stddev=[%.3f, %.3f]",
            self.learning_starts,
            self.critic_warmup_updates,
            self.updates_per_step,
            self.actor_update_interval,
            self.random_action_noise_scale,
            self.stddev_max,
            self.stddev_min,
        )

    def resume_training(self, run_results):
        super().resume_training(run_results)
        self.critic_warmup_done = self.update_count >= self.critic_warmup_updates

    def _current_stddev(self) -> float:
        if self.stddev_step <= 0:
            return self.stddev_min
        progress = min(max(self.cnt_train_step / float(self.stddev_step), 0.0), 1.0)
        return self.stddev_max + progress * (self.stddev_min - self.stddev_max)

    def _in_data_warmup(self) -> bool:
        return len(self.obs_buffer) < self.learning_starts

    def _sample_policy_action(self, obs_venv: Dict[str, np.ndarray], deterministic: bool) -> np.ndarray:
        with torch.no_grad():
            cond = self._obs_venv_to_torch(obs_venv)
            if not deterministic and self._in_data_warmup():
                if self.use_base_policy_for_warmup:
                    samples = self.model.sample_base_plus_uniform_noise(
                        cond,
                        noise_scale=self.random_action_noise_scale,
                        deterministic_base=False,
                    )
                else:
                    samples = (
                        torch.rand(
                            (self.n_envs, self.horizon_steps, self.action_dim),
                            device=self.device,
                            dtype=cond["state"].dtype,
                        )
                        * 2.0
                        - 1.0
                    ) * self.random_action_noise_scale
                    samples = samples.clamp(self.model.act_min, self.model.act_max)
            else:
                samples = self.model(
                    cond=cond,
                    deterministic=deterministic,
                    stddev=0.0 if deterministic else self._current_stddev(),
                )
        return samples.cpu().numpy()

    def agent_update(self, update_actor: Optional[bool] = None) -> Optional[Dict[str, float]]:
        batch = self._sample_batch()
        if batch is None:
            return None

        obs_b, actions_b, rewards_b, next_obs_b, terminated_b = batch
        loss_critic, critic_info = self.model.loss_critic_td3(
            obs_b,
            actions_b,
            rewards_b,
            next_obs_b,
            terminated_b,
            self.gamma,
            target_stddev=self.target_stddev,
            target_stddev_clip=self.target_stddev_clip,
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

        self.update_count += 1
        if update_actor is None:
            do_actor_update = self.update_count % self.actor_update_interval == 0
        else:
            do_actor_update = bool(update_actor)

        info = dict(critic_info)
        if critic_grad_norm is not None:
            info["critic_grad_norm"] = float(critic_grad_norm)

        if do_actor_update:
            loss_actor, actor_info = self.model.loss_actor_td3(
                obs_b,
                action_l2_weight=self.action_l2_weight,
            )
            self.actor_optimizer.zero_grad()
            loss_actor.backward()
            if self.actor_max_grad_norm:
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.residual_net.parameters(),
                    self.actor_max_grad_norm,
                )
            else:
                actor_grad_norm = None
            self.actor_optimizer.step()
            info.update(actor_info)
            if actor_grad_norm is not None:
                info["actor_grad_norm"] = float(actor_grad_norm)

        if self.update_count % self.target_update_interval == 0:
            self.model.update_targets(self.tau)

        info["alpha"] = 0.0
        info["log_alpha"] = 0.0
        info["energy_alpha"] = 0.0
        info["log_energy_alpha"] = 0.0
        info["kinetic"] = info.get("residual_l2", 0.0)
        info["target_kinetic"] = self.model.residual_action_scale
        info["bc_anchor"] = 0.0
        info["bc_anchor_coef"] = 0.0
        info["replay_size"] = len(self.obs_buffer)
        info["actor_update_enabled"] = float(do_actor_update)
        info["critic_warmup_done"] = float(self.critic_warmup_done)
        info["data_warmup_enabled"] = float(self._in_data_warmup())
        info["learning_starts"] = float(self.learning_starts)
        info["stddev"] = self._current_stddev()
        info["random_action_noise_scale"] = self.random_action_noise_scale
        info["residual_action_scale"] = self.model.residual_action_scale
        return info

    def _run_critic_warmup(self):
        if self.critic_warmup_done or self.critic_warmup_updates <= 0:
            self.critic_warmup_done = True
            return []

        log.info("Residual TD3 critic warmup: running %d critic-only updates.", self.critic_warmup_updates)
        update_infos = []
        log_every = max(self.critic_warmup_updates // 20, 1)
        for idx in range(self.critic_warmup_updates):
            update_info = self.agent_update(update_actor=False)
            if update_info is not None and (idx % log_every == 0 or idx == self.critic_warmup_updates - 1):
                update_infos.append(update_info)
        self.critic_warmup_done = True
        log.info("Residual TD3 critic warmup completed.")
        return update_infos

    def collect_rollout(self, prev_obs_venv: Dict[str, np.ndarray], cnt_train_step: int):
        firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
        firsts_trajs[0] = 1
        reward_trajs = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        update_infos = []

        for step in range(self.n_steps):
            action_samples = self._sample_policy_action(prev_obs_venv, deterministic=False)
            action_venv = action_samples[:, : self.act_steps]
            obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.venv.step(action_venv)
            done_venv = terminated_venv | truncated_venv

            self._add_to_replay(
                prev_obs_venv,
                action_samples,
                obs_venv,
                reward_venv,
                terminated_venv,
                truncated_venv,
                info_venv,
            )
            reward_trajs[step] = reward_venv
            firsts_trajs[step + 1] = done_venv
            prev_obs_venv = obs_venv
            cnt_train_step += self.n_envs * self.act_steps
            self.cnt_train_step = cnt_train_step

            if len(self.obs_buffer) >= self.learning_starts:
                if not self.critic_warmup_done:
                    update_infos.extend(self._run_critic_warmup())
                for _ in range(self.updates_per_step):
                    update_info = self.agent_update()
                    if update_info is not None:
                        update_infos.append(update_info)

        episode_info = self.summarize_episode_reward(firsts_trajs, reward_trajs)
        self.train_num_episode_total += int(episode_info["train/num_episode"])
        self.train_num_success_total += int(episode_info["train/num_success"])
        episode_info["train/cumulative_success_rate"] = (
            self.train_num_success_total / self.train_num_episode_total
            if self.train_num_episode_total > 0
            else 0.0
        )
        episode_info["train/cumulative_num_episode"] = self.train_num_episode_total
        episode_info["train/cumulative_num_success"] = self.train_num_success_total
        train_info = {
            "cnt_train_step": cnt_train_step,
            **episode_info,
            **self.average_update_infos(update_infos),
        }
        train_info["train/replay_size"] = len(self.obs_buffer)
        train_info["train/data_warmup_enabled"] = float(self._in_data_warmup())
        train_info["train/critic_warmup_done"] = float(self.critic_warmup_done)
        return train_info, prev_obs_venv
