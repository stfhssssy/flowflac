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

"""FLAC fine-tuning runner for low-dimensional ReinFlow tasks."""

import logging
import copy
import math
import os
import pickle
from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import wandb
from torch import Tensor

from agent.finetune.reinflow.train_agent import TrainAgent
from model.flow.ft_flac.flac_flow import FLACFlow
from util.timer import Timer

log = logging.getLogger(__name__)


class TrainFLACFlowAgent(TrainAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.env.get("use_image_obs", False) and self.__class__ is TrainFLACFlowAgent:
            raise NotImplementedError("TrainFLACFlowAgent currently supports low-dimensional observations only.")
        if self.horizon_steps != self.act_steps:
            raise ValueError(
                "The first FLAC integration requires horizon_steps == act_steps so Q(s,a) matches the executed action chunk."
            )

        self.model: FLACFlow
        self.model.to(self.device)
        self.reference_actor = copy.deepcopy(self.model.actor).to(self.device).eval()
        for param in self.reference_actor.parameters():
            param.requires_grad_(False)

        self.gamma = cfg.train.gamma
        self.tau = cfg.train.tau
        self.replay_size = cfg.train.replay_size
        self.start_steps = cfg.train.start_steps
        self.update_after_steps = cfg.train.get("update_after_steps", self.start_steps)
        self.actor_update_after_steps = cfg.train.get("actor_update_after_steps", self.update_after_steps)
        self.updates_per_step = cfg.train.get("updates_per_step", 1)
        self.target_update_interval = cfg.train.get("target_update_interval", 2)
        self.actor_update_interval = cfg.train.get("actor_update_interval", self.target_update_interval)
        self.critic_max_grad_norm = cfg.train.get("critic_max_grad_norm", self.max_grad_norm)
        self.actor_max_grad_norm = cfg.train.get("actor_max_grad_norm", self.max_grad_norm)
        self.scale_reward_factor = cfg.train.get("scale_reward_factor", 1.0)
        self.n_eval_episode = cfg.train.get("n_eval_episode", self.n_envs)
        self.n_steps_eval = cfg.train.get("n_steps_eval", self.max_episode_steps)
        self.skip_initial_eval = cfg.train.get("skip_initial_eval", True)
        self.log_alpha_value = float(cfg.train.get("init_log_alpha", -2.0))
        self.auto_alpha = cfg.train.get("auto_alpha", True)
        self.actor_q_coef = float(cfg.train.get("actor_q_coef", 1.0))
        self.bc_anchor_coef = float(cfg.train.get("bc_anchor_coef", 0.0))
        self.bc_anchor_decay = cfg.train.get("bc_anchor_decay", "none")
        self.target_kinetic_mode = cfg.train.get("target_kinetic_mode", "coef")
        self.target_kinetic_stat = cfg.train.get("target_kinetic_stat", "mean")
        self.target_kinetic_scale = float(cfg.train.get("target_kinetic_scale", 1.0))
        self.measure_kinetic_steps = int(cfg.train.get("measure_kinetic_steps", 0))
        self.init_kinetic_stats = {}
        self.target_kinetic = self._init_target_kinetic(cfg)
        self.update_count = 0

        self.actor_optimizer = torch.optim.AdamW(
            self.model.actor.parameters(),
            lr=cfg.train.actor_lr,
            weight_decay=cfg.train.get("actor_weight_decay", 0.0),
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.model.critic.parameters(),
            lr=cfg.train.critic_lr,
            weight_decay=cfg.train.get("critic_weight_decay", 0.0),
        )
        self.log_alpha = torch.tensor(
            [self.log_alpha_value],
            requires_grad=self.auto_alpha,
            device=self.device,
            dtype=torch.float32,
        )
        self.alpha_optimizer = (
            torch.optim.Adam([self.log_alpha], lr=cfg.train.get("alpha_lr", cfg.train.actor_lr * 0.1))
            if self.auto_alpha
            else None
        )

        self.obs_buffer = deque(maxlen=self.replay_size)
        self.next_obs_buffer = deque(maxlen=self.replay_size)
        self.action_buffer = deque(maxlen=self.replay_size)
        self.reward_buffer = deque(maxlen=self.replay_size)
        # Match PPO/GAE semantics: only true environment termination masks bootstrap.
        # Time-limit truncation is used for episode bookkeeping, not Bellman targets.
        self.bootstrap_terminal_buffer = deque(maxlen=self.replay_size)
        self.cnt_train_step = 0
        self.train_num_episode_total = 0
        self.train_num_success_total = 0

    def _obs_venv_to_torch(self, obs_venv: Dict[str, np.ndarray]) -> Dict[str, Tensor]:
        return {
            "state": torch.from_numpy(np.asarray(obs_venv["state"])).float().to(self.device)
        }

    def _init_target_kinetic(self, cfg) -> float:
        if self.resume and self.resume_path and os.path.exists(self.resume_path):
            data = torch.load(self.resume_path, weights_only=True, map_location="cpu")
            if "target_kinetic" in data:
                self.init_kinetic_stats = data.get("init_kinetic_stats", {})
                target_kinetic = float(data["target_kinetic"])
                log.info(
                    "FLAC kinetic target %.6f loaded from resume checkpoint %s | init stats %s",
                    target_kinetic,
                    self.resume_path,
                    self.init_kinetic_stats,
                )
                return target_kinetic

        if self.target_kinetic_mode == "coef":
            target_kinetic = cfg.train.get("target_kinetic_coef", 2.5) * self.horizon_steps * self.action_dim
        elif self.target_kinetic_mode == "fixed":
            target_kinetic = float(cfg.train.target_kinetic)
        elif self.target_kinetic_mode == "measured":
            if self.measure_kinetic_steps <= 0:
                raise ValueError("target_kinetic_mode=measured requires train.measure_kinetic_steps > 0.")
            self.init_kinetic_stats = self.measure_init_kinetic(self.measure_kinetic_steps)
            stat_key = f"init_kinetic_{self.target_kinetic_stat}"
            if stat_key not in self.init_kinetic_stats:
                raise ValueError(
                    f"Unknown target_kinetic_stat={self.target_kinetic_stat}. "
                    "Expected one of mean, p75, p90."
                )
            target_kinetic = self.target_kinetic_scale * self.init_kinetic_stats[stat_key]
        else:
            raise ValueError(
                f"Unknown target_kinetic_mode={self.target_kinetic_mode}. "
                "Expected coef, fixed, or measured."
            )

        log.info(
            "FLAC kinetic target %.6f | mode %s | stat %s | scale %.3f | init stats %s",
            target_kinetic,
            self.target_kinetic_mode,
            self.target_kinetic_stat,
            self.target_kinetic_scale,
            self.init_kinetic_stats,
        )
        if self.use_wandb:
            wandb_info = {
                "train/target_kinetic": float(target_kinetic),
                "train/target_kinetic_scale": self.target_kinetic_scale,
            }
            for key, value in self.init_kinetic_stats.items():
                wandb_info[f"train/{key}"] = value
            wandb.log(wandb_info, step=0)
        return float(target_kinetic)

    @torch.no_grad()
    def measure_init_kinetic(self, num_steps: int) -> Dict[str, float]:
        was_training = self.model.training
        self.model.eval()
        self.reference_actor.eval()

        kinetics = []
        obs_venv = self.reset_env_all(options_venv=[{} for _ in range(self.n_envs)])
        cnt_steps = 0
        while cnt_steps < num_steps:
            cond = self._obs_venv_to_torch(obs_venv)
            actions, kinetic = self.model.sample_with_actor(
                self.reference_actor,
                cond,
                deterministic=False,
            )
            kinetics.append(kinetic.detach().cpu())
            action_venv = actions[:, : self.act_steps].detach().cpu().numpy()
            obs_venv, _, _, _, _ = self.venv.step(action_venv)
            cnt_steps += self.n_envs * self.act_steps

        if was_training:
            self.model.train()
        self.reference_actor.eval()

        kinetics = torch.cat(kinetics, dim=0)
        stats = {
            "init_kinetic_mean": float(kinetics.mean().item()),
            "init_kinetic_p75": float(torch.quantile(kinetics, 0.75).item()),
            "init_kinetic_p90": float(torch.quantile(kinetics, 0.90).item()),
            "init_kinetic_num_samples": float(kinetics.numel()),
        }
        return stats

    def _load_existing_run_results(self):
        if not (self.resume and os.path.exists(self.result_path)):
            return []
        with open(self.result_path, "rb") as f:
            run_results = pickle.load(f)
        if not isinstance(run_results, list):
            raise ValueError(f"Expected {self.result_path} to contain a list, got {type(run_results)}.")
        return run_results

    def _infer_resume_step(self, run_results, completed_itr: int) -> int:
        for info in reversed(run_results):
            if int(info.get("itr", -1)) <= completed_itr and "step" in info:
                return int(info["step"])
        return completed_itr * self.n_envs * self.act_steps * self.n_steps

    def _recover_cumulative_train_counts(self, run_results, completed_itr: int):
        for info in reversed(run_results):
            if int(info.get("itr", -1)) <= completed_itr and "train/cumulative_num_episode" in info:
                self.train_num_episode_total = int(info.get("train/cumulative_num_episode", 0))
                self.train_num_success_total = int(info.get("train/cumulative_num_success", 0))
                return

    def resume_training(self, run_results):
        log.info("Resuming FLAC training from %s", self.resume_path)
        data = torch.load(self.resume_path, weights_only=True, map_location=self.device)
        completed_itr = int(data["itr"])
        if run_results:
            kept_results = [info for info in run_results if int(info.get("itr", -1)) <= completed_itr]
            if len(kept_results) != len(run_results):
                log.info("Trimmed %d result entries after resume itr %d.", len(run_results) - len(kept_results), completed_itr)
                run_results[:] = kept_results

        self.model.load_state_dict(data["model"], strict=True)
        if "actor_optimizer" in data:
            self.actor_optimizer.load_state_dict(data["actor_optimizer"])
        if "critic_optimizer" in data:
            self.critic_optimizer.load_state_dict(data["critic_optimizer"])
        if "log_alpha" in data:
            self.log_alpha.data.copy_(data["log_alpha"].to(self.device).view_as(self.log_alpha))
        if self.alpha_optimizer is not None and "alpha_optimizer" in data:
            self.alpha_optimizer.load_state_dict(data["alpha_optimizer"])

        self.target_kinetic = float(data.get("target_kinetic", self.target_kinetic))
        self.init_kinetic_stats = data.get("init_kinetic_stats", self.init_kinetic_stats)
        self.cnt_train_step = int(data.get("cnt_train_step", self._infer_resume_step(run_results, completed_itr)))
        self.update_count = int(data.get("update_count", 0))
        self.train_num_episode_total = int(data.get("train_num_episode_total", self.train_num_episode_total))
        self.train_num_success_total = int(data.get("train_num_success_total", self.train_num_success_total))
        if "train_num_episode_total" not in data or "train_num_success_total" not in data:
            self._recover_cumulative_train_counts(run_results, completed_itr)

        start_next_itr = self.cfg.get("resume_start_next_itr", True)
        self.itr = completed_itr + (1 if start_next_itr else 0)
        if self.itr >= self.n_train_itr:
            log.warning(
                "Resume start itr %d is >= train.n_train_itr %d. Increase train.n_train_itr to continue.",
                self.itr,
                self.n_train_itr,
            )
        log.info(
            "Resumed FLAC checkpoint completed_itr=%d | next_itr=%d | step=%d | update_count=%d | cumulative_success=%d/%d",
            completed_itr,
            self.itr,
            self.cnt_train_step,
            self.update_count,
            self.train_num_success_total,
            self.train_num_episode_total,
        )

    def _get_bc_anchor_coef(self) -> float:
        if self.bc_anchor_coef <= 0:
            return 0.0
        if self.bc_anchor_decay == "cosine":
            denom = max(self.n_train_itr - 1, 1)
            progress = min(max(self.itr / denom, 0.0), 1.0)
            return self.bc_anchor_coef * 0.5 * (1.0 + math.cos(math.pi * progress))
        if self.bc_anchor_decay == "linear":
            denom = max(self.n_train_itr - 1, 1)
            progress = min(max(self.itr / denom, 0.0), 1.0)
            return self.bc_anchor_coef * (1.0 - progress)
        if self.bc_anchor_decay in ("none", "fixed", None):
            return self.bc_anchor_coef
        raise ValueError(f"Unknown bc_anchor_decay={self.bc_anchor_decay}.")

    def _prepare_video_options(self):
        options_venv = [{} for _ in range(self.n_envs)]
        if self.itr % self.render_freq == 0 and self.render_video:
            for env_ind in range(self.n_render):
                options_venv[env_ind]["video_path"] = os.path.join(
                    self.render_dir, f"itr-{self.itr}_trial-{env_ind}.mp4"
                )
        return options_venv

    def _ensure_action_batch(self, action_samples):
        action_samples = np.asarray(action_samples, dtype=np.float32)
        if action_samples.ndim == 2:
            action_samples = action_samples[None]
        return action_samples

    def _sample_policy_action(self, obs_venv: Dict[str, np.ndarray], deterministic: bool) -> np.ndarray:
        with torch.no_grad():
            cond = self._obs_venv_to_torch(obs_venv)
            samples = self.model(cond=cond, deterministic=deterministic)
        return samples.cpu().numpy()

    def _add_to_replay(
        self,
        prev_obs_venv: Dict[str, np.ndarray],
        action_samples: np.ndarray,
        obs_venv: Dict[str, np.ndarray],
        reward_venv: np.ndarray,
        terminated_venv: np.ndarray,
        truncated_venv: np.ndarray,
        info_venv,
    ):
        for i in range(self.n_envs):
            self.obs_buffer.append(prev_obs_venv["state"][i].copy())
            if "final_obs" in info_venv[i]:
                self.next_obs_buffer.append(info_venv[i]["final_obs"]["state"].copy())
            else:
                self.next_obs_buffer.append(obs_venv["state"][i].copy())
            self.action_buffer.append(action_samples[i].copy())
            self.reward_buffer.append(float(reward_venv[i]) * self.scale_reward_factor)
            self.bootstrap_terminal_buffer.append(float(terminated_venv[i]))

    def _sample_batch(self) -> Optional[Tuple[Dict[str, Tensor], Tensor, Tensor, Dict[str, Tensor], Tensor]]:
        if len(self.obs_buffer) < self.batch_size:
            return None

        inds = np.random.choice(len(self.obs_buffer), self.batch_size, replace=False)
        obs_b = torch.from_numpy(np.asarray([self.obs_buffer[i] for i in inds])).float().to(self.device)
        next_obs_b = torch.from_numpy(np.asarray([self.next_obs_buffer[i] for i in inds])).float().to(self.device)
        actions_b = torch.from_numpy(np.asarray([self.action_buffer[i] for i in inds])).float().to(self.device)
        rewards_b = torch.from_numpy(np.asarray([self.reward_buffer[i] for i in inds])).float().to(self.device)
        terminated_b = torch.from_numpy(np.asarray([self.bootstrap_terminal_buffer[i] for i in inds])).float().to(self.device)

        return {"state": obs_b}, actions_b, rewards_b, {"state": next_obs_b}, terminated_b

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
            bc_anchor_coef = self._get_bc_anchor_coef()
            loss_actor, kinetic, actor_info = self.model.loss_actor(
                obs_b,
                alpha,
                actor_q_coef=self.actor_q_coef,
                bc_anchor_coef=bc_anchor_coef,
                reference_actor=self.reference_actor if bc_anchor_coef > 0 else None,
            )
            self.actor_optimizer.zero_grad()
            loss_actor.backward()
            if self.actor_max_grad_norm:
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.actor.parameters(),
                    self.actor_max_grad_norm,
                )
            else:
                actor_grad_norm = None
            self.actor_optimizer.step()

            info.update(actor_info)
            if actor_grad_norm is not None:
                info["actor_grad_norm"] = float(actor_grad_norm)
            if self.auto_alpha:
                loss_alpha = self.model.loss_alpha(self.log_alpha, kinetic, self.target_kinetic)
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
        info["target_kinetic"] = self.target_kinetic
        if "kinetic" in info:
            info["kinetic_minus_target"] = info["kinetic"] - self.target_kinetic
            info["kinetic_ratio"] = info["kinetic"] / max(self.target_kinetic, 1e-8)
        info["bc_anchor_coef"] = self._get_bc_anchor_coef()
        info["replay_size"] = len(self.obs_buffer)
        info["actor_update_enabled"] = float(do_actor_update)
        return info

    def run(self):
        timer = Timer()
        run_results = self._load_existing_run_results()
        if self.resume:
            self.resume_training(run_results)
        cnt_train_step = self.cnt_train_step
        last_train_info = {}
        prev_obs_venv = self.reset_env_all(options_venv=self._prepare_video_options())

        while self.itr < self.n_train_itr:
            options_venv = self._prepare_video_options()
            eval_mode = (
                self.itr % self.val_freq == 0
                and not self.force_train
                and not (self.skip_initial_eval and self.itr == 0)
            )

            if eval_mode:
                self.model.eval()
                eval_info = self.evaluate(options_venv)
                log_info = {"itr": self.itr, "step": cnt_train_step, **eval_info}
            else:
                self.model.train()
                if self.reset_at_iteration or self.itr == 0:
                    prev_obs_venv = self.reset_env_all(options_venv=options_venv)

                rollout_info, prev_obs_venv = self.collect_rollout(prev_obs_venv, cnt_train_step)
                cnt_train_step = rollout_info.pop("cnt_train_step")
                last_train_info = rollout_info
                log_info = {"itr": self.itr, "step": cnt_train_step, **last_train_info}

            run_results.append(log_info)
            self.log_iteration(timer, eval_mode, log_info)
            with open(self.result_path, "wb") as f:
                pickle.dump(run_results, f)

            if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
                self.save_model()

            self.itr += 1

    def collect_rollout(self, prev_obs_venv: Dict[str, np.ndarray], cnt_train_step: int):
        firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
        firsts_trajs[0] = 1
        reward_trajs = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        update_infos = []

        for step in range(self.n_steps):
            if cnt_train_step < self.start_steps:
                action_samples = self._ensure_action_batch(self.venv.action_space.sample())
            else:
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

            if cnt_train_step >= self.update_after_steps:
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
        return train_info, prev_obs_venv

    def evaluate(self, options_venv):
        eval_summaries = []
        num_eval_done = 0
        total_eval_target = max(int(self.n_eval_episode), 1)

        while num_eval_done < total_eval_target:
            n_eval_envs = min(total_eval_target - num_eval_done, self.n_envs)
            eval_env_mask = np.zeros(self.n_envs, dtype=bool)
            eval_env_mask[:n_eval_envs] = True

            prev_obs_venv = self.reset_env_all(options_venv=options_venv)
            firsts_trajs = np.zeros((self.n_steps_eval + 1, self.n_envs))
            reward_trajs = np.zeros((self.n_steps_eval, self.n_envs), dtype=np.float32)
            firsts_trajs[0, eval_env_mask] = 1
            finished_eval_env = np.zeros(self.n_envs, dtype=bool)

            for step in range(self.n_steps_eval):
                action_samples = self._sample_policy_action(prev_obs_venv, deterministic=True)
                action_venv = action_samples[:, : self.act_steps]
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.venv.step(action_venv)
                done_venv = terminated_venv | truncated_venv
                active_eval_env = eval_env_mask & ~finished_eval_env
                reward_trajs[step, active_eval_env] = reward_venv[active_eval_env]
                newly_finished_eval_env = done_venv & active_eval_env
                firsts_trajs[step + 1, newly_finished_eval_env] = 1
                finished_eval_env |= newly_finished_eval_env
                prev_obs_venv = obs_venv
                if np.all(finished_eval_env[eval_env_mask]):
                    break

            batch_summary = self.summarize_episode_reward(
                firsts_trajs[: step + 2],
                reward_trajs[: step + 1],
                prefix="eval/",
            )
            eval_summaries.append(batch_summary)
            num_episode = int(batch_summary["eval/num_episode"])
            if num_episode == 0:
                break
            num_eval_done += num_episode

        return self.aggregate_eval_summaries(eval_summaries)

    def aggregate_eval_summaries(self, summaries):
        if not summaries:
            return self.summarize_episode_reward(
                np.zeros((1, self.n_envs)),
                np.zeros((0, self.n_envs), dtype=np.float32),
                prefix="eval/",
            )

        total_episode = sum(int(summary["eval/num_episode"]) for summary in summaries)
        total_success = sum(int(summary["eval/num_success"]) for summary in summaries)
        if total_episode == 0:
            return summaries[-1]

        weighted_keys = [
            "eval/avg_episode_reward",
            "eval/avg_best_reward",
            "eval/avg_best_reward_raw",
            "eval/avg_episode_length",
        ]
        result = {
            "eval/num_episode": total_episode,
            "eval/num_success": total_success,
            "eval/success_rate": total_success / total_episode,
        }
        for key in weighted_keys:
            result[key] = sum(summary[key] * summary["eval/num_episode"] for summary in summaries) / total_episode
        return result

    def summarize_episode_reward(self, firsts_trajs, reward_trajs, prefix="train/"):
        episodes_start_end = []
        for env_ind in range(self.n_envs):
            env_steps = np.where(firsts_trajs[:, env_ind] == 1)[0]
            for i in range(len(env_steps) - 1):
                start = env_steps[i]
                end = env_steps[i + 1]
                if end - start > 1:
                    episodes_start_end.append((env_ind, start, end - 1))

        if not episodes_start_end:
            return {
                f"{prefix}num_episode": 0,
                f"{prefix}num_success": 0,
                f"{prefix}avg_episode_reward": 0.0,
                f"{prefix}avg_best_reward": 0.0,
                f"{prefix}avg_best_reward_raw": 0.0,
                f"{prefix}success_rate": 0.0,
                f"{prefix}avg_episode_length": 0.0,
            }

        reward_trajs_split = [
            reward_trajs[start : end + 1, env_ind]
            for env_ind, start, end in episodes_start_end
        ]
        episode_reward = np.asarray([np.sum(reward_traj) for reward_traj in reward_trajs_split])
        episode_best_reward_raw = np.asarray([np.max(reward_traj) for reward_traj in reward_trajs_split])
        episode_best_reward = episode_best_reward_raw / self.act_steps
        episode_success = episode_best_reward_raw >= self.best_reward_threshold_for_success
        episode_lengths = np.asarray([end - start + 1 for _, start, end in episodes_start_end]) * self.act_steps
        return {
            f"{prefix}num_episode": len(reward_trajs_split),
            f"{prefix}num_success": int(np.sum(episode_success)),
            f"{prefix}avg_episode_reward": float(np.mean(episode_reward)),
            f"{prefix}avg_best_reward": float(np.mean(episode_best_reward)),
            f"{prefix}avg_best_reward_raw": float(np.mean(episode_best_reward_raw)),
            f"{prefix}success_rate": float(np.mean(episode_success)),
            f"{prefix}avg_episode_length": float(np.mean(episode_lengths)),
        }

    def average_update_infos(self, update_infos):
        if not update_infos:
            return {}
        keys = sorted({key for info in update_infos for key in info})
        avg_info = {}
        for key in keys:
            values = [info[key] for info in update_infos if key in info]
            avg_info[f"loss/{key}" if key.startswith("loss") else f"train/{key}"] = float(np.mean(values))
        return avg_info

    def log_iteration(self, timer, eval_mode: bool, info: Dict[str, float]):
        if self.itr % self.log_freq != 0:
            return
        elapsed = timer()
        if eval_mode:
            log.info(
                "eval itr %d | success %.3f | reward %.3f | best %.3f | t %.2f",
                self.itr,
                info.get("eval/success_rate", 0.0),
                info.get("eval/avg_episode_reward", 0.0),
                info.get("eval/avg_best_reward_raw", info.get("eval/avg_best_reward", 0.0)),
                elapsed,
            )
        else:
            log.info(
                "train itr %d | step %d | replay %d | reward %.3f | best %.3f | success %.3f | cum_success %.3f | critic %.3e | actor %.3e | alpha %.3e | kinetic %.3e/%.3e | bc %.3e | t %.2f",
                self.itr,
                int(info.get("step", 0)),
                int(info.get("train/replay_size", 0)),
                info.get("train/avg_episode_reward", 0.0),
                info.get("train/avg_best_reward_raw", info.get("train/avg_best_reward", 0.0)),
                info.get("train/success_rate", 0.0),
                info.get("train/cumulative_success_rate", 0.0),
                info.get("loss/loss_critic", 0.0),
                info.get("loss/loss_actor", 0.0),
                info.get("train/alpha", self.log_alpha.exp().item()),
                info.get("train/kinetic", 0.0),
                info.get("train/target_kinetic", self.target_kinetic),
                info.get("train/bc_anchor", 0.0),
                elapsed,
            )
        if self.use_wandb:
            wandb.log(info, step=self.itr, commit=not eval_mode)

    def save_model(self):
        data = {
            "itr": self.itr,
            "model": self.model.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "target_kinetic": self.target_kinetic,
            "init_kinetic_stats": self.init_kinetic_stats,
            "cnt_train_step": self.cnt_train_step,
            "update_count": self.update_count,
            "train_num_episode_total": self.train_num_episode_total,
            "train_num_success_total": self.train_num_success_total,
        }
        if self.alpha_optimizer is not None:
            data["alpha_optimizer"] = self.alpha_optimizer.state_dict()

        save_path = os.path.join(self.checkpoint_dir, "last.pt")
        torch.save(data, save_path)
        if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
            save_path = os.path.join(self.checkpoint_dir, f"state_{self.itr}.pt")
            torch.save(data, save_path)
            log.info("Saved FLAC model to %s", save_path)
