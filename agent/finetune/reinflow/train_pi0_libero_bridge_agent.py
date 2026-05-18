"""Fixed-task LIBERO training loop for DSRL-style Pi0 bridge fine-tuning."""

from __future__ import annotations

from collections import deque
import logging
import os
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torch
import wandb

from model.flow.ft_flac.pi0_libero_adapter import Pi0LiberoAdapter, quat_to_axis_angle
from util.reproducibility import set_seed_everywhere

log = logging.getLogger(__name__)


def _to_numpy_obs(raw_obs: Dict[str, Any], image_size: int) -> Dict[str, np.ndarray]:
    img = np.ascontiguousarray(raw_obs["agentview_image"][::-1, ::-1])
    if image_size > 0 and (img.shape[0] != image_size or img.shape[1] != image_size):
        img = np.asarray(Image.fromarray(img).resize((image_size, image_size), Image.BILINEAR))
    state = np.concatenate(
        (
            np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float32),
            quat_to_axis_angle(raw_obs["robot0_eef_quat"]),
            np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float32),
        )
    ).astype(np.float32)
    return {
        "pixels": np.clip(img, 0, 255).astype(np.uint8),
        "state": state,
    }


class TrainPi0LiberoBridgeAgent:
    """A compact PyTorch trainer for frozen Pi0 + trainable flow bridge.

    The environment and task are fixed, matching the DSRL Pi0 LIBERO setting:
    ``libero_90`` task id 57 by default.  Pi0 is used only during rollout/eval
    to produce frozen base action chunks; replay caches those chunks for all
    gradient updates.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device
        self.seed = int(cfg.get("seed", 42))
        set_seed_everywhere(self.seed)
        np.random.seed(self.seed)

        self.logdir = cfg.logdir
        self.checkpoint_dir = os.path.join(self.logdir, "checkpoint")
        self.result_path = os.path.join(self.logdir, "result.pkl")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.use_wandb = cfg.get("wandb", None) is not None
        if self.use_wandb:
            offline_mode = bool(cfg.wandb.get("offline_mode", False))
            wandb_dir = cfg.wandb.get("dir", "./wandb_offline" if offline_mode else "./wandb")
            os.makedirs(wandb_dir, exist_ok=True)
            wandb.init(
                entity=cfg.wandb.get("entity", None),
                project=cfg.wandb.project,
                name=cfg.wandb.run,
                config=OmegaConf.to_container(cfg, resolve=True),
                mode="offline" if offline_mode else "online",
                dir=wandb_dir,
                id=cfg.wandb.get("id", None),
                resume=cfg.wandb.get("resume", None),
            )

        self._build_env()
        self.model = hydra.utils.instantiate(cfg.model).to(self.device)
        self.pi0 = Pi0LiberoAdapter(
            openpi_root=cfg.pi0.openpi_root,
            config_name=cfg.pi0.config_name,
            checkpoint_dir=cfg.pi0.checkpoint_dir,
            prompt=self.task_description,
            horizon_steps=cfg.horizon_steps,
            action_dim=cfg.action_dim,
            deterministic_eval=cfg.pi0.get("deterministic_eval", False),
        )

        self.batch_size = int(cfg.train.batch_size)
        self.replay_size = int(cfg.train.replay_size)
        self.obs_buffer = deque(maxlen=self.replay_size)
        self.next_obs_buffer = deque(maxlen=self.replay_size)
        self.base_action_buffer = deque(maxlen=self.replay_size)
        self.next_base_action_buffer = deque(maxlen=self.replay_size)
        self.action_buffer = deque(maxlen=self.replay_size)
        self.reward_buffer = deque(maxlen=self.replay_size)
        self.done_buffer = deque(maxlen=self.replay_size)

        self.gamma = float(cfg.train.gamma)
        self.tau = float(cfg.train.tau)
        self.actor_q_coef = float(cfg.train.get("actor_q_coef", 1.0))
        self.target_bridge_energy = float(cfg.train.target_bridge_energy)
        self.auto_alpha = bool(cfg.train.get("auto_alpha", True))
        self.log_alpha = torch.tensor(
            float(cfg.train.init_log_alpha),
            device=self.device,
            requires_grad=True,
        )

        self.actor_optimizer = torch.optim.AdamW(
            self.model.actor_parameters(),
            lr=float(cfg.train.actor_lr),
            weight_decay=float(cfg.train.get("actor_weight_decay", 0.0)),
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.model.critic_parameters(),
            lr=float(cfg.train.critic_lr),
            weight_decay=float(cfg.train.get("critic_weight_decay", 0.0)),
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=float(cfg.train.alpha_lr))

        self.n_train_itr = int(cfg.train.n_train_itr)
        self.val_freq = int(cfg.train.val_freq)
        self.save_model_freq = int(cfg.train.save_model_freq)
        self.n_eval_episode = int(cfg.train.n_eval_episode)
        self.start_online_updates = int(cfg.train.start_online_updates)
        self.updates_per_step = int(cfg.train.updates_per_step)
        self.actor_update_interval = int(cfg.train.actor_update_interval)
        self.target_update_interval = int(cfg.train.target_update_interval)
        self.critic_max_grad_norm = float(cfg.train.get("critic_max_grad_norm", 0.0))
        self.actor_max_grad_norm = float(cfg.train.get("actor_max_grad_norm", 0.0))
        self.exploration_noise = float(cfg.train.get("exploration_noise", 0.0))

        self.itr = 0
        self.update_count = 0
        self.total_env_steps = 0
        self.total_train_episodes = 0
        self.total_train_successes = 0

    def _build_env(self) -> None:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        import pathlib

        benchmark_dict = benchmark.get_benchmark_dict()
        self.task_suite = benchmark_dict[self.cfg.env.suite]()
        self.task = self.task_suite.get_task(int(self.cfg.env.task_id))
        self.task_description = str(self.task.language)
        bddl_file = pathlib.Path(get_libero_path("bddl_files")) / self.task.problem_folder / self.task.bddl_file
        env_args = {
            "bddl_file_name": bddl_file,
            "camera_heights": int(self.cfg.env.render_resolution),
            "camera_widths": int(self.cfg.env.render_resolution),
        }
        self.env = OffScreenRenderEnv(**env_args)
        self.env.seed(self.seed)
        self.eval_env = OffScreenRenderEnv(**env_args)
        self.eval_env.seed(self.seed + 1000)
        self.max_timesteps = int(self.cfg.env.max_timesteps)
        self.env_max_reward = float(self.cfg.env.get("max_reward", 1.0))
        self.image_size = int(self.cfg.env.image_size)
        log.info("Loaded LIBERO task %s/%d: %s", self.cfg.env.suite, int(self.cfg.env.task_id), self.task_description)

    def _torch_obs_batch(self, obs_list: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
        return {
            "pixels": torch.from_numpy(np.asarray([obs["pixels"] for obs in obs_list])).to(self.device),
            "state": torch.from_numpy(np.asarray([obs["state"] for obs in obs_list])).float().to(self.device),
        }

    def _sample_batch(self):
        if len(self.obs_buffer) < self.batch_size:
            return None
        inds = np.random.choice(len(self.obs_buffer), self.batch_size, replace=False)
        obs = self._torch_obs_batch([self.obs_buffer[i] for i in inds])
        next_obs = self._torch_obs_batch([self.next_obs_buffer[i] for i in inds])
        base_actions = torch.from_numpy(np.asarray([self.base_action_buffer[i] for i in inds])).float().to(self.device)
        next_base_actions = torch.from_numpy(np.asarray([self.next_base_action_buffer[i] for i in inds])).float().to(self.device)
        actions = torch.from_numpy(np.asarray([self.action_buffer[i] for i in inds])).float().to(self.device)
        rewards = torch.from_numpy(np.asarray([self.reward_buffer[i] for i in inds])).float().to(self.device)
        dones = torch.from_numpy(np.asarray([self.done_buffer[i] for i in inds])).float().to(self.device)
        return obs, actions, rewards, next_obs, dones, base_actions, next_base_actions

    def _add_transition(
        self,
        obs: Dict[str, np.ndarray],
        base_action: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        next_raw_obs: Dict[str, Any],
    ) -> None:
        next_obs = _to_numpy_obs(next_raw_obs, self.image_size)
        next_base_action = self.pi0.infer_base_action(next_raw_obs, deterministic=False)
        self.obs_buffer.append(obs)
        self.next_obs_buffer.append(next_obs)
        self.base_action_buffer.append(base_action.astype(np.float32, copy=True))
        self.next_base_action_buffer.append(next_base_action.astype(np.float32, copy=True))
        self.action_buffer.append(action.astype(np.float32, copy=True))
        self.reward_buffer.append(float(reward))
        self.done_buffer.append(float(done))

    def _sample_bridge_action(
        self,
        obs_np: Dict[str, np.ndarray],
        base_action: np.ndarray,
        *,
        deterministic: bool,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        obs_t = self._torch_obs_batch([obs_np])
        base_t = torch.from_numpy(base_action[None]).float().to(self.device)
        with torch.no_grad():
            action_t, info = self.model.sample_from_base(
                obs_t,
                base_t,
                deterministic=deterministic,
                return_info=True,
            )
            if not deterministic and self.exploration_noise > 0:
                action_t = (action_t + self.exploration_noise * torch.randn_like(action_t)).clamp(-1.0, 1.0)
        info_np = {
            "bridge_energy": float(info["bridge_energy"].mean().item()),
            "residual_norm_mean": float(info["residual_norm_mean"]),
        }
        return action_t[0].cpu().numpy().astype(np.float32), info_np

    def _set_critic_requires_grad(self, requires_grad: bool) -> None:
        for param in list(self.model.critic.parameters()) + list(self.model.critic_encoder.parameters()):
            param.requires_grad_(requires_grad)

    def agent_update(self) -> Optional[Dict[str, float]]:
        batch = self._sample_batch()
        if batch is None:
            return None
        obs, actions, rewards, next_obs, dones, base_actions, next_base_actions = batch
        alpha = self.log_alpha.exp()

        loss_critic, critic_info = self.model.loss_critic(
            obs,
            actions,
            rewards,
            next_obs,
            dones,
            base_actions,
            next_base_actions,
            self.gamma,
            alpha,
        )
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        if self.critic_max_grad_norm > 0:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.critic_parameters(), self.critic_max_grad_norm)
        else:
            critic_grad_norm = None
        self.critic_optimizer.step()

        self.update_count += 1
        info = dict(critic_info)
        if critic_grad_norm is not None:
            info["critic_grad_norm"] = float(critic_grad_norm)

        do_actor_update = self.update_count % self.actor_update_interval == 0
        if do_actor_update:
            self._set_critic_requires_grad(False)
            loss_actor, energy, actor_info = self.model.loss_actor(
                obs,
                base_actions,
                alpha,
                actor_q_coef=self.actor_q_coef,
            )
            self.actor_optimizer.zero_grad()
            loss_actor.backward()
            if self.actor_max_grad_norm > 0:
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.actor_parameters(), self.actor_max_grad_norm)
            else:
                actor_grad_norm = None
            self.actor_optimizer.step()
            self._set_critic_requires_grad(True)
            info.update(actor_info)
            if actor_grad_norm is not None:
                info["actor_grad_norm"] = float(actor_grad_norm)

            if self.auto_alpha:
                loss_alpha = self.model.loss_alpha(self.log_alpha, energy, self.target_bridge_energy)
                self.alpha_optimizer.zero_grad()
                loss_alpha.backward()
                self.alpha_optimizer.step()
                info["loss_alpha"] = float(loss_alpha.item())

        if self.update_count % self.target_update_interval == 0:
            self.model.update_target_critic(self.tau)

        info["alpha"] = float(self.log_alpha.exp().item())
        info["target_bridge_energy"] = self.target_bridge_energy
        info["target_kinetic"] = self.target_bridge_energy
        info["replay_size"] = len(self.obs_buffer)
        info["actor_update_enabled"] = float(do_actor_update)
        return info

    def collect_episode(self, *, train: bool) -> Dict[str, float]:
        raw_obs = self.env.reset()
        episode_reward = 0.0
        success = False
        transitions = 0
        update_infos = []
        last_bridge_info = {}

        t = 0
        while t < self.max_timesteps:
            obs_np = _to_numpy_obs(raw_obs, self.image_size)
            base_action = self.pi0.infer_base_action(raw_obs, deterministic=not train)
            action_chunk, bridge_info = self._sample_bridge_action(
                obs_np,
                base_action,
                deterministic=not train,
            )
            last_bridge_info = bridge_info
            chunk_reward = 0.0
            done = False
            steps_this_chunk = min(int(self.cfg.act_steps), action_chunk.shape[0], self.max_timesteps - t)
            for chunk_step in range(steps_this_chunk):
                next_raw_obs, reward, done, _ = self.env.step(action_chunk[chunk_step])
                reward = float(reward)
                episode_reward += reward
                chunk_reward = max(chunk_reward, reward)
                t += 1
                self.total_env_steps += int(train)
                raw_obs = next_raw_obs
                if done:
                    break

            success = success or bool(done) or chunk_reward >= self.env_max_reward
            if train:
                self._add_transition(obs_np, base_action, action_chunk, chunk_reward, success, raw_obs)
                transitions += 1
                if len(self.obs_buffer) >= self.start_online_updates:
                    for _ in range(self.updates_per_step):
                        update_info = self.agent_update()
                        if update_info is not None:
                            update_infos.append(update_info)
            if done:
                break

        info = {
            "episode_reward": episode_reward,
            "success": float(success),
            "episode_length": float(t),
            "num_transition": float(transitions),
            **last_bridge_info,
        }
        if update_infos:
            keys = set().union(*[item.keys() for item in update_infos])
            for key in keys:
                values = [item[key] for item in update_infos if key in item]
                if values:
                    info[key] = float(np.mean(values))
        return info

    def evaluate(self) -> Dict[str, float]:
        rewards, successes, lengths = [], [], []
        env = self.env
        self.env = self.eval_env
        try:
            for _ in range(self.n_eval_episode):
                info = self.collect_episode(train=False)
                rewards.append(info["episode_reward"])
                successes.append(info["success"])
                lengths.append(info["episode_length"])
        finally:
            self.env = env
        return {
            "eval/avg_episode_reward": float(np.mean(rewards)) if rewards else 0.0,
            "eval/success_rate": float(np.mean(successes)) if successes else 0.0,
            "eval/num_success": int(np.sum(successes)),
            "eval/num_episode": len(successes),
            "eval/avg_episode_length": float(np.mean(lengths)) if lengths else 0.0,
        }

    def save_model(self) -> None:
        data = {
            "itr": self.itr,
            "model": self.model.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "update_count": self.update_count,
            "total_env_steps": self.total_env_steps,
        }
        path = os.path.join(self.checkpoint_dir, f"state_{self.itr}.pt")
        torch.save(data, path)
        log.info("Saved Pi0 LIBERO bridge model to %s", path)

    def run(self) -> None:
        results = []
        start_time = time.time()
        while self.itr < self.n_train_itr:
            if self.itr % self.val_freq == 0:
                self.model.eval()
                eval_info = self.evaluate()
                log_info = {"itr": self.itr, "env_steps": self.total_env_steps, **eval_info}
                log.info(
                    "eval itr %d | success %.3f | reward %.3f | len %.1f",
                    self.itr,
                    eval_info["eval/success_rate"],
                    eval_info["eval/avg_episode_reward"],
                    eval_info["eval/avg_episode_length"],
                )
            else:
                self.model.train()
                train_info = self.collect_episode(train=True)
                self.total_train_episodes += 1
                self.total_train_successes += int(train_info["success"] > 0.5)
                log_info = {
                    "itr": self.itr,
                    "env_steps": self.total_env_steps,
                    "train/success_rate": train_info["success"],
                    "train/avg_episode_reward": train_info["episode_reward"],
                    "train/avg_episode_length": train_info["episode_length"],
                    "train/cumulative_success_rate": self.total_train_successes / max(self.total_train_episodes, 1),
                    "train/replay_size": len(self.obs_buffer),
                }
                for key, value in train_info.items():
                    if key not in {"success", "episode_reward", "episode_length"}:
                        log_info[f"train/{key}"] = value
                log.info(
                    "train itr %d | env_steps %d | replay %d | success %.3f | cum %.3f | critic %.3e | actor %.3e | alpha %.3e | energy %.3e/%.3e | t %.1f",
                    self.itr,
                    self.total_env_steps,
                    len(self.obs_buffer),
                    log_info["train/success_rate"],
                    log_info["train/cumulative_success_rate"],
                    log_info.get("train/loss_critic", 0.0),
                    log_info.get("train/loss_actor", 0.0),
                    log_info.get("train/alpha", self.log_alpha.exp().item()),
                    log_info.get("train/bridge_energy", log_info.get("train/kinetic", 0.0)),
                    self.target_bridge_energy,
                    time.time() - start_time,
                )

            results.append(log_info)
            with open(self.result_path, "wb") as f:
                pickle.dump(results, f)
            if self.use_wandb:
                wandb.log(log_info, step=self.itr)
            if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
                self.save_model()
            self.itr += 1

