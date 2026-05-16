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

"""FLAC fine-tuning runner for image-conditioned ReinFlow tasks."""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from agent.finetune.reinflow.train_flac_flow_agent import TrainFLACFlowAgent


class TrainFLACImgFlowAgent(TrainFLACFlowAgent):
    def __init__(self, cfg):
        if not cfg.env.get("use_image_obs", False):
            raise ValueError("TrainFLACImgFlowAgent requires env.use_image_obs=true.")
        super().__init__(cfg)
        self.obs_keys = [key for key in ("state", "rgb") if key in cfg.shape_meta.obs]
        if self.obs_keys != ["state", "rgb"]:
            raise ValueError("TrainFLACImgFlowAgent expects shape_meta.obs to contain state and rgb.")

    def _copy_obs_at_env(self, obs_venv: Dict[str, np.ndarray], env_ind: int) -> Dict[str, np.ndarray]:
        return {
            "state": obs_venv["state"][env_ind].astype(np.float32, copy=True),
            "rgb": np.clip(obs_venv["rgb"][env_ind], 0, 255).astype(np.uint8, copy=True),
        }

    def _copy_single_obs(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {
            "state": obs["state"].astype(np.float32, copy=True),
            "rgb": np.clip(obs["rgb"], 0, 255).astype(np.uint8, copy=True),
        }

    def _obs_batch_to_torch(self, obs_batch: Dict[str, np.ndarray]) -> Dict[str, Tensor]:
        return {
            "state": torch.from_numpy(obs_batch["state"]).float().to(self.device),
            "rgb": torch.from_numpy(obs_batch["rgb"]).float().to(self.device),
        }

    def _obs_venv_to_torch(self, obs_venv: Dict[str, np.ndarray]) -> Dict[str, Tensor]:
        return self._obs_batch_to_torch(
            {
                "state": np.asarray(obs_venv["state"]),
                "rgb": np.asarray(obs_venv["rgb"]),
            }
        )

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
            self.obs_buffer.append(self._copy_obs_at_env(prev_obs_venv, i))
            if "final_obs" in info_venv[i]:
                self.next_obs_buffer.append(self._copy_single_obs(info_venv[i]["final_obs"]))
            else:
                self.next_obs_buffer.append(self._copy_obs_at_env(obs_venv, i))
            self.action_buffer.append(action_samples[i].astype(np.float32, copy=True))
            self.reward_buffer.append(float(reward_venv[i]) * self.scale_reward_factor)
            self.bootstrap_terminal_buffer.append(float(terminated_venv[i]))

    def _sample_batch(self) -> Optional[Tuple[Dict[str, Tensor], Tensor, Tensor, Dict[str, Tensor], Tensor]]:
        if len(self.obs_buffer) < self.batch_size:
            return None

        inds = np.random.choice(len(self.obs_buffer), self.batch_size, replace=False)
        obs_b = {
            "state": np.asarray([self.obs_buffer[i]["state"] for i in inds]),
            "rgb": np.asarray([self.obs_buffer[i]["rgb"] for i in inds]),
        }
        next_obs_b = {
            "state": np.asarray([self.next_obs_buffer[i]["state"] for i in inds]),
            "rgb": np.asarray([self.next_obs_buffer[i]["rgb"] for i in inds]),
        }
        actions_b = torch.from_numpy(np.asarray([self.action_buffer[i] for i in inds])).float().to(self.device)
        rewards_b = torch.from_numpy(np.asarray([self.reward_buffer[i] for i in inds])).float().to(self.device)
        terminated_b = torch.from_numpy(np.asarray([self.bootstrap_terminal_buffer[i] for i in inds])).float().to(self.device)

        return self._obs_batch_to_torch(obs_b), actions_b, rewards_b, self._obs_batch_to_torch(next_obs_b), terminated_b
