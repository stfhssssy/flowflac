"""Frozen OpenPI / Pi0 adapter for fixed-task LIBERO rollouts."""

from __future__ import annotations

import math
import pathlib
import sys
from typing import Any, Dict, Optional

import numpy as np


def _ensure_openpi_importable(openpi_root: str) -> None:
    root = pathlib.Path(openpi_root).expanduser().resolve()
    for rel in ("src", "packages/openpi-client/src"):
        path = str(root / rel)
        if path not in sys.path:
            sys.path.insert(0, path)


def quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denom = math.sqrt(max(1.0 - quat[3] * quat[3], 0.0))
    if math.isclose(denom, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(quat[3]) / denom).astype(np.float32)


class Pi0LiberoAdapter:
    """Small wrapper around OpenPI's inference policy.

    The adapter is intentionally rollout-only.  Training should cache the Pi0
    base action in replay and never call Pi0 from gradient updates.
    """

    def __init__(
        self,
        openpi_root: str,
        config_name: str,
        checkpoint_dir: str,
        prompt: str,
        horizon_steps: int,
        action_dim: int = 7,
        deterministic_eval: bool = False,
    ):
        _ensure_openpi_importable(openpi_root)
        from openpi.policies import policy_config
        from openpi.shared import download
        from openpi.training import config as openpi_config

        self.prompt = str(prompt)
        self.horizon_steps = int(horizon_steps)
        self.action_dim = int(action_dim)
        self.deterministic_eval = bool(deterministic_eval)

        self.train_config = openpi_config.get_config(config_name)
        resolved_checkpoint = download.maybe_download(checkpoint_dir)
        self.policy = policy_config.create_trained_policy(
            self.train_config,
            resolved_checkpoint,
            default_prompt=self.prompt,
        )
        self.model_action_horizon = int(getattr(self.train_config.model, "action_horizon", self.horizon_steps))
        self.model_action_dim = int(getattr(self.train_config.model, "action_dim", 32))

    def build_input(self, raw_obs: Dict[str, Any]) -> Dict[str, Any]:
        from openpi_client import image_tools

        img = np.ascontiguousarray(raw_obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(raw_obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
        wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, 224, 224))
        state = np.concatenate(
            (
                np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float32),
                quat_to_axis_angle(raw_obs["robot0_eef_quat"]),
                np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float32),
            )
        ).astype(np.float32)
        return {
            "observation/image": img,
            "observation/wrist_image": wrist_img,
            "observation/state": state,
            "prompt": self.prompt,
        }

    def infer_base_action(
        self,
        raw_obs: Dict[str, Any],
        *,
        deterministic: bool = False,
        noise: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        element = self.build_input(raw_obs)
        if noise is None and deterministic and self.deterministic_eval:
            noise = np.zeros(
                (self.model_action_horizon, self.model_action_dim),
                dtype=np.float32,
            )
        outputs = self.policy.infer(element, noise=noise)
        actions = np.asarray(outputs["actions"], dtype=np.float32)
        if actions.ndim != 2:
            raise ValueError(f"Expected Pi0 actions [H, A], got shape {actions.shape}.")
        if actions.shape[0] < self.horizon_steps:
            pad = np.repeat(actions[-1:, :], self.horizon_steps - actions.shape[0], axis=0)
            actions = np.concatenate([actions, pad], axis=0)
        actions = actions[: self.horizon_steps, : self.action_dim]
        return np.clip(actions, -1.0, 1.0).astype(np.float32)

