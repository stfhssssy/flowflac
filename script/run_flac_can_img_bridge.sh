#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/ssy/ReinFlow"
RUN_TAG="$(date +%Y%m%d_%H%M%S)_$$"
TEE_LOG="/tmp/flac_can_img_bridge_${RUN_TAG}.log"

cd "${REPO_DIR}"

echo "Writing tee log to ${TEE_LOG}"

python3 script/run.py \
  --config-path=/home/ssy/ReinFlow/cfg/robomimic/finetune/can \
  --config-name=ft_flac_bridge_reflow_mlp_img \
  name=can_ft_flac_bridge_reflow_mlp_img_seed42 \
  env.n_envs=50 \
  train.n_train_itr=1001 \
  train.n_eval_episode=50 \
  train.save_model_freq=20 \
  train.val_freq=10 \
  train.base_policy_warmup_steps=75000 \
  train.base_policy_warmup_noise=0.03 \
  train.update_after_steps=75001 \
  train.actor_update_after_steps=75001 \
  train.updates_per_step=1 \
  train.actor_update_interval=20 \
  train.actor_lr=1e-6 \
  train.actor_max_grad_norm=0.0 \
  train.actor_q_coef=1.0 \
  train.target_bridge_energy=0.01 \
  train.target_kinetic=0.01 \
  train.exploration_noise=0.01 \
  model.distributional_critic=false \
  model.critic.output_dim=1 \
  2>&1 | tee "${TEE_LOG}"
