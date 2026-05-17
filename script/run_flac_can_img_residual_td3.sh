#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/ssy/ReinFlow"
RUN_TAG="$(date +%Y%m%d_%H%M%S)_$$"
TEE_LOG="/tmp/flac_can_img_residual_td3_${RUN_TAG}.log"

cd "${REPO_DIR}"

echo "Writing tee log to ${TEE_LOG}"

python3 script/run.py \
  --config-path=/home/ssy/ReinFlow/cfg/robomimic/finetune/can \
  --config-name=ft_residual_td3_reflow_mlp_img \
  name=can_ft_residual_td3_reflow_mlp_img_seed42 \
  env.n_envs=50 \
  train.n_train_itr=1001 \
  train.n_eval_episode=50 \
  train.save_model_freq=20 \
  train.val_freq=10 \
  train.learning_starts=10000 \
  train.critic_warmup_updates=10000 \
  train.updates_per_step=4 \
  train.actor_update_interval=4 \
  train.random_action_noise_scale=0.2 \
  train.stddev_max=0.05 \
  train.stddev_min=0.05 \
  model.residual_action_scale=0.1 \
  model.policy_gradient_type=mean \
  2>&1 | tee "${TEE_LOG}"
