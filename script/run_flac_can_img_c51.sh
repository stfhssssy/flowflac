#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/ssy/ReinFlow"
RUN_TAG="$(date +%Y%m%d_%H%M%S)_$$"
TEE_LOG="/tmp/flac_can_img_c51_${RUN_TAG}.log"

cd "${REPO_DIR}"

echo "Writing tee log to ${TEE_LOG}"

python3 script/run.py \
  --config-path=/home/ssy/ReinFlow/cfg/robomimic/finetune/can \
  --config-name=ft_flac_reflow_mlp_img \
  name=can_ft_flac_reflow_mlp_img_c51_seed42 \
  env.n_envs=50 \
  train.n_train_itr=10000 \
  train.n_eval_episode=50 \
  train.measure_kinetic_steps=5000 \
  train.save_model_freq=20 \
  train.updates_per_step=1 \
  train.actor_update_interval=50 \
  model.distributional_critic=true \
  model.critic.output_dim=101 \
  model.critic_num_atoms=101 \
  model.critic_v_min=-150.0 \
  model.critic_v_max=150.0 \
  2>&1 | tee "${TEE_LOG}"
