#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/ssy/ReinFlow"
RUN_TAG="$(date +%Y%m%d_%H%M%S)_$$"
TEE_LOG="/tmp/flac_can_img_u4_actor25_${RUN_TAG}.log"

cd "${REPO_DIR}"

echo "Writing tee log to ${TEE_LOG}"

python3 script/run.py \
  --config-path=/home/ssy/ReinFlow/cfg/robomimic/finetune/can \
  --config-name=ft_flac_reflow_mlp_img \
  name=can_ft_flac_reflow_mlp_img_u4_actor25_seed42 \
  env.n_envs=50 \
  train.n_train_itr=1001 \
  train.n_eval_episode=50 \
  train.measure_kinetic_steps=5000 \
  train.save_model_freq=20 \
  train.updates_per_step=4 \
  train.actor_update_interval=25 \
  model.distributional_critic=false \
  model.critic.output_dim=1 \
  2>&1 | tee "${TEE_LOG}"
