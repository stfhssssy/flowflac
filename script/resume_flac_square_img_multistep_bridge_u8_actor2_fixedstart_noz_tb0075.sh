#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_DIR="${REPO_DIR}/log/robomimic/finetune/square_ft_flac_multistep_bridge_reflow_mlp_img_s4_u8_actor2_fixedstart_noz_tb0075_seed42/2026-05-22_23-13-28_42"
RESUME_PATH="${RUN_DIR}/checkpoint/last.pt"
RUN_TAG="$(date +%Y%m%d_%H%M%S)_$$"
TEE_LOG="/tmp/resume_flac_square_img_multistep_bridge_u8_actor2_fixedstart_noz_tb0075_${RUN_TAG}.log"
MUJOCO_BIN="/home/ssy/.mujoco/mujoco210/bin"
NVIDIA_LIB_DIR="/usr/lib/nvidia"

cd "${REPO_DIR}"

if [ ! -f "${RESUME_PATH}" ]; then
  echo "Missing resume checkpoint: ${RESUME_PATH}" >&2
  exit 1
fi

echo "Writing tee log to ${TEE_LOG}"
echo "Resuming from ${RESUME_PATH}"

append_ld_library_path() {
  local lib_dir="$1"
  if [ -d "${lib_dir}" ]; then
    if [ -n "${LD_LIBRARY_PATH:-}" ]; then
      export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${lib_dir}"
    else
      export LD_LIBRARY_PATH="${lib_dir}"
    fi
  fi
}

append_ld_library_path "${MUJOCO_BIN}"
append_ld_library_path "${NVIDIA_LIB_DIR}"

python3 script/run.py \
  --config-path="${REPO_DIR}/cfg/robomimic/finetune/square" \
  --config-name=ft_flac_multistep_bridge_reflow_mlp_img \
  name=square_ft_flac_multistep_bridge_reflow_mlp_img_s4_u8_actor2_fixedstart_noz_tb0075_seed42 \
  logdir="${RUN_DIR}" \
  +resume_path="${RESUME_PATH}" \
  base_policy_path=null \
  +wandb.id=9gdeo7qn \
  +wandb.resume=must \
  env.n_envs=50 \
  train.n_train_itr=1001 \
  train.n_eval_episode=50 \
  train.save_model_freq=20 \
  train.val_freq=10 \
  train.base_policy_warmup_steps=75000 \
  train.base_policy_warmup_noise=0.03 \
  train.update_after_steps=75001 \
  train.actor_update_after_steps=75001 \
  train.updates_per_step=8 \
  train.actor_update_interval=2 \
  train.actor_lr=6e-6 \
  train.actor_max_grad_norm=0.0 \
  train.actor_q_coef=1.0 \
  train.alpha_lr=5e-5 \
  train.target_bridge_energy=0.0075 \
  train.target_kinetic=0.0075 \
  train.init_log_alpha=3.0 \
  train.exploration_noise=0.03 \
  model.bridge_steps=4 \
  model.bridge_velocity_scale=0.3 \
  model.distributional_critic=false \
  model.critic.output_dim=1 \
  "$@" \
  2>&1 | tee "${TEE_LOG}"
