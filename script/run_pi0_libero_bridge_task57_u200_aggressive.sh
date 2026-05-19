#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/ssy/ReinFlow"
RUN_TAG="$(date +%Y%m%d_%H%M%S)_$$"
TEE_LOG="/tmp/pi0_libero_bridge_task57_u200_aggressive_${RUN_TAG}.log"

cd "${REPO_DIR}"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/home/ssy/ReinFlow/openpi}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"
export PYTHONPATH="/home/ssy/ReinFlow/LIBERO:${PYTHONPATH:-}"

echo "Writing tee log to ${TEE_LOG}"

python3 script/run.py \
  --config-path=/home/ssy/ReinFlow/cfg/libero/finetune \
  --config-name=ft_pi0_libero_multistep_bridge \
  name=pi0_libero90_task57_dsrl_bridge_s4_u200_aggressive_seed42 \
  sim_device=cuda:${MUJOCO_EGL_DEVICE_ID} \
  env.task_id=57 \
  train.n_train_itr=1001 \
  train.save_model_freq=20 \
  train.val_freq=10 \
  train.updates_per_step=200 \
  train.actor_update_interval=1 \
  train.actor_lr=3e-4 \
  train.alpha_lr=3e-4 \
  train.init_log_alpha=2.0 \
  train.target_bridge_energy=0.01 \
  train.exploration_noise=0.05 \
  model.bridge_velocity_scale=0.6 \
  2>&1 | tee "${TEE_LOG}"
