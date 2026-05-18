#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/ssy/ReinFlow"
RUN_TAG="$(date +%Y%m%d_%H%M%S)_$$"
TEE_LOG="/tmp/pi0_libero_bridge_task57_${RUN_TAG}.log"

cd "${REPO_DIR}"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/home/ssy/ReinFlow/openpi}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

echo "Writing tee log to ${TEE_LOG}"

python3 script/run.py \
  --config-path=/home/ssy/ReinFlow/cfg/libero/finetune \
  --config-name=ft_pi0_libero_multistep_bridge \
  name=pi0_libero90_task57_dsrl_bridge_s4_seed42 \
  env.task_id=57 \
  2>&1 | tee "${TEE_LOG}"

