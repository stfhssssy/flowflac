#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${GPU_ID:-0}"
if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
  GPU_ID="$1"
  shift
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export REINFLOW_LOG_DIR="${REINFLOW_LOG_DIR:-${REPO_DIR}/log}"
export REINFLOW_DATA_DIR="${REINFLOW_DATA_DIR:-${REPO_DIR}/data}"
export SQUARE_WANDB_ENTITY="${SQUARE_WANDB_ENTITY:-siyuan-ntu}"
export SQUARE_WANDB_PROJECT="${SQUARE_WANDB_PROJECT:-robomimic-square-finetune}"
export REINFLOW_WANDB_ENTITY="${REINFLOW_WANDB_ENTITY:-${SQUARE_WANDB_ENTITY}}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}${MUJOCO_BIN:-$HOME/.mujoco/mujoco210/bin}:${NVIDIA_LIB_DIR:-/usr/lib/nvidia}"

cd "${REPO_DIR}"

python3 script/run.py \
  --config-path="${REPO_DIR}/cfg/robomimic/finetune/square" \
  --config-name=ft_ppo_shortcut_mlp_img \
  device=cuda:0 \
  "sim_device=${GPU_ID}" \
  "+env.render_gpu_devices=[${GPU_ID}]" \
  "wandb.entity=${SQUARE_WANDB_ENTITY}" \
  "wandb.project=${SQUARE_WANDB_PROJECT}" \
  'wandb.run=${now:%Y-%m-%d}_${now:%H-%M-%S}_${name}' \
  "$@"
