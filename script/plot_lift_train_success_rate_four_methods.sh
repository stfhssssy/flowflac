#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_PREFIX="${1:-${REPO_DIR}/visualize/local/robomimic-img/lift-img/lift_four_methods_train_success_rate_smooth5}"

latest_result() {
  local run_root="$1"
  find "${run_root}" -mindepth 2 -maxdepth 2 -name result.pkl -type f -printf '%T@ %p\n' \
    | sort -nr \
    | awk 'NR==1 {print $2}'
}

cd "${REPO_DIR}"

REFLOW_RESULT="$(latest_result "${REPO_DIR}/log/robomimic/finetune/lift_ft_reflow_mlp_img_ta4_td1_tdf1")"
SHORTCUT_RESULT="$(latest_result "${REPO_DIR}/log/robomimic/finetune/lift_ft_shortcut_mlp_img_ta4_td1_tdf1")"
DPPO_RESULT="$(latest_result "${REPO_DIR}/log/robomimic/finetune/lift_ft_diffusion_mlp_img_ta4_td100_tdf5_seed42")"
GAUSSIAN_RESULT="$(latest_result "${REPO_DIR}/log/robomimic/finetune/lift_ft_gaussian_mlp_img_ta4")"

python3 agent/eval/visualize/plot_finetune_result.py \
  --series "${REFLOW_RESULT}" "ReinFlow-R" \
  --series "${SHORTCUT_RESULT}" "ReinFlow-S" \
  --series "${DPPO_RESULT}" "DPPO" \
  --series "${GAUSSIAN_RESULT}" "Gaussian" \
  --output-prefix "${OUT_PREFIX}" \
  --title "Lift train success rate" \
  --x-axis step \
  --data-source train \
  --smooth-window 5
