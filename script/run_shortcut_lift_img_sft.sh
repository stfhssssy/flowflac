#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${REPO_DIR}"

python3 script/run.py \
  --config-path="${REPO_DIR}/cfg/robomimic/pretrain/lift" \
  --config-name=pre_shortcut_mlp_img \
  "$@"
