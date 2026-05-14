#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" train/train_airnet_pl.py \
  --config configs/0_pod_config.json \
  --workspace workspace/debug \
  --batch_size_forward 2 \
  --gradient_accumulation_steps 1 \
  --enable_gradient_checkpointing True \
  --enable_mixed_precision False \
  --pod True \
  --max_epochs 30 \
  --num_iterations 1000 \
  --log_every 20 \
  --vis_every 50 \
  --num_vis_images 32 \
  --enable_tensorboard True \
  --seed 0
