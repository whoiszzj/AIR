#!/usr/bin/env bash

# SLURM resource configuration. Replace the TODO fields according to your
# cluster account, GPU partition, and preferred log directory before running
# `sbatch run_slurm.sh`.
#SBATCH --job-name=TODO_air_train
#SBATCH --partition=TODO_gpu_partition
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --comment=TODO_slurm_comment
#SBATCH -o /TODO/path/to/AIR/slurm-%j.out

set -euo pipefail

# -------------------------
# User configuration
# -------------------------
PROJECT_DIR="${PROJECT_DIR:-/TODO/path/to/AIR}"
CONDA_SH="${CONDA_SH:-/TODO/path/to/anaconda3/etc/profile.d/conda.sh}"
# Default conda env set as AIR
CONDA_ENV="${CONDA_ENV:-AIR}"

CONFIG="${CONFIG:-configs/3_div2k_refine_config.json}"
WORKSPACE="${WORKSPACE:-workspace/debug}"
# Optional. Set this to resume/inherit from a previous checkpoint.
# Leave empty to train from scratch. Compatible parameters will be loaded
# even if the network has been slightly modified.
CHECKPOINT="${CHECKPOINT:-}"

MAX_EPOCHS="${MAX_EPOCHS:-30}"
NUM_ITERATIONS="${NUM_ITERATIONS:-10000}"
BATCH_SIZE_FORWARD="${BATCH_SIZE_FORWARD:-5}"
VIS_EVERY="${VIS_EVERY:-2000}"
DEVICES_PER_NODE="${DEVICES_PER_NODE:-2}"

ENABLE_MIXED_PRECISION="${ENABLE_MIXED_PRECISION:-False}"
POD="${POD:-False}"

export PROJECT_DIR
export CONDA_SH
export CONDA_ENV
export CONFIG
export WORKSPACE
export CHECKPOINT
export MAX_EPOCHS
export NUM_ITERATIONS
export BATCH_SIZE_FORWARD
export VIS_EVERY
export DEVICES_PER_NODE
export ENABLE_MIXED_PRECISION
export POD

# -------------------------
# Environment setup
# -------------------------
if [[ -f "${CONDA_SH}" ]]; then
  source "${CONDA_SH}"
else
  source "${HOME}/.bashrc" || true
fi
conda activate "${CONDA_ENV}"

cd "${PROJECT_DIR}"

export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export NCCL_DEBUG=warn
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "========== SLURM multi-node job =========="
echo "JOB_ID=${SLURM_JOB_ID:-N/A}"
echo "NODELIST=${SLURM_JOB_NODELIST:-N/A}"
echo "NNODES=${SLURM_NNODES:-2}"
echo "DEVICES_PER_NODE=${DEVICES_PER_NODE}"
echo "CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}"
echo "Workdir=$(pwd)"
echo "Config=${CONFIG}"
echo "Workspace=${WORKSPACE}"
echo "Checkpoint=${CHECKPOINT:-<none>}"
echo "Start time: $(date)"
echo "-----------------------------------------------------"
echo "Using srun with Lightning DDP..."
echo "-----------------------------------------------------"

srun --kill-on-bad-exit=1 bash -lc '
  set -euo pipefail

  if [[ -f "${CONDA_SH}" ]]; then
    source "${CONDA_SH}"
  else
    source "${HOME}/.bashrc" || true
  fi
  conda activate "${CONDA_ENV}"

  echo "[node ${SLURM_NODEID:-?} task ${SLURM_PROCID:-?}] host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-unset}"
  python3 -V || true
  nvidia-smi -L || true

  train_args=(
    --config "${CONFIG}"
    --workspace "${WORKSPACE}"
    --max_epochs "${MAX_EPOCHS}"
    --num_iterations "${NUM_ITERATIONS}"
    --batch_size_forward "${BATCH_SIZE_FORWARD}"
    --pod "${POD}"
    --enable_mixed_precision "${ENABLE_MIXED_PRECISION}"
    --vis_every "${VIS_EVERY}"
    --num_nodes "${SLURM_NNODES:-2}"
    --devices "${DEVICES_PER_NODE}"
  )

  if [[ -n "${CHECKPOINT}" ]]; then
    train_args+=(--checkpoint "${CHECKPOINT}")
  fi

  python3 train/train_airnet_pl.py "${train_args[@]}"
'

echo "Finish time: $(date)"
echo "Logs: ${PROJECT_DIR}/slurm-${SLURM_JOB_ID:-N/A}.out"
