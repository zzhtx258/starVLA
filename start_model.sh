#!/usr/bin/env bash
set -euo pipefail

# Start the StarVLA policy server.
# Usage: GPU_ID=0 bash start_model.sh

WORK_ROOT="${WORK_ROOT:-/group/ycyang/anupam}"
STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
STARVLA_RESULTS="${STARVLA_RESULTS:-${STARVLA_DIR}/playground/results}"
MODELS_ROOT="${MODELS_ROOT:-${STARVLA_RESULTS}/models/hub}"
STARVLA_ENV="${STARVLA_ENV:-${WORK_ROOT}/miniconda3/envs/starvla}"
STARVLA_PYTHON="${STARVLA_PYTHON:-${STARVLA_ENV}/bin/python}"
# This cache is shared with other local projects and is intentionally kept
# outside STARVLA_RESULTS. Override HF_HOME if a private cache is preferred.
HF_HOME="${HF_HOME:-${WORK_ROOT}/huggingface_data}"
CKPT="${CKPT:-${MODELS_ROOT}/Qwen2.5-VL-GR00T-LIBERO-4in1/checkpoints/steps_30000_pytorch_model.pt}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-6694}"
USE_BF16="${USE_BF16:-1}"

if [[ ! -x "${STARVLA_PYTHON}" ]]; then
  echo "StarVLA Python not found: ${STARVLA_PYTHON}" >&2
  exit 1
fi

if [[ ! -f "${CKPT}" ]]; then
  echo "Checkpoint not found: ${CKPT}" >&2
  exit 1
fi

export STARVLA_DIR STARVLA_RESULTS MODELS_ROOT STARVLA_PYTHON HF_HOME CKPT GPU_ID PORT USE_BF16
export NO_ALBUMENTATIONS_UPDATE=1

echo "Starting StarVLA model server"
echo "  GPU:        ${GPU_ID}"
echo "  port:       ${PORT}"
echo "  checkpoint: ${CKPT}"

cd "${STARVLA_DIR}"
exec bash examples/simBenchmarks/LIBERO/eval_files/run_policy_server.sh
