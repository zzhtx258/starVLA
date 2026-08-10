#!/usr/bin/env bash
set -euo pipefail

# Run LIBERO evaluation against the policy server started by start_model.sh.
#
# Usage:
#   bash run_libero.sh                    # libero_10, 5 trials/task, 10 tasks
#   bash run_libero.sh libero_goal 1 10   # suite, trials/task, max tasks

WORK_ROOT="${WORK_ROOT:-/group/ycyang/anupam}"
STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
LIBERO_HOME="${LIBERO_HOME:-${WORK_ROOT}/LIBERO}"
LIBERO_ENV="${LIBERO_ENV:-${WORK_ROOT}/miniconda3/envs/libero}"
LIBERO_PYTHON="${LIBERO_PYTHON:-${LIBERO_ENV}/bin/python}"
CKPT="${CKPT:-${WORK_ROOT}/starvla-models/Qwen2.5-VL-GR00T-LIBERO-4in1/checkpoints/steps_30000_pytorch_model.pt}"

TASK_SUITE="${1:-${TASK_SUITE:-libero_10}}"
NUM_TRIALS="${2:-${NUM_TRIALS:-5}}"
MAX_TASKS="${3:-${MAX_TASKS:-10}}"
GPU_ID="${GPU_ID:-0}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-6694}"
UNNORM_KEY="${UNNORM_KEY:-franka}"
VIDEO_OUT="${VIDEO_OUT:-${WORK_ROOT}/starvla-runs/eval_groot_${TASK_SUITE}_${MAX_TASKS}x${NUM_TRIALS}}"

if [[ ! -x "${LIBERO_PYTHON}" ]]; then
  echo "LIBERO Python not found: ${LIBERO_PYTHON}" >&2
  exit 1
fi

if [[ ! -d "${LIBERO_HOME}" ]]; then
  echo "LIBERO source not found: ${LIBERO_HOME}" >&2
  exit 1
fi

if [[ ! -f "${CKPT}" ]]; then
  echo "Checkpoint not found: ${CKPT}" >&2
  exit 1
fi

export PYTHONNOUSERSITE=1
export PYTHONPATH="${LIBERO_HOME}:${STARVLA_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/libero}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

echo "Starting LIBERO evaluation"
echo "  suite:       ${TASK_SUITE}"
echo "  trials/task: ${NUM_TRIALS}"
echo "  max tasks:   ${MAX_TASKS}"
echo "  server:      ${HOST}:${PORT}"
echo "  videos:      ${VIDEO_OUT}"

cd "${STARVLA_DIR}"
exec "${LIBERO_PYTHON}" \
  examples/simBenchmarks/LIBERO/eval_files/eval_libero.py \
  --args.host "${HOST}" \
  --args.port "${PORT}" \
  --args.task-suite-name "${TASK_SUITE}" \
  --args.num-trials-per-task "${NUM_TRIALS}" \
  --args.max-tasks "${MAX_TASKS}" \
  --args.unnorm-key "${UNNORM_KEY}" \
  --args.pretrained-path "${CKPT}" \
  --args.video-out-path "${VIDEO_OUT}"
