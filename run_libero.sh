#!/usr/bin/env bash
set -euo pipefail

# Run LIBERO evaluation against the policy server started by start_model.sh.
#
# Usage:
#   bash run_libero.sh                    # libero_10, 5 trials/task, 10 tasks
#   bash run_libero.sh libero_goal 1 10   # suite, trials/task, max tasks
#   TASK_IDS=8 NUM_TRIALS=50 bash run_libero.sh
#   TASK_IDS=8 NUM_TRIALS=50 REPLAN_INTERVAL=4 bash run_libero.sh
#   TASK_IDS=8 REPLAN_INTERVAL=6 ACTION_ENSEMBLE=0 bash run_libero.sh

WORK_ROOT="${WORK_ROOT:-/group/ycyang/anupam}"
STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
STARVLA_RESULTS="${STARVLA_RESULTS:-${STARVLA_DIR}/playground/results}"
MODELS_ROOT="${MODELS_ROOT:-${STARVLA_RESULTS}/models/hub}"
RUNS_ROOT="${RUNS_ROOT:-${STARVLA_RESULTS}/runs}"
LIBERO_HOME="${LIBERO_HOME:-${WORK_ROOT}/LIBERO}"
LIBERO_ENV="${LIBERO_ENV:-${WORK_ROOT}/miniconda3/envs/libero}"
LIBERO_PYTHON="${LIBERO_PYTHON:-${LIBERO_ENV}/bin/python}"
CKPT="${CKPT:-${MODELS_ROOT}/Qwen2.5-VL-GR00T-LIBERO-4in1/checkpoints/steps_30000_pytorch_model.pt}"

TASK_SUITE="${1:-${TASK_SUITE:-libero_10}}"
NUM_TRIALS="${2:-${NUM_TRIALS:-5}}"
MAX_TASKS="${3:-${MAX_TASKS:-10}}"
TASK_IDS="${TASK_IDS:-}"
REPLAN_INTERVAL="${REPLAN_INTERVAL:-0}"
ACTION_ENSEMBLE="${ACTION_ENSEMBLE:-0}"
GPU_ID="${GPU_ID:-0}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-6694}"
UNNORM_KEY="${UNNORM_KEY:-franka}"
if [[ -n "${TASK_IDS}" ]]; then
  TASK_LABEL="task${TASK_IDS//,/-}"
else
  TASK_LABEL="${MAX_TASKS}tasks"
fi
if [[ "${REPLAN_INTERVAL}" -gt 0 ]]; then
  if [[ "${ACTION_ENSEMBLE}" == "1" ]]; then
    REPLAN_LABEL="replan${REPLAN_INTERVAL}_ensemble"
  else
    REPLAN_LABEL="replan${REPLAN_INTERVAL}_direct"
  fi
else
  REPLAN_LABEL="replan_full"
fi
MODEL_ROOT="${CKPT%%/checkpoints/*}"
MODEL_LABEL="${MODEL_LABEL:-$(basename "${MODEL_ROOT}")}"
VIDEO_OUT="${VIDEO_OUT:-${RUNS_ROOT}/${MODEL_LABEL}/${TASK_SUITE}_${TASK_LABEL}_${NUM_TRIALS}trials_${REPLAN_LABEL}}"

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
echo "  task IDs:    ${TASK_IDS:-all selected by max tasks}"
echo "  replan:      ${REPLAN_INTERVAL} (0 means full action chunk)"
echo "  ensemble:    ${ACTION_ENSEMBLE} (0 = direct replacement, 1 = blend overlapping chunks)"
echo "  server:      ${HOST}:${PORT}"
echo "  videos:      ${VIDEO_OUT}"

cd "${STARVLA_DIR}"
CMD=(
  "${LIBERO_PYTHON}"
  examples/simBenchmarks/LIBERO/eval_files/eval_libero.py \
  --args.host "${HOST}" \
  --args.port "${PORT}" \
  --args.task-suite-name "${TASK_SUITE}" \
  --args.num-trials-per-task "${NUM_TRIALS}" \
  --args.max-tasks "${MAX_TASKS}" \
  --args.replan-interval "${REPLAN_INTERVAL}" \
  --args.action-ensemble "${ACTION_ENSEMBLE}" \
  --args.unnorm-key "${UNNORM_KEY}" \
  --args.pretrained-path "${CKPT}" \
  --args.video-out-path "${VIDEO_OUT}"
)

if [[ -n "${TASK_IDS}" ]]; then
  CMD+=(--args.task-ids "${TASK_IDS}")
fi

exec "${CMD[@]}"
