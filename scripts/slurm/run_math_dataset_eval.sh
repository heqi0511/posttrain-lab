#!/usr/bin/env bash
set -euo pipefail

ENV_PATH="${POSTTRAIN_LAB_ENV:-/fs/nexus-scratch/qhe123/envs/posttrain-lab-trl}"

DATASET_ID="${DATASET_ID:?DATASET_ID is required}"
DATASET_CONFIG="${DATASET_CONFIG:-default}"
SPLIT="${SPLIT:-train}"
MODEL_NAME="${MODEL_NAME:?MODEL_NAME is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
SAMPLE_SIZE="${SAMPLE_SIZE:-50}"
SEED="${SEED:-20260529}"
SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-5000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"
ENABLE_THINKING="${ENABLE_THINKING:-false}"
REWARD_VERSION="${REWARD_VERSION:-math_boxed_v001}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export HF_HOME="${HF_HOME:-/fs/nexus-scratch/qhe123/.cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/fs/nexus-scratch/qhe123/pip-cache}"
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false

echo "host=$(hostname)"
echo "pwd=$(pwd)"
echo "dataset_id=${DATASET_ID}"
echo "dataset_config=${DATASET_CONFIG}"
echo "split=${SPLIT}"
echo "model_name=${MODEL_NAME}"
echo "output_dir=${OUTPUT_DIR}"
echo "sample_size=${SAMPLE_SIZE}"
echo "seed=${SEED}"
echo "max_new_tokens=${MAX_NEW_TOKENS}"
echo "temperature=${TEMPERATURE}"
echo "top_p=${TOP_P}"
echo "batch_size=${BATCH_SIZE}"
echo "torch_dtype=${TORCH_DTYPE}"
echo "enable_thinking=${ENABLE_THINKING}"
echo "reward_version=${REWARD_VERSION}"
echo "env=${ENV_PATH}"
echo "job_id=${SLURM_JOB_ID:-none}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

"${ENV_PATH}/bin/python" - <<'PY'
import datasets
import torch
import transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "cuda_build", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
print("transformers", transformers.__version__)
print("datasets", datasets.__version__)
PY

"${ENV_PATH}/bin/python" -m posttrain_lab.eval.math_dataset_eval \
  --dataset-id "${DATASET_ID}" \
  --dataset-config "${DATASET_CONFIG}" \
  --split "${SPLIT}" \
  --model-name "${MODEL_NAME}" \
  --output-dir "${OUTPUT_DIR}" \
  --sample-size "${SAMPLE_SIZE}" \
  --seed "${SEED}" \
  --shuffle-buffer-size "${SHUFFLE_BUFFER_SIZE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --batch-size "${BATCH_SIZE}" \
  --torch-dtype "${TORCH_DTYPE}" \
  --enable-thinking "${ENABLE_THINKING}" \
  --reward-version "${REWARD_VERSION}" \
  --trust-remote-code
