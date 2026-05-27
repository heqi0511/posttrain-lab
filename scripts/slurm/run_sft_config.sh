#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:?usage: run_sft_config.sh <config.yaml>}"
ENV_PATH="${POSTTRAIN_LAB_ENV:-/fs/nexus-scratch/qhe123/envs/posttrain-lab-trl}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export HF_HOME="${HF_HOME:-/fs/nexus-scratch/qhe123/.cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/fs/nexus-scratch/qhe123/pip-cache}"
export PYTHONPATH="${PYTHONPATH:-src}"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false

echo "host=$(hostname)"
echo "pwd=$(pwd)"
echo "config=${CONFIG_PATH}"
echo "env=${ENV_PATH}"
echo "job_id=${SLURM_JOB_ID:-none}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

"${ENV_PATH}/bin/python" - <<'PY'
import torch
import transformers
import datasets
import trl
import peft
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "cuda_build", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
print("transformers", transformers.__version__)
print("datasets", datasets.__version__)
print("trl", trl.__version__)
print("peft", peft.__version__)
PY

"${ENV_PATH}/bin/python" -m posttrain_lab.train.train_sft --config "${CONFIG_PATH}"
