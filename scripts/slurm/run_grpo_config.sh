#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:?usage: run_grpo_config.sh <config.yaml>}"
ENV_PATH="${POSTTRAIN_LAB_ENV:-/fs/nexus-scratch/qhe123/envs/posttrain-lab-trl}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export HF_HOME="${HF_HOME:-/fs/nexus-scratch/qhe123/.cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/fs/nexus-scratch/qhe123/pip-cache}"
export PYTHONPATH="${PYTHONPATH:-src}"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"

echo "host=$(hostname)"
echo "pwd=$(pwd)"
echo "config=${CONFIG_PATH}"
echo "env=${ENV_PATH}"
echo "job_id=${SLURM_JOB_ID:-none}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "slurm_gpus_on_node=${SLURM_GPUS_ON_NODE:-unset}"
echo "slurm_cpus_per_task=${SLURM_CPUS_PER_TASK:-unset}"
nvidia-smi || true

"${ENV_PATH}/bin/python" - <<'PY'
import importlib
import torch

print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "cuda_build", torch.version.cuda)
print("cuda_device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        print(f"gpu[{index}]", torch.cuda.get_device_name(index))
for name in ["transformers", "datasets", "trl", "peft", "accelerate", "sympy"]:
    module = importlib.import_module(name)
    print(name, getattr(module, "__version__", "unknown"))
try:
    module = importlib.import_module("latex2sympy2")
    print("latex2sympy2", getattr(module, "__version__", "unknown"))
except Exception as exc:
    print("latex2sympy2", "MISSING", type(exc).__name__)
PY

NPROC_PER_NODE="${NPROC_PER_NODE:-}"
if [[ -z "${NPROC_PER_NODE}" ]]; then
  NPROC_PER_NODE="$("${ENV_PATH}/bin/python" - <<'PY'
import torch
print(max(1, torch.cuda.device_count()))
PY
)"
fi

echo "nproc_per_node=${NPROC_PER_NODE}"

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  "${ENV_PATH}/bin/python" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    -m posttrain_lab.train.train_grpo \
    --config "${CONFIG_PATH}"
else
  "${ENV_PATH}/bin/python" -m posttrain_lab.train.train_grpo --config "${CONFIG_PATH}"
fi
