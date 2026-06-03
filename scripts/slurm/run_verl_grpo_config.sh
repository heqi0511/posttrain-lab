#!/usr/bin/env bash
set -euo pipefail

CONFIG_SCRIPT="${1:?usage: run_verl_grpo_config.sh <config.sh>}"
ENV_PATH="${POSTTRAIN_LAB_VERL_ENV:-/fs/nexus-scratch/qhe123/envs/posttrain-lab-verl}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export POSTTRAIN_LAB_REPO="${POSTTRAIN_LAB_REPO:-$(pwd)}"
export VERL_REPO_ROOT="${VERL_REPO_ROOT:-/fs/nexus-scratch/qhe123/src/verl}"
export HF_HOME="${HF_HOME:-/fs/nexus-scratch/qhe123/.cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/fs/nexus-scratch/qhe123/pip-cache}"
export PYTHONPATH="${POSTTRAIN_LAB_REPO}/src:${VERL_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export PATH="${ENV_PATH}/bin:${PATH}"

echo "host=$(hostname)"
echo "pwd=$(pwd)"
echo "config_script=${CONFIG_SCRIPT}"
echo "env=${ENV_PATH}"
echo "verl_repo=${VERL_REPO_ROOT}"
echo "posttrain_lab_repo=${POSTTRAIN_LAB_REPO}"
echo "job_id=${SLURM_JOB_ID:-none}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "slurm_gpus_on_node=${SLURM_GPUS_ON_NODE:-unset}"
echo "slurm_cpus_per_task=${SLURM_CPUS_PER_TASK:-unset}"
nvidia-smi || true

python - <<'PY'
import importlib
import torch

print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "cuda_build", torch.version.cuda)
print("cuda_device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        print(f"gpu[{index}]", torch.cuda.get_device_name(index))
for name in ["transformers", "verl", "vllm", "pyarrow", "flash_attn", "latex2sympy2_extended"]:
    module = importlib.import_module(name)
    print(name, getattr(module, "__version__", "unknown"))
PY

bash "${CONFIG_SCRIPT}"
