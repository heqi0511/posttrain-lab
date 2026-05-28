#!/bin/bash
#SBATCH --job-name=gsm8k-sweep
#SBATCH --partition=cbcb-heng
#SBATCH --account=cbcb-heng
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=08:00:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-5
#SBATCH --output=/fs/nexus-scratch/qhe123/posttrain-lab-worktrees/5a83769-gsm8k-frontier-audit/slurm/gsm8k_sweep_%A_%a.out
#SBATCH --error=/fs/nexus-scratch/qhe123/posttrain-lab-worktrees/5a83769-gsm8k-frontier-audit/slurm/gsm8k_sweep_%A_%a.err

set -euo pipefail

WORKTREE=/fs/nexus-scratch/qhe123/posttrain-lab-worktrees/5a83769-gsm8k-frontier-audit
PYTHON=/fs/nexus-scratch/qhe123/envs/posttrain-lab-trl/bin/python3
CONFIG=configs/audit/gsm8k_qwen3_4b_prompt_sweep.yaml
VARIANTS=(current_128 strong_128 strong_512 strong_1024 strong_nonthinking_512 strong_thinking_1024)
VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}

cd "$WORKTREE"
export PYTHONPATH=src
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/fs/nexus-scratch/qhe123/hf-cache

echo "host=$(hostname)"
echo "worktree=$WORKTREE"
echo "variant=$VARIANT"
git rev-parse HEAD
$PYTHON --version
nvidia-smi

$PYTHON runs/audit/gsm8k_qwen3_4b_prompt_sweep/run_prompt_sweep.py --config "$CONFIG" --variant "$VARIANT"
