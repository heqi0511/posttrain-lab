#!/usr/bin/env bash
set -euo pipefail

# Paper-style small GRPO smoke for Section 4.1 / Appendix D.5:
# Qwen2.5-Math-1.5B, DAPO17k, n=4, max prompt 1024, max response 2048,
# lr=2e-6, clip ratio 0.22, KL disabled. Batch sizes are reduced for Nexus.

POSTTRAIN_LAB_REPO="${POSTTRAIN_LAB_REPO:-$(pwd)}"
VERL_REPO_ROOT="${VERL_REPO_ROOT:-/fs/nexus-scratch/qhe123/src/verl}"
MODEL_PATH="${MODEL_PATH:-/fs/nexus-scratch/qhe123/models/Qwen2.5-Math-1.5B}"
TRAIN_FILE="${TRAIN_FILE:-/fs/nexus-scratch/qhe123/datasets/verl_parquet/dapo_math_raw_17k_boxed/train.parquet}"
MATH500_FILE="${MATH500_FILE:-/fs/nexus-scratch/qhe123/datasets/verl_parquet/math500/test.parquet}"
AMC23_FILE="${AMC23_FILE:-/fs/nexus-scratch/qhe123/datasets/verl_parquet/amc23/test.parquet}"
OLYMPIAD_FILE="${OLYMPIAD_FILE:-/fs/nexus-scratch/qhe123/datasets/verl_parquet/olympiadbench/test.parquet}"
REWARD_FN_PATH="${REWARD_FN_PATH:-${POSTTRAIN_LAB_REPO}/src/posttrain_lab/rewards/verl_math_reward.py}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${POSTTRAIN_LAB_REPO}/runs/verl/qwen25_math_1_5b_dapo_grpo_smoke}"

NGPUS_PER_NODE="${NGPUS_PER_NODE:-4}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-64}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-64}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-4}"
SAVE_FREQ="${SAVE_FREQ:-2}"
TEST_FREQ="${TEST_FREQ:-2}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.5}"

export PYTHONPATH="${POSTTRAIN_LAB_REPO}/src:${VERL_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-/fs/nexus-scratch/qhe123/.cache/huggingface}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_ALLREDUCE_USE_SYMM_MEM="${VLLM_ALLREDUCE_USE_SYMM_MEM:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

mkdir -p "${OUTPUT_ROOT}"

cd "${VERL_REPO_ROOT}"

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="['${MATH500_FILE}','${AMC23_FILE}','${OLYMPIAD_FILE}']" \
  data.train_max_samples="${TRAIN_MAX_SAMPLES}" \
  data.val_max_samples="${VAL_MAX_SAMPLES}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length=1024 \
  data.max_response_length=2048 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  data.dataloader_num_workers=2 \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=0.000002 \
  actor_rollout_ref.actor.optim.clip_grad=1.0 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.actor.ppo_epochs=2 \
  actor_rollout_ref.actor.clip_ratio=0.22 \
  actor_rollout_ref.actor.clip_ratio_low=0.22 \
  actor_rollout_ref.actor.clip_ratio_high=0.22 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.temperature=0.8 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.max_model_len=3072 \
  actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enable_prefix_caching=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.8 \
  actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
  actor_rollout_ref.rollout.val_kwargs.n=16 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096 \
  reward.reward_manager.name=dapo \
  reward.custom_reward_function.path="${REWARD_FN_PATH}" \
  reward.custom_reward_function.name=compute_score_verl_style \
  +reward.custom_reward_function.reward_kwargs.allow_symbolic_equivalence=True \
  +reward.custom_reward_function.reward_kwargs.symbolic_equivalence_engine=sympy \
  +reward.custom_reward_function.reward_kwargs.max_symbolic_expr_chars=200 \
  +reward.custom_reward_function.reward_kwargs.max_symbolic_ast_nodes=128 \
  +reward.custom_reward_function.reward_kwargs.max_symbolic_collection_size=32 \
  +reward.custom_reward_function.reward_kwargs.format_reward=0.1 \
  trainer.critic_warmup=0 \
  trainer.logger=console \
  trainer.project_name=posttrain_lab_verl \
  trainer.experiment_name=qwen25_math_1_5b_dapo_grpo_smoke \
  trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
  trainer.nnodes=1 \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  trainer.total_epochs=1 \
  trainer.val_before_train=True \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.default_local_dir="${OUTPUT_ROOT}/checkpoints" \
  trainer.rollout_data_dir="${OUTPUT_ROOT}/rollouts" \
  trainer.validation_data_dir="${OUTPUT_ROOT}/validation" \
  trainer.log_val_generations=8 \
  trainer.resume_mode=disable \
  "$@"
