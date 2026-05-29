# Reference Repositories

Date: 2026-05-29

Scope: reference study for the current Posttrain Lab path: GSM8K-style math data,
Qwen3-4B, boxed-answer reward, TRL-first SFT/GRPO, frontier audit before larger
RLVR runs. This document is for design guidance only. Do not vendor code from
these repositories without a separate review.

## Relevance Ranking

| Rank | Repo | Why it matters now |
| ---: | --- | --- |
| 1 | [huggingface/open-r1](https://github.com/huggingface/open-r1) | Closest to our current workflow: reasoning data, SFT, GRPO, reward functions, pass-rate filtering, and eval scripts. |
| 2 | [huggingface/trl](https://github.com/huggingface/trl) | Our MVP backend. It defines the trainer APIs and config surface for SFT, GRPO, PPO, DPO, and reward modeling. |
| 3 | [hiyouga/EasyR1](https://github.com/hiyouga/EasyR1) | Practical Qwen3-4B math GRPO examples, prompt templates, reward functions, and verl-style scaling path. |
| 4 | [verl-project/verl](https://github.com/verl-project/verl) | Best reference for later migration after reward/eval/frontier audit stabilize. Too heavy for MVP, but important for scale. |
| 5 | [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | Strong reference for Ray/vLLM RLHF systems, experience generation, KL control, and custom reward functions. |
| 6 | [axolotl-ai-cloud/axolotl](https://github.com/axolotl-ai-cloud/axolotl) | Useful config and dataset ergonomics for SFT/LoRA/QLoRA and newer GRPO support, but broader than our focused lab. |
| 7 | [allenai/reward-bench](https://github.com/allenai/reward-bench) | Useful for reward-model evaluation discipline and standardized reporting, less central for deterministic GSM8K rewards. |
| 8 | [RLHFlow/RLHF-Reward-Modeling](https://github.com/RLHFlow/RLHF-Reward-Modeling) | Good reward-modeling and PRM/ORM reference; mostly future work because our first reward is deterministic. |
| 9 | [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | Mature SFT/DPO/RM/PPO training UX, but it is less aligned with our TRL-first GRPO path. |
| 10 | [MajoRoth/hack-verifiable-environments](https://github.com/MajoRoth/hack-verifiable-environments) | Useful reward-hacking evaluation concepts for future verifiable environments. Not needed for immediate GSM8K GRPO. |
| 11 | [JonathanGabor/EvilGenie](https://github.com/JonathanGabor/EvilGenie) | Relevant reward-hacking benchmark, but deprecated in favor of `evilgenie_inspect`; use only as conceptual reference. |
| 12 | [healthylaife/Composite-LLM-Reward-Model](https://github.com/healthylaife/Composite-LLM-Reward-Model) | Relevant paper artifact for composite reward hacking mitigation, but the repo is minimal and not a training framework. |

## Capability Matrix

Legend: Yes = first-class support; Partial = useful examples or related tooling;
No = not a meaningful repo focus for our purpose.

| Repo | SFT | GRPO | PPO | DPO | Reward modeling | Eval | Reward hacking detection / mitigation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `huggingface/trl` | Yes | Yes | Yes | Yes | Yes | Partial | No |
| `huggingface/open-r1` | Yes | Yes | No | No | Partial verifier rewards | Yes | Partial via pass-rate filtering |
| `verl-project/verl` | Partial | Yes | Yes | No | Partial reward managers | Partial | No |
| `OpenRLHF/OpenRLHF` | Yes | Yes | Yes | Yes | Yes | Partial | No |
| `axolotl-ai-cloud/axolotl` | Yes | Yes | No | Yes | Yes | Partial | No |
| `hiyouga/LLaMA-Factory` | Yes | No | Yes | Yes | Yes | Partial | No |
| `hiyouga/EasyR1` | No | Yes | No | No | Partial verifier rewards | Partial | No |
| `allenai/reward-bench` | No | No | No | Partial DPO evaluation | Yes, evaluation | Yes | Partial safety/reward stress tests |
| `RLHFlow/RLHF-Reward-Modeling` | No | No | No | No | Yes | Partial | Partial reward-hacking mitigation ideas |
| `healthylaife/Composite-LLM-Reward-Model` | No | No | No | No | Partial | Partial | Yes, composite reward mitigation |
| `JonathanGabor/EvilGenie` | No | No | No | No | No | Yes | Yes |
| `MajoRoth/hack-verifiable-environments` | No | No | No | No | No | Yes | Yes |

## Repository Notes

### 1. huggingface/open-r1

- Problem solved: open reproduction path for R1-style reasoning, including SFT,
  GRPO, synthetic data generation, reward functions, and evaluation.
- Supports: SFT and GRPO directly; eval and pass-rate filtering directly; no
  general PPO/DPO/RM stack.
- Files worth reading:
  - [`src/open_r1/grpo.py`](https://github.com/huggingface/open-r1/blob/main/src/open_r1/grpo.py)
  - [`src/open_r1/sft.py`](https://github.com/huggingface/open-r1/blob/main/src/open_r1/sft.py)
  - [`src/open_r1/rewards.py`](https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py)
  - [`scripts/pass_rate_filtering/compute_pass_rate.py`](https://github.com/huggingface/open-r1/blob/main/scripts/pass_rate_filtering/compute_pass_rate.py)
  - [`scripts/pass_rate_filtering/README.md`](https://github.com/huggingface/open-r1/blob/main/scripts/pass_rate_filtering/README.md)
  - [`recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo.yaml`](https://github.com/huggingface/open-r1/blob/main/recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo.yaml)
- Patterns to adopt:
  - Treat pass-rate/frontier filtering as a data stage before RL.
  - Keep reward functions in one explicit module with tests.
  - Keep recipe configs separate from code.
  - Keep benchmark/eval scripts separate from training scripts.
- Too heavy or irrelevant now:
  - Code execution rewards, E2B routing, Morph/Piston clients, and large-scale
    distillation are not needed for GSM8K boxed-answer RLVR.
- Relation to our modules:
  - Data: strong reference for pass-rate filtering and decontamination scripts.
  - Reward: useful contrast to our stricter exactly-one-boxed parser.
  - Eval: useful for fixed benchmark scripts.
  - Frontier audit: strongest direct reference.
  - SFT: useful minimal SFT script shape.
  - GRPO: useful TRL-based GRPO script shape.
  - Run card: less explicit than our run-card requirement.
  - Reward hacking: filtering helps avoid no-signal prompts but is not a full
    hacking detector.

### 2. huggingface/trl

- Problem solved: general post-training trainer library on top of Transformers,
  Accelerate, and PEFT.
- Supports: SFT, GRPO, PPO, DPO, reward modeling, online DPO, and related
  trainers. Eval is not the main product, but tests and examples are extensive.
- Files worth reading:
  - [`trl/trainer/sft_trainer.py`](https://github.com/huggingface/trl/blob/main/trl/trainer/sft_trainer.py)
  - [`trl/trainer/sft_config.py`](https://github.com/huggingface/trl/blob/main/trl/trainer/sft_config.py)
  - [`trl/trainer/grpo_trainer.py`](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py)
  - [`trl/trainer/grpo_config.py`](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_config.py)
  - [`trl/trainer/reward_trainer.py`](https://github.com/huggingface/trl/blob/main/trl/trainer/reward_trainer.py)
  - [`examples/scripts/sft.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py)
  - [`examples/scripts/grpo_2048.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/grpo_2048.py)
  - [`tests/test_grpo_trainer.py`](https://github.com/huggingface/trl/blob/main/tests/test_grpo_trainer.py)
- Patterns to adopt:
  - Follow trainer config names closely instead of inventing incompatible config
    abstractions.
  - Keep dry-run/tiny-model tests around config parsing and trainer construction.
  - Log generation length, reward stats, and parse failures outside the trainer
    when the trainer does not expose exactly what we need.
- Too heavy or irrelevant now:
  - Async GRPO, VLM trainers, OpenEnv agents, and PPO examples are not needed
    until our GSM8K reward/eval loop is stable.
- Relation to our modules:
  - Data: consumes datasets but does not define our strict JSONL schema.
  - Reward: callable reward interface is directly relevant.
  - Eval: requires our own eval runner.
  - Frontier audit: sampling behavior should match TRL generation settings.
  - SFT: primary backend.
  - GRPO: primary MVP backend.
  - Run card: our repo must add run-card discipline around TRL.
  - Reward hacking: TRL is mechanism, not mitigation.

### 3. hiyouga/EasyR1

- Problem solved: efficient multimodal RL training framework based on verl,
  with practical Qwen and Qwen-VL GRPO examples.
- Supports: GRPO and related RL algorithms such as DAPO, Reinforce++, ReMax,
  RLOO, GSPO, and CISPO. It is not an SFT-first framework.
- Files worth reading:
  - [`examples/qwen3_4b_math_grpo.sh`](https://github.com/hiyouga/EasyR1/blob/main/examples/qwen3_4b_math_grpo.sh)
  - [`examples/qwen3_4b_math_grpo_lora.sh`](https://github.com/hiyouga/EasyR1/blob/main/examples/qwen3_4b_math_grpo_lora.sh)
  - [`examples/reward_function/math.py`](https://github.com/hiyouga/EasyR1/blob/main/examples/reward_function/math.py)
  - [`examples/format_prompt/math.jinja`](https://github.com/hiyouga/EasyR1/blob/main/examples/format_prompt/math.jinja)
  - [`verl/trainer/data_loader.py`](https://github.com/hiyouga/EasyR1/blob/main/verl/trainer/data_loader.py)
  - [`verl/trainer/metrics.py`](https://github.com/hiyouga/EasyR1/blob/main/verl/trainer/metrics.py)
- Patterns to adopt:
  - Keep the math prompt template explicit and versioned.
  - Separate format reward from accuracy reward in reports, even if our actual
    reward remains a single deterministic score.
  - Use Qwen3-4B-specific examples as a migration reference after TRL.
- Too heavy or irrelevant now:
  - The verl fork, Ray/vLLM execution, multimodal support, and Docker/Apptainer
    environment are too heavy for the current TRL MVP.
- Relation to our modules:
  - Data: useful target format for RL prompts.
  - Reward: useful reference for reporting format and accuracy separately.
  - Eval: not enough as our official eval.
  - Frontier audit: useful sampling and prompt settings reference.
  - SFT: out of scope.
  - GRPO: strong Qwen3-4B reference.
  - Run card: we still need our own run-card artifacts.
  - Reward hacking: no direct detector.

### 4. verl-project/verl

- Problem solved: scalable, production-oriented RL post-training framework with
  HybridFlow, distributed rollout/training, and multiple inference backends.
- Supports: GRPO and PPO directly; reward managers and reward-score utilities;
  eval and data preprocessing examples. SFT is not the central path.
- Files worth reading:
  - [`examples/grpo_trainer/README.md`](https://github.com/verl-project/verl/blob/main/examples/grpo_trainer/README.md)
  - [`examples/grpo_trainer/run_qwen3_4b_fsdp.sh`](https://github.com/verl-project/verl/blob/main/examples/grpo_trainer/run_qwen3_4b_fsdp.sh)
  - [`examples/data_preprocess/gsm8k.py`](https://github.com/verl-project/verl/blob/main/examples/data_preprocess/gsm8k.py)
  - [`verl/trainer/config/ppo_trainer.yaml`](https://github.com/verl-project/verl/blob/main/verl/trainer/config/ppo_trainer.yaml)
  - [`verl/workers/reward_manager/`](https://github.com/verl-project/verl/tree/main/verl/workers/reward_manager)
  - [`verl/utils/reward_score/gsm8k.py`](https://github.com/verl-project/verl/blob/main/verl/utils/reward_score/gsm8k.py)
- Patterns to adopt:
  - Keep rollout, reward, actor, reference, and trainer configs clearly
    separated.
  - Treat engine/backends as replaceable infrastructure, not mixed into reward
    or eval semantics.
  - Use explicit reward manager boundaries if we migrate.
- Too heavy or irrelevant now:
  - FSDP/Megatron/Ray/vLLM/SGlang/large-model placement logic is premature
    before our reward, eval, and frontier selection are stable.
- Relation to our modules:
  - Data: useful GSM8K preprocessing reference.
  - Reward: reward managers are a future architecture reference.
  - Eval: partial only.
  - Frontier audit: useful once audit needs distributed rollout.
  - SFT: not our SFT source.
  - GRPO: future scale backend.
  - Run card: our artifacts should wrap verl jobs if migrated.
  - Reward hacking: no direct detector.

### 5. OpenRLHF/OpenRLHF

- Problem solved: scalable RLHF/agentic RL framework using Ray, vLLM, and
  DeepSpeed, with SFT/RM/DPO/PPO and newer RL variants.
- Supports: SFT, reward modeling, DPO, PPO, GRPO-family algorithms, custom
  reward functions, Ray/vLLM rollout, and Slurm examples.
- Files worth reading:
  - [`examples/python/math_reward_func.py`](https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/python/math_reward_func.py)
  - [`examples/scripts/train_sft.sh`](https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/scripts/train_sft.sh)
  - [`examples/scripts/train_rm.sh`](https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/scripts/train_rm.sh)
  - [`examples/scripts/train_ppo_with_reward_fn.sh`](https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/scripts/train_ppo_with_reward_fn.sh)
  - [`openrlhf/trainer/ppo_utils/experience_maker.py`](https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/trainer/ppo_utils/experience_maker.py)
  - [`openrlhf/trainer/ppo_utils/kl_controller.py`](https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/trainer/ppo_utils/kl_controller.py)
  - [`openrlhf/datasets/prompts_dataset.py`](https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/datasets/prompts_dataset.py)
- Patterns to adopt:
  - Study experience generation and KL logging before scaling beyond TRL.
  - Keep reward functions injectable and isolated from policy code.
  - Treat Slurm scripts as reproducible launch artifacts.
- Too heavy or irrelevant now:
  - Ray/vLLM/DeepSpeed hybrid-engine complexity is not justified for the first
    GSM8K Qwen3-4B experiments.
- Relation to our modules:
  - Data: prompt dataset handling is relevant.
  - Reward: custom reward function boundary is relevant.
  - Eval: not a replacement for our frozen eval suite.
  - Frontier audit: rollout infrastructure is a future reference.
  - SFT: reference, but we stay with TRL for MVP.
  - GRPO: future scale/backend reference.
  - Run card: our repo should keep stronger run-card requirements.
  - Reward hacking: no direct detector.

### 6. axolotl-ai-cloud/axolotl

- Problem solved: broad, config-first LLM fine-tuning framework with LoRA,
  QLoRA, full fine-tuning, preference tuning, reward modeling, and newer RL
  support.
- Supports: SFT, DPO, reward modeling, GRPO/GDPO; no central PPO path for our
  purpose.
- Files worth reading:
  - [`docs/grpo.qmd`](https://github.com/axolotl-ai-cloud/axolotl/blob/main/docs/grpo.qmd)
  - [`src/axolotl/core/trainers/grpo/trainer.py`](https://github.com/axolotl-ai-cloud/axolotl/blob/main/src/axolotl/core/trainers/grpo/trainer.py)
  - [`src/axolotl/core/trainers/grpo/async_trainer.py`](https://github.com/axolotl-ai-cloud/axolotl/blob/main/src/axolotl/core/trainers/grpo/async_trainer.py)
  - [`src/axolotl/core/trainers/grpo/replay_buffer.py`](https://github.com/axolotl-ai-cloud/axolotl/blob/main/src/axolotl/core/trainers/grpo/replay_buffer.py)
  - [`examples/qwen3/qlora-fsdp.yaml`](https://github.com/axolotl-ai-cloud/axolotl/blob/main/examples/qwen3/qlora-fsdp.yaml)
  - [`examples/qwen3/reward-model.yaml`](https://github.com/axolotl-ai-cloud/axolotl/blob/main/examples/qwen3/reward-model.yaml)
- Patterns to adopt:
  - Config-first training recipes with strong examples.
  - Clear separation between dataset format docs and trainer internals.
  - Optional callbacks for generation/perplexity can inspire later reporting.
- Too heavy or irrelevant now:
  - Very broad model zoo, FSDP2, async GRPO, replay buffers, and production
    config surface exceed our current scope.
- Relation to our modules:
  - Data: dataset-format documentation is useful.
  - Reward: reward-model configs are future work.
  - Eval: only partial.
  - Frontier audit: replay-buffer and async sampling ideas are future work.
  - SFT: useful config ergonomics reference.
  - GRPO: useful but not our backend.
  - Run card: our explicit run cards are still needed.
  - Reward hacking: no direct detector.

### 7. allenai/reward-bench

- Problem solved: standardized benchmark and leaderboard for reward-model
  evaluation, including safety and preference sets.
- Supports: reward-model evaluation, DPO/implicit reward evaluation, generative
  judge evaluation, and analysis scripts. It does not train SFT/GRPO/PPO.
- Files worth reading:
  - [`rewardbench/rewardbench.py`](https://github.com/allenai/reward-bench/blob/main/rewardbench/rewardbench.py)
  - [`rewardbench/utils.py`](https://github.com/allenai/reward-bench/blob/main/rewardbench/utils.py)
  - [`rewardbench/generative.py`](https://github.com/allenai/reward-bench/blob/main/rewardbench/generative.py)
  - [`scripts/run_rm.py`](https://github.com/allenai/reward-bench/blob/main/scripts/run_rm.py)
  - [`scripts/run_dpo.py`](https://github.com/allenai/reward-bench/blob/main/scripts/run_dpo.py)
  - [`scripts/run_v2.py`](https://github.com/allenai/reward-bench/blob/main/scripts/run_v2.py)
  - [`scripts/configs/eval_configs.yaml`](https://github.com/allenai/reward-bench/blob/main/scripts/configs/eval_configs.yaml)
- Patterns to adopt:
  - Keep evaluation formatting and model adapters separate from metrics.
  - Report category-level scores rather than only one aggregate.
  - Preserve failed examples and analysis outputs.
- Too heavy or irrelevant now:
  - Preference-model benchmarking is not needed for deterministic boxed GSM8K
    reward, but it will matter if we add learned reward models.
- Relation to our modules:
  - Data: less relevant.
  - Reward: strong future reference for learned reward evaluation.
  - Eval: strong reporting discipline reference.
  - Frontier audit: not directly related.
  - SFT/GRPO: not related.
  - Run card: useful for standardized report fields.
  - Reward hacking: partial safety/stress-test reference, not direct hacking.

### 8. RLHFlow/RLHF-Reward-Modeling

- Problem solved: recipes for Bradley-Terry reward models, pairwise preference
  models, ArmoRM, process/outcome math reward models, and related evaluation.
- Supports: reward modeling strongly; reward-bench evaluation scripts; no SFT or
  GRPO trainer path.
- Files worth reading:
  - [`math-rm/README.md`](https://github.com/RLHFlow/RLHF-Reward-Modeling/blob/main/math-rm/README.md)
  - [`math-rm/scalar_orm_train.py`](https://github.com/RLHFlow/RLHF-Reward-Modeling/blob/main/math-rm/scalar_orm_train.py)
  - [`math-rm/orm_evaluate.py`](https://github.com/RLHFlow/RLHF-Reward-Modeling/blob/main/math-rm/orm_evaluate.py)
  - [`math-rm/prm_evaluate.py`](https://github.com/RLHFlow/RLHF-Reward-Modeling/blob/main/math-rm/prm_evaluate.py)
  - [`armo-rm/stage-1_train.py`](https://github.com/RLHFlow/RLHF-Reward-Modeling/blob/main/armo-rm/stage-1_train.py)
  - [`armo-rm/stage-2_train.py`](https://github.com/RLHFlow/RLHF-Reward-Modeling/blob/main/armo-rm/stage-2_train.py)
  - [`useful_code/eval_reward_bench_bt.py`](https://github.com/RLHFlow/RLHF-Reward-Modeling/blob/main/useful_code/eval_reward_bench_bt.py)
- Patterns to adopt:
  - Separate ORM/PRM training, evaluation, and reward-card documentation.
  - Track length bias and reward-model-specific failure modes.
  - Treat reward modeling as a later milestone, not mixed into verifier reward.
- Too heavy or irrelevant now:
  - Learned reward models and PRMs are overkill while boxed-answer exact reward
    still works for GSM8K.
- Relation to our modules:
  - Data: future preference/process data reference.
  - Reward: future learned RM/PRM reference.
  - Eval: reward-model eval reference.
  - Frontier audit: not directly relevant.
  - SFT/GRPO: not directly relevant.
  - Run card: useful for reward-model cards.
  - Reward hacking: includes mitigation ideas around reward modeling, not our
    immediate deterministic verifier.

### 9. hiyouga/LLaMA-Factory

- Problem solved: unified efficient fine-tuning for many LLMs/VLMs, with CLI,
  Web UI, data registry, LoRA/QLoRA, SFT, reward modeling, DPO, and PPO.
- Supports: SFT, DPO, PPO, reward modeling; no first-class GRPO path in the
  inspected tree.
- Files worth reading:
  - [`src/llamafactory/train/sft/trainer.py`](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/train/sft/trainer.py)
  - [`src/llamafactory/train/rm/trainer.py`](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/train/rm/trainer.py)
  - [`src/llamafactory/train/ppo/trainer.py`](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/train/ppo/trainer.py)
  - [`src/llamafactory/train/dpo/trainer.py`](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/train/dpo/trainer.py)
  - [`data/dataset_info.json`](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/dataset_info.json)
  - [`examples/train_lora/qwen3_lora_sft.yaml`](https://github.com/hiyouga/LLaMA-Factory/blob/main/examples/train_lora/qwen3_lora_sft.yaml)
  - [`examples/train_lora/qwen3_lora_reward.yaml`](https://github.com/hiyouga/LLaMA-Factory/blob/main/examples/train_lora/qwen3_lora_reward.yaml)
- Patterns to adopt:
  - Dataset registry metadata can inspire a future `data/registry.yaml`.
  - Good example coverage for Qwen3 LoRA/QLoRA.
  - Clean workflow separation by training method.
- Too heavy or irrelevant now:
  - GUI, broad model support, and PPO/RM workflows are not needed for our
    immediate TRL GRPO path.
- Relation to our modules:
  - Data: dataset registry is useful.
  - Reward: RM examples are future work.
  - Eval: partial only.
  - Frontier audit: not relevant.
  - SFT: useful Qwen3 LoRA reference.
  - GRPO: not directly relevant.
  - Run card: our own artifacts remain stricter.
  - Reward hacking: no direct detector.

### 10. MajoRoth/hack-verifiable-environments

- Problem solved: benchmark framework for evaluating reward hacking in
  verifiable text environments using filesystem wrappers and hidden-solution or
  logical-bug settings.
- Supports: reward-hacking evaluation; no SFT, GRPO, PPO, DPO, or reward-model
  training.
- Files worth reading:
  - [`hvta/FilesystemWrapper/filesystem_wrapper.py`](https://github.com/MajoRoth/hack-verifiable-environments/blob/main/hvta/FilesystemWrapper/filesystem_wrapper.py)
  - [`hvta/FilesystemWrapper/virtual_fs.py`](https://github.com/MajoRoth/hack-verifiable-environments/blob/main/hvta/FilesystemWrapper/virtual_fs.py)
  - [`hvta/FilesystemWrapper/logical_bug_envs/README.md`](https://github.com/MajoRoth/hack-verifiable-environments/blob/main/hvta/FilesystemWrapper/logical_bug_envs/README.md)
  - [`examples/run_hidden_solution_benchmark.py`](https://github.com/MajoRoth/hack-verifiable-environments/blob/main/examples/run_hidden_solution_benchmark.py)
  - [`examples/test_1p.py`](https://github.com/MajoRoth/hack-verifiable-environments/blob/main/examples/test_1p.py)
- Patterns to adopt:
  - Add hidden holdout checks when rewards depend on tool or environment access.
  - Separate success score from reward-hack flags.
  - Log concrete hacking traces, not only aggregate rates.
- Too heavy or irrelevant now:
  - Filesystem-wrapped TextArena environments are not needed for boxed GSM8K.
- Relation to our modules:
  - Data: future adversarial/eval data reference.
  - Reward: future reward hacking tests.
  - Eval: useful separate "task success vs hack" reporting idea.
  - Frontier audit: not directly relevant.
  - SFT/GRPO: not directly relevant.
  - Run card: future hacking-rate field.
  - Reward hacking: strong conceptual reference.

### 11. JonathanGabor/EvilGenie

- Problem solved: reward-hacking benchmark based on modified LiveCodeBench-style
  programming tasks where agents can exploit tests or harnesses instead of
  solving the intended task.
- Supports: reward-hacking detection/evaluation; no training framework. The
  README states the repo is deprecated and points to `evilgenie_inspect` for the
  current implementation.
- Files worth reading:
  - [`src/reward_hacking_detector.py`](https://github.com/JonathanGabor/EvilGenie/blob/main/src/reward_hacking_detector.py)
  - [`src/llm_reward_hacking_detector.py`](https://github.com/JonathanGabor/EvilGenie/blob/main/src/llm_reward_hacking_detector.py)
  - [`src/reward_hacking_reporter.py`](https://github.com/JonathanGabor/EvilGenie/blob/main/src/reward_hacking_reporter.py)
  - [`src/run_holdout_evaluation.py`](https://github.com/JonathanGabor/EvilGenie/blob/main/src/run_holdout_evaluation.py)
  - [`src/canonical_splits/README.md`](https://github.com/JonathanGabor/EvilGenie/blob/main/src/canonical_splits/README.md)
  - [`test_config.yaml`](https://github.com/JonathanGabor/EvilGenie/blob/main/test_config.yaml)
- Patterns to adopt:
  - Keep canonical splits immutable.
  - Run holdout evaluation after suspected reward hacking.
  - Preserve examples where a high proxy score conflicts with true task success.
- Too heavy or irrelevant now:
  - Agent CLI execution, arbitrary code execution, and LiveCodeBench setup are
    not relevant to GSM8K math reward.
- Relation to our modules:
  - Data: canonical split discipline is relevant.
  - Reward: adversarial reward-hacking examples are relevant.
  - Eval: holdout evaluation concept is relevant.
  - Frontier audit: not directly relevant.
  - SFT/GRPO: not directly relevant.
  - Run card: add reward-hacking caveats for code/tool tasks later.
  - Reward hacking: strong but deprecated conceptual reference.

### 12. healthylaife/Composite-LLM-Reward-Model

- Problem solved: paper artifact for "Reward Hacking Mitigation using
  Verifiable Composite Rewards."
- Supports: composite reward mitigation concept; no reusable SFT/GRPO/PPO/DPO
  training framework in the inspected repo.
- Files worth reading:
  - [`README.md`](https://github.com/healthylaife/Composite-LLM-Reward-Model/blob/main/README.md)
  - [`main.ipynb`](https://github.com/healthylaife/Composite-LLM-Reward-Model/blob/main/main.ipynb)
  - [`ACMBCB25_LLM.pdf`](https://github.com/healthylaife/Composite-LLM-Reward-Model/blob/main/ACMBCB25_LLM.pdf)
- Patterns to adopt:
  - Keep component rewards separately reported before combining.
  - Treat composite rewards as experiment-defining changes requiring review.
  - Use verifiable sub-rewards only when their failure modes are documented.
- Too heavy or irrelevant now:
  - The repo is too small to guide our training architecture.
- Relation to our modules:
  - Data: not directly relevant.
  - Reward: future composite reward design reference.
  - Eval: not enough for our eval suite.
  - Frontier audit: not directly relevant.
  - SFT/GRPO: not relevant.
  - Run card: would require component-reward logging.
  - Reward hacking: relevant mitigation concept, not immediately adoptable.

## Design Changes To Consider

At most five changes should be considered from this study:

1. Make frontier audit a first-class pipeline artifact.
   - Inspired by `open-r1` pass-rate filtering.
   - Keep `rollout_audit_summary.json`, per-prompt CSV, sample rollouts, and
     exclusion reasons as stable outputs before GRPO.

2. Add a prompt/reward version registry.
   - Inspired by EasyR1 prompt templates and reward functions.
   - Record prompt template version, reward version, parser version, and
     thinking mode in every audit, SFT, GRPO, and eval report.

3. Separate format, parse, and correctness reporting everywhere.
   - Inspired by EasyR1 and RewardBench.
   - Keep the actual reward semantics unchanged, but always report
     `format_success_rate`, `parse_failure_rate`, `correctness_given_parse`,
     `reward_mean`, and length/truncation metrics separately.

4. Add a minimal dataset registry for staged data only.
   - Inspired by LLaMA-Factory and Axolotl.
   - Track staged dataset path, source, split policy, hash, license, and intended
     use. Do not include or rewrite `data/raw/`.

5. Add a reward-hacking checklist to reward cards and run cards.
   - Inspired by EvilGenie, hack-verifiable-environments, RLHFlow, and the
     composite reward repo.
   - For each reward version, record known hacks, adversarial fixtures, hidden
     holdout status, and whether high reward can diverge from heldout success.

## Immediate Recommendation

For the current GSM8K + Qwen3-4B workflow, keep TRL as the backend and use
`open-r1` plus EasyR1 as the main references. Do not migrate to verl/OpenRLHF
until the frontier audit produces stable mixed groups, the boxed-answer reward
is final for this phase, and the base/SFT/SFT+RLVR eval comparison is reliable.

