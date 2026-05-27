# Experiment Log

Status: scaffold plus SFT overfit-32, Qwen3-0.6B overfit-32, and real Qwen3-0.6B smoke-1k paths.

Record training and eval runs here with links to run directories, run cards, resolved configs, metrics, sample generations, representative failures, and comparison notes.

## SFT Overfit-32

- Config: `configs/sft/overfit32.yaml`
- Command: `make sft-overfit32`
- Default mode: dry-run with synthetic temporary data under `/tmp/posttrain_lab_sft/`
- Output path: `runs/sft/overfit32/`
- Required artifacts: `resolved_config.yaml`, `run_card.md`, `metrics.jsonl`, `sample_generations.jsonl`
- Real TRL/PEFT training requires setting `dry_run: false`, installing `trl`, `peft`, `datasets`, and `transformers`, and using an available local or Hugging Face model path.

## SFT Smoke-1k

- Config: `configs/sft/smoke_1k.yaml`
- Command: `make sft-smoke`
- Default mode: real TRL/PEFT LoRA run with `Qwen/Qwen3-0.6B`.
- Data source: synthetic schema-valid boxed-format addition examples generated under `runs/sft/smoke_1k_boxed/data/` when missing.
- Output path: `runs/sft/smoke_1k_boxed/`
- This is a smoke run, not a final training result.
- Required artifacts: `resolved_config.yaml`, `run_card.md`, `metrics.jsonl`, `sample_generations.jsonl`, `eval_diff.md`
- The target first runs the fixed dry-run eval baseline, then runs eval-after-train on the saved adapter without modifying baseline eval config files.
- Required checks: train loss, validation loss, format success, target eval score, average output length, and 20 sampled generations.

### 2026-05-27 Nexus Real Smoke Run

- Slurm job: `6914860` on `cbcb-heng`, RTX A6000, completed successfully in `00:05:34`.
- Git commit: `4b052bc31422f0c2d3a8d7d32323c99b988c3259`.
- Output path: `/fs/nexus-scratch/qhe123/posttrain-lab-worktrees/4b052bc-sft-smoke1k-real/runs/sft/smoke_1k/`.
- Train examples: `1000`; validation examples: `64`; max steps: `1000`.
- First logged loss: `6.3056`; final trainer loss: `0.2896`; minimum logged loss: `0.2529`.
- Final validation loss: `1.5160`.
- Eval-after-train: exact match `0.0`, format success `0.0`, parse failure rate `1.0`, average output length `20.0`.
- Manual generation check saved `20` sampled generations. For arithmetic training prompts, generations generally returned the correct numeric answer after an empty `<think>...</think>` block.
- Caveat: this run is superseded by the boxed-format smoke config because the fixed baseline eval prompt requests boxed format, but this earlier smoke data taught plain numeric answers.

### 2026-05-27 Nexus Boxed-Format Smoke Run

- Slurm job: `6915024` on `cbcb-heng`, RTX A6000, completed successfully in `00:05:36`.
- Git commit: `7a25cad69f8553e44352af4e319acee1036a1620`.
- Output path: `/fs/nexus-scratch/qhe123/posttrain-lab-worktrees/7a25cad-sft-smoke1k-boxed/runs/sft/smoke_1k_boxed/`.
- Train examples: `1000`; validation examples: `64`; max steps: `1000`.
- Data format: synthetic addition examples requiring boxed final answers, generated under `runs/sft/smoke_1k_boxed/data/`.
- First logged loss: `0.3065`; final trainer loss: `0.2657`; minimum logged step loss: `0.2291`.
- Final validation loss: `0.2316`.
- Eval-after-train on the fixed baseline eval: exact match `1.0`, format success `1.0`, parse failure rate `0.0`, average output length `9.0`.
- Raw fixed-eval generation for `2 + 2`: `\boxed{4}`.
- Manual generation check saved `20` sampled generations; all inspected samples used boxed answer format without `<think>...</think>` wrappers.
- Note: this run fixes the previous target-eval mismatch by changing the smoke training target format and disabling Qwen thinking mode for generation/eval. It does not change fixed eval prompts, labels, metrics, or baseline comparison settings.

## Qwen3-0.6B Overfit-32

- Config: `configs/sft/qwen3_0_6b_overfit32.yaml`
- Command: `make sft-overfit32-qwen3`
- Intended runtime: Nexus GPU via `scripts/slurm/run_sft_config.sh`
- Output path: `runs/sft/qwen3_0_6b_overfit32/`
- This is an overfit gate, not a final training result.
- Required checks: training loss curve, 32-sample recall/generation inspection, `run_card.md`, `trainer_log.jsonl`, `metrics.jsonl`, `sample_generations.jsonl`.

### 2026-05-27 Nexus Run

- Slurm job: `6914656` on `cbcb-heng`, RTX A6000, completed successfully.
- Base model: `Qwen/Qwen3-0.6B`.
- Output path: `/fs/nexus-scratch/qhe123/posttrain-lab/runs/sft/qwen3_0_6b_overfit32/`.
- Run card: `runs/sft/qwen3_0_6b_overfit32/run_card.md`.
- First logged loss: `6.4811`; final logged step loss: `0.1304`; minimum logged step loss: `0.1252`; trainer final loss: `0.2417`.
- Normalized train-set recall job: `6914743`, completed successfully.
- Normalized answer recall: `32/32`; raw exact recall: `0/32` because Qwen3 generated an empty `<think>...</think>` block before the numeric answer.
- Caveat: the in-training sample generation check produced repeated `user` tokens, so post-save adapter reload checks should be preferred for judging this run.

## RLVR / GRPO Smoke

- Config: `configs/rlvr/toy_math_grpo.yaml`
- Command: `make rlvr-smoke`
- Default mode: dry-run toy math smoke with synthetic RLVR JSONL generated under `runs/rlvr/toy_math_grpo/data/` when missing.
- Output path: `runs/rlvr/toy_math_grpo/`
- Reward version: `math_boxed_v001`
- This is only an RL loop smoke test, not a model-quality result.
- Required artifacts: `resolved_config.yaml`, `run_card.md`, `metrics.jsonl`, `eval_report.json`, `sample_rollouts.jsonl`, `trainer_log.jsonl`.
- Required metrics: reward mean, reward std, zero reward rate, perfect reward rate, parse failure rate, and average completion length.
- Real TRL GRPO training requires setting `dry_run: false`, installing compatible `trl`, `datasets`, `transformers`, and `peft`, and using an approved short GPU run.

### 2026-05-27 Nexus Qwen3-0.6B Real TRL Smoke

- Slurm job: `6915223` on `cbcb-heng`, RTX A6000, completed successfully in `00:00:31`.
- Git commit: `a1f3e7e801b8efa98cadb17e9802dd232c15d272`.
- Config: `configs/rlvr/qwen3_0_6b_grpo_smoke.yaml`.
- Output path: `/fs/nexus-scratch/qhe123/posttrain-lab-worktrees/a1f3e7e-rlvr-grpo-smoke/runs/rlvr/qwen3_0_6b_grpo_smoke/`.
- Train examples: `8`; max steps: `1`; num generations: `2`; max completion length: `16`.
- Reward version: `math_boxed_v001`.
- Metrics: reward mean `0.0`, reward std `0.0`, zero reward rate `1.0`, perfect reward rate `0.0`, parse failure rate `1.0`, average completion length `35.25`.
- Trainer loop status: TRL `GRPOTrainer` ran and saved adapter/checkpoint artifacts successfully.
- Rollout failure pattern: Qwen3-0.6B produced explanatory text and incomplete or non-final-only boxed expressions, e.g. `2 + 3 = 5\n\nFinal answer: $\\boxed{5}$`, which the tightened reward rejects as `boxed_not_final_only`.
- Interpretation: this is a successful real RL loop smoke test, not a successful reward-learning result. Next RLVR smoke should either start from the boxed SFT adapter or use a prompt/generation setting that produces final-only boxed completions before increasing steps.

### 2026-05-27 Nexus Qwen3-0.6B SFT-Init TRL Smoke

- Slurm job: `6915257` on `cbcb-heng`, RTX A6000, completed successfully in `00:01:08`.
- Git commit: `cd324fea27a6dbfa6eb1a1f033e368ca7e0fd42c`.
- Config: `configs/rlvr/qwen3_0_6b_grpo_smoke.yaml`.
- Source policy: `Qwen/Qwen3-0.6B` initialized from boxed SFT adapter at `runs/sft/smoke_1k_boxed`.
- Server adapter artifact: `/fs/nexus-scratch/qhe123/posttrain-lab-worktrees/7a25cad-sft-smoke1k-boxed/runs/sft/smoke_1k_boxed/`.
- Output path: `/fs/nexus-scratch/qhe123/posttrain-lab-worktrees/cd324fe-rlvr-grpo-sft-init/runs/rlvr/qwen3_0_6b_grpo_smoke_sft_init/`.
- Train examples: `8`; max steps: `1`; num generations: `2`; max completion length: `32`.
- Reward version: `math_boxed_v001`.
- Rollout-format gate: passed before TRL GRPO training with parse failure rate `0.0`, perfect reward rate `1.0`, zero reward rate `0.0`, reward mean `1.0`, reward std `0.0`, and average completion length `9.75`.
- Training metrics: reward mean `1.0`, reward std `0.0`, zero reward rate `0.0`, perfect reward rate `1.0`, parse failure rate `0.0`, average completion length `9.75`, final loss `0.0`.
- Sample rollouts were final-only boxed answers such as `\boxed{5}`, `\boxed{11}`, `\boxed{17}`, and `\boxed{23}` with parsed answers matching the verifier labels.
- Trainer loop status: TRL `GRPOTrainer` ran from the SFT-initialized adapter and saved adapter/checkpoint artifacts successfully.
- Interpretation: this verifies the rollout-format gate, SFT-adapter initialization path, and real TRL GRPO loop. It is still a smoke test, not a model-quality result.
- Caveat: all sampled completions were already perfect, so reward std and advantages were `0.0`; the run did not provide a meaningful learning signal. The next non-smoke RLVR run needs harder or more varied prompts that preserve format compliance while producing nonzero reward variance.

## RLVR Small Math-1k

- Config: `configs/rlvr/math_1k_grpo.yaml`
- Command: `make rlvr-small`
- Comparison command: `make compare-runs`
- Intended policy: `Qwen/Qwen3-0.6B` initialized from boxed SFT adapter at `runs/sft/smoke_1k_boxed`.
- Training set: synthetic RLVR JSONL generated at `runs/rlvr/math_1k_grpo/data/rlvr_math_1k.jsonl` when missing.
- Problem style: mixed arithmetic requiring exactly one final boxed answer.
- Output path: `runs/rlvr/math_1k_grpo/`.
- Eval paths: `runs/rlvr/math_1k_grpo/evals/base`, `runs/rlvr/math_1k_grpo/evals/sft`, and `runs/rlvr/math_1k_grpo/evals/sft_rlvr`.
- Comparison report: `runs/rlvr/math_1k_grpo/comparison_report.md`.
- Failure cases: `runs/rlvr/math_1k_grpo/failure_cases.jsonl`.
- Required comparison metrics: target accuracy, format success, parse failure rate, average output length, and general regression score when available.
- Required interpretation: state whether SFT+RLVR improves heldout eval over SFT; training reward alone is not evidence of improvement.
- Current status: implementation ready; no real small RLVR GPU run has been launched in this entry.
