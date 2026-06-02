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

## OpenR1 Math SFT-1k

- Config: `configs/sft/openr1_math_1k.yaml`
- Command: `make sft-openr1-math-1k`
- Base model: `Qwen/Qwen3-0.6B`
- Dataset: `open-r1/Mixture-of-Thoughts`, config `math`, source split `train`.
- Staged data path: `runs/sft/openr1_math_1k/data/sft_openr1_math_1k.jsonl`
- Output path: `runs/sft/openr1_math_1k/`
- This is a smoke-scale real SFT run intended to produce candidate checkpoints for later GRPO initialization, not a final SFT model.
- Checkpoint policy: save every `100` steps and keep up to `10` checkpoints.
- Required artifacts: `trainer_log.jsonl`, `loss_curve.csv`, `checkpoint_manifest.json`, `resolved_config.yaml`, `run_card.md`, `metrics.jsonl`, `sample_generations.jsonl`, and `eval_diff.md`.
- Selection policy: stream source records, deterministically shuffle with seed `17` and buffer `10000`, then stage `1000` train and `128` validation examples.
- Integrity note: `data/raw/`, fixed eval prompts, reward semantics, and existing train/val/test fixtures are unchanged.

### 2026-05-29 Nexus OpenR1 Math SFT-1k Run

- Slurm job: `6930990` on `cbcb-heng`, RTX A6000, completed successfully in `00:14:37`.
- Git commit: `454caf6a581cd37ff0eb935049cde0a2b4b38d11`.
- Worktree: `/fs/nexus-scratch/qhe123/posttrain-lab-worktrees/454caf6-openr1-sft`.
- Output path: `/fs/nexus-scratch/qhe123/posttrain-lab-worktrees/454caf6-openr1-sft/runs/sft/openr1_math_1k/`.
- Staged data: `1000` train and `128` validation examples from `open-r1/Mixture-of-Thoughts`, config `math`, source split `train`.
- Data hash: `46bea95d594bc995aae1e7488d29adde45b5bc168602fc454dd69ca47ebde863`.
- Config hash: `f6fb8b7c2e3e66c493d214e4454b6d370e0c5c81afcfe0bead1f44feb1463eb8`.
- Training: `1000` max steps, LoRA, save every `100` steps, `10` checkpoints saved from `checkpoint-100` through `checkpoint-1000`.
- First logged train loss: `0.5500`; final trainer loss: `0.5673`; final logged step loss: `0.5585`.
- Validation loss by checkpoint: `100=0.5800`, `200=0.5691`, `300=0.5661`, `400=0.5636`, `500=0.5621`, `600=0.5607`, `700=0.5601`, `800=0.5594`, `900=0.5589`, `1000=0.5586`.
- Best validation checkpoint in this run: `checkpoint-1000` with eval loss `0.5586` and eval mean token accuracy `0.8249`.
- Practical GRPO initialization note: `checkpoint-1000` is the best candidate by validation loss; `checkpoint-900` is a conservative alternative because the improvement from `900` to `1000` is small.
- Fixed baseline eval after training: exact match `0.0`, format success `0.0`, parse failure rate `1.0`, average output length `178.0`.
- Raw fixed-eval generation solved `2 + 2` and included `\boxed{4}`, but also included reasoning text, so it failed the strict final-only baseline format.
- Manual generation check saved `20` sampled generations for review in `sample_generations.jsonl`.
- Interpretation: this run did not show validation-loss overfitting within `1000` steps. It produced useful reasoning-style checkpoints, but it did not yet satisfy the strict concise boxed-answer eval contract needed for reward/eval comparability.

### 2026-05-29 OpenR1 Long-Context Smoke Revision

- New config: `configs/sft/openr1_math_1k_len8192.yaml`.
- New command: `make sft-openr1-math-1k-long`.
- Purpose: rerun the same OpenR1 math 1k smoke with less truncation and reasoning-compatible evaluation.
- Training max sequence length increased from `2048` to `8192`.
- Generation check and eval `max_new_tokens` increased from `256` to `2048`.
- Generation check and eval set `enable_thinking: true` to match the OpenR1-style reasoning data.
- Eval prompt source changed to the staged validation split for this run only; fixed baseline eval prompts are not modified.
- Eval metric uses boxed-answer extraction (`boxed_math_match`) instead of full-generation exact match, while preserving raw generations for review.
- Output path will be `runs/sft/openr1_math_1k_len8192/`, so the previous `runs/sft/openr1_math_1k/` run is not overwritten.

## Qwen3-4B Short Boxed / Format-Repair SFT

- Config: `configs/sft/qwen3_4b_openr1_format_repair_tiny.yaml`
- Command: `make sft-qwen3-4b-format-repair-tiny`
- Base model: `Qwen/Qwen3-4B`
- Dataset: `open-r1/Mixture-of-Thoughts`, config `math`, source split `train`.
- Purpose: teach concise final-only boxed-answer format before testing whether GRPO has nonzero reward variance.
- Output path: `runs/sft/qwen3_4b_openr1_format_repair_tiny/`.
- This is a smoke-scale format-repair run, not a final SFT model.
- Integrity note: `data/raw/`, fixed eval prompts, reward semantics, and train/val/test splits are unchanged.

### 2026-05-29 Nexus Qwen3-4B Format-Repair Run

- Slurm job: `6932612` on `cbcb-heng`, RTX A5000, completed successfully in `00:05:20`.
- Git commit: `c1bc24a67fcc07cb4c6ae4a7d8ff50fcc475d99c`.
- Worktree: `/fs/nexus-scratch/qhe123/posttrain-lab-worktrees/c1bc24a-qwen3-4b-format-repair`.
- Output path: `/fs/nexus-scratch/qhe123/posttrain-lab-worktrees/c1bc24a-qwen3-4b-format-repair/runs/sft/qwen3_4b_openr1_format_repair_tiny/`.
- Staged data: `512` train and `128` validation examples.
- Data hash: `6775d0c1361482ddbe029c2a40a54ab983b8288e0347049f342c58ed47b7bc30`.
- Config hash: `0d7dac866e88a64cb95ae03e792a2285da6ef8c356488968f558281ad11b6d48`.
- Training: `300` max steps, LoRA rank `16`, max sequence length `2048`, save/eval every `50` steps.
- Validation loss by checkpoint: `50=1.532`, `100=1.247`, `150=1.165`, `200=1.091`, `250=1.080`, `300=1.076`.
- Selected checkpoint by minimum eval loss: `checkpoint-300`; server marker: `selected_checkpoint.json`.
- Final train loss: `1.3789`; final validation loss: `1.0760`.
- Eval-after-train on 16 validation prompts: boxed-answer match `0.3125`, answer parse failure rate `0.0`, average completion length `14.94`.
- Interpretation: the format-repair run fixed boxed-format parseability on the sampled eval set, but math correctness remains limited on these harder OpenR1 math prompts.

### 2026-05-29 Qwen3-4B GRPO Feasibility Audit

- Audit job: `6932640` on `cbcb-heng`, RTX A5000, completed successfully in `00:01:06`.
- Policy checkpoint: `runs/sft/qwen3_4b_openr1_format_repair_tiny/checkpoint-300`.
- Audit data: first `16` train prompts converted from the staged SFT data to RLVR JSONL under `runs/rlvr/qwen3_4b_format_repair_feasibility/`.
- Rollout settings: `4` completions per prompt, temperature `0.9`, top-p `0.95`, max new tokens `64`, thinking disabled.
- Reward: `math_boxed_v001`; no reward semantics were changed.
- Summary: `13` all-zero prompts, `1` all-one prompt, `2` mixed prompts.
- Effective mixed group rate: `0.125`; selected frontier prompt rate: `0.125`.
- Parse failure rate: `0.015625`; average completion length: `13.95`; mean unique parsed answers per prompt: `3.44`.
- Filtered frontier data: `2` prompts; excluded data: `14` prompts. The filtered RLVR JSONL validated successfully.
- Interpretation: this checkpoint is format-compliant enough for reward parsing, but it is not yet a good GRPO starting point on this prompt sample because most groups still have zero reward variance. Do not launch a larger GRPO run from this result alone.

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

## E2E Toy Math Post-Training Smoke

- Config: `configs/e2e/toy_math_posttraining.yaml`
- Command: `make e2e-smoke`
- Real-run command: `make e2e-smoke REAL_RUN=1`
- Default mode: safe dry-run with tiny fixtures; it does not load a model or start GPU training.
- Real-run model path: `Qwen/Qwen3-0.6B` from config, only used with the explicit real-run flag.
- Pipeline stages: validate SFT/RLVR data, eval base, SFT overfit-32, eval SFT, reward checks, RLVR/GRPO toy smoke, eval RLVR, and write base/SFT/SFT+RLVR comparison.
- Output path: `runs/e2e/toy_math_posttraining/`.
- Required root artifacts: `baseline_eval_report.json`, `sft_run_card.md`, `sft_eval_report.json`, `rlvr_run_card.md`, `rlvr_eval_report.json`, `comparison_report.md`, `sample_generations.jsonl`, and `sample_rollouts.jsonl`.
- Fail-fast gates: data validation errors, reward check failures, output length above config, or parse failure rate above the stage-specific config.
- Baseline eval may use a looser parse-failure threshold so base-model format failures can be recorded for comparison; SFT and RLVR remain strict by default.
- Current status: safe-mode CI smoke passed; Nexus real-model smoke completed.

### 2026-05-27 Nexus Qwen3-0.6B Real E2E Smoke

- Final Slurm job: `6916056` on `cbcb-heng`, RTX A5000, completed successfully in `00:02:18`.
- Git commit: `6709384a5bbf0888378ea2ec96aadaf954f2454d`.
- Output path: `/fs/nexus-scratch/qhe123/posttrain-lab-worktrees/6709384-e2e-real-run3/runs/e2e/toy_math_posttraining/`.
- Run mode: `real-model run`.
- Base eval: exact match `0.0`, format success `0.0`, parse failure rate `1.0`, average output length `36.5`.
- SFT eval: exact match `1.0`, format success `1.0`, parse failure rate `0.0`, average output length `9.0`.
- SFT training: `32` train examples, `2` validation examples, `200` max steps, final train loss `0.1998`, validation loss `0.4440`.
- RLVR smoke: `8` train prompts, `1` GRPO step, `2` generations, max completion length `32`.
- RLVR metrics: reward mean `1.0`, reward std `0.0`, zero reward rate `0.0`, perfect reward rate `1.0`, parse failure rate `0.0`, average completion length `9.75`.
- RLVR eval: exact match `1.0`, format success `1.0`, parse failure rate `0.0`, average output length `9.0`.
- Comparison conclusion: SFT fixed the toy heldout boxed-answer eval relative to base; SFT+RLVR did not improve heldout target accuracy beyond SFT on this tiny eval.
- Sample generations and rollouts were final-only boxed answers such as `\boxed{4}`, `\boxed{5}`, and `\boxed{11}`.
- Failed setup attempts before the final run: job `6915931` stopped at the baseline parse-failure gate, and job `6915984` stopped because 2-step SFT still had parse failure rate `1.0`.
- Caveat: this is still a toy smoke run. RLVR started from an already format-perfect SFT adapter, so reward std was `0.0` and the GRPO step did not provide meaningful learning pressure.

### 2026-05-27 Nexus Qwen3-0.6B Diverse Math Real E2E Smoke

- Slurm job: `6916399` on `cbcb-heng`, RTX A5000, completed successfully in `00:03:32`.
- Git commit: `e57542b1974706b832e4cb3c795596cf93dc1c94`.
- Output path: `/fs/nexus-scratch/qhe123/posttrain-lab-worktrees/e57542b-e2e-diverse-sft-real/runs/e2e/diverse_math_posttraining_real/`.
- Dataset version: `synthetic-e2e-diverse-v1`.
- SFT staged data: `80` train and `10` validation examples covering arithmetic, fractions, linear equations, and algebra simplification across easy/medium/hard difficulties.
- Heldout eval: `10` prompts, disjoint from SFT train and validation prompts.
- SFT training: `300` max steps, final train loss `0.3376`, validation loss `0.3068`.
- Base eval: exact match `0.0`, format success `0.0`, parse failure rate `1.0`, average output length `73.3`.
- SFT eval: exact match `0.6`, format success `1.0`, parse failure rate `0.0`, average output length `10.6`.
- SFT+RLVR eval: exact match `0.6`, format success `1.0`, parse failure rate `0.0`, average output length `10.6`.
- RLVR smoke metrics: reward mean `0.9583`, reward std `0.1998`, zero reward rate `0.0417`, perfect reward rate `0.9583`, parse failure rate `0.0`, average completion length `10.25`.
- SFT got integer arithmetic and algebra simplification examples correct, but missed heldout fraction addition and harder linear equations.
- Interpretation: the enriched fixture is no longer trivially saturated by this small SFT run. It verifies format learning, but target accuracy is not yet maxed out.
- Caveat: this is still a small synthetic smoke dataset, not a reliable benchmark. The Slurm script and run directory were untracked server artifacts; code ran from the recorded commit.

### 2026-05-27 Nexus Qwen3-0.6B Expanded RLVR Real E2E Smoke

- Final Slurm job: `6916959` on `cbcb-heng`, RTX A5000, completed successfully in `00:03:34`.
- Git commit: `5c2676ec942bc269856d2026bee7a495769971ad`.
- Output path: `/fs/nexus-scratch/qhe123/posttrain-lab-worktrees/5c2676e-e2e-sft-rlvr-expanded-cuda-gate/runs/e2e/rlvr_expanded_sft_to_rlvr_real_cuda_gate/`.
- Dataset version: `synthetic-e2e-diverse-v2`.
- Staged RLVR data: `80` train prompts; rollout-format gate evaluated `32` prompts before GRPO.
- SFT training: `80` train examples, `10` validation examples, `300` max steps, final train loss `0.3286`, validation loss `0.3162`.
- Base eval: exact match `0.0`, format success `0.0`, parse failure rate `1.0`, average output length `73.3`.
- SFT eval: exact match `0.6`, format success `1.0`, parse failure rate `0.0`, average output length `10.5`.
- Rollout-format gate: passed with reward mean `0.21875`, reward std `0.4134`, zero reward rate `0.78125`, perfect reward rate `0.21875`, parse failure rate `0.0`, average completion length `10.8125`.
- RLVR training: `50` GRPO steps, `4` generations, max completion length `32`, reward mean `0.2125`, reward std `0.4091`, zero reward rate `0.7875`, perfect reward rate `0.2125`, parse failure rate `0.0`, average completion length `10.65`.
- SFT+RLVR eval: exact match `0.6`, format success `1.0`, parse failure rate `0.0`, average output length `10.5`.
- Heldout conclusion: SFT+RLVR did not improve target accuracy versus SFT on this eval.
- Main diagnosis: prompt-level rollouts often had identical rewards inside each GRPO group, with trainer logs showing `frac_reward_zero_std=1.0`, `grad_norm=0.0`, and loss `0.0` on many steps. The run had aggregate reward variance, but little or no within-prompt advantage signal.
- Representative failures unchanged after RLVR: fraction addition and harder linear equations, for example `3/4 + 1/6` predicted `\boxed{5/12}` instead of `\boxed{11/12}` and `5x - 4 = 2x + 17` predicted `\boxed{6}` instead of `\boxed{7}`.
- Cancelled setup attempt: job `6916896` was stopped after SFT because the pre-RL rollout gate path was running inefficiently on CPU. Commit `5c2676e` fixed gate rollout device placement and reran from a clean worktree.
- Caveat: this is still a smoke-scale synthetic run. Next RLVR iteration should increase within-prompt sampling diversity or use a policy/init where the four sampled completions for the same prompt receive mixed rewards.

### 2026-05-27 Nexus Qwen3-0.6B Frontier GRPO Smoke

- Git commit: `0b8b437b6238fe2c46e7af3c4af0229c2b1a6d8a`.
- Worktree: `/fs/nexus-scratch/qhe123/posttrain-lab-worktrees/0b8b437-frontier-grpo-smoke/`.
- Source policy: `Qwen/Qwen3-0.6B` initialized from the boxed SFT adapter from run `6916959`.
- Frontier audit job: `6917321` on `cbcb-heng`, RTX A5000, completed successfully in `00:03:38`.
- Audit config: `32` train prompts, `16` completions per prompt, temperature `0.9`, top_p `0.95`, max new tokens `384`.
- Audit result: `19` all-zero prompts, `4` all-one prompts, `9` mixed prompts, effective mixed group rate `0.28125`.
- Filtered GRPO set: `4` prompts kept with `0.2 <= reward_mean <= 0.8`, parse failure rate `<= 0.5`, and unique answer count `>= 3`.
- Filtered dataset path: `runs/rlvr/frontier_prompt_audit_real_32/frontier_grpo_train.jsonl`; schema validation passed.
- Cancelled setup attempt: audit job `6917291` was stopped before summary output because `96` prompts was too large for a smoke run.
- GRPO job: `6917423` on `cbcb-heng`, RTX A5000, completed successfully in `00:01:03`.
- GRPO config: `4` frontier prompts, `30` steps, `8` generations, temperature `0.9`, top_p `0.95`, max completion length `384`.
- Early stop rule: stop after `frac_reward_zero_std > 0.8` for `20` consecutive logged steps; it did not trigger because almost all groups were mixed.
- Rollout-format gate before training: reward mean `0.34375`, reward std `0.4750`, zero reward rate `0.65625`, perfect reward rate `0.34375`, parse failure rate `0.0`, effective mixed group rate `1.0`.
- Training metrics: reward mean `0.3708`, reward std `0.4132`, zero reward rate `0.5`, perfect reward rate `0.5`, parse failure rate `0.0`, average completion length `7.575`.
- Signal metrics: average `frac_reward_zero_std` `0.0333`, effective mixed group rate `0.9667`, nonzero grad step rate `0.9667`.
- Trainer log detail: `29/30` logged steps had mixed reward groups, and all `29` mixed steps had nonzero `grad_norm`.
- Compared with run `6916959`, average `frac_reward_zero_std` dropped from about `0.84` to `0.0333`, so frontier selection materially improved within-group advantage signal.
- Sample rollouts contain multiple mixed reward vectors, for example `[1,0,0,1,1,0,0,0]`, `[0,1,0,0,1,1,1,0]`, and `[0,1,0,0,0,0,0,0]`.
- Heldout eval was not run in this smoke. This result only proves that the frontier-selected GRPO loop now receives a nonzero training signal.

### 2026-05-28 Nexus GSM8K Qwen3-4B Frontier Scout

- Worktree: `/fs/nexus-scratch/qhe123/posttrain-lab-worktrees/5a83769-gsm8k-frontier-audit/`.
- Data source: official GSM8K train split converted to RLVR JSONL at `data/rlvr_prompts/gsm8k_train.jsonl`; official GSM8K test was not used.
- Long audit jobs `6922910` and `6922911` were cancelled because the first implementation sampled one completion per `generate()` call and was too slow.
- Commit `83adafa04fa14379bf531cd60c82e02ecda0d9ee` batched audit generation with `num_return_sequences`, added scout configs, and wrote partial audit outputs every 10-25 prompts.
- Initial scout jobs `6923245` and `6923246` were cancelled after 20 prompts because the old final-only parser rejected normal reasoning before a final boxed answer.
- Commit `0a5d68404fa14379bf531cd60c82e02ecda0d9ee` updated `math_boxed_v001` to strip complete `<think>...</think>` blocks, reject unclosed think blocks, allow ordinary reasoning before exactly one visible final `\boxed{...}`, and still reject multiple boxed answers.
- Reward tests and audit/config tests passed locally and on the server before rerunning the scout.
- Rerun thinking=false job `6923399` completed in `00:08:22` on RTX A5000 with Qwen/Qwen3-4B, `100` prompts, `8` completions per prompt, temperature `0.9`, top_p `0.95`, and max new tokens `128`.
- Thinking=false summary: all-zero `36`, all-one `27`, mixed `37`, mixed rate `0.37`, effective mixed group rate `0.37`, parse failure rate `0.55875`, average completion length `317.60`, selected prompts `0`.
- Filtered frontier output: `runs/rlvr/gsm8k_frontier_scout_thinking_false/frontier_grpo_train.jsonl`; it is empty but schema-valid. All `100` prompts were excluded: `67` parse_fail, `27` all_one, and `6` low_diversity.
- Thinking=true rerun job `6923400` was stopped after 20 prompts to avoid wasted GPU. Partial summary: all-zero `20`, mixed `0`, parse failure rate `1.0`, caused by unclosed `<think>` generations under the short token budget.
- Main diagnosis: the parser fix worked and revealed real mixed reward groups for thinking=false, but the current prompt/model combination still often omits final boxed answers, appends units after boxed answers, or produces too little answer diversity for the strict frontier filter.
- Recommendation: do not start GRPO from this filtered GSM8K scout yet. First run a short boxed-format SFT warmup or strengthen the prompt/decoding so parse failure drops below the frontier threshold; use thinking=false for now unless thinking=true is given a much larger completion budget and a format gate.

### 2026-05-28 GSM8K Parse-Failure Taxonomy

- Diagnostic command: `make diagnose-parse-failures`.
- Input completions: `runs/rlvr/gsm8k_frontier_scout_thinking_false/sample_rollouts_for_review.jsonl`.
- Output files: `data/reports/parse_failure_taxonomy/parse_failure_summary.json`, `parse_failure_by_completion.csv`, and `parse_failure_examples.md`.
- Scope caveat: the scout saved review rollouts for the first `20` prompts only, so this taxonomy covers `160` completions rather than the full `800` audit completions.
- Parse failures in the review file: `94 / 160 = 58.75%`.
- Category breakdown: truncated_before_final `36` (`38.30%`), no_boxed_answer `35` (`37.23%`), final_answer_unboxed `15` (`15.96%`), parser_too_strict `5` (`5.32%`), malformed_boxed `3` (`3.19%`), all other categories `0`.
- Truncation caveat: the audit file did not record generated token counts, so truncation was estimated from completion character length against `max_new_tokens=128`.
- Recommended next action: increase `max_new_tokens`, strengthen the final-answer prompt, and run a boxed-format SFT warmup before any GSM8K GRPO smoke. Do not start GRPO from the current filtered set.

### 2026-05-28 GSM8K Qwen3-4B Prompt Sweep

- Goal: no-training prompt/generation sweep before SFT or GRPO.
- Config: `configs/audit/gsm8k_qwen3_4b_prompt_sweep.yaml`.
- Runner: `runs/audit/gsm8k_qwen3_4b_prompt_sweep/run_prompt_sweep.py`.
- Model: `Qwen/Qwen3-4B`; sample: fixed random `300` GSM8K train prompts; generations per prompt: `8`; temperature `0.9`; top_p `0.95`.
- Full Slurm array: job `6927608`, variants `current_128`, `strong_128`, `strong_512`, `strong_1024`, `strong_nonthinking_512`, and `strong_thinking_1024`.
- Full array status: queued until after the 2026-05-28 16:45-20:00 monthly maintenance window; no training is involved.
- Short pre-maintenance array: job `6927609`, variants `current_128` and `strong_128`, completed successfully.
- `current_128` result: reward_mean `0.3842`, format_success_rate `0.3983`, parse_failure_rate `0.6017`, correctness_given_parse `0.9644`, any_correct_prompt_rate `0.5533`, all_correct_prompt_rate `0.1900`, mixed_prompt_rate `0.3633`, all_zero_rate `0.4467`, avg_completion_length `328.14`, truncation_rate `0.7533`.
- `strong_128` default-thinking result: reward_mean `0.0`, format_success_rate `0.0`, parse_failure_rate `1.0`, truncation_rate `1.0`, with `2399/2400` failures from `unclosed_think_block`.
- Interim interpretation: `current_128` is dominated by truncation, while `strong_128` without explicit non-thinking mode triggers thinking-mode interference. The final recommendation should wait for the 512/1024 and explicit non-thinking/thinking variants.

### 2026-05-29 Frontier Audit Default Disabled

- Decision: pause frontier audit as a default workflow because the GSM8K/Qwen3-4B audit path is dominated by rollout inference cost and should not run accidentally during ordinary smoke work.
- Make targets now require `RUN_FRONTIER_AUDIT=1` before running frontier rollout sampling or frontier-dependent GRPO smoke targets.
- Affected targets: `rlvr-frontier-audit`, `rlvr-frontier-smoke`, `rlvr-gsm8k-scout-*`, and `rlvr-gsm8k-audit-*`.
- This does not delete the audit implementation, change reward semantics, alter GSM8K conversion, modify eval prompts, or change train/validation/test split policy.
- Current interpretation for data choice: GSM8K is useful as a sanity benchmark, but Qwen/Qwen3-4B appears too often correct once formatting and truncation are controlled, so it is a weak main source for GRPO advantage signal. Harder math data from the TRL/open-r1 style workflow should be evaluated next, with source/license/split review before adding it as training data.

### 2026-05-29 DeepMath/OpenR1 Qwen3 Pilot Eval

- Goal: estimate whether `Qwen/Qwen3-0.6B` and `Qwen/Qwen3-4B` can solve sampled examples from `trl-lib/DeepMath-103K` and `open-r1/OpenR1-Math-220k`.
- Git commit: `52d723f455e0c1cb5897e6433d6f6a3ce01323eb`.
- Worktree: `/fs/nexus-scratch/qhe123/posttrain-lab-worktrees/52d723f-math-dataset-eval`.
- Eval command: `scripts/slurm/run_math_dataset_eval.sh`.
- Settings: `50` streamed train examples per dataset/model, seed `20260529`, greedy decoding, `max_new_tokens=2048`, `enable_thinking=false`, `math_boxed_v001` strict reward, no training.
- Slurm jobs: smoke `6930645`; pilot jobs `6930647`, `6930648`, `6930649`, and `6930650`, all completed successfully on `cbcb-heng`.
- Results were synced locally under `runs/eval/math_dataset_pilot/`; generated run artifacts are not source files and are not committed.

| Model | Dataset | Strict boxed accuracy | Parse failure rate | Correctness given parse | Truncation rate | Avg completion tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `Qwen/Qwen3-0.6B` | `trl-lib/DeepMath-103K` | `0.18` | `0.12` | `0.2045` | `0.06` | `496.5` |
| `Qwen/Qwen3-0.6B` | `open-r1/OpenR1-Math-220k` | `0.12` | `0.12` | `0.1364` | `0.04` | `340.5` |
| `Qwen/Qwen3-4B` | `trl-lib/DeepMath-103K` | `0.36` | `0.32` | `0.5294` | `0.12` | `947.3` |
| `Qwen/Qwen3-4B` | `open-r1/OpenR1-Math-220k` | `0.26` | `0.46` | `0.4815` | `0.24` | `1123.4` |

- Relaxed diagnostic accuracy, using the existing compatibility scorer that allows repeated identical boxed answers and non-final boxed answers, was `0.18`, `0.12`, `0.44`, and `0.32` for the rows above. This diagnostic did not change reward semantics.
- Interpretation: both datasets are much harder than GSM8K for the current models. `Qwen/Qwen3-0.6B` solves too few examples to be a good direct GRPO policy without SFT/warmup. `Qwen/Qwen3-4B` has meaningful math signal, especially on parseable outputs, but strict-format failures and truncation remain material.
- Recommendation: use these datasets as the next source pool, but start with format/prompt stabilization and a small SFT warmup before GRPO. For RLVR, prefer `Qwen/Qwen3-4B` or an SFT-initialized 0.6B policy; do not treat 0.6B base as ready to solve these datasets directly.

### 2026-05-29 OpenR1 Qwen3-0.6B SFT Smoke Long Context

- Goal: rerun the 1k SFT smoke with OpenR1-style reasoning enabled, longer context, longer evaluation generation, and boxed-final-answer scoring.
- Config: `configs/sft/openr1_math_1k_len8192.yaml`.
- Data source: `open-r1/Mixture-of-Thoughts`, `math` config, streamed `train` split, seed `17`; staged as `1000` train and `128` validation examples.
- Length policy: train/eval sequence length `8192`; generation check and post-train eval `max_new_tokens=2048`; thinking remains enabled.
- Eval policy: validation examples are converted into fixed eval prompts; scoring compares the visible final `\boxed{...}` answer against the target answer, not the full reasoning trace.
- First Slurm attempt `6931368` failed at step `100` during validation loss with CUDA OOM because the default eval batch size was `8`.
- Follow-up fix: set `per_device_eval_batch_size=1` for long-context SFT validation before rerunning. This does not change the data, reward semantics, eval prompts, or train/validation/test split policy.
- Second Slurm attempt `6931381` completed the 1000-step training phase and wrote checkpoints through `checkpoint-1000`, but was cancelled during post-training sample generation after it reused the training model with `use_cache=False`.
- Follow-up fix: temporarily restore `use_cache=True` for generation checks and write `sample_generations.jsonl` incrementally so long generations leave inspectable partial output.
- Completed Slurm run `6931421` on commit `a105ae2c7cfcd8ffd3316dd9bae488fe84504681`; wall time `01:11:05`.
- Training artifacts: `checkpoint-100` through `checkpoint-1000`, `loss_curve.csv`, `trainer_log.jsonl`, `sample_generations.jsonl`, `metrics.jsonl`, `run_card.md`, `eval/raw_generations.jsonl`, `eval/metrics.json`, and `eval_diff.md`.
- Train loss summary: first logged loss `0.5809`, final train loss `0.5075`, last logged loss `0.5040`.
- Validation loss curve by checkpoint: `100=0.5072`, `200=0.4980`, `300=0.4954`, `400=0.4933`, `500=0.4921`, `600=0.4913`, `700=0.4905`, `800=0.4899`, `900=0.4894`, `1000=0.4894`.
- Best validation checkpoint for this smoke is `checkpoint-1000` by eval loss, but this is not a final model-selection decision.
- Post-train validation generation eval: `answer_match=0.0`, `answer_parse_failure_rate=0.875`, `completion_length_mean=5493.75` characters across `8` validation prompts.
- Failure diagnosis: `7/8` eval generations had `unclosed_think_block` under `max_new_tokens=2048`; `1/8` parsed but had an answer mismatch. The main remaining issue is long-thinking truncation/format closure, not SFT trainer execution.
- Manual-review samples: `20` random generations saved; character length range `3203` to `9189`, mean `5694.8`.

### 2026-06-02 Qwen2.5-Math Eval Loader Prep

- Goal: prepare the eval stage for the Qwen2.5-Math-1.5B DAPO-style workflow before training.
- Change: `math_dataset_eval` now supports local JSONL/JSON/Parquet paths, recursive local dataset directories, RLVR `verifier.answer`, DAPO `reward_model.ground_truth`, OlympiadBench list-style `final_answer`, and an explicit `paper_math` prompt template.
- Safety: no reward semantics, eval labels, existing baseline prompt, train/validation/test split, or raw data were changed.
- Smoke checks: `make test-eval`, `make eval-baseline`, and `make eval-qwen25-math-dry` passed locally.

### 2026-05-29 OpenR1 Qwen3-0.6B Format-Repair SFT

- Goal: continue from the long-context OpenR1 SFT adapter and teach concise final-only boxed outputs before GRPO.
- Config: `configs/sft/openr1_math_format_repair_1k.yaml`.
- Command: `make sft-openr1-format-repair` via Slurm script `scripts/slurm/run_sft_config.sh`.
- Code commit: `ae9054adbac32cbe335aa8627c4b4779380d6d31`.
- Slurm job: `6932546` on `cbcb-heng`, RTX A5000, completed successfully in `00:05:40`.
- Worktree: `/fs/nexus-scratch/qhe123/posttrain-lab-worktrees/ae9054a-openr1-format-repair-sft`.
- Output path: `/fs/nexus-scratch/qhe123/posttrain-lab-worktrees/ae9054a-openr1-format-repair-sft/runs/sft/openr1_math_format_repair_1k/`.
- Parent adapter: `/fs/nexus-scratch/qhe123/posttrain-lab-worktrees/a105ae2-openr1-sft-long-cache/runs/sft/openr1_math_1k_len8192/checkpoint-1000`.
- Data source: same `open-r1/Mixture-of-Thoughts`, `math` config, streamed `train` split, seed `17`; staged as `1000` train and `128` validation examples.
- Target conversion: source assistant reasoning is rewritten to one assistant message containing only the visible final `\boxed{...}` answer. This did not modify raw data, eval prompts, reward semantics, or train/validation/test split policy.
- Training settings: `500` steps, max sequence length `2048`, LoRA continuation, `enable_thinking=false` for generation and eval, eval `max_new_tokens=256`.
- Train loss summary: first logged loss `2.099`, trainer final loss `1.206`, last logged step loss `1.198`.
- Validation loss curve by checkpoint: `50=1.2236`, `100=1.2116`, `150=1.1947`, `200=1.1869`, `250=1.1820`, `300=1.1741`, `350=1.1714`, `400=1.1664`, `450=1.1652`, `500=1.1632`.
- Post-train validation generation eval on `16` fixed validation prompts: `answer_match=0.125`, `answer_parse_failure_rate=0.0`, average completion length `15.56` characters.
- Manual-review samples: `20` random generations saved; `19/20` had a parseable boxed answer. One sample emitted an overlong unclosed boxed number, so the next GRPO step should keep parse-failure and output-length gates.
- Checkpoint retained for next GRPO: `checkpoint-500`, because it had the best validation loss and final eval had zero parse failures. The root adapter at `runs/sft/openr1_math_format_repair_1k/` is equivalent to the final saved adapter for loading, but `checkpoint-500` is the explicit immutable selection.
- Interpretation: this run fixed the main format/truncation problem from the long SFT run, improving parse failure from `0.875` to `0.0` on validation eval and reducing average eval output length from about `5494` characters to `15.56`. It did not solve math accuracy on the harder OpenR1 validation slice; most wrong cases are short parseable boxed guesses. GRPO should treat this as a format-stabilized policy, not a strong math policy.

### 2026-05-29 Format-Repair Checkpoint Tiny GRPO Feasibility Audit

- Goal: quickly test whether the format-repair `checkpoint-500` can be used as a GRPO starting policy without running trainer steps.
- Job: Slurm `6932587`, `grpo-feas-v2`, completed in about `9` seconds of audit runtime.
- Worktree: `/fs/nexus-scratch/qhe123/posttrain-lab-worktrees/ae9054a-openr1-format-repair-sft`.
- Output path: `runs/rlvr/format_repair_checkpoint_feasibility_v2/`.
- Policy checkpoint: `runs/sft/openr1_math_format_repair_1k/checkpoint-500`.
- Prompt source: temporary RLVR JSONL converted from the format-repair SFT staged `train` split; no raw data, eval prompts, reward semantics, or train/validation/test split policy were modified.
- Scale: `8` train prompts, `4` sampled completions per prompt, `32` total completions, `temperature=0.9`, `top_p=0.95`, `max_new_tokens=64`, `enable_thinking=false`.
- Result: parse failure rate `0.03125`, average completion length `16.94`, all-zero prompts `6/8`, all-one prompts `0/8`, mixed prompts `2/8`, effective mixed group rate `0.25`.
- Frontier output: `2` prompts passed the tiny frontier filter and the filtered RLVR JSONL validated successfully.
- Interpretation: the checkpoint is feasible for a very small GRPO smoke because it loads, generates concise boxed answers, and has nonzero mixed groups. It is not ready for direct larger GRPO because most sampled prompts are all-zero and would provide no within-group advantage.

### 2026-05-29 Qwen3-4B SymPy-Boxed SFT

- Goal: train `Qwen/Qwen3-4B` on the curated OpenR1/DeepMath SymPy-compatible boxed-answer SFT dataset, first with a small smoke run and then with a full run to convergence.
- Code commit: `1581aadfd67af4cd1c5cc80a259146fff13ab6da`.
- Worktree: `/fs/nexus-scratch/qhe123/posttrain-lab-worktrees/4affb7b-sympy-boxed-data`.
- Dataset: `data/staged/openr1_deepmath_sympy_boxed_v1/`; train `5000`, val `500`, test `500`; no `data/raw` files were modified.
- Data hashes: train `b9adac8b3c861177a4e805fd6ff16e623603b0077be14ebb15c446badb5b1741`; val `fd36e947110862b8b6e934a4f7ac73b1a19c15c2d59e84d8817a9b262e95505c`.
- SymPy engine: post-train eval used `allow_symbolic_equivalence=true` and `symbolic_equivalence_engine=sympy`; reward semantics were not changed during the run.
- Smoke config: `configs/sft/qwen3_4b_sympy_boxed_smoke.yaml`; Slurm job `6933159`, completed in `00:02:55` on RTX A5000.
- Smoke result: final train loss `1.8372`, validation loss `1.2814`, validation generation `answer_match=0.0625`, parse failure rate `0.0`, average completion length `15.875`.
- Full config: `configs/sft/qwen3_4b_sympy_boxed_full.yaml`; Slurm job `6933180`, completed in `01:08:31` on RTX A5000.
- Full training settings: LoRA rank `16`, max sequence length `2048`, `1800` optimizer steps, batch size `1`, gradient accumulation `4`, learning rate `2e-5`, bf16, `enable_thinking=false`.
- Loss curve summary: first train loss `2.8176`, last logged train loss `0.9427`, trainer final loss `0.9780`; first eval loss `1.1502`, final/best eval loss `0.8978`.
- Validation loss kept improving slowly through the final checkpoint; best checkpoint is `runs/sft/qwen3_4b_sympy_boxed_full/checkpoint-1800`.
- Post-train validation generation eval on `64` fixed validation prompts: `answer_match=0.15625`, parse failure rate `0.0`, average completion length `13.70` characters.
- Run artifacts: `runs/sft/qwen3_4b_sympy_boxed_full/run_card.md`, `resolved_config.yaml`, `metrics.jsonl`, `loss_curve.csv`, `selected_checkpoint.json`, `sample_generations.jsonl`, and `eval/metrics.json`.
- GRPO starting checkpoint: use `runs/sft/qwen3_4b_sympy_boxed_full/checkpoint-1800` as the next parent adapter unless a later heldout eval contradicts this validation-loss choice.
- Interpretation: the run successfully stabilized concise boxed output with zero parse failures and improved validation loss, but math accuracy on this harder validation slice remains modest. The next step should be a very small no-training rollout audit or GRPO smoke from `checkpoint-1800`, with reward vectors checked before any larger RLVR run.

### 2026-05-29 Qwen3-4B SymPy-Boxed Checkpoint RLVR Audit

- Goal: run a small no-training rollout audit to decide whether `Qwen/Qwen3-4B` plus the SymPy-boxed SFT adapter is suitable for the next GRPO step.
- Code commit used on server: `49866a60ebad4ef5355b1a4f4d414e55b36c9e50`.
- Worktree: `/fs/nexus-scratch/qhe123/posttrain-lab-worktrees/4affb7b-sympy-boxed-data`.
- Parent adapter: `runs/sft/qwen3_4b_sympy_boxed_full/checkpoint-1800`.
- Audit output path: `runs/rlvr/qwen3_4b_sympy_boxed_checkpoint_audit/`.
- Slurm job: `6933650`, completed in `00:01:46` on RTX A5000; no trainer step was executed.
- Prompt source: temporary RLVR JSONL converted from the SFT train split under `runs/`; `data/raw`, eval prompts, reward semantics, and train/validation/test split policy were not modified.
- Scale: `24` train prompts, `8` sampled completions per prompt, `192` total completions, `temperature=0.9`, `top_p=0.95`, `max_new_tokens=64`, `enable_thinking=false`.
- Reward config: `math_boxed_v001` with `allow_symbolic_equivalence=true` and `symbolic_equivalence_engine=sympy`.
- Result: total reward mean `0.125`, reward std `0.3307`, parse failure rate `0.0`, average completion length `13.875`.
- Prompt buckets: `17/24` all-zero, `1/24` all-one, `6/24` mixed; effective mixed group rate `0.25`.
- Frontier filter result: `3/24` prompts kept (`12.5%` selected); the filtered RLVR JSONL validated successfully.
- Interpretation: the checkpoint is suitable for a very small GRPO smoke because it produces stable boxed outputs and has some mixed reward groups. It is not yet suitable for direct larger GRPO on unfiltered prompts because most prompts are all-zero and provide no within-group advantage.

### 2026-05-29 Qwen3-4B SymPy-Boxed GRPO Smoke

- Goal: start from `runs/sft/qwen3_4b_sympy_boxed_full/checkpoint-1800` and run a minimal real GRPO smoke to test whether the current checkpoint produces useful training signal.
- Code commit used on server: `e91096b015d55ab5fa5833174d4a7aad7266e0eb`.
- Worktree: `/fs/nexus-scratch/qhe123/posttrain-lab-worktrees/4affb7b-sympy-boxed-data`.
- Output path: `runs/rlvr/qwen3_4b_sympy_boxed_grpo_smoke/`.
- Slurm jobs: training `6933656` completed in `00:02:03`; small eval `6933664` completed in `00:01:09`.
- Training data: temporary smoke JSONL under `runs/`, made by repeating the `3` frontier prompts kept by the prior audit to `24` rows with unique IDs. No raw data, eval prompts, reward semantics, or train/validation/test split policy were modified.
- GRPO settings: `12` steps, `num_generations=8`, `per_device_train_batch_size=8`, `max_completion_length=64`, `temperature=0.9`, `top_p=0.95`, `learning_rate=5e-7`, `beta=0.0`, `enable_thinking=false`.
- Reward config: `math_boxed_v001` with `allow_symbolic_equivalence=true` and `symbolic_equivalence_engine=sympy`.
- Rollout-format gate before training: reward mean `0.2083`, reward std `0.4061`, parse failure rate `0.0`, effective mixed group rate `1.0`, average completion length `16.71`.
- Trainer signal: `12/12` trainer steps logged, mean reward `0.2604`, mean reward std `0.4135`, frac reward zero std `0.0833`, effective mixed group rate `0.9167`, nonzero grad step rate `0.9167`, parse failure rate `0.0`, average completion length `9.71`.
- Post-training train-prompt sample: reward mean `0.2917`, parse failure rate `0.0`, effective mixed group rate `1.0`.
- Small heldout eval on a fixed random sample of `32` eval prompts: SFT answer match `0.21875`; SFT+GRPO answer match `0.1875`; both had format success `0.96875` and parse failure rate `0.03125`.
- Interpretation: the GRPO loop is technically working and provides nonzero advantage/gradient on the selected frontier prompts, but this tiny run did not improve heldout accuracy. Do not continue training longer on the same `3` prompts. The next RLVR run should use a larger and more diverse frontier set, then repeat a short GRPO smoke with heldout eval before scaling steps.

### 2026-05-30 Qwen3-4B OpenR1 CN Math SFT

- Goal: build a staged OpenR1 CN math Algebra/Number Theory pool and train `Qwen/Qwen3-4B` with concise boxed-answer SFT for the next RLVR stage.
- Code commit used on server: `b85c43e51353a6d36f0b174fccf8548591571974`.
- Worktree: `/fs/nexus-scratch/qhe123/posttrain-lab-worktrees/4affb7b-sympy-boxed-data`.
- Data source: `open-r1/OpenR1-Math-220k`, config `extended`, source split `train`, sources `cn_k12`, `cn_contest`, and `amc_aime`, domains `Algebra` and `Number Theory`.
- Staged pools: RLVR under `data/rlvr_prompts/openr1_cn_math_alg_nt_v1/`; SFT under `data/staged/openr1_cn_math_alg_nt_sft_v1/`.
- Split counts: train `18500`, validation `2300`, test `2300`; the originally planned `20000/2000/2000` split was reduced because the strict parser-compatible filter produced `23236` usable examples.
- Data validation: `make validate-data` passed on the generated RLVR and SFT train/validation/test JSONL files.
- SFT config: `configs/sft/qwen3_4b_openr1_cn_math_sft.yaml`.
- Slurm job: `6934082`, completed successfully on `cbcb-heng` in `04:14:34` on RTX A5000.
- Training settings: LoRA rank `16`, max sequence length `2048`, `6000` optimizer steps, batch size `1`, gradient accumulation `4`, learning rate `2e-5`, bf16, `enable_thinking=false`.
- Early stopping was enabled but did not stop the run before `max_steps`; validation loss flattened near the end rather than clearly worsening for enough evaluation windows.
- Loss curve summary: first train loss `3.3663`, last logged train loss `0.6169`, trainer final loss `0.6099`; first eval loss `0.6675`, final eval loss `0.5744`.
- Best validation checkpoint: `runs/sft/qwen3_4b_openr1_cn_math_sft/checkpoint-5250`, with eval loss `0.5743` and eval mean token accuracy `0.8182`.
- Post-train validation generation eval on `64` fixed validation prompts: `answer_match=0.375`, parse failure rate `0.0`, average completion length `15.42` characters.
- Run artifacts synced locally: `runs/sft/qwen3_4b_openr1_cn_math_sft/loss_curve.csv`, `trainer_log.jsonl`, `selected_checkpoint.json`, `metrics.jsonl`, `run_card.md`, `sample_generations.jsonl`, `eval/metrics.json`, and Slurm logs. Large adapter checkpoint weights remain on the server scratch path.
- No `data/raw` files, eval prompts, reward semantics, or existing train/validation/test splits were modified. This run created a new staged dataset version and documented its split policy.
- Interpretation: the run substantially reduced validation loss and produced short parseable boxed outputs, with zero parse failures on the fixed validation-generation eval. Accuracy is meaningfully higher than the previous SymPy-boxed SFT validation sample (`0.375` vs `0.15625`), but validation loss plateaued after about step `4500`; use `checkpoint-5250` as the selected checkpoint for a small GRPO feasibility audit rather than continuing SFT blindly.

### 2026-05-30 Qwen3-4B OpenR1 CN Math Checkpoint-5250 Micro Rollout Audit

- Goal: confirm `math_boxed_v001` reward behavior and rollout readiness before any GRPO trainer step from the selected SFT checkpoint.
- Code commit used on server: `af5ea5bd0cb3eca5246e4e5176ab4f8f993062ee`.
- Worktree: `/fs/nexus-scratch/qhe123/posttrain-lab-worktrees/4affb7b-sympy-boxed-data`.
- Policy checkpoint: `runs/sft/qwen3_4b_openr1_cn_math_sft/checkpoint-5250`.
- Prompt source: fixed random sample of `12` train prompts from `data/rlvr_prompts/openr1_cn_math_alg_nt_v1/train.jsonl`, written under `runs/rlvr/qwen3_4b_openr1_cn_math_ckpt5250_micro_audit/sampled_train_prompts.jsonl`.
- Slurm job: `6938512`, completed successfully on `cbcb-heng` in `00:01:14` on RTX A5000; no trainer step was executed.
- Reward tests: local `make test-rewards` passed with `25 passed, 5 skipped`; server `make test-rewards PYTHON=/fs/nexus-scratch/qhe123/envs/posttrain-lab-trl/bin/python` passed with `30 passed`.
- Rollout settings: `8` sampled completions per prompt, `96` total completions, `temperature=0.9`, `top_p=0.95`, `max_new_tokens=64`, `enable_thinking=false`, generation batch size `4`.
- Reward config: `math_boxed_v001` with `allow_symbolic_equivalence=true` and `symbolic_equivalence_engine=sympy`; reward semantics were not changed.
- Audit summary: reward mean `0.1458`, reward std `0.3529`, zero reward rate `0.8542`, perfect reward rate `0.1458`, parse failure rate `0.0`, average completion length `15.59` characters, max completion length `44` characters.
- Prompt buckets: `9/12` all-zero, `1/12` all-one, `2/12` mixed; effective mixed group rate `0.1667`.
- Frontier filter result: `1/12` prompts kept; the filtered `frontier_grpo_train.jsonl` validated successfully.
- Representative mixed prompt: for `x^2-3=0`, one exact target-form set answer received reward `1`, while variants such as `\sqrt{3}`, `\pm\sqrt{3}`, or differently formatted assignment sets received reward `0`. This confirms the current reward is strict and deterministic, but it may under-credit mathematically equivalent multi-solution formats.
- Interpretation: the selected SFT checkpoint produces stable single-boxed parseable outputs, so reward parsing is ready. The reward signal is too sparse for direct GRPO on arbitrary sampled prompts; only a frontier-filtered prompt pool or a reward-normalization/parser-format review for multi-solution answers should precede larger GRPO.

### 2026-05-30 Qwen3-4B Single-Expression Filter Eval and Rollout Audit

- Goal: apply `single_expression_no_assignment_v1`, re-evaluate the current `checkpoint-5250`, and rerun a small rollout audit before deciding on GRPO.
- Code commit used on server: `61db40d852b60ccc18bca5f293c247d0804887ce`.
- Worktree: `/fs/nexus-scratch/qhe123/posttrain-lab-worktrees/4affb7b-sympy-boxed-data`.
- Filtered RLVR output: `data/rlvr_prompts/openr1_cn_math_alg_nt_single_expr_v1/`.
- Filtered SFT output: `data/staged/openr1_cn_math_alg_nt_single_expr_sft_v1/`.
- Filter policy: reject targets containing assignment/equation `=` or plus-minus macros/symbols. This did not modify `data/raw`, reward semantics, eval prompts, or existing split membership.
- Filter result: input train/validation/test `18500/2300/2300`; output train/validation/test `16904/2104/2115`; rejected `1977` assignment targets and `0` plus-minus targets. All generated RLVR/SFT JSONL files validated successfully.
- Reward tests: server `tests/test_math_reward.py` passed with `30 passed`.
- Evaluation: `Qwen/Qwen3-4B` plus `runs/sft/qwen3_4b_openr1_cn_math_sft/checkpoint-5250` on a fixed random sample of `512` filtered validation prompts.
- Eval settings: greedy decoding, `max_new_tokens=64`, `enable_thinking=false`, `math_boxed_v001`, `allow_symbolic_equivalence=true`, `symbolic_equivalence_engine=sympy`.
- Eval result: `answer_match=0.328125`, parse failure rate `0.0`, average completion length `13.51` characters.
- Rollout audit: `24` filtered train prompts, `8` sampled completions per prompt, `192` total completions, `temperature=0.9`, `top_p=0.95`, `max_new_tokens=64`.
- Audit result: reward mean `0.1458`, reward std `0.3529`, parse failure rate `0.0104`, average completion length `15.65` characters.
- Prompt buckets: `7/24` all-zero, `4/24` all-one, `13/24` mixed; effective mixed group rate `0.5417`; frontier filter kept `7/24` prompts and the filtered frontier JSONL validated.
- Interpretation: filtering assignment-like targets materially improved rollout usefulness compared with the prior micro audit (`mixed_rate` from `0.1667` to `0.5417`, selected prompt rate from `0.0833` to `0.2917`) while preserving low parse failures. A tiny GRPO smoke on the frontier-kept prompts is now reasonable, but a larger GRPO run should wait for a larger filtered/frontier pool and heldout comparison.

### 2026-05-30 Qwen3-4B Single-Expression Frontier GRPO Smoke

- Goal: run a minimal real GRPO smoke from `runs/sft/qwen3_4b_openr1_cn_math_sft/checkpoint-5250` using only the frontier prompts kept from the single-expression rollout audit.
- Code commit used on server: `7ac6495bd1a84d293b62c05f2000246cce4dfcce`.
- Worktree: `/fs/nexus-scratch/qhe123/posttrain-lab-worktrees/4affb7b-sympy-boxed-data`.
- Output path: `runs/rlvr/qwen3_4b_openr1_cn_math_single_expr_grpo_smoke_v4/`.
- Slurm job: `6938749`, completed successfully on `cbcb-heng` in `00:02:37` on RTX A5000.
- Failed setup attempts: jobs `6938744`, `6938745`, and `6938746` failed before any trainer step because the temporary Slurm-side data-prep snippet had quoting/schema issues. They did not change reward semantics, eval prompts, raw data, or official splits.
- Training data: temporary RLVR JSONL under `runs/`, made by repeating the `7` single-expression frontier prompts to `24` rows with unique IDs. The source frontier file was `runs/rlvr/qwen3_4b_openr1_cn_math_ckpt5250_single_expr_micro_audit/frontier_grpo_train.jsonl`.
- GRPO settings: `12` steps, `num_generations=8`, `per_device_train_batch_size=8`, `max_completion_length=64`, `temperature=0.9`, `top_p=0.95`, `learning_rate=5e-7`, `beta=0.0`, `enable_thinking=false`.
- Reward config: `math_boxed_v001` with `allow_symbolic_equivalence=true` and `symbolic_equivalence_engine=sympy`; reward semantics were not changed.
- Rollout-format gate before training: reward mean `0.3571`, reward std `0.4792`, parse failure rate `0.0`, effective mixed group rate `0.5714`, average completion length `16.0`.
- Trainer signal: `12/12` trainer steps logged, final loss `0.00699`, mean reward `0.4271`, reward std `0.3605`, frac reward zero std `0.25`, effective mixed group rate `0.75`, nonzero grad step rate `0.75`, parse failure rate `0.0`, average completion length `10.15`.
- Post-training sample rollouts: `7` prompts with `8` completions each; several prompts had mixed reward vectors, such as `[1,1,1,1,0,1,0,1]`, `[1,0,1,0,1,0,1,0]`, and `[0,1,1,1,1,0,1,0]`.
- Heldout eval was not run in this smoke. This run only confirms that the filtered frontier prompts can drive nonzero GRPO advantage and gradients.
- Interpretation: this is the first clean 4B GRPO smoke on the single-expression filtered data. The RL loop is technically ready for a slightly larger frontier-prompt experiment, but model-quality claims still require a frozen heldout eval comparison against the SFT checkpoint.

### 2026-05-30 Qwen3-4B Single-Expression Small GRPO With Heldout Comparison

- Goal: run a small real GRPO experiment from `runs/sft/qwen3_4b_openr1_cn_math_sft/checkpoint-5250` and compare the frozen SFT checkpoint against SFT+GRPO on the same heldout validation sample.
- Code commit used on server: `08ca504e9e26aa233fd21cf4a2a45739894e6622`.
- Worktree: `/fs/nexus-scratch/qhe123/posttrain-lab-worktrees/4affb7b-sympy-boxed-data`.
- Data source: `data/rlvr_prompts/openr1_cn_math_alg_nt_single_expr_v1/`; fixed train sample `256` prompts and fixed heldout validation sample `256` prompts, both sampled with seed `20260530`.
- Parent adapter: `runs/sft/qwen3_4b_openr1_cn_math_sft/checkpoint-5250`.
- Failed first attempt: Slurm job `6938794` ran `97/100` steps but failed with CUDA OOM before final save/eval-after. It produced only the frozen SFT before-eval result: `answer_match=0.3125`, parse failure rate `0.0`, average completion length `13.8359`.
- Completed run: Slurm job `6938810`, output path `runs/rlvr/qwen3_4b_openr1_cn_math_single_expr_grpo_small_v2/`, completed successfully in `00:08:08` on RTX A5000.
- GRPO settings: `80` steps, `num_generations=8`, `per_device_train_batch_size=8`, `max_completion_length=64`, `temperature=0.9`, `top_p=0.95`, `learning_rate=3e-7`, `beta=0.0`, `gradient_checkpointing=true`, `enable_thinking=false`.
- Reward config: `math_boxed_v001` with `allow_symbolic_equivalence=true` and `symbolic_equivalence_engine=sympy`; reward semantics were not changed.
- Rollout-format gate before training: reward mean `0.3594`, reward std `0.4798`, parse failure rate `0.0`, frac reward zero std `0.6875`, effective mixed group rate `0.3125`, average completion length `13.0547`.
- Trainer signal: reward mean `0.3531`, reward std `0.2137`, frac reward zero std `0.5125`, effective mixed group rate `0.4875`, nonzero grad step rate `0.4875`, parse failure rate `0.0`, average completion length `8.3016`, final loss `0.01485`.
- Heldout eval before GRPO on the fixed `256` validation prompts: `answer_match=0.3125`, parse failure rate `0.0`, average completion length `13.8359`.
- Heldout eval after GRPO on the same `256` validation prompts: `answer_match=0.3242`, parse failure rate `0.0`, average completion length `13.7930`.
- Heldout delta: answer accuracy `+0.0117` absolute, parse failure rate unchanged at `0.0`, average output length `-0.0430`.
- Note: `eval_runner`'s original raw `format_success` field is not useful for this run because the configured regex used a single backslash in `\boxed`, which Python regex interpreted as `\b` word-boundary. `math_boxed_v001` parsing still succeeded with zero parse failures.
- Format-success recheck after the eval bug fix: SFT before `256/256 = 1.0`; SFT+GRPO after `256/256 = 1.0`. Recheck artifacts are `runs/rlvr/qwen3_4b_openr1_cn_math_single_expr_grpo_small_v2/format_success_recheck.json`, `format_success_recheck.md`, and `comparison_metrics_format_fixed.json`.
- Run artifacts synced locally: `comparison_metrics.json`, `comparison_report.md`, `metrics.jsonl`, `trainer_log.jsonl`, `run_card.md`, `sample_rollouts.jsonl`, heldout eval metrics/generations, and Slurm logs. Large checkpoint weights remain on the server scratch path.
- No `data/raw` files, eval prompts, reward semantics, or existing train/validation/test splits were modified.
- Interpretation: the small GRPO run technically worked and produced nonzero advantage on about half of trainer steps. Heldout answer accuracy improved slightly, but the gain is small on only `256` eval examples, so this is evidence that the setup is viable rather than proof of a robust model-quality improvement. The next run should keep the same frozen heldout comparison but increase training prompts and steps only after fixing the format-success reporting issue.

### 2026-05-30 Qwen3-4B Single-Expression Medium GRPO With Heldout Comparison

- Goal: increase the prior small GRPO run by one scale step, save checkpoints, and decide whether the current SFT checkpoint is suitable for further GRPO scaling.
- Code commit used on server: `5049624a267c1984cfd3285bdd397c7280233100`.
- Worktree: `/fs/nexus-scratch/qhe123/posttrain-lab-worktrees/4affb7b-sympy-boxed-data`.
- Output path: `runs/rlvr/qwen3_4b_openr1_cn_math_single_expr_grpo_medium_v1/`.
- Slurm job: `6938848`, completed successfully on `cbcb-heng` in `00:25:53` on RTX A5000.
- Parent adapter: `runs/sft/qwen3_4b_openr1_cn_math_sft/checkpoint-5250`.
- Data source: `data/rlvr_prompts/openr1_cn_math_alg_nt_single_expr_v1/`; fixed train sample `1024` prompts and fixed heldout validation sample `512` prompts, both sampled with seed `20260530`.
- Data hashes: train `0d26027244296f400ef251c96cbfb9896a2ed94f7740399beb5437b7903ea90c`; heldout eval `c543c2f5bb8804eb05ba81001b5025e5957268df750a825c2f9e789fca83a953`.
- GRPO settings: `300` steps, `num_generations=8`, `per_device_train_batch_size=8`, `max_completion_length=64`, `temperature=0.9`, `top_p=0.95`, `learning_rate=3e-7`, `beta=0.0`, `gradient_checkpointing=true`, `enable_thinking=false`.
- Reward config: `math_boxed_v001` with `allow_symbolic_equivalence=true` and `symbolic_equivalence_engine=sympy`; reward semantics were not changed.
- Checkpoints saved on server: `checkpoint-100`, `checkpoint-200`, `checkpoint-300`, plus the final root adapter under the run directory.
- Rollout-format gate before training: reward mean `0.3594`, reward std `0.4798`, parse failure rate `0.0`, frac reward zero std `0.5625`, effective mixed group rate `0.4375`, average completion length `14.20`.
- Trainer signal: `300/300` trainer steps logged, final loss `0.00628`, reward mean `0.33`, reward std `0.2145`, frac reward zero std `0.51`, effective mixed group rate `0.49`, nonzero grad step rate `0.49`, parse failure rate `0.0`, average completion length `8.58`.
- Heldout eval before GRPO on the fixed `512` validation prompts: `answer_match=0.3125`, `format_success=0.9980`, parse failure rate `0.0`, average completion length `13.56`.
- Heldout eval after GRPO on the same `512` validation prompts: `answer_match=0.3184`, `format_success=0.9980`, parse failure rate `0.0`, average completion length `13.61`.
- Heldout delta: answer accuracy `+0.0059` absolute, format success unchanged, parse failure rate unchanged at `0.0`, average output length `+0.0449`.
- Run artifacts synced locally: `comparison_metrics.json`, `comparison_report.md`, `metrics.jsonl`, `trainer_log.jsonl`, `run_card.md`, `sample_rollouts.jsonl`, heldout eval metrics/generations, and Slurm logs. Large adapter checkpoint weights remain on the server scratch path.
- No `data/raw` files, eval prompts, reward semantics, or existing train/validation/test splits were modified.
- Interpretation: the medium run is technically healthy and did not regress format, parseability, or output length. The heldout accuracy gain is positive but very small on `512` examples, so it supports one more cautious scale-up rather than a large run. The next run should keep the same parent SFT checkpoint and frozen eval protocol, increase data/steps moderately, and consider evaluating intermediate checkpoints before committing to longer GRPO.

### 2026-05-30 Qwen3-4B Single-Expression GRPO Continuation to 2000 Steps

- Goal: add a heldout sampled eval and continue GRPO from the medium run to a total of about `2000` GRPO steps from the selected SFT checkpoint.
- Code commit used on server: `7d0455bfe275bff68ba727e9a488840acd9df838`.
- Worktree: `/fs/nexus-scratch/qhe123/posttrain-lab-worktrees/4affb7b-sympy-boxed-data`.
- Output path: `runs/rlvr/qwen3_4b_openr1_cn_math_single_expr_grpo_2k_v1/`.
- Slurm job: `6938965`, completed successfully on `cbcb-heng` in `01:23:08` on RTX A5000.
- Parent adapter: `runs/rlvr/qwen3_4b_openr1_cn_math_single_expr_grpo_medium_v1/checkpoint-300`.
- Data source: `data/rlvr_prompts/openr1_cn_math_alg_nt_single_expr_v1/`; fixed train sample `4096` prompts and fixed heldout validation sample `512` prompts, both sampled with seed `20260530`.
- Data hashes: train `498c15e0138c0fa3009dd3f4e3eaac2e4f56c27c1e1e02e29800e005acbb61f1`; heldout eval `f55130fff2bcd1e304df5bf0ac8e8015d536cb011bb8d00206a2522942a884ef`.
- GRPO settings: `1700` additional steps, `num_generations=8`, `per_device_train_batch_size=8`, `max_completion_length=64`, `temperature=0.9`, `top_p=0.95`, `learning_rate=2e-7`, `beta=0.0`, `gradient_checkpointing=true`, `enable_thinking=false`.
- Total GRPO steps from the selected SFT checkpoint: `2000` (`300` medium-run steps plus `1700` continuation steps).
- Checkpoints saved on server: `checkpoint-250`, `checkpoint-500`, `checkpoint-750`, `checkpoint-1000`, `checkpoint-1250`, `checkpoint-1500`, `checkpoint-1700`, plus the final root adapter under the run directory.
- Reward config: `math_boxed_v001` with `allow_symbolic_equivalence=true` and `symbolic_equivalence_engine=sympy`; reward semantics were not changed.
- Rollout-format gate before continuation: reward mean `0.2949`, reward std `0.4560`, parse failure rate `0.0`, frac reward zero std `0.5156`, effective mixed group rate `0.4844`, average completion length `13.59`.
- Trainer signal over `1700` steps: final loss `0.00719`, reward mean `0.3319`, reward std `0.1815`, frac reward zero std `0.5924`, effective mixed group rate `0.4076`, nonzero grad step rate `0.4076`, parse failure rate `0.0`, average completion length `8.65`.
- Greedy heldout eval on `512` prompts before continuation: `answer_match=0.3125`, `format_success=1.0`, parse failure rate `0.0`, average completion length `13.80`.
- Greedy heldout eval after continuation: `answer_match=0.3164`, `format_success=1.0`, parse failure rate `0.0`, average completion length `13.50`.
- Greedy heldout delta: answer accuracy `+0.0039` absolute, format success unchanged, parse failure rate unchanged at `0.0`, average output length `-0.3008`.
- Sampled heldout eval settings: same `512` prompts, `8` completions per prompt, temperature `0.9`, top-p `0.95`, max new tokens `64`, same reward and symbolic-equivalence settings.
- Sampled heldout eval before continuation: sampled accuracy `0.2632`, `pass_at_8=0.5020`, format success `1.0`, parse failure rate `0.0`, mixed prompt rate `0.3711`, mean unique answer count `4.8125`.
- Sampled heldout eval after continuation: sampled accuracy `0.2830`, `pass_at_8=0.5098`, format success `0.9998`, parse failure rate `0.00024`, mixed prompt rate `0.3672`, mean unique answer count `4.3574`.
- Sampled heldout delta: sampled accuracy `+0.0198`, `pass_at_8 +0.0078`, parse failure rate `+0.00024`, average completion length `-0.2778`, mean unique answer count `-0.4551`.
- Greedy answer changes: `7` heldout examples changed from wrong to correct, `5` changed from correct to wrong, and `464/512` greedy generations were unchanged.
- Sampled pass@8 changes: `38` prompts changed from no-pass to pass, `34` changed from pass to no-pass, and `338/512` prompts had no sampled reward-mean change.
- Run artifacts synced locally: `comparison_metrics.json`, `comparison_report.md`, greedy and sampled eval metrics/generations, `metrics.jsonl`, `trainer_log.jsonl`, `run_card.md`, `sample_rollouts.jsonl`, and Slurm logs. Large adapter checkpoint weights remain on the server scratch path.
- No `data/raw` files, eval prompts, reward semantics, or existing train/validation/test splits were modified.
- Interpretation: continuing to 2000 total GRPO steps improved sampled accuracy more than greedy accuracy, suggesting the policy distribution shifted modestly toward correct answers while top-1 behavior barely moved. The gain remains small and noisy; continuing much longer on the same setup is unlikely to produce a large greedy jump without improving prompt selection, data difficulty balance, or adding checkpoint-level selection/eval.

### 2026-06-02 Qwen2.5-Math-1.5B DAPO GRPO Prep

- Goal: prepare a higher-resource GRPO experiment based on the Section 4.1 / Appendix D Qwen2.5-Math-1.5B setup, adapted to Nexus resources.
- Added DAPO staging command: `make dapo-rlvr-data`, which converts the downloaded DAPO-Math Raw 17k parquet into strict RLVR JSONL under `data/rlvr_prompts/dapo_math_raw_17k/train.jsonl`.
- Added config: `configs/rlvr/qwen25_math_1_5b_dapo_grpo_paperish.yaml`.
- Paper-aligned settings in the config: Qwen2.5-Math-1.5B, DAPO17k, `num_generations=4`, `max_completion_length=2048`, `learning_rate=2e-6`, `epsilon=0.22`, `beta=0.0`, `temperature=0.8`, `num_iterations=2`, DAPO loss, and full finetuning via `peft.method: none`.
- Nexus adaptation: intended `4` GPU launch uses `per_device_train_batch_size=16`, giving a target global prompt batch of about `64` prompts under `torchrun --nproc_per_node=4`.
- Added multi-GPU Slurm runner: `scripts/slurm/run_grpo_config.sh`.
- Added submit target: `make submit-qwen25-dapo-grpo`, defaulting to `cbcb-heng`, `gpu:rtx6000ada:4`, `16` CPU, `192G` memory, and `12:00:00` walltime.
- Server staging check at commit `cedb421`: `make dapo-rlvr-data` produced `17,917` RLVR train rows, schema validation passed, and `make rlvr-qwen25-dapo-dry` confirmed no model loading or training in dry-run mode.
- Safety: no training was started; no `data/raw`, eval prompts, reward semantics, or existing train/validation/test splits were modified.
