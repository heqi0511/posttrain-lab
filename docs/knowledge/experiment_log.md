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
