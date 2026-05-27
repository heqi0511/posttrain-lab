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
- Data source: synthetic schema-valid SFT examples generated under `runs/sft/smoke_1k/data/` when missing.
- Output path: `runs/sft/smoke_1k/`
- This is a smoke run, not a final training result.
- Required artifacts: `resolved_config.yaml`, `run_card.md`, `metrics.jsonl`, `sample_generations.jsonl`, `eval_diff.md`
- The target first runs the fixed dry-run eval baseline, then runs eval-after-train on the saved adapter without modifying baseline eval config files.
- Required checks: train loss, validation loss, format success, target eval score, average output length, and 20 sampled generations.

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
