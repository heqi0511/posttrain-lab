# Experiment Log

Status: scaffold plus SFT overfit-32 and smoke-1k dry-run paths.

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
- Default mode: dry-run with synthetic temporary data under `/tmp/posttrain_lab_sft/`
- Output path: `runs/sft/smoke_1k/`
- This is a smoke run, not a final training result.
- Required artifacts: `resolved_config.yaml`, `run_card.md`, `metrics.jsonl`, `sample_generations.jsonl`, `eval_diff.md`
- The target first runs the fixed dry-run eval baseline and then runs eval-after-train without modifying baseline eval config files.
