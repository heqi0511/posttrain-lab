# Experiment Log

Status: scaffold plus SFT overfit-32 dry-run path.

Record training and eval runs here with links to run directories, run cards, resolved configs, metrics, sample generations, representative failures, and comparison notes.

## SFT Overfit-32

- Config: `configs/sft/overfit32.yaml`
- Command: `make sft-overfit32`
- Default mode: dry-run with synthetic temporary data under `/tmp/posttrain_lab_sft/`
- Output path: `runs/sft/overfit32/`
- Required artifacts: `resolved_config.yaml`, `run_card.md`, `metrics.jsonl`, `sample_generations.jsonl`
- Real TRL/PEFT training requires setting `dry_run: false`, installing `trl`, `peft`, `datasets`, and `transformers`, and using an available local or Hugging Face model path.
