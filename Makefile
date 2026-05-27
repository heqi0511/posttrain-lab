.PHONY: format lint test validate-data eval-baseline sft-smoke sft-overfit32 sft-overfit32-qwen3 rlvr-smoke

format:
	@echo "format placeholder: no formatter configured yet"

lint:
	@echo "lint placeholder: no linter configured yet"

test:
	PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q

validate-data:
	PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python3 -m posttrain_lab.data.validate --type sft --path tests/fixtures/sft_good.jsonl
	PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python3 -m posttrain_lab.data.validate --type rlvr --path tests/fixtures/rlvr_good.jsonl

eval-baseline:
	@mkdir -p /tmp/posttrain_lab_eval
	@printf '%s\n' '{"id":"baseline-001","prompt":"Return the answer to 2 + 2 in boxed format.","answer":"\\boxed{4}","mock_generation":"\\boxed{4}"}' > /tmp/posttrain_lab_eval/baseline_prompts.jsonl
	PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python3 -m posttrain_lab.eval.eval_runner --config configs/eval/baseline.yaml

sft-smoke: eval-baseline
	PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python3 -m posttrain_lab.train.train_sft --config configs/sft/smoke_1k.yaml

sft-overfit32:
	PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python3 -m posttrain_lab.train.train_sft --config configs/sft/overfit32.yaml

sft-overfit32-qwen3:
	PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python3 -m posttrain_lab.train.train_sft --config configs/sft/qwen3_0_6b_overfit32.yaml

rlvr-smoke:
	@echo "rlvr-smoke placeholder: RLVR training is not implemented yet"
