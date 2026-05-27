.PHONY: format lint test test-rewards test-eval validate-data eval-baseline sft-smoke sft-overfit32 sft-overfit32-qwen3 rlvr-smoke rlvr-smoke-qwen3 rlvr-small compare-runs

format:
	@echo "format placeholder: no formatter configured yet"

lint:
	@echo "lint placeholder: no linter configured yet"

test:
	PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q

test-rewards:
	PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q tests/test_math_reward.py

test-eval:
	PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q tests/test_eval_runner.py

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
	PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python3 -m posttrain_lab.train.train_grpo --config configs/rlvr/toy_math_grpo.yaml

rlvr-smoke-qwen3:
	PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python3 -m posttrain_lab.train.train_grpo --config configs/rlvr/qwen3_0_6b_grpo_smoke.yaml

rlvr-small: eval-baseline
	PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python3 -m posttrain_lab.train.train_grpo --config configs/rlvr/math_1k_grpo.yaml

compare-runs:
	PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python3 -m posttrain_lab.eval.compare_runs --base runs/rlvr/math_1k_grpo/evals/base --sft runs/rlvr/math_1k_grpo/evals/sft --rlvr runs/rlvr/math_1k_grpo/evals/sft_rlvr --output runs/rlvr/math_1k_grpo/comparison_report.md --failure-output runs/rlvr/math_1k_grpo/failure_cases.jsonl
