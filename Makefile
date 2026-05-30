.PHONY: format lint test test-rewards test-eval validate-data validate-sympy-boxed-data build-sympy-boxed-data openr1-level-rlvr-data openr1-cn-math-data check-leakage eval-baseline eval-math-dataset-dry sft-smoke sft-openr1-math-1k sft-openr1-math-1k-long sft-openr1-format-repair sft-qwen3-4b-format-repair-tiny sft-qwen3-4b-sympy-boxed-smoke sft-qwen3-4b-sympy-boxed-full sft-qwen3-4b-cn-math sft-overfit32 sft-overfit32-qwen3 rlvr-smoke rlvr-smoke-qwen3 rlvr-frontier-audit rlvr-frontier-smoke gsm8k-rlvr-data rlvr-gsm8k-scout-thinking-false rlvr-gsm8k-scout-thinking-true rlvr-gsm8k-scout rlvr-gsm8k-audit-thinking-false rlvr-gsm8k-audit-thinking-true rlvr-gsm8k-audit diagnose-parse-failures rlvr-small compare-runs e2e-smoke
PYTHON ?= python3
RUN_FRONTIER_AUDIT ?= 0
SYMPY_BOXED_DATA_DIR ?= data/staged/openr1_deepmath_sympy_boxed_v1
OPENR1_LEVEL_RLVR_DIR ?= data/rlvr_prompts/openr1_math_l2_l3_alg_nt_v1
OPENR1_CN_MATH_RLVR_DIR ?= data/rlvr_prompts/openr1_cn_math_alg_nt_v1
OPENR1_CN_MATH_SFT_DIR ?= data/staged/openr1_cn_math_alg_nt_sft_v1

format:
	@echo "format placeholder: no formatter configured yet"

lint:
	@echo "lint placeholder: no linter configured yet"

test:
	PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m pytest -q

test-rewards:
	PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m pytest -q tests/test_math_reward.py

test-eval:
	PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m pytest -q tests/test_eval_runner.py tests/test_math_dataset_eval.py

validate-data:
	PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m posttrain_lab.data.validate --type sft --path tests/fixtures/sft_good.jsonl
	PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m posttrain_lab.data.validate --type rlvr --path tests/fixtures/rlvr_good.jsonl
	PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m posttrain_lab.data.validate --type sft --path data/fixtures/e2e/sft_seed.jsonl
	PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m posttrain_lab.data.validate --type rlvr --path data/fixtures/e2e/rlvr_seed.jsonl
	@if [ -f $(SYMPY_BOXED_DATA_DIR)/train.jsonl ]; then $(MAKE) validate-sympy-boxed-data; fi
	@if [ -f data/rlvr_prompts/frontier_grpo_train.jsonl ]; then PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m posttrain_lab.data.validate --type rlvr --path data/rlvr_prompts/frontier_grpo_train.jsonl; fi
	@if [ -f data/rlvr_prompts/gsm8k_train.jsonl ]; then PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m posttrain_lab.data.validate --type rlvr --path data/rlvr_prompts/gsm8k_train.jsonl; fi
	@if [ -f $(OPENR1_LEVEL_RLVR_DIR)/train.jsonl ]; then PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m posttrain_lab.data.validate --type rlvr --path $(OPENR1_LEVEL_RLVR_DIR)/train.jsonl; fi
	@if [ -f $(OPENR1_LEVEL_RLVR_DIR)/val.jsonl ]; then PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m posttrain_lab.data.validate --type rlvr --path $(OPENR1_LEVEL_RLVR_DIR)/val.jsonl; fi
	@if [ -f $(OPENR1_LEVEL_RLVR_DIR)/test.jsonl ]; then PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m posttrain_lab.data.validate --type rlvr --path $(OPENR1_LEVEL_RLVR_DIR)/test.jsonl; fi
	@if [ -f $(OPENR1_CN_MATH_RLVR_DIR)/train.jsonl ]; then PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m posttrain_lab.data.validate --type rlvr --path $(OPENR1_CN_MATH_RLVR_DIR)/train.jsonl; fi
	@if [ -f $(OPENR1_CN_MATH_RLVR_DIR)/val.jsonl ]; then PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m posttrain_lab.data.validate --type rlvr --path $(OPENR1_CN_MATH_RLVR_DIR)/val.jsonl; fi
	@if [ -f $(OPENR1_CN_MATH_RLVR_DIR)/test.jsonl ]; then PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m posttrain_lab.data.validate --type rlvr --path $(OPENR1_CN_MATH_RLVR_DIR)/test.jsonl; fi
	@if [ -f $(OPENR1_CN_MATH_SFT_DIR)/train.jsonl ]; then PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m posttrain_lab.data.validate --type sft --path $(OPENR1_CN_MATH_SFT_DIR)/train.jsonl; fi
	@if [ -f $(OPENR1_CN_MATH_SFT_DIR)/val.jsonl ]; then PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m posttrain_lab.data.validate --type sft --path $(OPENR1_CN_MATH_SFT_DIR)/val.jsonl; fi
	@if [ -f $(OPENR1_CN_MATH_SFT_DIR)/test.jsonl ]; then PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m posttrain_lab.data.validate --type sft --path $(OPENR1_CN_MATH_SFT_DIR)/test.jsonl; fi
	@if [ -f runs/rlvr/gsm8k_frontier_audit_thinking_false/frontier_grpo_train.jsonl ]; then PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m posttrain_lab.data.validate --type rlvr --path runs/rlvr/gsm8k_frontier_audit_thinking_false/frontier_grpo_train.jsonl; fi
	@if [ -f runs/rlvr/gsm8k_frontier_audit_thinking_true/frontier_grpo_train.jsonl ]; then PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m posttrain_lab.data.validate --type rlvr --path runs/rlvr/gsm8k_frontier_audit_thinking_true/frontier_grpo_train.jsonl; fi

validate-sympy-boxed-data:
	PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m posttrain_lab.data.validate --type sft --path $(SYMPY_BOXED_DATA_DIR)/train.jsonl
	PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m posttrain_lab.data.validate --type sft --path $(SYMPY_BOXED_DATA_DIR)/val.jsonl
	PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m posttrain_lab.data.validate --type sft --path $(SYMPY_BOXED_DATA_DIR)/test.jsonl

build-sympy-boxed-data:
	PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m posttrain_lab.data.math_sft_curation --output-dir $(SYMPY_BOXED_DATA_DIR)

openr1-level-rlvr-data:
	PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m posttrain_lab.data.openr1_level_filter --output-dir $(OPENR1_LEVEL_RLVR_DIR)

openr1-cn-math-data:
	PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m posttrain_lab.data.openr1_level_filter --dataset-config extended --levels all --sources cn_k12,cn_contest,amc_aime --domains Algebra,Number\ Theory --output-dir $(OPENR1_CN_MATH_RLVR_DIR) --sft-output-dir $(OPENR1_CN_MATH_SFT_DIR) --train-count 20000 --val-count 2000 --test-count 2000

check-leakage:
	@echo "check-leakage placeholder: no leakage checker configured yet"

eval-baseline:
	@mkdir -p /tmp/posttrain_lab_eval
	@printf '%s\n' '{"id":"baseline-001","prompt":"Return the answer to 2 + 2 in boxed format.","answer":"\\boxed{4}","mock_generation":"\\boxed{4}"}' > /tmp/posttrain_lab_eval/baseline_prompts.jsonl
	PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m posttrain_lab.eval.eval_runner --config configs/eval/baseline.yaml

eval-math-dataset-dry:
	PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m posttrain_lab.eval.math_dataset_eval --dataset-id dummy --dataset-config default --split train --model-name dummy --output-dir /tmp/posttrain_lab_math_dataset_eval --sample-size 1 --dry-run

sft-smoke: eval-baseline
	PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m posttrain_lab.train.train_sft --config configs/sft/smoke_1k.yaml

sft-openr1-math-1k: eval-baseline
	PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m posttrain_lab.train.train_sft --config configs/sft/openr1_math_1k.yaml

sft-openr1-math-1k-long:
	PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m posttrain_lab.train.train_sft --config configs/sft/openr1_math_1k_len8192.yaml

sft-openr1-format-repair:
	PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m posttrain_lab.train.train_sft --config configs/sft/openr1_math_format_repair_1k.yaml

sft-qwen3-4b-format-repair-tiny:
	PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m posttrain_lab.train.train_sft --config configs/sft/qwen3_4b_openr1_format_repair_tiny.yaml

sft-qwen3-4b-sympy-boxed-smoke:
	PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m posttrain_lab.train.train_sft --config configs/sft/qwen3_4b_sympy_boxed_smoke.yaml

sft-qwen3-4b-sympy-boxed-full:
	PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m posttrain_lab.train.train_sft --config configs/sft/qwen3_4b_sympy_boxed_full.yaml

sft-qwen3-4b-cn-math:
	PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m posttrain_lab.train.train_sft --config configs/sft/qwen3_4b_openr1_cn_math_sft.yaml

sft-overfit32:
	PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m posttrain_lab.train.train_sft --config configs/sft/overfit32.yaml

sft-overfit32-qwen3:
	PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m posttrain_lab.train.train_sft --config configs/sft/qwen3_0_6b_overfit32.yaml

rlvr-smoke:
	PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m posttrain_lab.train.train_grpo --config configs/rlvr/toy_math_grpo.yaml

rlvr-smoke-qwen3:
	PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m posttrain_lab.train.train_grpo --config configs/rlvr/qwen3_0_6b_grpo_smoke.yaml

rlvr-frontier-audit:
	@if [ "$(RUN_FRONTIER_AUDIT)" != "1" ]; then \
		echo "frontier audit is disabled by default because rollout sampling is expensive."; \
		echo "Run 'make rlvr-frontier-audit RUN_FRONTIER_AUDIT=1' to opt in."; \
	else \
		PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m posttrain_lab.data.rollout_audit --config configs/rlvr/frontier_prompt_audit.yaml; \
	fi

rlvr-frontier-smoke:
	@if [ "$(RUN_FRONTIER_AUDIT)" != "1" ]; then \
		echo "frontier GRPO smoke is disabled by default because it depends on frontier audit output."; \
		echo "Run 'make rlvr-frontier-smoke RUN_FRONTIER_AUDIT=1' to opt in."; \
	else \
		$(MAKE) rlvr-frontier-audit RUN_FRONTIER_AUDIT=1; \
		PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m posttrain_lab.train.train_grpo --config configs/rlvr/frontier_grpo_smoke.yaml; \
	fi

gsm8k-rlvr-data:
	PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m posttrain_lab.data.gsm8k --output data/rlvr_prompts/gsm8k_train.jsonl --summary data/rlvr_prompts/gsm8k_train_summary.json

rlvr-gsm8k-scout-thinking-false:
	@if [ "$(RUN_FRONTIER_AUDIT)" != "1" ]; then \
		echo "GSM8K frontier scout is disabled by default because it runs model rollouts."; \
		echo "Run 'make rlvr-gsm8k-scout-thinking-false RUN_FRONTIER_AUDIT=1' to opt in."; \
	else \
		$(MAKE) gsm8k-rlvr-data; \
		PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m posttrain_lab.data.rollout_audit --config configs/rlvr/gsm8k_frontier_scout_thinking_false.yaml; \
	fi

rlvr-gsm8k-scout-thinking-true:
	@if [ "$(RUN_FRONTIER_AUDIT)" != "1" ]; then \
		echo "GSM8K frontier scout is disabled by default because it runs model rollouts."; \
		echo "Run 'make rlvr-gsm8k-scout-thinking-true RUN_FRONTIER_AUDIT=1' to opt in."; \
	else \
		$(MAKE) gsm8k-rlvr-data; \
		PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m posttrain_lab.data.rollout_audit --config configs/rlvr/gsm8k_frontier_scout_thinking_true.yaml; \
	fi

rlvr-gsm8k-scout: rlvr-gsm8k-scout-thinking-false rlvr-gsm8k-scout-thinking-true

rlvr-gsm8k-audit-thinking-false:
	@if [ "$(RUN_FRONTIER_AUDIT)" != "1" ]; then \
		echo "GSM8K frontier audit is disabled by default because it runs model rollouts."; \
		echo "Run 'make rlvr-gsm8k-audit-thinking-false RUN_FRONTIER_AUDIT=1' to opt in."; \
	else \
		$(MAKE) gsm8k-rlvr-data; \
		PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m posttrain_lab.data.rollout_audit --config configs/rlvr/gsm8k_frontier_audit_thinking_false.yaml; \
	fi

rlvr-gsm8k-audit-thinking-true:
	@if [ "$(RUN_FRONTIER_AUDIT)" != "1" ]; then \
		echo "GSM8K frontier audit is disabled by default because it runs model rollouts."; \
		echo "Run 'make rlvr-gsm8k-audit-thinking-true RUN_FRONTIER_AUDIT=1' to opt in."; \
	else \
		$(MAKE) gsm8k-rlvr-data; \
		PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m posttrain_lab.data.rollout_audit --config configs/rlvr/gsm8k_frontier_audit_thinking_true.yaml; \
	fi

rlvr-gsm8k-audit: rlvr-gsm8k-audit-thinking-false rlvr-gsm8k-audit-thinking-true

diagnose-parse-failures:
	PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m posttrain_lab.diagnostics.parse_failure_taxonomy --output-dir data/reports/parse_failure_taxonomy

rlvr-small: eval-baseline
	PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m posttrain_lab.train.train_grpo --config configs/rlvr/math_1k_grpo.yaml

compare-runs:
	PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m posttrain_lab.eval.compare_runs --base runs/rlvr/math_1k_grpo/evals/base --sft runs/rlvr/math_1k_grpo/evals/sft --rlvr runs/rlvr/math_1k_grpo/evals/sft_rlvr --output runs/rlvr/math_1k_grpo/comparison_report.md --failure-output runs/rlvr/math_1k_grpo/failure_cases.jsonl

e2e-smoke:
	PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m posttrain_lab.pipelines.e2e_smoke --config configs/e2e/toy_math_posttraining.yaml $(if $(REAL_RUN),--real-run,)
