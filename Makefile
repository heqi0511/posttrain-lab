.PHONY: format lint test validate-data eval-baseline sft-overfit32 rlvr-smoke

format:
	@echo "format placeholder: no formatter configured yet"

lint:
	@echo "lint placeholder: no linter configured yet"

test:
	PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q

validate-data:
	@echo "validate-data placeholder: validators are not implemented yet"

eval-baseline:
	@echo "eval-baseline placeholder: eval runner is not implemented yet"

sft-overfit32:
	@echo "sft-overfit32 placeholder: SFT training is not implemented yet"

rlvr-smoke:
	@echo "rlvr-smoke placeholder: RLVR training is not implemented yet"
