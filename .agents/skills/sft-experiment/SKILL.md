---
name: sft-experiment
description: Use when preparing, configuring, smoke-testing, debugging, or reviewing supervised fine-tuning, TRL SFTTrainer, LoRA/QLoRA, format learning, chat templates, or overfit checks.
---

# SFT Experiment

## When to Use

Use for SFT configs, SFT smoke runs, overfit checks, formatting behavior, tool-call format learning, chat-template checks, and SFT run-card review.

Do not use for reward design, RLVR rollouts, eval metric changes, or data split changes.

## Required Workflow

1. Confirm the base model, tokenizer, chat template, context length, and adapter strategy are documented in `docs/knowledge/model_choices.md`.
2. Verify the training data passed data-curation checks and has stable hashes.
3. Run `make sft-smoke`, then `make sft-overfit32`; do not start any larger SFT run until overfit-32 succeeds or the user explicitly overrides it.
4. Prefer explicit configs over CLI-only defaults.
5. Inspect sample generations for format compliance before judging quality.
6. Run eval only through the eval-regression skill.
7. Write run artifacts and update `docs/knowledge/experiment_log.md`.

## Required Commands

```bash
make validate-data
make sft-smoke
make sft-overfit32
make test-eval
```

If config generation exists:

```bash
python .agents/skills/sft-experiment/scripts/make_sft_config.py <args>
python .agents/skills/sft-experiment/scripts/smoke_train.py <config>
```

## Invariants / Forbidden Actions

- Never launch full SFT or server GPU jobs without explicit approval.
- Never launch larger SFT runs before `make sft-overfit32` passes or the user explicitly overrides the failed gate.
- Never change tokenizer or chat template without human review.
- Never train on eval prompts, hidden tests, or reward-only fixtures.
- Never compare runs without recording data/config/model hashes.
- Never treat loss reduction as sufficient evidence of task improvement.

## Common Failure Modes

- Chat template mismatch between training and inference.
- Labels include prompt tokens that should be masked.
- Sequence truncation removes final answers or tool-call fields.
- Data collator pads or masks incorrectly.
- Overfit-32 fails or is skipped, but a larger run is launched anyway.
- Reported improvement comes from format leakage or changed eval settings.

## References

- `docs/knowledge/project_overview.md`
- `docs/knowledge/model_choices.md`
- `docs/knowledge/data_card.md`
- `docs/knowledge/eval_card.md`
- `docs/knowledge/experiment_log.md`
- `.agents/skills/sft-experiment/references/sft_recipe.md`
