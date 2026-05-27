---
name: rlvr-experiment
description: Use when preparing, configuring, smoke-testing, debugging, or reviewing RLVR, GRPO, TRL RL training, rollout sampling, KL control, verifier rewards, or policy checkpoints.
---

# RLVR Experiment

## When to Use

Use for RLVR/GRPO configs, rollout settings, verifier integration, KL policy, sampling parameters, smoke runs, checkpoint review, and RLVR run-card review.

Do not use for SFT data formatting, reward semantic changes, eval metric changes, or raw data edits.

## Required Workflow

1. Confirm SFT baseline, reward verifier, and eval baseline are stable.
2. Use TRL for the initial MVP; discuss verl migration only after data, rewards, and evals are stable.
3. Run reward tests and eval baseline before RLVR smoke.
4. Start with tiny prompt sets and short rollouts before expensive runs.
5. Record rollout sampling, KL, reward version, policy checkpoint, and seed.
6. Compare against the frozen SFT baseline with unchanged eval settings.
7. Request human review before server GPU runs or algorithm-defining changes.

## Required Commands

```bash
make test-rewards
make test-eval
make eval-baseline
make rlvr-smoke
```

If smoke tooling exists:

```bash
python .agents/skills/rlvr-experiment/scripts/smoke_grpo.py <config>
```

## Invariants / Forbidden Actions

- Never launch full RLVR/GRPO or server GPU jobs without explicit approval.
- Never change reward semantics inside an RLVR experiment.
- Never tune eval prompts, labels, or metrics to improve RLVR results.
- Never compare RLVR against a moving SFT baseline.
- Never overwrite policy checkpoints, rollout logs, or failed generations.

## Common Failure Modes

- Reward hacking produces high reward but wrong answers.
- KL, sampling, or reward scale changes are not recorded.
- Verifier timeout or crash is interpreted as a valid low reward.
- Rollout prompts leak validation or hidden eval content.
- Baseline changes between SFT and RLVR comparison.
- Smoke run passes format but fails correctness or eval regression.

## References

- `docs/knowledge/project_overview.md`
- `docs/knowledge/model_choices.md`
- `docs/knowledge/reward_card.md`
- `docs/knowledge/eval_card.md`
- `docs/knowledge/experiment_log.md`
- `.agents/skills/rlvr-experiment/references/grpo_recipe.md`
