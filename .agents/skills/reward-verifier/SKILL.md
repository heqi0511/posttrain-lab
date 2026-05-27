---
name: reward-verifier
description: Use when designing, testing, auditing, or debugging reward functions, verifiers, math answer checking, code execution checks, schema rewards, reward fixtures, or reward cards.
---

# Reward Verifier

## When to Use

Use for deterministic rewards, math answer equivalence, code verification, schema rewards, fixture tests, reward aggregation, and reward-card updates.

Do not use for SFT training, rollout scheduling, eval prompt changes, or data split changes.

## Required Workflow

1. Define the reward contract: inputs, outputs, scale, partial credit, and invalid-output handling.
2. Add or update fixtures before changing reward semantics.
3. Test correct, incorrect, malformed, adversarial, and edge-case outputs.
4. Keep verifiers deterministic, side-effect-free, offline, and fast.
5. Record semantic changes in `docs/knowledge/reward_card.md`.
6. Request human review before changing semantics or aggregation.

## Required Commands

```bash
make test-rewards
```

If verifier tests exist:

```bash
python .agents/skills/reward-verifier/scripts/run_verifier_tests.py
```

For RLVR readiness:

```bash
make rlvr-smoke
```

## Invariants / Forbidden Actions

- Never make network calls from rewards or verifiers.
- Never depend on wall-clock time, external mutable state, or non-fixed randomness.
- Never change reward semantics without tests and review.
- Never expose hidden eval answers through reward fixtures.
- Never reward formatting in a way that hides wrong answers unless explicitly intended.

## Common Failure Modes

- Equivalent math answers marked wrong due to brittle string matching.
- Wrong answers accepted because extraction grabs the wrong final field.
- Malformed outputs receive high reward.
- Reward scale changes silently destabilize RLVR.
- Verifier depends on package versions, timeouts, network, or filesystem state.
- Reward fixtures overlap with eval hidden answers.

## References

- `docs/knowledge/reward_card.md`
- `docs/knowledge/eval_card.md`
- `docs/knowledge/failure_taxonomy.md`
- `docs/knowledge/experiment_log.md`
- `.agents/skills/reward-verifier/references/reward_design.md`
