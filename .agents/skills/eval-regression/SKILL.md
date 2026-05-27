---
name: eval-regression
description: Use when running, changing, auditing, or comparing eval suites, metrics, baselines, regression reports, answer extraction, pass/fail gates, or model comparison results.
---

# Eval Regression

## When to Use

Use for eval harness changes, metric changes, answer extraction, baseline runs, regression comparisons, failure taxonomy updates, and eval-card maintenance.

Do not use for training data edits, reward semantic changes, or training loop changes.

## Required Workflow

1. Treat eval prompts, labels, metrics, and baseline logic as frozen unless review approves a change.
2. Confirm candidate and baseline runs use the same eval version.
3. Run a small eval smoke before full evaluation.
4. Report aggregate metrics and representative failures.
5. Compare against prior baseline without changing prompts or answer keys.
6. Update `docs/knowledge/eval_card.md` and `docs/knowledge/experiment_log.md` when eval behavior changes.
7. Request human review before changing any eval-defining artifact.

## Required Commands

```bash
make test-eval
make eval-baseline
```

If comparison tooling exists:

```bash
python .agents/skills/eval-regression/scripts/run_eval_suite.py <config>
python .agents/skills/eval-regression/scripts/compare_runs.py <baseline> <candidate>
```

## Invariants / Forbidden Actions

- Never change eval prompts, labels, or metrics to make a model look better.
- Never expose hidden tests or labels in prompts, training data, or reward fixtures.
- Never compare runs produced by different eval versions without saying so.
- Never drop failed examples from reports.
- Never use eval results as training examples without explicit review.

## Common Failure Modes

- Answer extraction changes inflate accuracy without model improvement.
- Baseline and candidate use different prompts, seeds, or decoding settings.
- Hidden answers leak into data, prompts, or reward fixtures.
- Metric aggregation hides severe subgroup regressions.
- Evaluation silently skips failed or timed-out examples.
- Reported gains are within noise but described as decisive.

## References

- `docs/knowledge/eval_card.md`
- `docs/knowledge/failure_taxonomy.md`
- `docs/knowledge/experiment_log.md`
- `docs/knowledge/model_choices.md`
- `.agents/skills/eval-regression/references/eval_protocol.md`
