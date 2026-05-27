## Non-negotiable rules

- Never modify files under `data/raw/` unless explicitly instructed.
- Never change train/val/test splits without updating `docs/knowledge/data_card.md` and asking for review.
- Never modify eval prompts, labels, or metrics to make a model look better.
- Never modify reward functions without adding or updating unit tests.
- Never expose hidden tests or ground-truth answers in prompts.
- Never start long training runs unless explicitly requested. A long run means any server GPU job, any run expected to exceed 10 minutes locally, or any command that downloads large models or datasets.
- Never delete failed runs, bad metrics, or failure samples.
- Never make network calls inside reward functions or verifiers.
- Never commit secrets, API keys, model tokens, or private credentials.

## Project invariants

- Base model: `Qwen/Qwen3-0.6B-Base`.
- First task family: math only.
- First training framework: TRL for the MVP.
- Migration target: verl only after data, rewards, and evals are stable enough to justify the move.
- Stage 1 objective: use SFT to teach output format, math-domain behavior, tool-use protocol, and final-answer schema.
- Stage 2 objective: use RLVR to improve correctness with deterministic verifiers.

## Default workflow

For any non-trivial change:

1. Inspect the relevant skill.
2. Make the smallest change that satisfies the task.
3. Add or update tests.
4. Run the required checks.
5. Summarize changed files, commands run, and remaining risks.

Prefer smoke tests before full runs.
Prefer deterministic tests before expensive training.
Prefer explicit configs over hidden defaults.

If the relevant skill is missing or empty, inspect `docs/knowledge/` and mention the missing skill in the summary.

## Required checks

If a listed `make` target exists, run it. If it does not exist yet, state that explicitly and prefer adding the target before relying on an ad hoc command.

For general code changes:

```bash
make format
make lint
make test
```

For data schema or dataset changes:

```bash
make validate-data
make check-leakage
```

For reward/verifier changes:

```bash
make test-rewards
```

For eval changes:

```bash
make test-eval
make eval-baseline
```

For SFT training code changes:

```bash
make sft-overfit32
```

For RLVR training code changes:

```bash
make rlvr-smoke
```

## Reward and eval integrity

- Reward functions and verifiers must be deterministic, side-effect-free, and independent of wall-clock time.
- Reward functions and verifiers must not call external APIs, download resources, or depend on network access.
- Eval sets, hidden tests, and ground-truth answers must not enter SFT or RLVR training data.
- Eval prompts, labels, and metrics must not be changed for the purpose of improving reported model performance.
- Failed runs must be preserved. New runs should write to new directories rather than overwriting prior outputs.

## Experiment logging

Every training run must write:

- `run_card.md`
- `resolved_config.yaml`
- `metrics.jsonl`
- `eval_report.json`
- `sample_generations.jsonl`

Every `run_card.md` must include:

- base model
- parent checkpoint, if applicable
- checkpoint or adapter path
- git commit
- launch command
- environment and dependency versions
- hardware or server node
- random seed
- whether this was a smoke run
- training duration
- data path
- data hash
- config path
- config hash
- reward version, if applicable
- eval version
- final metrics
- known caveats

## Human review required

Ask for review before:

- changing reward semantics
- changing eval prompts, labels, or metrics
- changing train/val/test split logic
- adding new training datasets
- launching runs expected to use significant GPU time
- changing base model family
- changing tokenizer or chat template
- deleting or overwriting previous runs
