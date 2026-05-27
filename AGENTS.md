# Post-Training Agent Rules

## Principles

- Correctness over speed.
- Reproducibility over convenience.
- Eval integrity over metric improvement.
- Smoke tests before full training.
- Human review before expensive or experiment-defining changes.

## Scope

This repo prepares post-training workflows for approximately 5B-class models. The intended path is data curation -> SFT format/domain learning -> reward/verifier hardening -> RLVR/GRPO correctness improvement -> eval regression. Start with TRL MVPs; consider verl only after data, reward, and eval protocols are stable.

## Non-Negotiable Rules

- Do not implement or launch training code unless the user explicitly asks.
- Do not modify `data/raw/` unless explicitly instructed.
- Do not change train/val/test splits without updating `docs/knowledge/data_card.md` and asking for review.
- Do not modify eval prompts, labels, answer extraction, metrics, baselines, filters, decoding settings, or aggregation to improve reported results.
- Do not accept eval changes whose main effect is better metrics unless the change fixes a documented eval bug and has human review.
- Do not expose hidden tests or ground-truth answers in prompts, training data, or reward fixtures.
- Do not change reward semantics without tests and human review.
- Do not patch rewards in response to RLVR failures without separating reward fixes from policy training.
- Do not make network calls inside reward functions or verifiers.
- Do not delete failed runs, bad metrics, or failure samples.
- Do not overwrite prior runs; write new outputs to new run directories.
- Do not commit secrets, API keys, model tokens, private data, or credentials.
- Do not start long runs without explicit approval. Long runs include any server GPU job, local run expected to exceed 10 minutes, or command that downloads large models or datasets.
- Do not hand-edit experiment code on the server; server changes must be reproduced locally, committed, pushed, then checked out on the server.

## Responsibility Boundaries

- Data work owns schema, provenance, deduplication, leakage checks, and split records.
- SFT work owns format learning, supervised configs, overfit-32 gates, and SFT run artifacts.
- Reward work owns deterministic verifier semantics, fixtures, reward hacking checks, and reward cards.
- RLVR work owns rollout policy, KL/sampling settings, parse failure rate, completion length, and policy checkpoints.
- Eval work owns frozen prompts, labels, extraction, metrics, baselines, and regression reports.

## Default Workflow

For non-trivial work:

1. Inspect the relevant skill under `.agents/skills/`.
2. Check `docs/knowledge/` for project facts and prior decisions.
3. Make the smallest change that satisfies the task.
4. Add or update deterministic tests when behavior changes.
5. Run the required checks, or state clearly when a target does not exist yet.
6. Summarize changed files, commands run, and remaining risks.

## Post-Change Report

After modifying code, configs, docs, or experiment scripts, explain the change in beginner-friendly Chinese. Include:

1. Which files changed, and what each file is for.
2. Whether the task goal was completed.
3. Which acceptance criteria passed, failed, or were not applicable.
4. Which commands were run and what the results were.
5. Whether `data/raw`, eval prompts, reward semantics, or train/val/test splits changed.
6. Remaining risks or caveats.
7. Which files the user should open to review.

## Code Sync And Server Runs

- Treat the local repo as the only development source.
- The standard path is: local edit -> local checks -> `git commit` -> `git push` -> server `git fetch`/`git checkout <commit>` -> Slurm run.
- Server experiment code must run from a clean git worktree at a known commit.
- Prefer checking out the exact commit recorded in the run card instead of relying on branch tip state.
- Use `rsync` for data, logs, and result artifacts only; do not use it for code sync unless explicitly marked as an emergency workaround.
- Before launching server training, verify server `git status --porcelain` is empty and `git rev-parse HEAD` matches the intended commit.
- If a server-side debug change is necessary, copy the finding back into the local repo and repeat the commit/push/checkout path before treating results as canonical.
- Generated outputs under `runs/` are experiment artifacts, not source code; do not commit them unless the user explicitly asks.

## Required Checks

If a listed target does not exist yet, do not invent a hidden equivalent; report the gap.

| Task type | Required checks |
| --- | --- |
| General code or config | `make format`, `make lint`, `make test` |
| Data schema or dataset staging | `make validate-data`, `make check-leakage` |
| Reward or verifier | `make test-rewards` |
| Eval harness, metrics, prompts | `make test-eval`, `make eval-baseline` |
| SFT workflow | `make sft-smoke`, `make sft-overfit32` before any larger run |
| RLVR/GRPO workflow | `make rlvr-smoke`, `make test-rewards`, `make test-eval` |

## Human Review Required

Ask for review before:

- changing base model family, tokenizer, chat template, or context length assumptions
- changing reward semantics or reward aggregation
- changing eval prompts, labels, answer extraction, metrics, decoding settings, filters, or baseline comparison logic
- changing train/val/test split logic
- adding, removing, or materially filtering training datasets
- launching any server GPU job or long local run
- changing RL algorithm, KL policy, rollout sampling policy, or verifier contract
- deleting, hiding, or overwriting run outputs or failure cases

## Experiment Logging

Every training or eval run must write:

- `run_card.md`
- `resolved_config.yaml`
- `metrics.jsonl`
- `eval_report.json`
- `sample_generations.jsonl` when generations are produced

Every `run_card.md` must include:

- base model, checkpoint or adapter path, and parent checkpoint if applicable
- git commit, launch command, environment/dependency versions, hardware or server node
- random seed, smoke/full-run label, start time, duration, and failure status if failed
- data paths and hashes, config path and hash, reward version, eval version, decoding settings
- for generation runs: parse failure rate and completion length summary
- final metrics, representative failures, and known caveats

## Skill Map

- Data work: `.agents/skills/data-curation/SKILL.md`
- SFT work: `.agents/skills/sft-experiment/SKILL.md`
- Reward/verifier work: `.agents/skills/reward-verifier/SKILL.md`
- RLVR/GRPO work: `.agents/skills/rlvr-experiment/SKILL.md`
- Eval work: `.agents/skills/eval-regression/SKILL.md`
