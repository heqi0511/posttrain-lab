# Data Card

Status: strict schema validators implemented for SFT and RLVR JSONL fixtures.

No real datasets are included yet. Raw data must remain read-only unless explicitly reviewed. Future updates should record source, license, schema, split logic, hashes, filters, deduplication, and leakage checks.

## Common Rules

- Files are UTF-8 JSONL with exactly one JSON object per line.
- Required `split` values are `train`, `val`, or `test`.
- `id` values must be non-empty strings and unique within a file.
- `metadata` must contain non-empty string fields: `source`, `domain`, `difficulty`, and `license`.
- Empty message or prompt content is invalid.
- Validation errors report file path and line number.

## SFT JSONL Schema

Required top-level fields: `id`, `split`, `messages`, `metadata`.

`messages` must be a non-empty list of objects with exactly `role` and `content`. Valid roles are `system`, `user`, and `assistant`; at least one assistant message is required.

Example:

```json
{"id":"sft-001","split":"train","messages":[{"role":"system","content":"You are careful."},{"role":"user","content":"Compute 2 + 2."},{"role":"assistant","content":"4"}],"metadata":{"source":"unit-test","domain":"math","difficulty":"easy","license":"synthetic"}}
```

## RLVR JSONL Schema

Required top-level fields: `id`, `split`, `prompt`, `verifier`, `metadata`.

`prompt` must be a non-empty list of user-only messages with exactly `role` and `content`. Assistant messages in RLVR prompts are invalid.

`verifier` must contain non-empty string fields: `type` and `answer`.

Example:

```json
{"id":"rlvr-001","split":"train","prompt":[{"role":"user","content":"Compute 5 + 7."}],"verifier":{"type":"math_boxed_v001","answer":"12"},"metadata":{"source":"unit-test","domain":"math","difficulty":"easy","license":"synthetic"}}
```

## Validation Commands

```bash
python -m posttrain_lab.data.validate --type sft --path <file>
python -m posttrain_lab.data.validate --type rlvr --path <file>
make validate-data
```

## RLVR Frontier Prompt Audit

Status: offline prompt selection tooling is available for GRPO signal checks, but it is disabled by default because rollout sampling is compute-expensive. It samples completions before training, computes per-prompt reward statistics, and writes a schema-valid filtered RLVR train set only when explicitly enabled.

Opt-in command:

```bash
make rlvr-frontier-audit RUN_FRONTIER_AUDIT=1
```

Default config: `configs/rlvr/frontier_prompt_audit.yaml`.

Default safety settings:

- `dry_run: true`; no model is loaded and no trainer step is executed.
- `selection.max_prompts: 20`.
- `rollout.completions_per_prompt: 16`.
- Source data: `data/fixtures/e2e/rlvr_seed.jsonl`.
- Filtered train output: `data/rlvr_prompts/frontier_grpo_train.jsonl`.

Audit artifacts:

- `data/rlvr_prompts/frontier_audit/rollout_audit_summary.json`
- `data/rlvr_prompts/frontier_audit/rollout_audit_by_prompt.csv`
- `data/rlvr_prompts/frontier_audit/sample_rollouts_for_review.jsonl`
- `data/rlvr_prompts/frontier_excluded.jsonl`

Filtering policy:

- Keep prompts with `0.2 <= reward_mean <= 0.8`.
- Require `parse_failure_rate <= 0.5`.
- Require `unique_answer_count >= 3`.
- Excluded prompts are labeled as `all_zero`, `all_one`, `parse_fail`, or `low_diversity`.

Integrity notes:

- Running `make rlvr-frontier-audit`, `make rlvr-frontier-smoke`, `make rlvr-gsm8k-scout`, or `make rlvr-gsm8k-audit` without `RUN_FRONTIER_AUDIT=1` must not sample rollouts or launch training.
- The audit does not modify reward semantics.
- The filtered output preserves the original RLVR record schema and validates through `make validate-data` when present.
- Real model audits should be treated as GPU inference jobs; cache and review audit outputs before launching GRPO.

## GSM8K RLVR Staging

Status: GSM8K conversion workflow is available for frontier audit with `Qwen/Qwen3-4B`; no GSM8K files are committed by default.

Conversion command:

```bash
make gsm8k-rlvr-data
```

Output files:

- `data/rlvr_prompts/gsm8k_train.jsonl`
- `data/rlvr_prompts/gsm8k_train_summary.json`

Split policy:

- Only the official GSM8K `train` split is converted for RLVR training, audit, and filtering.
- The converter refuses `test` split conversion for this workflow.
- Audit configs read only `data/rlvr_prompts/gsm8k_train.jsonl`.
- Official GSM8K test examples must remain out of training, rollout audit, frontier filtering, and GRPO train sets.

Answer parsing:

- The verifier answer is parsed from the final `####` field in each GSM8K answer.
- Commas and currency markers are removed from the verifier answer because the prompt asks the model not to use commas in boxed numeric answers.
- Prompts ask for exactly one final `\boxed{...}` answer and no reasoning, matching `math_boxed_v001`.

Pilot frontier audit configs are opt-in and should be treated as GPU inference jobs:

- `configs/rlvr/gsm8k_frontier_scout_thinking_false.yaml`
- `configs/rlvr/gsm8k_frontier_scout_thinking_true.yaml`
- `configs/rlvr/gsm8k_frontier_audit_thinking_false.yaml`
- `configs/rlvr/gsm8k_frontier_audit_thinking_true.yaml`

Scout audit settings:

- Model: `Qwen/Qwen3-4B`.
- Prompt count: `100` train prompts.
- Completions per prompt: `8`.
- Sampling: temperature `0.9`, top_p `0.95`, max new tokens `128`.
- Generation batch size: `8` completions per prompt.
- Audit outputs flush every `10` prompts so interrupted jobs still expose partial summaries.

Full pilot audit settings:

- Model: `Qwen/Qwen3-4B`.
- Prompt count: `300` train prompts.
- Completions per prompt: `16`.
- Sampling: temperature `0.9`, top_p `0.95`.
- Generation batch size: `8` completions per prompt.
- Audit outputs flush every `25` prompts.
- Frontier filter: `0.2 <= reward_mean <= 0.8`, `parse_failure_rate <= 0.2`, `unique_answer_count >= 3`.
- The audit is inference-only and must not execute trainer steps.

## OpenR1 SFT Staging

Status: OpenR1 math SFT staging is configured for a small real SFT run; no OpenR1 raw data is committed.

Config:

- `configs/sft/openr1_math_1k.yaml`

Source:

- Dataset: `open-r1/Mixture-of-Thoughts`
- Config: `math`
- Source split: `train`
- Staged output: `runs/sft/openr1_math_1k/data/sft_openr1_math_1k.jsonl`
- Manifest: `runs/sft/openr1_math_1k/data/sft_openr1_math_1k.manifest.json`

Split policy:

- Deterministically shuffle source records with the SFT config seed.
- Stage the first `1000` usable records as train.
- Stage the next `128` usable records as validation.
- Do not use OpenR1 records for eval prompts in this step.

Schema mapping:

- Source `messages` is normalized to project SFT `messages`.
- Source `source` is stored in `metadata.source` when present.
- Metadata uses `domain: math`, `difficulty: mixed`, and `license: source-dataset-card`.
- Records that do not normalize to user/assistant chat messages are skipped and counted in the manifest.

Integrity notes:

- `data/raw/` remains unchanged.
- Eval prompts, eval labels, reward semantics, and existing train/val/test fixtures are unchanged.
- The staged JSONL must pass the strict SFT validator before training.

## Synthetic E2E Math Fixture

Status: `synthetic-e2e-diverse-v2` is a toy fixture for pipeline, SFT smoke testing, and small RLVR signal checks, not a real math benchmark.

Files:

- `data/fixtures/e2e/sft_seed.jsonl`
- `data/fixtures/e2e/rlvr_seed.jsonl`
- `data/fixtures/e2e/eval_math.jsonl`

Split policy for the E2E SFT smoke path:

- Train: `80` SFT records.
- Validation: `10` SFT records.
- Heldout eval: `10` JSONL eval prompts.
- Intended ratio: `80/10/10` across train/validation/eval.
- RLVR fixture: `96` train, `12` validation, and `12` test prompts; the current E2E smoke stages the first `80` train prompts.
- RLVR pre-training gate: checks `32` rollout samples for zero parse failures and rejects near-saturated reward signal above configured reward ceilings.

Coverage:

- Easy: integer addition, subtraction, multiplication, and division.
- Medium: simple fractions, one-step and two-step linear equations, and combining like terms.
- Hard: equations with parentheses, equations with variables on both sides, and algebraic simplification with distribution.

Integrity notes:

- `data/raw/` remains unchanged.
- Eval prompts are held out from SFT train and validation staged data.
- The E2E pipeline fails fast if normalized prompts overlap between SFT train, SFT validation, RLVR train, and heldout eval where applicable.
- All answers are synthetic and final-only boxed for SFT/eval; RLVR verifier answers store the unboxed value used by `math_boxed_v001`.
